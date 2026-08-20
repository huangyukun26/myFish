from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from build_balanced_feature_holdout import merge_payloads, parse_paths
from train_text_to_visual_adapter import build_visual_prototypes, normalize, train_model


def load_class_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def genus(name: str) -> str:
    parts = str(name).split()
    return parts[0] if parts else ""


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % folds


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(part.strip()) for part in value.split(",") if part.strip()}, reverse=True)
    if 1.0 not in values:
        values.insert(0, 1.0)
    return values


@dataclass
class MetricAccumulator:
    rows: int = 0
    top1: int = 0
    top5: int = 0
    top20: int = 0
    changed: int = 0
    wins: int = 0
    losses: int = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        base_top1: torch.Tensor,
    ) -> None:
        prediction_top1 = predictions[:, 0]
        base_correct = base_top1 == targets
        prediction_correct = prediction_top1 == targets
        self.rows += targets.numel()
        self.top1 += int(prediction_correct.sum().item())
        self.top5 += int((predictions[:, :5] == targets[:, None]).any(dim=1).sum().item())
        self.top20 += int((predictions[:, :20] == targets[:, None]).any(dim=1).sum().item())
        self.changed += int((prediction_top1 != base_top1).sum().item())
        self.wins += int((prediction_correct & ~base_correct).sum().item())
        self.losses += int((~prediction_correct & base_correct).sum().item())

    def as_dict(self) -> dict[str, float | int]:
        denominator = max(1, self.rows)
        return {
            "rows": self.rows,
            "top1": self.top1 / denominator,
            "top5": self.top5 / denominator,
            "top20": self.top20 / denominator,
            "changed": self.changed,
            "wins": self.wins,
            "losses": self.losses,
            "net": self.wins - self.losses,
        }


def candidate_classes_for_fold(
    heldout_classes: list[str],
    distractor_classes: list[str],
    candidate_count: int,
    fold_seed: int,
) -> list[str]:
    heldout_set = set(heldout_classes)
    distractors = [name for name in distractor_classes if name not in heldout_set]
    needed = candidate_count - len(heldout_classes)
    if needed < 0:
        raise ValueError(
            f"Fold has {len(heldout_classes)} true classes, more than candidate_count={candidate_count}"
        )
    generator = torch.Generator().manual_seed(fold_seed)
    order = torch.randperm(len(distractors), generator=generator).tolist()
    selected = [distractors[idx] for idx in order[:needed]]
    if len(selected) != needed:
        raise RuntimeError("Not enough distractor classes to construct the requested candidate pool")
    return heldout_classes + selected


def evaluate_fold(
    *,
    image_payload: dict,
    text_classes: list[str],
    text_features: torch.Tensor,
    distractor_classes: list[str],
    heldout_classes: list[str],
    candidate_count: int,
    blend_original_values: list[float],
    min_count: int,
    hidden_dim: int,
    residual_scale: float,
    dropout: float,
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    contrastive_weight: float,
    seed: int,
    device: torch.device,
    max_queries_per_class: int,
) -> dict:
    heldout_set = set(heldout_classes)
    train_classes, target_prototypes, train_counts = build_visual_prototypes(
        image_payload,
        text_classes,
        exclude_classes=heldout_set,
        exclude_genera=set(),
        min_count=min_count,
    )
    text_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    train_indices = torch.tensor([text_to_idx[name] for name in train_classes], dtype=torch.long)
    torch.manual_seed(seed)
    model, losses = train_model(
        text_features[train_indices],
        target_prototypes,
        hidden_dim=hidden_dim,
        residual_scale=residual_scale,
        dropout=dropout,
        epochs=epochs,
        batch_size=train_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        contrastive_weight=contrastive_weight,
        device=device,
    )
    with torch.inference_mode():
        adapted_all = model(text_features.to(device)).cpu()

    candidates = candidate_classes_for_fold(
        heldout_classes,
        distractor_classes,
        candidate_count,
        seed,
    )
    candidate_to_idx = {name: idx for idx, name in enumerate(candidates)}
    candidate_text_indices = torch.tensor([text_to_idx[name] for name in candidates], dtype=torch.long)

    query_indices = [
        idx for idx, label in enumerate(image_payload["labels"]) if label in heldout_set
    ]
    if max_queries_per_class > 0:
        by_class: dict[str, list[int]] = {}
        for idx in query_indices:
            by_class.setdefault(image_payload["labels"][idx], []).append(idx)
        generator = torch.Generator().manual_seed(seed + 7919)
        capped_indices = []
        for class_name in sorted(by_class):
            indices = by_class[class_name]
            order = torch.randperm(len(indices), generator=generator).tolist()
            capped_indices.extend(indices[local_idx] for local_idx in order[:max_queries_per_class])
        query_indices = sorted(capped_indices)
    if not query_indices:
        raise RuntimeError("Fold has no query images")
    query_features = normalize(image_payload["features"][torch.tensor(query_indices)])
    query_targets = torch.tensor(
        [candidate_to_idx[image_payload["labels"][idx]] for idx in query_indices],
        dtype=torch.long,
    )

    candidate_matrices: dict[float, torch.Tensor] = {}
    original = text_features[candidate_text_indices]
    adapted = adapted_all[candidate_text_indices]
    for blend_original in blend_original_values:
        candidate_matrices[blend_original] = normalize(
            blend_original * original + (1.0 - blend_original) * adapted
        ).half().to(device)

    accumulators = {value: MetricAccumulator() for value in blend_original_values}
    with torch.inference_mode():
        for start in range(0, query_features.shape[0], eval_batch_size):
            query_batch = query_features[start : start + eval_batch_size].half().to(device)
            target_batch = query_targets[start : start + eval_batch_size].to(device)
            base_scores = query_batch @ candidate_matrices[1.0].T
            base_predictions = base_scores.topk(20, dim=1).indices
            base_top1 = base_predictions[:, 0]
            for blend_original in blend_original_values:
                if blend_original == 1.0:
                    predictions = base_predictions
                else:
                    scores = query_batch @ candidate_matrices[blend_original].T
                    predictions = scores.topk(20, dim=1).indices
                accumulators[blend_original].update(predictions, target_batch, base_top1)

    return {
        "heldout_classes": len(heldout_classes),
        "query_rows": len(query_indices),
        "max_queries_per_class": max_queries_per_class,
        "candidate_classes": len(candidates),
        "adapter_train_classes": len(train_classes),
        "adapter_train_min_count": int(train_counts.min().item()) if train_counts.numel() else 0,
        "adapter_train_median_count": float(train_counts.float().median().item()) if train_counts.numel() else 0,
        "final_loss": losses[-1] if losses else None,
        "metrics": {str(value): accumulator.as_dict() for value, accumulator in accumulators.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", required=True, help="Comma-separated labeled image caches")
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--distractor-classes", type=Path, required=True)
    parser.add_argument("--distractor-pool", choices=["provided", "all_text"], default="provided")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-modes", default="species,genus")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=11598)
    parser.add_argument("--blend-original", default="1.0,0.9,0.75,0.5,0.0")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-queries-per-class", type=int, default=0)
    args = parser.parse_args()

    modes = [part.strip() for part in args.holdout_modes.split(",") if part.strip()]
    invalid_modes = sorted(set(modes) - {"species", "genus"})
    if invalid_modes:
        raise ValueError(f"Unsupported holdout modes: {invalid_modes}")
    blend_values = parse_float_list(args.blend_original)
    image_payload = merge_payloads(parse_paths(args.image_features))
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    text_classes = list(text_payload["classes"])
    text_features = normalize(text_payload["features"])
    distractor_classes = load_class_list(args.distractor_classes)
    missing_text = sorted(
        (set(image_payload["labels"]) | set(distractor_classes)) - set(text_classes)
    )
    if missing_text:
        raise RuntimeError(f"{len(missing_text)} classes lack text features; first={missing_text[:5]}")

    labeled_classes = sorted(set(image_payload["labels"]))
    if args.distractor_pool == "all_text":
        distractor_classes = list(text_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, list[dict]] = {}
    for mode in modes:
        fold_rows = []
        for fold in range(args.folds):
            heldout_classes = [
                name
                for name in labeled_classes
                if stable_fold(name if mode == "species" else genus(name), args.folds, args.seed) == fold
            ]
            row = evaluate_fold(
                image_payload=image_payload,
                text_classes=text_classes,
                text_features=text_features,
                distractor_classes=distractor_classes,
                heldout_classes=heldout_classes,
                candidate_count=args.candidate_count,
                blend_original_values=blend_values,
                min_count=args.min_count,
                hidden_dim=args.hidden_dim,
                residual_scale=args.residual_scale,
                dropout=args.dropout,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                contrastive_weight=args.contrastive_weight,
                seed=args.seed + 1000 * (modes.index(mode) + 1) + fold,
                device=device,
                max_queries_per_class=args.max_queries_per_class,
            )
            row["fold"] = fold
            print(json.dumps({"mode": mode, **row}, ensure_ascii=False), flush=True)
            fold_rows.append(row)
        results[mode] = fold_rows

    aggregate: dict[str, dict] = {}
    for mode, rows in results.items():
        mode_summary = {}
        for blend_original in blend_values:
            key = str(blend_original)
            metrics = [row["metrics"][key] for row in rows]
            total_rows = sum(int(item["rows"]) for item in metrics)
            top1_correct = sum(float(item["top1"]) * int(item["rows"]) for item in metrics)
            top5_correct = sum(float(item["top5"]) * int(item["rows"]) for item in metrics)
            top20_correct = sum(float(item["top20"]) * int(item["rows"]) for item in metrics)
            mode_summary[key] = {
                "rows": total_rows,
                "top1": top1_correct / max(1, total_rows),
                "top5": top5_correct / max(1, total_rows),
                "top20": top20_correct / max(1, total_rows),
                "changed": sum(int(item["changed"]) for item in metrics),
                "wins": sum(int(item["wins"]) for item in metrics),
                "losses": sum(int(item["losses"]) for item in metrics),
                "net": sum(int(item["net"]) for item in metrics),
                "worst_fold_top1": min(float(item["top1"]) for item in metrics),
                "worst_fold_net": min(int(item["net"]) for item in metrics),
            }
        aggregate[mode] = mode_summary

    output = {
        "config": {
            **vars(args),
            "text_features": str(args.text_features),
            "distractor_classes": str(args.distractor_classes),
            "out": str(args.out),
        },
        "image_rows": len(image_payload["labels"]),
        "labeled_classes": len(labeled_classes),
        "text_classes": len(text_classes),
        "device": str(device),
        "results": results,
        "aggregate": aggregate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()

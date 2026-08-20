from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def row_standardize(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def build_prototypes(payload: dict) -> torch.Tensor:
    features = F.normalize(payload["features"].float(), dim=1)
    class_ids = payload["class_ids"].long()
    class_count = len(payload["classes"])
    sums = torch.zeros(class_count, features.shape[1])
    counts = torch.zeros(class_count)
    sums.index_add_(0, class_ids, features)
    counts.index_add_(0, class_ids, torch.ones_like(class_ids, dtype=torch.float32))
    if bool(counts.eq(0).any()):
        raise RuntimeError(f"{int(counts.eq(0).sum())} classes have no auxiliary support")
    return F.normalize(sums / counts[:, None], dim=1)


def genus_folds(labels: list[str], fold_count: int) -> torch.Tensor:
    values = []
    for label in labels:
        genus = label.split(maxsplit=1)[0]
        digest = hashlib.sha1(genus.encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:4], "little") % fold_count)
    return torch.tensor(values, dtype=torch.long)


def metrics(prediction: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> dict:
    correct = prediction.eq(target)
    base_correct = baseline.eq(target)
    return {
        "rows": int(target.numel()),
        "top1": float(correct.float().mean().item()),
        "changed": int(prediction.ne(baseline).sum().item()),
        "wins": int((correct & ~base_correct).sum().item()),
        "losses": int((~correct & base_correct).sum().item()),
        "net": int((correct.sum() - base_correct.sum()).item()),
    }


def masked_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    if not bool(mask.any()):
        return {"rows": 0, "top1": 0.0, "changed": 0, "wins": 0, "losses": 0, "net": 0}
    return metrics(prediction[mask], target[mask], baseline[mask])


def read_aspects(path: Path, image_ids: list[str]) -> torch.Tensor:
    by_id = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_id[row["image_id"]] = float(row["aspect_ratio"])
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} rows lack crop metadata; first={missing[:5]}")
    return torch.tensor([by_id[image_id] for image_id in image_ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--aux-support", type=Path, required=True)
    parser.add_argument("--aux-query", type=Path, required=True)
    parser.add_argument("--crop-metadata", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-oof-logits", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--weights", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0"
    )
    args = parser.parse_args()

    base = load(args.base_logits)
    support = load(args.aux_support)
    query = load(args.aux_query)
    classes = list(base["classes"])
    if classes != list(support["classes"]) or classes != list(query["classes"]):
        raise RuntimeError("Base and auxiliary class orders differ")
    if list(base["image_ids"]) != list(query["image_ids"]):
        raise RuntimeError("Base logits and auxiliary query row orders differ")

    target = base["class_ids"].long()
    base_scores = row_standardize(base["logits"])
    prototype = build_prototypes(support)
    query_features = F.normalize(query["features"].float(), dim=1)
    aux_scores = row_standardize(query_features @ prototype.T)
    base_prediction = base_scores.argmax(dim=1)
    aux_prediction = aux_scores.argmax(dim=1)

    weights = [float(value) for value in args.weights.split(",") if value.strip()]
    predictions = [(base_scores + weight * aux_scores).argmax(dim=1) for weight in weights]
    grid = [
        {"aux_weight": weight, **metrics(prediction, target, base_prediction)}
        for weight, prediction in zip(weights, predictions)
    ]
    folds = genus_folds(list(base["labels"]), args.folds)
    oof_prediction = torch.empty_like(target)
    oof_scores = torch.empty_like(base_scores)
    selected_weights = []
    fold_rows = []
    for fold in range(args.folds):
        heldout = folds.eq(fold)
        train = ~heldout
        train_accuracies = [
            float(prediction[train].eq(target[train]).float().mean().item())
            for prediction in predictions
        ]
        selected = max(
            range(len(weights)), key=lambda idx: (train_accuracies[idx], -weights[idx])
        )
        selected_weight = weights[selected]
        selected_weights.append(selected_weight)
        oof_prediction[heldout] = predictions[selected][heldout]
        oof_scores[heldout] = base_scores[heldout] + selected_weight * aux_scores[heldout]
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(heldout.sum().item()),
                "selected_aux_weight": selected_weight,
                "train_top1": train_accuracies[selected],
                "heldout_top1": float(
                    predictions[selected][heldout].eq(target[heldout]).float().mean().item()
                ),
            }
        )

    best = max(grid, key=lambda row: (float(row["top1"]), -float(row["aux_weight"])))
    output: dict[str, object] = {
        "rows": len(target),
        "classes": len(classes),
        "base": metrics(base_prediction, target, base_prediction),
        "auxiliary_standalone": metrics(aux_prediction, target, base_prediction),
        "best_global": best,
        "genus_grouped_oof": {
            **metrics(oof_prediction, target, base_prediction),
            "selected_weights": selected_weights,
            "folds": fold_rows,
        },
        "weight_grid": grid,
    }
    if args.crop_metadata is not None:
        aspects = read_aspects(args.crop_metadata, list(base["image_ids"]))
        buckets = {
            "aspect_lt1.5": aspects.lt(1.5),
            "aspect_1.5_2.25": aspects.ge(1.5) & aspects.lt(2.25),
            "aspect_ge2.25": aspects.ge(2.25),
        }
        output["aspect_buckets"] = {
            name: masked_metrics(oof_prediction, target, base_prediction, mask)
            for name, mask in buckets.items()
        }
        selective_rows = {}
        for threshold in (1.5, 2.25):
            mask = aspects.ge(threshold)
            selective = base_prediction.clone()
            selective[mask] = oof_prediction[mask]
            selective_rows[f"apply_oof_aspect_ge{threshold}"] = metrics(
                selective, target, base_prediction
            )
        output["aspect_selective"] = selective_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_oof_logits is not None:
        args.save_oof_logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **{key: base[key] for key in ("class_ids", "labels", "image_ids", "classes")},
                "logits": oof_scores,
                "source_base_logits": str(args.base_logits),
                "source_aux_support": str(args.aux_support),
                "source_aux_query": str(args.aux_query),
                "selected_weights": selected_weights,
            },
            args.save_oof_logits,
        )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

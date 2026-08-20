from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_feature_prototypes import build_prototypes, prepare_train_classes


BUCKETS = (
    ("count_2", 0, 2),
    ("count_3_5", 3, 5),
    ("count_6_10", 6, 10),
    ("count_11_20", 11, 20),
    ("count_21_50", 21, 50),
    ("count_51_plus", 51, 10**9),
)


def normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1)


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def row_standardize(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def align_text_features(payload: dict, classes: list[str]) -> torch.Tensor:
    text_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in classes if name not in text_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} classes lack text features; first={missing[:5]}")
    indices = torch.tensor([text_to_idx[name] for name in classes], dtype=torch.long)
    return normalize(payload["features"][indices])


def reorder_payload_like(payload: dict, target_image_ids: list[str], split: str) -> dict:
    source_image_ids = list(payload["image_ids"])
    if source_image_ids == target_image_ids:
        return payload
    if len(set(source_image_ids)) != len(source_image_ids):
        raise RuntimeError(f"Alignment {split} cache contains duplicate image IDs")
    source_index = {image_id: idx for idx, image_id in enumerate(source_image_ids)}
    missing = [image_id for image_id in target_image_ids if image_id not in source_index]
    if missing:
        raise RuntimeError(
            f"Alignment {split} cache lacks {len(missing)} classifier rows; first={missing[:5]}"
        )
    indices = torch.tensor([source_index[image_id] for image_id in target_image_ids], dtype=torch.long)
    reordered = dict(payload)
    reordered["image_ids"] = list(target_image_ids)
    for key in ("features", "class_ids"):
        value = payload.get(key)
        if value is not None:
            reordered[key] = value[indices]
    labels = payload.get("labels")
    if labels is not None:
        reordered["labels"] = [labels[idx] for idx in indices.tolist()]
    return reordered


def score_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    full_counts: torch.Tensor,
    base_top1: torch.Tensor | None = None,
) -> dict:
    top = scores.topk(min(20, scores.shape[1]), dim=1).indices
    top1 = top[:, 0]
    correct = top1 == targets
    result: dict[str, object] = {
        "rows": targets.numel(),
        "top1": float(correct.float().mean().item()),
        "top5": float((top[:, :5] == targets[:, None]).any(dim=1).float().mean().item()),
        "top20": float((top == targets[:, None]).any(dim=1).float().mean().item()),
    }
    if base_top1 is not None:
        base_correct = base_top1 == targets
        result.update(
            {
                "changed": int((top1 != base_top1).sum().item()),
                "wins": int((correct & ~base_correct).sum().item()),
                "losses": int((~correct & base_correct).sum().item()),
            }
        )
        result["net"] = int(result["wins"]) - int(result["losses"])

    sample_counts = full_counts[targets]
    bucket_metrics = {}
    for name, low, high in BUCKETS:
        mask = (sample_counts >= low) & (sample_counts <= high)
        if not bool(mask.any()):
            continue
        bucket_correct = correct[mask]
        bucket_row: dict[str, float | int] = {
            "rows": int(mask.sum().item()),
            "top1": float(bucket_correct.float().mean().item()),
        }
        if base_top1 is not None:
            bucket_base_correct = base_top1[mask] == targets[mask]
            bucket_row["wins"] = int((bucket_correct & ~bucket_base_correct).sum().item())
            bucket_row["losses"] = int((~bucket_correct & bucket_base_correct).sum().item())
            bucket_row["net"] = int(bucket_row["wins"]) - int(bucket_row["losses"])
        bucket_metrics[name] = bucket_row
    result["frequency_buckets"] = bucket_metrics
    result["worst_bucket_top1"] = min(float(row["top1"]) for row in bucket_metrics.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--mlp-logits", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--alignment-train-cache", type=Path, default=None)
    parser.add_argument("--alignment-val-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prototype-weights", default="0,0.1,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--text-weights", default="0,0.05,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--save-selected-logits", type=Path, default=None)
    parser.add_argument("--selected-prototype-weight", type=float, default=0.1)
    parser.add_argument("--selected-text-weight", type=float, default=0.75)
    args = parser.parse_args()

    train_payload = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_payload = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    alignment_train_payload = (
        torch.load(args.alignment_train_cache, map_location="cpu", weights_only=False)
        if args.alignment_train_cache is not None
        else train_payload
    )
    alignment_val_payload = (
        torch.load(args.alignment_val_cache, map_location="cpu", weights_only=False)
        if args.alignment_val_cache is not None
        else val_payload
    )
    mlp_payload = torch.load(args.mlp_logits, map_location="cpu", weights_only=False)
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    alignment_train_payload = reorder_payload_like(
        alignment_train_payload, list(train_payload["image_ids"]), "train"
    )
    alignment_val_payload = reorder_payload_like(
        alignment_val_payload, list(val_payload["image_ids"]), "validation"
    )
    _train_ids, classes = prepare_train_classes(train_payload)
    _alignment_train_ids, alignment_classes = prepare_train_classes(alignment_train_payload)
    if alignment_classes != classes:
        raise RuntimeError("Alignment and classifier cache class orders differ")
    if classes != list(mlp_payload["classes"]):
        raise RuntimeError("MLP and train cache class orders differ")
    if list(val_payload["image_ids"]) != list(mlp_payload["image_ids"]):
        raise RuntimeError("MLP logits and validation cache row orders differ")
    if list(train_payload["image_ids"]) != list(alignment_train_payload["image_ids"]):
        raise RuntimeError("Alignment and classifier train cache row orders differ")
    if list(val_payload["image_ids"]) != list(alignment_val_payload["image_ids"]):
        raise RuntimeError("Alignment and classifier validation cache row orders differ")

    query = normalize(alignment_val_payload["features"])
    targets = val_payload.get("class_ids", mlp_payload["class_ids"]).long()
    full_counts = val_payload.get("full_class_counts")
    if full_counts is None:
        train_counts = torch.bincount(train_payload["class_ids"].long(), minlength=len(classes))
        val_counts = torch.bincount(targets, minlength=len(classes))
        full_counts = train_counts + val_counts
    full_counts = full_counts.long()

    prototypes, _counts = build_prototypes(alignment_train_payload)
    text_features = align_text_features(text_payload, classes)
    mlp_scores = row_standardize(mlp_payload["logits"])
    prototype_scores = row_standardize(query @ prototypes.T)
    text_scores = row_standardize(query @ text_features.T)
    base_top1 = mlp_scores.argmax(dim=1)

    standalone = {
        "mlp": score_metrics(mlp_scores, targets, full_counts),
        "prototype": score_metrics(prototype_scores, targets, full_counts, base_top1),
        "text": score_metrics(text_scores, targets, full_counts, base_top1),
    }
    fusion_rows = []
    for prototype_weight in parse_floats(args.prototype_weights):
        for text_weight in parse_floats(args.text_weights):
            if prototype_weight == 0 and text_weight == 0:
                continue
            fused = mlp_scores + prototype_weight * prototype_scores + text_weight * text_scores
            metrics = score_metrics(fused, targets, full_counts, base_top1)
            fusion_rows.append(
                {
                    "prototype_weight": prototype_weight,
                    "text_weight": text_weight,
                    **metrics,
                }
            )
    fusion_rows.sort(
        key=lambda row: (
            float(row["top1"]),
            float(row["worst_bucket_top1"]),
            int(row["net"]),
        ),
        reverse=True,
    )

    prototype_top1 = prototype_scores.argmax(dim=1)
    text_top1 = text_scores.argmax(dim=1)
    base_correct = base_top1 == targets
    complementarity = {
        "base_wrong_prototype_correct": int((~base_correct & (prototype_top1 == targets)).sum().item()),
        "base_wrong_text_correct": int((~base_correct & (text_top1 == targets)).sum().item()),
        "base_wrong_either_correct": int(
            (~base_correct & ((prototype_top1 == targets) | (text_top1 == targets))).sum().item()
        ),
        "all_three_agree": int(
            ((base_top1 == prototype_top1) & (base_top1 == text_top1)).sum().item()
        ),
    }
    output = {
        "train_cache": str(args.train_cache),
        "val_cache": str(args.val_cache),
        "mlp_logits": str(args.mlp_logits),
        "text_features": str(args.text_features),
        "alignment_train_cache": str(args.alignment_train_cache or args.train_cache),
        "alignment_val_cache": str(args.alignment_val_cache or args.val_cache),
        "rows": targets.numel(),
        "classes": len(classes),
        "standalone": standalone,
        "complementarity": complementarity,
        "best_fusions": fusion_rows[:20],
        "all_fusions": fusion_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_selected_logits is not None:
        selected_logits = (
            mlp_scores
            + args.selected_prototype_weight * prototype_scores
            + args.selected_text_weight * text_scores
        )
        args.save_selected_logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": selected_logits,
                "class_ids": targets,
                "labels": list(val_payload["labels"]),
                "image_ids": list(val_payload["image_ids"]),
                "classes": classes,
                "full_class_counts": full_counts,
                "source_mlp_logits": str(args.mlp_logits),
                "prototype_weight": args.selected_prototype_weight,
                "text_weight": args.selected_text_weight,
            },
            args.save_selected_logits,
        )
    print(json.dumps({**{k: output[k] for k in ("rows", "classes", "standalone", "complementarity")}, "best_fusions": fusion_rows[:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

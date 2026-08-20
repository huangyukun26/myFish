from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_pair(train_path: Path, query_path: Path) -> tuple[dict, dict]:
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    query = torch.load(query_path, map_location="cpu", weights_only=False)
    return train, query


def prototypes(payload: dict) -> torch.Tensor:
    features = F.normalize(payload["features"].float(), dim=1)
    class_ids = payload["class_ids"].long()
    class_count = len(payload["classes"])
    sums = torch.zeros(class_count, features.shape[1])
    counts = torch.zeros(class_count)
    sums.index_add_(0, class_ids, features)
    counts.index_add_(0, class_ids, torch.ones_like(class_ids, dtype=torch.float32))
    if (counts == 0).any():
        raise RuntimeError(f"{int((counts == 0).sum())} classes have no support rows")
    return F.normalize(sums / counts[:, None], dim=1)


def cosine_logits(train: dict, query: dict, device: torch.device) -> torch.Tensor:
    support = prototypes(train).to(device)
    features = F.normalize(query["features"].float(), dim=1).to(device)
    return features @ support.T


def row_standardize(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True).clamp_min(1e-6)


def prediction_metrics(pred: torch.Tensor, target: torch.Tensor, base_pred: torch.Tensor) -> dict:
    correct = pred.eq(target)
    base_correct = base_pred.eq(target)
    return {
        "top1": float(correct.float().mean().item()),
        "changed": int(pred.ne(base_pred).sum().item()),
        "wins": int((correct & ~base_correct).sum().item()),
        "losses": int((~correct & base_correct).sum().item()),
        "net": int((correct.sum() - base_correct.sum()).item()),
    }


def load_class_counts(path: Path, class_count: int) -> torch.Tensor:
    counts = torch.zeros(class_count, dtype=torch.long)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            class_id = int(row["class_id"])
            if 0 <= class_id < class_count:
                counts[class_id] += 1
    if (counts == 0).any():
        raise RuntimeError(f"{int((counts == 0).sum())} classes are absent from {path}")
    return counts


def frequency_bucket_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_pred: torch.Tensor,
    class_counts: torch.Tensor,
) -> dict:
    sample_counts = class_counts.to(target.device)[target]
    buckets = [
        ("count_2", sample_counts.eq(2)),
        ("count_3_5", sample_counts.ge(3) & sample_counts.le(5)),
        ("count_6_10", sample_counts.ge(6) & sample_counts.le(10)),
        ("count_11_20", sample_counts.ge(11) & sample_counts.le(20)),
        ("count_21_50", sample_counts.ge(21) & sample_counts.le(50)),
        ("count_51_plus", sample_counts.ge(51)),
    ]
    result = {}
    for name, mask in buckets:
        rows = int(mask.sum().item())
        if rows == 0:
            continue
        metrics = prediction_metrics(pred[mask], target[mask], base_pred[mask])
        result[name] = {"rows": rows, **metrics}
    return result


def topk_metrics(scores: torch.Tensor, target: torch.Tensor) -> dict:
    indices = scores.topk(min(20, scores.shape[1]), dim=1).indices
    return {
        "top1": float(indices[:, 0].eq(target).float().mean().item()),
        "top5": float(indices[:, :5].eq(target[:, None]).any(dim=1).float().mean().item()),
        "top20": float(indices.eq(target[:, None]).any(dim=1).float().mean().item()),
    }


def genus_folds(labels: list[str], fold_count: int) -> torch.Tensor:
    folds = []
    for label in labels:
        genus = label.split(maxsplit=1)[0]
        digest = hashlib.sha1(genus.encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:4], "little") % fold_count)
    return torch.tensor(folds, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-query", type=Path, required=True)
    parser.add_argument("--alt-train", type=Path, required=True)
    parser.add_argument("--alt-query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--count-manifest",
        type=Path,
        default=None,
        help="Optional full labeled manifest used to report metrics by class-frequency bucket.",
    )
    args = parser.parse_args()

    base_train, base_query = load_pair(args.base_train, args.base_query)
    alt_train, alt_query = load_pair(args.alt_train, args.alt_query)
    if base_query["image_ids"] != alt_query["image_ids"]:
        raise RuntimeError("Base and alternate query image order differs")
    if base_train["classes"] != alt_train["classes"]:
        raise RuntimeError("Base and alternate class order differs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = base_query["class_ids"].long().to(device)
    base_scores = cosine_logits(base_train, base_query, device)
    alt_scores = cosine_logits(alt_train, alt_query, device)
    base_z = row_standardize(base_scores)
    alt_z = row_standardize(alt_scores)
    base_pred = base_scores.argmax(dim=1)
    alt_pred = alt_scores.argmax(dim=1)
    class_counts = (
        load_class_counts(args.count_manifest, len(base_train["classes"]))
        if args.count_manifest is not None
        else None
    )

    weights = [round(value / 10, 1) for value in range(11)]
    predictions: list[torch.Tensor] = []
    fusion_rows = []
    for weight in weights:
        pred = ((1.0 - weight) * base_z + weight * alt_z).argmax(dim=1)
        predictions.append(pred)
        row = {"alt_weight": weight, **prediction_metrics(pred, target, base_pred)}
        if class_counts is not None:
            row["frequency_buckets"] = frequency_bucket_metrics(pred, target, base_pred, class_counts)
        fusion_rows.append(row)

    folds = genus_folds(list(base_query["labels"]), args.folds).to(device)
    oof_pred = torch.empty_like(base_pred)
    fold_rows = []
    for fold in range(args.folds):
        heldout = folds.eq(fold)
        train_mask = ~heldout
        train_accuracies = [float(pred[train_mask].eq(target[train_mask]).float().mean()) for pred in predictions]
        best_index = max(range(len(weights)), key=lambda idx: (train_accuracies[idx], -abs(weights[idx] - 0.5)))
        oof_pred[heldout] = predictions[best_index][heldout]
        fold_row = {
            "fold": fold,
            "rows": int(heldout.sum().item()),
            "selected_alt_weight": weights[best_index],
            "train_top1": train_accuracies[best_index],
            "heldout_top1": float(predictions[best_index][heldout].eq(target[heldout]).float().mean().item()),
        }
        if class_counts is not None:
            heldout_counts = class_counts.to(device)[target]
            count2_heldout = heldout & heldout_counts.eq(2)
            fold_row["count_2"] = {
                "rows": int(count2_heldout.sum().item()),
                **prediction_metrics(
                    predictions[best_index][count2_heldout],
                    target[count2_heldout],
                    base_pred[count2_heldout],
                ),
            }
        fold_rows.append(fold_row)

    best_row = max(fusion_rows, key=lambda row: (row["top1"], -abs(row["alt_weight"] - 0.5)))
    fixed_index = weights.index(0.5)
    fixed_scores = 0.5 * base_z + 0.5 * alt_z
    best_index = weights.index(best_row["alt_weight"])
    best_scores = (1.0 - weights[best_index]) * base_z + weights[best_index] * alt_z
    base_correct = base_pred.eq(target)
    alt_correct = alt_pred.eq(target)
    base_standalone = topk_metrics(base_scores, target)
    alternate_standalone = {
        **topk_metrics(alt_scores, target),
        **prediction_metrics(alt_pred, target, base_pred),
    }
    if class_counts is not None:
        base_standalone["frequency_buckets"] = frequency_bucket_metrics(
            base_pred, target, base_pred, class_counts
        )
        alternate_standalone["frequency_buckets"] = frequency_bucket_metrics(
            alt_pred, target, base_pred, class_counts
        )

    oof_summary = {
        **prediction_metrics(oof_pred, target, base_pred),
        "folds": fold_rows,
    }
    if class_counts is not None:
        oof_summary["frequency_buckets"] = frequency_bucket_metrics(
            oof_pred, target, base_pred, class_counts
        )

    summary = {
        "rows": len(base_query["image_ids"]),
        "classes": len(base_train["classes"]),
        "device": str(device),
        "count_manifest": str(args.count_manifest) if args.count_manifest is not None else None,
        "standalone": {
            "base": base_standalone,
            "alternate": alternate_standalone,
        },
        "complementarity": {
            "both_correct": int((base_correct & alt_correct).sum().item()),
            "base_only_correct": int((base_correct & ~alt_correct).sum().item()),
            "alternate_only_correct": int((~base_correct & alt_correct).sum().item()),
            "either_correct": int((base_correct | alt_correct).sum().item()),
            "either_correct_top1": float((base_correct | alt_correct).float().mean().item()),
            "prediction_agreement": int(base_pred.eq(alt_pred).sum().item()),
        },
        "fixed_half": {
            **topk_metrics(fixed_scores, target),
            **fusion_rows[fixed_index],
        },
        "gate_best": {
            **topk_metrics(best_scores, target),
            **best_row,
        },
        "genus_grouped_oof": oof_summary,
        "weight_grid": fusion_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

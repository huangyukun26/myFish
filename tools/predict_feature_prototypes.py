from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def classes_from_ids(labels: list[str], class_ids: torch.Tensor) -> list[str]:
    valid = class_ids[class_ids >= 0]
    if valid.numel() == 0:
        return []
    classes = [""] * (int(valid.max().item()) + 1)
    for label, class_id in zip(labels, class_ids.tolist()):
        if class_id >= 0 and label and not classes[class_id]:
            classes[class_id] = label
    return classes


def train_class_ids(payload: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    labels = list(payload.get("labels", []))
    if "class_ids" in payload:
        class_ids = payload["class_ids"].long()
        classes = list(payload.get("classes") or classes_from_ids(labels, class_ids))
        return class_ids, classes
    classes = sorted({label for label in labels if label})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    return torch.tensor([class_to_idx.get(label, -1) for label in labels], dtype=torch.long), classes


def query_class_ids(payload: dict[str, Any], classes: list[str]) -> torch.Tensor:
    if "class_ids" in payload:
        return payload["class_ids"].long()
    class_to_idx = {label: idx for idx, label in enumerate(classes) if label}
    return torch.tensor(
        [class_to_idx.get(label, -1) for label in payload.get("labels", [])],
        dtype=torch.long,
    )


def build_prototypes(train_payload: dict[str, Any]) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    features = normalize(train_payload["features"])
    class_ids, classes = train_class_ids(train_payload)
    sums = torch.zeros((len(classes), features.shape[1]), dtype=torch.float32)
    counts = torch.zeros(len(classes), dtype=torch.long)
    valid = (class_ids >= 0) & (class_ids < len(classes))
    for feature, class_id in zip(features[valid], class_ids[valid]):
        idx = int(class_id.item())
        sums[idx] += feature
        counts[idx] += 1
    prototypes = normalize(sums / counts[:, None].clamp_min(1).float())
    return prototypes, classes, counts


def write_predictions(
    *,
    path: Path,
    image_ids: list[str],
    labels: list[str],
    classes: list[str],
    top_indices: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fields = ["image_id", "prediction"]
        if any(labels):
            fields.append("label")
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row_idx, image_id in enumerate(image_ids):
            pred = classes[int(top_indices[row_idx, 0].item())]
            row = {"image_id": image_id, "prediction": pred}
            if any(labels):
                row["label"] = labels[row_idx]
            writer.writerow(row)


def write_topk(
    *,
    path: Path,
    image_ids: list[str],
    labels: list[str],
    classes: list[str],
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "top_classes", "top_scores", "margin_top1_top2"]
    if any(labels):
        fields.append("true_label")
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row_idx, image_id in enumerate(image_ids):
            indices = top_indices[row_idx].tolist()
            scores = top_scores[row_idx].tolist()
            margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
            row = {
                "image_id": image_id,
                "top_classes": json.dumps([classes[int(idx)] for idx in indices], ensure_ascii=False),
                "top_scores": json.dumps([float(score) for score in scores], ensure_ascii=False),
                "margin_top1_top2": float(margin),
            }
            if any(labels):
                row["true_label"] = labels[row_idx]
            writer.writerow(row)


def metrics(logits: torch.Tensor, class_ids: torch.Tensor, topk: int) -> dict[str, Any]:
    valid = (class_ids >= 0) & (class_ids < logits.shape[1])
    if not bool(valid.any()):
        return {}
    logits = logits[valid]
    labels = class_ids[valid]
    top_indices = logits.topk(min(topk, logits.shape[1]), dim=1).indices
    return {
        "known": int(labels.numel()),
        "top1": float((top_indices[:, 0] == labels).float().mean().item()),
        "top5": float((top_indices[:, : min(5, top_indices.shape[1])] == labels[:, None]).any(dim=1).float().mean().item()),
        f"top{topk}": float((top_indices == labels[:, None]).any(dim=1).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    prototypes, classes, counts = build_prototypes(train_payload)
    query_features = normalize(query_payload["features"])
    labels = list(query_payload.get("labels", [""] * len(query_payload["image_ids"])))
    query_ids = query_class_ids(query_payload, classes)
    top_scores_parts = []
    top_indices_parts = []
    metric_logits_parts = []
    for start in range(0, query_features.shape[0], args.batch_size):
        chunk = query_features[start : start + args.batch_size]
        logits = chunk @ prototypes.T
        scores, indices = logits.topk(min(args.topk, logits.shape[1]), dim=1)
        top_scores_parts.append(scores.cpu())
        top_indices_parts.append(indices.cpu())
        if labels and any(labels):
            metric_logits_parts.append(logits.cpu())
    top_scores = torch.cat(top_scores_parts, dim=0)
    top_indices = torch.cat(top_indices_parts, dim=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        path=args.out_dir / "predictions.csv",
        image_ids=list(query_payload["image_ids"]),
        labels=labels,
        classes=classes,
        top_indices=top_indices,
    )
    write_topk(
        path=args.out_dir / "topk.csv",
        image_ids=list(query_payload["image_ids"]),
        labels=labels,
        classes=classes,
        top_indices=top_indices,
        top_scores=top_scores,
    )
    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "query_rows": len(query_payload["image_ids"]),
        "classes_with_samples": int((counts > 0).sum().item()),
        "topk": args.topk,
        "predictions_csv": str(args.out_dir / "predictions.csv"),
        "topk_csv": str(args.out_dir / "topk.csv"),
    }
    if metric_logits_parts:
        summary.update(metrics(torch.cat(metric_logits_parts, dim=0), query_ids, args.topk))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


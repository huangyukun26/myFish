from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def prepare_train_classes(train_payload: dict) -> tuple[torch.Tensor, list[str]]:
    labels = list(train_payload.get("labels", []))
    if "class_ids" in train_payload:
        class_ids = train_payload["class_ids"].long()
        classes = list(train_payload.get("classes") or classes_from_ids(labels, class_ids))
        return class_ids, classes
    classes = sorted({label for label in labels if label})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    class_ids = torch.tensor([class_to_idx.get(label, -1) for label in labels], dtype=torch.long)
    return class_ids, classes


def query_class_ids(query_payload: dict, classes: list[str]) -> torch.Tensor:
    if "class_ids" in query_payload:
        return query_payload["class_ids"].long()
    class_to_idx = {label: idx for idx, label in enumerate(classes) if label}
    return torch.tensor(
        [class_to_idx.get(label, -1) for label in query_payload.get("labels", [])],
        dtype=torch.long,
    )


def build_prototypes(train_payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    features = normalize(train_payload["features"])
    class_ids, classes = prepare_train_classes(train_payload)
    class_count = len(classes)
    sums = torch.zeros((class_count, features.shape[1]), dtype=torch.float32)
    counts = torch.zeros(class_count, dtype=torch.long)
    valid = (class_ids >= 0) & (class_ids < class_count)
    for feature, class_id in zip(features[valid], class_ids[valid]):
        idx = int(class_id.item())
        sums[idx] += feature
        counts[idx] += 1
    prototypes = normalize(sums / counts[:, None].clamp_min(1).float())
    return prototypes, counts


def metrics(logits: torch.Tensor, class_ids: torch.Tensor) -> dict:
    valid = (class_ids >= 0) & (class_ids < logits.shape[1])
    logits = logits[valid]
    labels = class_ids[valid]
    if labels.numel() == 0:
        return {}
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
    top1 = top5[:, 0]
    return {
        "known": int(labels.numel()),
        "top1": float((top1 == labels).float().mean().item()),
        "top5": float((top5 == labels[:, None]).any(dim=1).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    prototypes, counts = build_prototypes(train_payload)
    _train_ids, classes = prepare_train_classes(train_payload)
    query_features = normalize(query_payload["features"])
    logits = query_features @ prototypes.T
    query_ids = query_class_ids(query_payload, classes)
    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "train_rows": len(train_payload["image_ids"]),
        "query_rows": len(query_payload["image_ids"]),
        "classes_with_samples": int((counts > 0).sum().item()),
        **metrics(logits, query_ids),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

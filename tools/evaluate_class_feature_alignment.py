from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


BUCKETS = (
    ("count_2", 0, 2),
    ("count_3_5", 3, 5),
    ("count_6_10", 6, 10),
    ("count_11_20", 11, 20),
    ("count_21_50", 21, 50),
    ("count_51_plus", 51, 10**9),
)


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [name for name, _idx in sorted(payload.items(), key=lambda item: int(item[1]))]
    return list(payload)


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise ValueError(f"Expected name=path, got {value}")
    return name.strip(), Path(path.strip())


def align_features(payload: dict[str, Any], classes: list[str]) -> torch.Tensor:
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in classes if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} classes missing; first={missing[:5]}")
    indices = torch.tensor([class_to_idx[name] for name in classes], dtype=torch.long)
    return F.normalize(payload["features"][indices].float(), dim=1)


def score(
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    class_features = class_features.to(device)
    chunks = []
    for start in range(0, len(image_features), batch_size):
        batch = image_features[start : start + batch_size].to(device)
        chunks.append((batch @ class_features.T).topk(min(20, len(class_features)), dim=1).indices.cpu())
    return torch.cat(chunks)


def metrics(
    topk: torch.Tensor,
    target: torch.Tensor,
    full_counts: torch.Tensor,
    base_top1: torch.Tensor | None,
) -> dict[str, Any]:
    prediction = topk[:, 0]
    correct = prediction == target
    row: dict[str, Any] = {
        "top1": float(correct.float().mean().item()),
        "top5": float((topk[:, :5] == target[:, None]).any(dim=1).float().mean().item()),
        "top20": float((topk == target[:, None]).any(dim=1).float().mean().item()),
    }
    if base_top1 is not None:
        base_correct = base_top1 == target
        row.update(
            {
                "changed": int((prediction != base_top1).sum().item()),
                "wins": int((correct & ~base_correct).sum().item()),
                "losses": int((~correct & base_correct).sum().item()),
            }
        )
        row["net"] = row["wins"] - row["losses"]
    sample_counts = full_counts[target]
    row["frequency_buckets"] = {
        name: {
            "rows": int(mask.sum().item()),
            "top1": float(correct[mask].float().mean().item()),
        }
        for name, low, high in BUCKETS
        if bool((mask := (sample_counts >= low) & (sample_counts <= high)).any())
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--class-features", action="append", required=True, help="name=path")
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    classes = load_classes(args.classes)
    image_features = F.normalize(image_payload["features"].float(), dim=1)
    target = image_payload["class_ids"].long()
    full_counts = image_payload.get("full_class_counts")
    if full_counts is None:
        full_counts = torch.bincount(target, minlength=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    for name, path in map(parse_named_path, args.class_features):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        predictions[name] = score(
            image_features,
            align_features(payload, classes),
            device,
            args.batch_size,
        )
        sources[name] = str(path)
    if args.base not in predictions:
        raise ValueError(f"Base {args.base!r} is not one of {list(predictions)}")
    base_top1 = predictions[args.base][:, 0]
    results = {
        name: metrics(topk, target, full_counts, None if name == args.base else base_top1)
        for name, topk in predictions.items()
    }
    output = {
        "image_features": str(args.image_features),
        "rows": len(target),
        "classes": len(classes),
        "device": str(device),
        "base": args.base,
        "sources": sources,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

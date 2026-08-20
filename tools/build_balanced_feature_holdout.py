from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_paths(value: str) -> list[Path]:
    paths = [Path(part.strip()) for part in value.split(",") if part.strip()]
    if not paths:
        raise ValueError("At least one input cache is required")
    return paths


def load_classes(path: Path | None, labels: list[str]) -> list[str]:
    if path is None:
        return sorted(set(labels))
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        try:
            return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
        except (TypeError, ValueError):
            return list(data)
    return list(data)


def merge_payloads(paths: list[Path]) -> dict[str, Any]:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    dims = {int(payload["features"].shape[1]) for payload in payloads}
    if len(dims) != 1:
        raise RuntimeError(f"Feature dimensions differ: {sorted(dims)}")

    seen_ids: set[str] = set()
    image_ids: list[str] = []
    labels: list[str] = []
    feature_rows: list[torch.Tensor] = []
    duplicate_ids: list[str] = []
    for payload in payloads:
        features = payload["features"].float()
        rows = list(zip(payload["image_ids"], payload["labels"]))
        if len(rows) != features.shape[0]:
            raise RuntimeError("Cache metadata and feature row counts differ")
        keep: list[int] = []
        for row_idx, (image_id, label) in enumerate(rows):
            if image_id in seen_ids:
                duplicate_ids.append(image_id)
                continue
            if not label:
                continue
            seen_ids.add(image_id)
            image_ids.append(image_id)
            labels.append(label)
            keep.append(row_idx)
        if keep:
            feature_rows.append(features[torch.tensor(keep, dtype=torch.long)])
    if not feature_rows:
        raise RuntimeError("No labeled feature rows were loaded")
    return {
        "image_ids": image_ids,
        "labels": labels,
        "features": torch.cat(feature_rows, dim=0),
        "duplicate_ids": duplicate_ids,
        "source_caches": [str(path) for path in paths],
    }


def frequency_bucket(count: int) -> str:
    if count <= 2:
        return "count_2"
    if count <= 5:
        return "count_3_5"
    if count <= 10:
        return "count_6_10"
    if count <= 20:
        return "count_11_20"
    if count <= 50:
        return "count_21_50"
    return "count_51_plus"


def choose_holdout(
    indices: list[int],
    features: torch.Tensor,
    *,
    strategy: str,
    rng: random.Random,
) -> int:
    if strategy == "random":
        return rng.choice(indices)
    class_features = F.normalize(features[indices].float(), dim=1)
    centroid = F.normalize(class_features.mean(dim=0), dim=0)
    similarities = class_features @ centroid
    if strategy == "farthest":
        local_idx = int(similarities.argmin().item())
    elif strategy == "nearest":
        local_idx = int(similarities.argmax().item())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return indices[local_idx]


def subset_payload(
    merged: dict[str, Any],
    indices: list[int],
    classes: list[str],
    class_to_idx: dict[str, int],
    full_counts: Counter[str],
    split: str,
) -> dict[str, Any]:
    index_tensor = torch.tensor(indices, dtype=torch.long)
    labels = [merged["labels"][idx] for idx in indices]
    return {
        "image_ids": [merged["image_ids"][idx] for idx in indices],
        "labels": labels,
        "class_ids": torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long),
        "classes": classes,
        "features": merged["features"][index_tensor],
        "full_class_counts": torch.tensor([full_counts.get(name, 0) for name in classes], dtype=torch.long),
        "split": split,
        "source_caches": merged["source_caches"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-caches", required=True, help="Comma-separated labeled feature caches")
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strategy", choices=["random", "farthest", "nearest"], default="random")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-class-count", type=int, default=2)
    args = parser.parse_args()

    paths = parse_paths(args.input_caches)
    merged = merge_payloads(paths)
    classes = load_classes(args.classes, merged["labels"])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    missing = sorted(set(merged["labels"]) - set(class_to_idx))
    if missing:
        raise RuntimeError(f"{len(missing)} labels are absent from class list; first={missing[:5]}")

    rows_by_class: dict[str, list[int]] = defaultdict(list)
    for row_idx, label in enumerate(merged["labels"]):
        rows_by_class[label].append(row_idx)
    full_counts = Counter({name: len(indices) for name, indices in rows_by_class.items()})
    too_small = [name for name, count in full_counts.items() if count < args.min_class_count]
    if too_small:
        raise RuntimeError(f"{len(too_small)} classes have fewer than {args.min_class_count} rows")

    rng = random.Random(args.seed)
    val_indices: list[int] = []
    for class_name in classes:
        indices = rows_by_class.get(class_name, [])
        if len(indices) < args.min_class_count:
            continue
        val_indices.append(
            choose_holdout(indices, merged["features"], strategy=args.strategy, rng=rng)
        )
    val_set = set(val_indices)
    train_indices = [idx for idx in range(len(merged["labels"])) if idx not in val_set]
    val_indices.sort()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = subset_payload(
        merged, train_indices, classes, class_to_idx, full_counts, "balanced_holdout_train"
    )
    val_payload = subset_payload(
        merged, val_indices, classes, class_to_idx, full_counts, "balanced_holdout_val"
    )
    torch.save(train_payload, args.out_dir / "train.pt")
    torch.save(val_payload, args.out_dir / "val.pt")

    class_buckets = Counter(frequency_bucket(count) for count in full_counts.values())
    val_buckets = Counter(frequency_bucket(full_counts[label]) for label in val_payload["labels"])
    summary = {
        "input_caches": [str(path) for path in paths],
        "strategy": args.strategy,
        "seed": args.seed,
        "rows": len(merged["labels"]),
        "train_rows": len(train_indices),
        "val_rows": len(val_indices),
        "classes": len(classes),
        "classes_with_rows": len(full_counts),
        "duplicate_image_ids_skipped": len(merged["duplicate_ids"]),
        "class_frequency_buckets": dict(sorted(class_buckets.items())),
        "val_frequency_buckets": dict(sorted(val_buckets.items())),
        "train_cache": str(args.out_dir / "train.pt"),
        "val_cache": str(args.out_dir / "val.pt"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gather_flags(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        output[row["filename"]].add(row["category"])
    return dict(output)


def variant_payload(
    source: dict[str, Any],
    indices: torch.Tensor,
    name: str,
    source_path: Path,
) -> dict[str, Any]:
    result = dict(source)
    source_rows = len(source["image_ids"])
    for key, value in list(source.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == source_rows:
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == source_rows:
            result[key] = [value[int(index)] for index in indices.tolist()]
    result["quality_variant"] = name
    result["quality_variant_source"] = str(source_path)
    result["quality_variant_source_indices"] = indices
    if "class_ids" in result and "classes" in result:
        counts = torch.bincount(result["class_ids"].long(), minlength=len(result["classes"]))
        result["full_class_counts"] = counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-seen-manifest", type=Path, required=True)
    parser.add_argument("--test-unseen-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_flags = read_csv(args.flags)
    flags = gather_flags(raw_flags)
    manifests = {
        "train": read_csv(args.train_manifest),
        "val": read_csv(args.val_manifest),
        "test_seen": read_csv(args.test_seen_manifest),
        "test_unseen": read_csv(args.test_unseen_manifest),
    }
    location: dict[str, tuple[str, dict[str, str]]] = {}
    for split, rows in manifests.items():
        for row in rows:
            location[row["image_id"]] = (split, row)

    split_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    for image_id, categories in sorted(flags.items()):
        if image_id not in location:
            unresolved.append(
                {"image_id": image_id, "categories": "|".join(sorted(categories))}
            )
            continue
        split, row = location[image_id]
        split_rows[split].append(
            {
                "image_id": image_id,
                "categories": "|".join(sorted(categories)),
                "label": row.get("label", ""),
                "class_id": row.get("class_id", ""),
            }
        )

    for split in ("train", "val", "test_seen", "test_unseen"):
        write_csv(
            args.out_dir / f"flags_{split}.csv",
            split_rows[split],
            ["image_id", "categories", "label", "class_id"],
        )
    write_csv(
        args.out_dir / "flags_unresolved.csv",
        unresolved,
        ["image_id", "categories"],
    )

    train_payload = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    train_ids = list(train_payload["image_ids"])
    train_flag_categories = {
        image_id: flags[image_id] for image_id in train_ids if image_id in flags
    }
    hard_categories = {"鱼的一部分", "目标过小", "保护色"}
    all_indices = torch.arange(len(train_ids), dtype=torch.long)
    flagged_indices = torch.tensor(
        [index for index, image_id in enumerate(train_ids) if image_id in train_flag_categories],
        dtype=torch.long,
    )
    hard_indices = torch.tensor(
        [
            index
            for index, image_id in enumerate(train_ids)
            if image_id in train_flag_categories
            and bool(train_flag_categories[image_id] & hard_categories)
        ],
        dtype=torch.long,
    )
    flagged_mask = torch.zeros(len(train_ids), dtype=torch.bool)
    hard_mask = torch.zeros(len(train_ids), dtype=torch.bool)
    flagged_mask[flagged_indices] = True
    hard_mask[hard_indices] = True
    variants = {
        "exclude_all_flags": all_indices[~flagged_mask],
        "exclude_hard3": all_indices[~hard_mask],
        "oversample_hard3_x2": torch.cat([all_indices, hard_indices]),
    }
    feature_dir = args.out_dir / "feature_variants"
    feature_dir.mkdir(parents=True, exist_ok=True)
    variant_summaries = {}
    for name, indices in variants.items():
        payload = variant_payload(train_payload, indices, name, args.train_cache)
        out = feature_dir / f"{name}.pt"
        torch.save(payload, out)
        unique_ids = len(set(payload["image_ids"]))
        variant_summaries[name] = {
            "path": str(out),
            "rows": len(payload["image_ids"]),
            "unique_image_ids": unique_ids,
            "duplicate_rows": len(payload["image_ids"]) - unique_ids,
            "classes_present": int(torch.unique(payload["class_ids"]).numel()),
        }

    category_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for image_id, categories in flags.items():
        split = location.get(image_id, ("unresolved", {}))[0]
        for category in categories:
            category_split_counts[category][split] += 1
    summary = {
        "source": str(args.flags),
        "raw_rows": len(raw_flags),
        "unique_image_ids": len(flags),
        "multi_category_ids": sum(len(values) > 1 for values in flags.values()),
        "split_unique_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "unresolved": len(unresolved),
        "category_split_counts": {
            category: dict(counts)
            for category, counts in sorted(category_split_counts.items())
        },
        "hard_categories": sorted(hard_categories),
        "flagged_train_rows": int(flagged_indices.numel()),
        "hard_train_rows": int(hard_indices.numel()),
        "test_flags_quarantined": len(split_rows["test_seen"]) + len(split_rows["test_unseen"]),
        "variants": variant_summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

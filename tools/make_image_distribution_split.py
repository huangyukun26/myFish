from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_val_count(class_size: int, val_fraction: float, min_train_per_class: int) -> int:
    max_val = max(0, class_size - min_train_per_class)
    if max_val == 0:
        return 0
    raw = class_size * val_fraction
    val_n = int(math.floor(raw))
    if val_n == 0 and raw >= 0.75:
        val_n = 1
    return min(max_val, val_n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("work/full_manifests/train.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-train-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace it.")
    if not (0 < args.val_fraction < 1):
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.min_train_per_class < 1:
        raise ValueError("--min-train-per-class must be positive")

    rows = read_rows(args.train_manifest)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[row["class_id"]].append(row)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    class_stats = []
    for class_id, items in sorted(by_class.items(), key=lambda pair: int(pair[0])):
        items = list(items)
        rng.shuffle(items)
        val_n = choose_val_count(len(items), args.val_fraction, args.min_train_per_class)
        val_rows.extend(items[:val_n])
        train_rows.extend(items[val_n:])
        class_stats.append({"class_id": class_id, "rows": len(items), "val": val_n, "train": len(items) - val_n})

    fieldnames = list(rows[0].keys())
    write_rows(args.output / "train.csv", train_rows, fieldnames)
    write_rows(args.output / "val.csv", val_rows, fieldnames)

    train_counts = Counter(row["label"] for row in train_rows)
    val_counts = Counter(row["label"] for row in val_rows)
    summary = {
        "source": str(args.train_manifest),
        "output": str(args.output),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "min_train_per_class": args.min_train_per_class,
        "source_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "classes": len(by_class),
        "train_classes": len(train_counts),
        "val_classes": len(val_counts),
        "val_rows_fraction": len(val_rows) / len(rows),
        "val_class_fraction": len(val_counts) / len(by_class),
        "classes_without_val": len(by_class) - len(val_counts),
        "top_val_classes": val_counts.most_common(20),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "class_stats.json").write_text(json.dumps(class_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

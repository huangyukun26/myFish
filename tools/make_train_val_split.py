from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("work/full_manifests/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("work/supervised_splits"))
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists. Pass --overwrite to replace split files.")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.train_manifest)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_id"]].append(row)

    rng = random.Random(args.seed)
    train_rows = []
    val_rows = []
    for class_id, items in sorted(by_class.items(), key=lambda pair: int(pair[0])):
        items = list(items)
        rng.shuffle(items)
        val_n = min(args.val_per_class, max(1, len(items) - 1))
        val_rows.extend(items[:val_n])
        train_rows.extend(items[val_n:])

    fieldnames = list(rows[0].keys())
    write_rows(args.output / "train.csv", train_rows, fieldnames)
    write_rows(args.output / "val.csv", val_rows, fieldnames)
    summary = {
        "source": str(args.train_manifest),
        "output": str(args.output),
        "classes": len(by_class),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "val_per_class": args.val_per_class,
        "seed": args.seed,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


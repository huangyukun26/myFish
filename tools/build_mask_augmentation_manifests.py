from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_key(image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).hexdigest()


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic train/validation manifests for SAM2 data-processing scouts."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rare-max-count", type=int, default=5)
    parser.add_argument("--pilot-train", type=int, default=512)
    parser.add_argument("--pilot-val", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    val_rows = read_rows(args.val)
    if not train_rows or not val_rows:
        raise RuntimeError("Both train and validation manifests must be non-empty")
    fieldnames = list(train_rows[0])
    if list(val_rows[0]) != fieldnames:
        raise RuntimeError("Train and validation manifests must have the same columns")

    counts = Counter(row["class_id"] for row in train_rows)
    rare_rows = [
        row for row in train_rows if counts[row["class_id"]] <= args.rare_max_count
    ]
    rare_rows.sort(key=lambda row: stable_key(row["image_id"], args.seed))
    val_rows.sort(key=lambda row: stable_key(row["image_id"], args.seed))

    rows_by_class: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        rows_by_class.setdefault(row["class_id"], []).append(row)
    support2_rows: list[dict[str, str]] = []
    for class_id in sorted(rows_by_class, key=int):
        class_rows = sorted(
            rows_by_class[class_id],
            key=lambda row: stable_key(row["image_id"], args.seed),
        )
        support2_rows.extend(class_rows[:2])
    support2_rows.sort(key=lambda row: stable_key(row["image_id"], args.seed))

    pilot_train = support2_rows[: min(args.pilot_train, len(support2_rows))]
    pilot_val = val_rows[: min(args.pilot_val, len(val_rows))]

    outputs = {
        "rare_train_full": args.out_dir / "rare_train_full.csv",
        "support2_train_full": args.out_dir / "support2_train_full.csv",
        "pilot_train": args.out_dir / "pilot_train.csv",
        "pilot_val": args.out_dir / "pilot_val.csv",
    }
    write_rows(outputs["rare_train_full"], rare_rows, fieldnames)
    write_rows(outputs["support2_train_full"], support2_rows, fieldnames)
    write_rows(outputs["pilot_train"], pilot_train, fieldnames)
    write_rows(outputs["pilot_val"], pilot_val, fieldnames)

    summary = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_classes": len(counts),
        "rare_max_count": args.rare_max_count,
        "rare_classes": sum(value <= args.rare_max_count for value in counts.values()),
        "rare_train_rows": len(rare_rows),
        "support2_train_rows": len(support2_rows),
        "support2_classes": len({row["class_id"] for row in support2_rows}),
        "pilot_train_rows": len(pilot_train),
        "pilot_val_rows": len(pilot_val),
        "seed": args.seed,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

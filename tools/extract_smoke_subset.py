from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from fishnet.zip_local import iter_local_entries, read_entry_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("work/smoke_subset"))
    parser.add_argument("--classes", type=int, default=16)
    parser.add_argument("--min-images-per-class", type=int, default=3)
    parser.add_argument("--max-images-per-class", type=int, default=6)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_path = args.dataset_root / "label_train.json"
    zip_path = args.dataset_root / "images.zip"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    by_class = defaultdict(list)
    entries = list(iter_local_entries(zip_path))
    for entry in entries:
        filename = Path(entry.name).name
        if filename in labels and entry.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            by_class[labels[filename]].append(entry)

    eligible = [(cls, items) for cls, items in by_class.items() if len(items) >= args.min_images_per_class]
    eligible.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    selected = eligible[: args.classes]
    if len(selected) < args.classes:
        raise RuntimeError(
            f"Only {len(selected)} eligible classes found in {zip_path}. "
            f"Need {args.classes} classes with >= {args.min_images_per_class} complete images."
        )

    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    class_to_idx = {cls: idx for idx, (cls, _items) in enumerate(selected)}

    with zip_path.open("rb") as fp:
        for cls, items in selected:
            items = list(items)
            rng.shuffle(items)
            items = items[: args.max_images_per_class]
            for item_index, entry in enumerate(items):
                source_name = Path(entry.name).name
                split = "val" if item_index < args.val_per_class else "train"
                raw = read_entry_data(fp, entry)
                out_name = f"{class_to_idx[cls]:03d}_{source_name}"
                out_path = image_dir / out_name
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        image.convert("RGB").save(out_path, quality=95)
                except Exception as exc:
                    print(f"skip corrupt image {entry.name}: {exc}")
                    continue
                rows.append(
                    {
                        "image_path": str(out_path.relative_to(args.output)).replace("\\", "/"),
                        "label": cls,
                        "class_id": class_to_idx[cls],
                        "split": split,
                        "source_name": source_name,
                    }
                )

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_path", "label", "class_id", "split", "source_name"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(args.dataset_root),
        "zip_path": str(zip_path),
        "local_zip_entries_seen": len(entries),
        "selected_classes": len(selected),
        "rows": len(rows),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "val_rows": sum(row["split"] == "val" for row in rows),
        "class_to_idx": class_to_idx,
    }
    (args.output / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "label", "class_id"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, default=Path("work/vendor"))
    parser.add_argument("--train-fraction", type=float, default=0.80)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vendor.resolve()))
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    records = table.to_pylist()
    labels = sorted({row["label"] for row in records})
    class_to_idx = {label: index for index, label in enumerate(labels)}
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    train_rows = []
    val_rows = []
    boxes = []
    seen = set()
    corrupt = []
    for index, row in enumerate(records):
        source_name = Path(row["path"]).name
        image_id = source_name
        if image_id in seen:
            image_id = f"{Path(source_name).stem}_{index}{Path(source_name).suffix}"
        seen.add(image_id)
        image_bytes = row["image"]["bytes"]
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception as error:
            corrupt.append({"image_id": image_id, "error": repr(error)})
            continue
        (image_dir / image_id).write_bytes(image_bytes)
        output_row = {
            "image_id": image_id,
            "label": row["label"],
            "class_id": class_to_idx[row["label"]],
        }
        if stable_hash("split:" + image_id) % 10000 < round(args.train_fraction * 10000):
            train_rows.append(output_row)
        else:
            val_rows.append(output_row)
        boxes.append(
            {
                "image_id": image_id,
                "crop_box": [0, 0, width, height],
                "fallback": False,
                "crop_area_fraction": 1.0,
                "source": "full-image public pretraining view",
            }
        )
    # Guarantee that every class has train exemplars even under a hash split.
    train_counts = {class_id: 0 for class_id in class_to_idx.values()}
    for row in train_rows:
        train_counts[row["class_id"]] += 1
    for class_id, count in train_counts.items():
        if count >= 2:
            continue
        candidates = [row for row in val_rows if row["class_id"] == class_id]
        for row in candidates[: 2 - count]:
            val_rows.remove(row)
            train_rows.append(row)
    train_rows.sort(key=lambda row: row["image_id"])
    val_rows.sort(key=lambda row: row["image_id"])
    write_manifest(args.out_dir / "train.csv", train_rows)
    write_manifest(args.out_dir / "val.csv", val_rows)
    (args.out_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "boxes.jsonl").open("w", encoding="utf-8") as handle:
        for row in boxes:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    provenance = {
        "dataset": "Oxford-IIIT Pet",
        "official_homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "license": "CC BY-SA 4.0; image copyright remains with original owners",
        "mirror_repository": "https://huggingface.co/datasets/enterprise-explorers/oxford-pets",
        "mirror_file_sha256": "1f890caead713365ad9ba5bb07c4e8d7c5148ce88f4431b5d228e3636154cd5d",
        "source_parquet": str(args.parquet.resolve()),
        "rows_in_parquet": len(records),
        "rows_extracted": len(train_rows) + len(val_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "classes": len(labels),
        "corrupt": corrupt,
        "purpose": "non-fish generic fine-grained query-exemplar comparator pretraining",
    }
    (args.out_dir / "SOURCE_AUDIT.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

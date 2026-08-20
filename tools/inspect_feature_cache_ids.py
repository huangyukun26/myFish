from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    image_ids = [str(value) for value in cache["image_ids"]]
    print(f"keys={sorted(cache)}")
    print(f"rows={len(image_ids)}")
    print(f"features_shape={tuple(cache['features'].shape)}")
    print(f"image_ids_sample={image_ids[: args.limit]}")
    classes = [str(value) for value in cache.get("classes", [])]
    print(f"classes={len(classes)} sample={classes[: args.limit]}")
    if "class_ids" in cache and len(cache["class_ids"]):
        class_ids = cache["class_ids"].long()
        print(f"class_ids_range=({int(class_ids.min())}, {int(class_ids.max())}) unique={int(class_ids.unique().numel())}")

    if args.manifest is None:
        return

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"manifest_fields={list(rows[0]) if rows else []}")
    print(f"manifest_sample={rows[: min(args.limit, len(rows))]}")

    cache_exact = set(image_ids)
    cache_name = {Path(value).name for value in image_ids}
    for field in rows[0] if rows else []:
        values = [str(row[field]) for row in rows]
        exact = sum(value in cache_exact for value in values)
        basename = sum(Path(value).name in cache_name for value in values)
        print(f"field={field!r} exact={exact}/{len(values)} basename={basename}/{len(values)}")


if __name__ == "__main__":
    main()

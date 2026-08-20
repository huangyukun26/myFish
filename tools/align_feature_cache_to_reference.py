from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    source_ids = list(source["image_ids"])
    reference_ids = list(reference["image_ids"])
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("Source cache contains duplicate image IDs")
    source_index = {image_id: idx for idx, image_id in enumerate(source_ids)}
    missing = [image_id for image_id in reference_ids if image_id not in source_index]
    if missing:
        raise RuntimeError(f"Source lacks {len(missing)} reference rows; first={missing[:5]}")
    indices = torch.tensor([source_index[image_id] for image_id in reference_ids], dtype=torch.long)
    selected_class_ids = source["class_ids"][indices].long()
    reference_class_ids = reference["class_ids"].long()
    if not torch.equal(selected_class_ids, reference_class_ids):
        raise RuntimeError("Source and reference class IDs differ after image-ID alignment")
    if list(source["classes"]) != list(reference["classes"]):
        raise RuntimeError("Source and reference class orders differ")

    output = dict(reference)
    output["features"] = source["features"][indices].contiguous()
    output["source_cache"] = str(args.source)
    output["reference_cache"] = str(args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(
        json.dumps(
            {
                "rows": len(reference_ids),
                "feature_dim": int(output["features"].shape[1]),
                "image_order_exact": list(output["image_ids"]) == reference_ids,
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    features = payload["features"]
    if not 0 <= args.start < args.end <= features.shape[1]:
        raise ValueError(
            f"Invalid feature slice [{args.start}:{args.end}] for dimension {features.shape[1]}"
        )
    output = dict(payload)
    output["features"] = features[:, args.start : args.end].contiguous()
    output["source_cache"] = str(args.input)
    output["feature_slice"] = [args.start, args.end]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(
        json.dumps(
            {
                "rows": len(output["image_ids"]),
                "input_dim": int(features.shape[1]),
                "output_dim": int(output["features"].shape[1]),
                "out": str(args.out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

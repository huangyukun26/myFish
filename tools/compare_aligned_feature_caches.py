from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--min-cosine", type=float, default=None)
    args = parser.parse_args()

    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    if list(left["image_ids"]) != list(right["image_ids"]):
        raise RuntimeError("image_ids differ")
    if list(left.get("labels", [])) != list(right.get("labels", [])):
        raise RuntimeError("labels differ")
    left_features = left["features"].float()
    right_features = right["features"].float()
    if left_features.shape != right_features.shape:
        raise RuntimeError(f"feature shapes differ: {left_features.shape} vs {right_features.shape}")
    cosine = (F.normalize(left_features, dim=1) * F.normalize(right_features, dim=1)).sum(dim=1)
    difference = (left_features - right_features).abs()
    summary = {
        "left": str(args.left),
        "right": str(args.right),
        "rows": len(cosine),
        "dim": int(left_features.shape[1]),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_p01": float(torch.quantile(cosine, 0.01).item()),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if args.min_cosine is not None and summary["cosine_min"] < args.min_cosine:
        raise RuntimeError(
            f"minimum cosine {summary['cosine_min']:.8f} is below required {args.min_cosine:.8f}"
        )


if __name__ == "__main__":
    main()

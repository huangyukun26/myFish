from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def parse_weights(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated feature caches")
    parser.add_argument("--weights", required=True, help="Comma-separated nonnegative weights")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.inputs)
    weights = parse_weights(args.weights)
    if len(paths) < 2 or len(paths) != len(weights):
        raise ValueError("--inputs and --weights need the same length of at least two")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be nonnegative with a positive sum")
    total = sum(weights)
    weights = [weight / total for weight in weights]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if list(payload["image_ids"]) != list(base["image_ids"]):
            raise RuntimeError(f"image_ids differ in {path}")
        if list(payload.get("labels", [])) != list(base.get("labels", [])):
            raise RuntimeError(f"labels differ in {path}")
        if payload["features"].shape != base["features"].shape:
            raise RuntimeError(f"feature shape differs in {path}")

    blended = sum(
        weight * F.normalize(payload["features"].float(), dim=1)
        for weight, payload in zip(weights, payloads)
    )
    output = {
        **base,
        "features": F.normalize(blended, dim=1),
        "blended_inputs": [str(path) for path in paths],
        "blend_weights": weights,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    summary = {
        "out": str(args.out),
        "inputs": [str(path) for path in paths],
        "weights": weights,
        "rows": len(output["image_ids"]),
        "dim": int(output["features"].shape[1]),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

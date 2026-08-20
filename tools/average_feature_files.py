from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def parse_weights(value: str, count: int) -> list[float]:
    if not value.strip():
        return [1.0 / count for _ in range(count)]
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}")
    total = sum(weights)
    if total == 0:
        raise ValueError("Weights must not sum to zero")
    return [weight / total for weight in weights]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated .pt feature files.")
    parser.add_argument("--weights", default="", help="Comma-separated weights; defaults to uniform.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.inputs)
    weights = parse_weights(args.weights, len(paths))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    image_ids = payloads[0]["image_ids"]
    labels = payloads[0].get("labels", [])
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["image_ids"] != image_ids:
            raise RuntimeError(f"image_ids differ in {path}")
    features = None
    for weight, payload in zip(weights, payloads):
        part = normalize(payload["features"]) * weight
        features = part if features is None else features + part
    out_payload = dict(payloads[0])
    out_payload["features"] = normalize(features)
    out_payload["ensemble_sources"] = [str(path) for path in paths]
    out_payload["ensemble_weights"] = weights
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.out)
    summary = {
        "inputs": [str(path) for path in paths],
        "weights": weights,
        "out": str(args.out),
        "rows": len(image_ids),
        "labels": int(sum(bool(label) for label in labels)) if labels else 0,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

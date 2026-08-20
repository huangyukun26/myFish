from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated feature cache .pt files.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.inputs)
    if len(paths) < 2:
        raise ValueError("--inputs needs at least two feature caches")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["image_ids"] != base["image_ids"]:
            raise RuntimeError(f"image_ids differ in {path}")
        if list(payload.get("labels", [])) != list(base.get("labels", [])):
            raise RuntimeError(f"labels differ in {path}")
    features = torch.stack([normalize(payload["features"]) for payload in payloads], dim=0).mean(dim=0)
    out_payload = {
        **base,
        "features": normalize(features),
        "averaged_inputs": [str(path) for path in paths],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.out)
    summary = {
        "out": str(args.out),
        "inputs": [str(path) for path in paths],
        "rows": len(out_payload["image_ids"]),
        "dim": int(out_payload["features"].shape[1]),
    }
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

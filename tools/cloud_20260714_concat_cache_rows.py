from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a = torch.load(args.first, map_location="cpu", weights_only=False)
    b = torch.load(args.second, map_location="cpu", weights_only=False)
    out = dict(a)
    for key in ("features", "class_ids"):
        out[key] = torch.cat([a[key], b[key]], dim=0)
    for key in ("image_ids", "labels"):
        out[key] = list(a[key]) + list(b[key])
    assert len(out["image_ids"]) == len(set(out["image_ids"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print({"rows": len(out["image_ids"]), "dim": out["features"].shape[1], "out": str(args.out)})


if __name__ == "__main__": main()

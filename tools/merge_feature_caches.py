from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ignore-classes", action="store_true")
    args = parser.parse_args()

    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.inputs]
    classes = payloads[0]["classes"]
    for path, payload in zip(args.inputs, payloads):
        if payload["classes"] != classes and not args.ignore_classes:
            raise ValueError(f"class list mismatch in {path}")

    merged = {
        "features": torch.cat([payload["features"].float() for payload in payloads], dim=0),
        "class_ids": torch.cat([payload["class_ids"].long() for payload in payloads], dim=0),
        "image_ids": [image_id for payload in payloads for image_id in payload["image_ids"]],
        "labels": [label for payload in payloads for label in payload["labels"]],
        "classes": classes if not args.ignore_classes else [],
        "sources": [str(path) for path in args.inputs],
        "env": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "rows": int(sum(payload["features"].shape[0] for payload in payloads)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(merged["image_ids"]),
                "dim": int(merged["features"].shape[1]),
                "sources": merged["sources"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

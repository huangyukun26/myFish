from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    classes = load_classes(args.classes_json)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    labels = list(payload["labels"])
    features = F.normalize(payload["features"].float(), dim=1)
    prototypes = torch.zeros((len(classes), features.shape[1]), dtype=torch.float32)
    counts = torch.zeros(len(classes), dtype=torch.long)
    missing = 0
    for idx, label in enumerate(labels):
        if not label:
            continue
        class_idx = class_to_idx.get(label)
        if class_idx is None:
            missing += 1
            continue
        prototypes[class_idx] += features[idx]
        counts[class_idx] += 1
    valid = counts > 0
    prototypes[valid] = F.normalize(prototypes[valid], dim=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "counts": counts,
            "classes": classes,
            "source_features": str(args.features),
        },
        args.out,
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "classes": len(classes),
                "valid_classes": int(valid.sum().item()),
                "missing_labels": missing,
                "dim": int(features.shape[1]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

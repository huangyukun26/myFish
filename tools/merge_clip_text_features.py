from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def parse_paths(value: str) -> List[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes-json", type=Path, required=True)
    parser.add_argument("--inputs", required=True, help="Comma-separated .pt files to merge.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    target_classes = load_classes(args.classes_json)
    input_paths = parse_paths(args.inputs)
    if not input_paths:
        raise ValueError("--inputs is empty")

    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in input_paths]
    dim = int(payloads[0]["features"].shape[1])
    class_to_feature = {}
    duplicate_classes = []
    for path, payload in zip(input_paths, payloads):
        features = payload["features"].float()
        if int(features.shape[1]) != dim:
            raise ValueError(f"Feature dim mismatch in {path}: {features.shape[1]} != {dim}")
        for class_name, feature in zip(payload["classes"], features):
            if class_name in class_to_feature:
                duplicate_classes.append(class_name)
            class_to_feature[class_name] = feature

    missing = [name for name in target_classes if name not in class_to_feature]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} classes; first={missing[:10]}")

    features = torch.stack([class_to_feature[name] for name in target_classes])
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    merged = dict(payloads[0])
    merged["classes"] = target_classes
    merged["features"] = features
    merged["merged_inputs"] = [str(path) for path in input_paths]
    merged["duplicate_classes"] = duplicate_classes[:100]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "classes": len(target_classes),
                "dim": dim,
                "inputs": [str(path) for path in input_paths],
                "duplicates": len(duplicate_classes),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if not isinstance(payload, dict) else list(payload.keys())


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated text feature .pt files with identical classes.")
    parser.add_argument("--weights", default="", help="Optional comma-separated weights, same length as inputs.")
    parser.add_argument("--target-classes", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.inputs)
    if not paths:
        raise ValueError("--inputs is empty")
    weights = [1.0 / len(paths) for _ in paths]
    if args.weights.strip():
        weights = [float(part.strip()) for part in args.weights.split(",") if part.strip()]
        if len(weights) != len(paths):
            raise ValueError("--weights length must match --inputs")
        total = sum(weights)
        weights = [value / total for value in weights]

    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    classes = load_classes(args.target_classes) if args.target_classes else list(payloads[0]["classes"])
    features = None
    for weight, payload in zip(weights, payloads):
        payload_classes = list(payload["classes"])
        if payload_classes == classes:
            aligned = payload["features"]
        elif args.target_classes:
            class_to_idx = {name: idx for idx, name in enumerate(payload_classes)}
            missing = [name for name in classes if name not in class_to_idx]
            if missing:
                raise RuntimeError(f"{len(missing)} target classes missing in {path}; first={missing[:5]}")
            indices = torch.tensor([class_to_idx[name] for name in classes], dtype=torch.long)
            aligned = payload["features"][indices]
        else:
            raise RuntimeError(f"Class order differs in {path}")
        part = normalize(aligned) * weight
        features = part if features is None else features + part
    if features is None:
        raise RuntimeError("No features")
    out = dict(payloads[0])
    out["classes"] = classes
    out["features"] = normalize(features)
    out["averaged_inputs"] = [str(path) for path in paths]
    out["average_weights"] = weights
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"out": str(args.out), "classes": len(classes), "weights": weights}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

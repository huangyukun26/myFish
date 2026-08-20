from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated aligned feature cache files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.inputs)
    if len(paths) < 2:
        raise ValueError("--inputs requires at least two caches")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if list(payload["image_ids"]) != list(base["image_ids"]):
            raise RuntimeError(f"image_ids differ in {path}")
        if list(payload.get("labels", [])) != list(base.get("labels", [])):
            raise RuntimeError(f"labels differ in {path}")
        if "class_ids" in base and not torch.equal(payload["class_ids"].long(), base["class_ids"].long()):
            raise RuntimeError(f"class_ids differ in {path}")

    class_lists = [list(payload.get("classes", [])) for payload in payloads]
    merged_classes = None
    if any(class_lists):
        lengths = {len(values) for values in class_lists if values}
        if len(lengths) != 1:
            raise RuntimeError(f"class-list lengths differ across inputs: {sorted(lengths)}")
        class_count = lengths.pop()
        merged_classes = []
        for class_id in range(class_count):
            names = {
                str(values[class_id])
                for values in class_lists
                if values and str(values[class_id])
            }
            if len(names) > 1:
                raise RuntimeError(
                    f"class name conflict at class_id={class_id}: {sorted(names)}"
                )
            merged_classes.append(next(iter(names)) if names else "")

    parts = [F.normalize(payload["features"].float(), dim=1) for payload in payloads]
    features = torch.cat(parts, dim=1) / (len(parts) ** 0.5)
    output = {
        **base,
        "features": features,
        "concatenated_inputs": [str(path) for path in paths],
        "component_dims": [int(part.shape[1]) for part in parts],
    }
    if merged_classes is not None:
        output["classes"] = merged_classes
        if "labels" in output and "class_ids" in output:
            for row, (label, class_id) in enumerate(
                zip(output["labels"], output["class_ids"].tolist())
            ):
                expected = merged_classes[int(class_id)]
                if expected and label and expected != label:
                    raise RuntimeError(
                        f"label/class_id mismatch at row={row}: {label!r} != {expected!r}"
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    summary = {
        "out": str(args.out),
        "inputs": [str(path) for path in paths],
        "rows": len(output["image_ids"]),
        "component_dims": output["component_dims"],
        "output_dim": int(features.shape[1]),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

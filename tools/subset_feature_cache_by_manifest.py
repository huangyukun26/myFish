from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_classes(path: Path | None, payloads: list[dict]) -> list[str]:
    if path is not None:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
        return list(data)
    for payload in payloads:
        if payload.get("classes"):
            return list(payload["classes"])
    return sorted({label for payload in payloads for label in payload.get("labels", []) if label})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-caches", required=True, help="Comma-separated feature caches")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--complement", action="store_true", help="Keep cache rows absent from the manifest")
    args = parser.parse_args()

    paths = [Path(value.strip()) for value in args.input_caches.split(",") if value.strip()]
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    classes = load_classes(args.classes, payloads)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    by_image_id: dict[str, tuple[str, int, torch.Tensor]] = {}
    duplicate_ids: list[str] = []
    for payload in payloads:
        row_count = len(payload["image_ids"])
        payload_labels = payload.get("labels") or [""] * row_count
        payload_class_ids = payload.get("class_ids")
        if payload_class_ids is None:
            payload_class_ids = torch.full((row_count,), -1, dtype=torch.long)
        if len(payload_labels) != row_count or len(payload_class_ids) != row_count:
            raise RuntimeError("Cache metadata length does not match feature rows")
        for image_id, label, class_id, feature in zip(
            payload["image_ids"], payload_labels, payload_class_ids, payload["features"]
        ):
            if image_id in by_image_id:
                duplicate_ids.append(image_id)
                continue
            by_image_id[image_id] = (label, int(class_id), feature)

    manifest_rows = read_manifest(args.manifest)
    missing = [row["image_id"] for row in manifest_rows if row["image_id"] not in by_image_id]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest images are absent from caches; first={missing[:5]}")
    if args.complement:
        excluded = {row["image_id"] for row in manifest_rows}
        rows = [
            {"image_id": image_id, "label": label}
            for image_id, (label, _class_id, _feature) in by_image_id.items()
            if image_id not in excluded
        ]
    else:
        rows = manifest_rows

    labels: list[str] = []
    source_class_ids: list[int] = []
    features: list[torch.Tensor] = []
    for row in rows:
        cached_label, cached_class_id, feature = by_image_id[row["image_id"]]
        manifest_label = row.get("label", "")
        if manifest_label and cached_label and manifest_label != cached_label:
            raise RuntimeError(
                f"Label mismatch for {row['image_id']}: manifest={manifest_label!r}, cache={cached_label!r}"
            )
        labels.append(manifest_label or cached_label)
        source_class_ids.append(cached_class_id)
        features.append(feature)

    labeled = [bool(label) for label in labels]
    if all(labeled):
        missing_classes = sorted({label for label in labels if label not in class_to_idx})
        if missing_classes:
            raise RuntimeError(f"Labels absent from class mapping: {missing_classes[:5]}")
        class_ids = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long)
    elif any(labeled):
        raise RuntimeError("Selected cache mixes labeled and unlabeled rows")
    else:
        # Test manifests intentionally carry empty labels; retain their source ids
        # (normally -1) while replacing only the class-name metadata.
        class_ids = torch.tensor(source_class_ids, dtype=torch.long)
    payload = {
        "image_ids": [row["image_id"] for row in rows],
        "labels": labels,
        "class_ids": class_ids,
        "classes": classes,
        "features": torch.stack(features),
        "source_caches": [str(path) for path in paths],
        "source_manifest": str(args.manifest),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    summary = {
        "rows": len(rows),
        "classes": len(classes),
        "unique_class_ids": int(class_ids.unique().numel()),
        "feature_dim": int(payload["features"].shape[1]),
        "duplicates_skipped": len(duplicate_ids),
        "complement": args.complement,
        "out": str(args.out),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

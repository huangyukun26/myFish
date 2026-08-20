#!/usr/bin/env python
"""Merge changed rows from seen/unseen package predictions into one submission."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def load_prediction(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("prediction.json") as f:
                return json.loads(f.read().decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def read_changed_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def read_submission_keys(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def apply_package(base: dict[str, str], package_prediction: Path, changed_rows: Path | None) -> tuple[dict[str, str], int]:
    if package_prediction is None:
        return base, 0
    override = load_prediction(package_prediction)
    if changed_rows is None:
        ids = [image_id for image_id, pred in override.items() if base.get(image_id) != pred]
    else:
        ids = read_changed_ids(changed_rows)
    changed = 0
    out = dict(base)
    for image_id in ids:
        new_pred = override[image_id]
        changed += int(out.get(image_id) != new_pred)
        out[image_id] = new_pred
    return out, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seen-package", type=Path, default=None)
    parser.add_argument("--seen-changed-rows", type=Path, default=None)
    parser.add_argument("--unseen-package", type=Path, default=None)
    parser.add_argument("--unseen-changed-rows", type=Path, default=None)
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    pred = load_prediction(args.base)
    pred, seen_changed = apply_package(pred, args.seen_package, args.seen_changed_rows)
    pred, unseen_changed = apply_package(pred, args.unseen_package, args.unseen_changed_rows)
    keys = read_submission_keys(args.submission_keys)
    missing = [image_id for image_id in keys if image_id not in pred]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} submission keys; first={missing[:10]}")
    ordered = {image_id: pred[image_id] for image_id in keys}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "prediction.json"
    out_zip = args.out_dir / "submission.zip"
    out_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    summary = {
        "base": str(args.base),
        "seen_package": str(args.seen_package) if args.seen_package else None,
        "seen_changed": seen_changed,
        "unseen_package": str(args.unseen_package) if args.unseen_package else None,
        "unseen_changed": unseen_changed,
        "total_changed": seen_changed + unseen_changed,
        "out_json": str(out_json),
        "out_zip": str(out_zip),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

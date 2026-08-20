"""Build a tiny official-data overlay for unambiguous byte-identical train/test images."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("runs/current_best_online_20260808_overall051/submission/prediction.json"),
    )
    parser.add_argument("--audit", type=Path, default=Path("runs/local_20260813_exact_dup_audit.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    changes = [
        row
        for row in audit
        if row["unique"] and not row["base_in_labels"] and row["train_label"] and row["image_id"] in base
    ]
    output = dict(base)
    for row in changes:
        output[row["image_id"]] = row["train_label"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction = args.out_dir / "prediction.json"
    prediction.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = args.out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction, "prediction.json")
    report = {
        "base": str(args.base),
        "audit": str(args.audit),
        "rule": "Only test_seen images byte-identical to one or more training images with exactly one observed training label; current prediction must differ.",
        "changes": len(changes),
        "rows": [
            {
                "image_id": row["image_id"],
                "current": row["base"],
                "replacement": row["train_label"],
                "matched_train_ids": row["train_ids"],
            }
            for row in changes
        ],
        "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
    (args.out_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

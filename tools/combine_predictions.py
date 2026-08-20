from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def read_prediction_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            yield row["image_id"], row["prediction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-csv", type=Path, required=True)
    parser.add_argument("--unseen-csv", type=Path, required=True)
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("runs/submission_baseline/prediction.json"))
    parser.add_argument("--out-zip", type=Path, default=Path("runs/submission_baseline/submission.zip"))
    args = parser.parse_args()

    predictions = {}
    for image_id, pred in read_prediction_csv(args.seen_csv):
        predictions[image_id] = pred
    for image_id, pred in read_prediction_csv(args.unseen_csv):
        predictions[image_id] = pred

    with args.submission_keys.open("r", encoding="utf-8", newline="") as fp:
        required = [row["image_id"] for row in csv.DictReader(fp)]
    required_set = set(required)
    missing = [image_id for image_id in required if image_id not in predictions]
    extra = [image_id for image_id in predictions if image_id not in required_set]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} required prediction keys; first={missing[:10]}")

    ordered = {image_id: predictions[image_id] for image_id in required}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out_json, arcname="prediction.json")
    summary = {
        "seen_csv": str(args.seen_csv),
        "unseen_csv": str(args.unseen_csv),
        "required": len(required),
        "written": len(ordered),
        "extra_ignored": len(extra),
        "out_json": str(args.out_json),
        "out_zip": str(args.out_zip),
    }
    (args.out_json.parent / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

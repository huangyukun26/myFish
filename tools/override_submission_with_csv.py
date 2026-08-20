from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def load_base(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("prediction.json") as fp:
                return json.loads(fp.read().decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def read_override(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return {row["image_id"]: row["prediction"] for row in csv.DictReader(fp)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override-csv", type=Path, required=True)
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = load_base(args.base)
    override = read_override(args.override_csv)
    changed = 0
    for image_id, pred in override.items():
        changed += int(predictions.get(image_id) != pred)
        predictions[image_id] = pred

    with args.submission_keys.open("r", encoding="utf-8", newline="") as fp:
        keys = [row["image_id"] for row in csv.DictReader(fp)]
    missing = [key for key in keys if key not in predictions]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} submission keys; first={missing[:10]}")
    ordered = {key: predictions[key] for key in keys}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "prediction.json"
    out_zip = args.out_dir / "submission.zip"
    out_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    summary = {
        "base": str(args.base),
        "override_csv": str(args.override_csv),
        "override_rows": len(override),
        "changed": changed,
        "keys": len(keys),
        "out_json": str(out_json),
        "out_zip": str(out_zip),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def load_predictions(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("prediction.json") as fp:
                return json.loads(fp.read().decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def load_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [row["image_id"] for row in csv.DictReader(fp)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override-submission", type=Path, required=True)
    parser.add_argument("--override-ids", type=Path, required=True)
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base = load_predictions(args.base)
    override = load_predictions(args.override_submission)
    override_ids = load_ids(args.override_ids)
    missing_override = [image_id for image_id in override_ids if image_id not in override]
    if missing_override:
        raise RuntimeError(f"Override submission missing {len(missing_override)} ids; first={missing_override[:10]}")

    changed = 0
    for image_id in override_ids:
        new_pred = override[image_id]
        changed += int(base.get(image_id) != new_pred)
        base[image_id] = new_pred

    keys = load_ids(args.submission_keys)
    missing = [image_id for image_id in keys if image_id not in base]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} submission keys; first={missing[:10]}")
    ordered = {image_id: base[image_id] for image_id in keys}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "prediction.json"
    out_zip = args.out_dir / "submission.zip"
    out_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    summary = {
        "base": str(args.base),
        "override_submission": str(args.override_submission),
        "override_ids": str(args.override_ids),
        "override_rows": len(override_ids),
        "changed": changed,
        "keys": len(keys),
        "out_json": str(out_json),
        "out_zip": str(out_zip),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

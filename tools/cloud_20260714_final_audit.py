from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / "runs" / "cloud_20260714_return"
BASE_ZIP = RETURN / "v12_balanced_q25_submission.zip"


def load_prediction(path: Path) -> tuple[list[str], dict[str, str], list[str]]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        data = json.loads(zf.read("prediction.json"))
    return names, data, list(data)


def main() -> None:
    _, _, reference_keys = load_prediction(BASE_ZIP)
    with (ROOT / "dataset" / "all_classes.pkl").open("rb") as fh:
        allowed = set(pickle.load(fh))
    rows = []
    for path in sorted(RETURN.glob("*_submission.zip")):
        names, prediction, keys = load_prediction(path)
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "zip_members": names,
                "rows": len(prediction),
                "ordered_keys_match_reference": keys == reference_keys,
                "invalid_label_count": sum(value not in allowed for value in prediction.values()),
                "valid": names == ["prediction.json"]
                and len(prediction) == 35665
                and keys == reference_keys
                and all(value in allowed for value in prediction.values()),
            }
        )
    output = {
        "reference": BASE_ZIP.name,
        "required_rows": 35665,
        "all_valid": all(row["valid"] for row in rows),
        "packages": rows,
    }
    out = RETURN / "FINAL_PACKAGE_AUDIT.json"
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

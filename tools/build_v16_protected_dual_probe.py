from __future__ import annotations

import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / "runs" / "cloud_20260714_return"
BASE_DIR = ROOT / "runs" / "submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox"
FIXED_DIR = ROOT / "runs" / "submission_20260702_seen_fixed_unseen_pair_o70species_avg_letterbox"
LETTERBOX_DIR = ROOT / "runs" / "submission_20260701_seen_concat_h4096_prior025_bm2_cm08_unseen_letterbox"
OUT = ROOT / "runs" / "submission_20260714_v16_protected_dual_probe"

CANDIDATES = [
    "v3_explore_top10_submission.zip",
    "v3_recommended_top5_submission.zip",
    "v3_safe_top3_submission.zip",
    "v3_ultrasafe_top2_submission.zip",
    "v4_v4_explore_submission.zip",
    "v4_v4_recommended_submission.zip",
    "v4_v4_safe_submission.zip",
    "v5_recommended_submission.zip",
    "v5_broader_submission.zip",
    "v10_balanced_strict_submission.zip",
    "v10_maxnet_strict_submission.zip",
]


def load_dir(folder: Path) -> dict[str, str]:
    path = folder / "prediction.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(folder / "submission.zip") as zf:
        return json.loads(zf.read("prediction.json"))


def load_zip(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def main() -> None:
    base = load_dir(BASE_DIR)
    fixed = load_dir(FIXED_DIR)
    letterbox = load_dir(LETTERBOX_DIR)
    candidates = [load_zip(RETURN / name) for name in CANDIDATES]
    seen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "test.pkl").read_bytes()))
    unseen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "unseen.pkl").read_bytes()))
    train_labels = json.loads((ROOT / "dataset" / "label_train.json").read_text(encoding="utf-8"))
    class_counts = collections.Counter(train_labels.values())
    known_genera = {label.split()[0] for label in train_labels.values()}

    # These rows produced the two real public gains that define the 7.2 baseline.
    protected_seen = {key for key in seen_ids if base[key] != fixed[key]}
    protected_unseen = {key for key in unseen_ids if base[key] != letterbox[key]}

    seen_changes: dict[str, str] = {}
    for key in seen_ids - protected_seen:
        proposed = [prediction[key] for prediction in candidates]
        alternate = proposed[0]
        if not all(value == alternate for value in proposed):
            continue
        if alternate == base[key]:
            continue
        if class_counts[alternate] != 2 or class_counts[base[key]] <= 2:
            continue
        if alternate.split()[0] != base[key].split()[0]:
            continue
        seen_changes[key] = alternate

    unseen_changes: dict[str, str] = {}
    for key in unseen_ids - protected_unseen:
        proposed = [prediction[key] for prediction in candidates]
        alternate = proposed[0]
        if not all(value == alternate for value in proposed):
            continue
        if alternate == base[key]:
            continue
        base_genus = base[key].split()[0]
        alternate_genus = alternate.split()[0]
        if base_genus not in known_genera or alternate_genus in known_genera:
            continue
        unseen_changes[key] = alternate

    assert len(seen_changes) == 101, len(seen_changes)
    assert len(unseen_changes) == 106, len(unseen_changes)
    assert not (set(seen_changes) & protected_seen)
    assert not (set(unseen_changes) & protected_unseen)

    output = dict(base)
    output.update(seen_changes)
    output.update(unseen_changes)
    assert list(output) == list(base)
    assert len(output) == 35665

    allowed = set(pickle.loads((ROOT / "dataset" / "all_classes.pkl").read_bytes()))
    assert all(label in allowed for label in output.values())

    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "prediction.json"
    prediction_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    submission_path = OUT / "submission.zip"
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, "prediction.json")

    summary = {
        "base": str(BASE_DIR.relative_to(ROOT)),
        "purpose": "Low-risk dual-split public probe; not a score projection.",
        "seen_changes": len(seen_changes),
        "unseen_changes": len(unseen_changes),
        "total_changes": len(seen_changes) + len(unseen_changes),
        "protected_seen_rows": len(protected_seen),
        "protected_unseen_rows": len(protected_unseen),
        "protected_rows_overwritten": 0,
        "seen_rule": "11-package unanimous alternate; alternate train count=2; base train count>2; same genus",
        "unseen_rule": "11-package unanimous alternate; known-genus to novel-genus; outside proven 7.2 consensus",
        "zip_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

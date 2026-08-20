from __future__ import annotations

import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

from build_v16_protected_dual_probe import (
    BASE_DIR,
    CANDIDATES,
    FIXED_DIR,
    LETTERBOX_DIR,
    RETURN,
    ROOT,
    load_dir,
    load_zip,
)


OUT = ROOT / "runs" / "submission_20260714_v17_microaligned_dual_probe"


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

    # Freeze the two branch-level changes that established the 7.2 public baseline.
    protected_seen = {key for key in seen_ids if base[key] != fixed[key]}
    protected_unseen = {key for key in unseen_ids if base[key] != letterbox[key]}

    seen_changes: dict[str, str] = {}
    for key in seen_ids - protected_seen:
        proposed = [prediction[key] for prediction in candidates]
        alternate = proposed[0]
        if not all(value == alternate for value in proposed) or alternate == base[key]:
            continue
        if alternate.split()[0] != base[key].split()[0]:
            continue
        # Unlike the failed macro-tail routing, never move probability toward a rarer class.
        if class_counts[alternate] < class_counts[base[key]]:
            continue
        seen_changes[key] = alternate

    unseen_changes: dict[str, str] = {}
    for key in unseen_ids - protected_unseen:
        proposed = [prediction[key] for prediction in candidates]
        alternate = proposed[0]
        if not all(value == alternate for value in proposed) or alternate == base[key]:
            continue
        if base[key].split()[0] not in known_genera:
            continue
        if alternate.split()[0] in known_genera:
            continue
        unseen_changes[key] = alternate

    assert len(seen_changes) == 43, len(seen_changes)
    assert len(unseen_changes) == 106, len(unseen_changes)
    assert not (set(seen_changes) & protected_seen)
    assert not (set(unseen_changes) & protected_unseen)

    output = dict(base)
    output.update(seen_changes)
    output.update(unseen_changes)
    assert list(output) == list(base) and len(output) == 35665
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
        "purpose": "Small dual-split causal probe; no absolute score projection.",
        "seen_changes": len(seen_changes),
        "unseen_changes": len(unseen_changes),
        "total_changes": len(seen_changes) + len(unseen_changes),
        "protected_seen_branch_rows": len(protected_seen),
        "protected_unseen_branch_rows": len(protected_unseen),
        "protected_rows_overwritten": 0,
        "seen_rule": "11-package unanimous; same genus; alternate train count >= base train count",
        "unseen_rule": "11-package unanimous structured route; known genus to novel genus",
        "maximum_metric_move_if_every_change_is_wrong": {
            "seen": len(seen_changes) / len(seen_ids),
            "unseen": len(unseen_changes) / len(unseen_ids),
        },
        "zip_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

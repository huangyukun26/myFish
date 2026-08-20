#!/usr/bin/env python
"""Guard a candidate submission against the online-best baseline.

This catches the failure mode where a candidate was generated from an older
baseline and silently reverts rows from the current online-best submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


def load_prediction(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if names != ["prediction.json"]:
                raise ValueError(f"{path} must contain exactly one root prediction.json, got {names}")
            return json.loads(zf.read("prediction.json").decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--online-best", type=Path, default=Path("runs/current_best_online_20260808_overall051/submission/prediction.json"))
    parser.add_argument("--previous-archive", type=Path, default=Path("runs/current_best_archive_20260730_seen078046/submission/prediction.json"))
    parser.add_argument(
        "--protected-reference",
        type=Path,
        action="append",
        default=[Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json")],
        help=(
            "Ancestor prediction whose rows changed by online-best are locked. "
            "May be passed repeatedly; the 2026-07-02 router parent is protected by default."
        ),
    )
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--test-seen", type=Path, default=Path("work/full_manifests/test_seen.csv"))
    parser.add_argument("--test-unseen", type=Path, default=Path("work/full_manifests/test_unseen.csv"))
    parser.add_argument("--max-seen-diff", type=int, default=50)
    parser.add_argument("--max-unseen-diff", type=int, default=50)
    parser.add_argument(
        "--max-touched-locked",
        type=int,
        default=0,
        help="Maximum changes to rows improved over any protected ancestor (default: 0).",
    )
    parser.add_argument("--max-direct-reverts", type=int, default=0)
    args = parser.parse_args()

    candidate = load_prediction(args.candidate)
    online = load_prediction(args.online_best)
    previous = load_prediction(args.previous_archive)
    keys = read_ids(args.submission_keys)
    seen = set(read_ids(args.test_seen))
    unseen = set(read_ids(args.test_unseen))

    key_set = set(keys)
    missing = [image_id for image_id in keys if image_id not in candidate]
    extra = [image_id for image_id in candidate if image_id not in key_set]
    if missing or extra:
        print(json.dumps({"status": "fail", "missing": missing[:10], "extra": extra[:10]}, ensure_ascii=False, indent=2))
        return 2

    diff_ids = [image_id for image_id in keys if candidate[image_id] != online[image_id]]
    seen_diff = [image_id for image_id in diff_ids if image_id in seen]
    unseen_diff = [image_id for image_id in diff_ids if image_id in unseen]

    protected_references = [load_prediction(path) for path in args.protected_reference]
    for path, reference in zip(args.protected_reference, protected_references):
        if set(reference) != key_set:
            raise ValueError(f"protected reference key set differs from submission keys: {path}")

    locked_ids = [image_id for image_id in keys if online[image_id] != previous[image_id]]
    ancestor_locked_ids = {
        image_id
        for reference in protected_references
        for image_id in keys
        if online[image_id] != reference[image_id]
    }
    all_locked_ids = set(locked_ids) | ancestor_locked_ids
    touched_locked = [image_id for image_id in all_locked_ids if candidate[image_id] != online[image_id]]
    direct_reverts = [image_id for image_id in touched_locked if candidate[image_id] == previous[image_id]]

    failures = []
    if len(seen_diff) > args.max_seen_diff:
        failures.append(f"seen_diff {len(seen_diff)} > {args.max_seen_diff}")
    if len(unseen_diff) > args.max_unseen_diff:
        failures.append(f"unseen_diff {len(unseen_diff)} > {args.max_unseen_diff}")
    if len(touched_locked) > args.max_touched_locked:
        failures.append(f"touched_locked {len(touched_locked)} > {args.max_touched_locked}")
    if len(direct_reverts) > args.max_direct_reverts:
        failures.append(f"direct_reverts {len(direct_reverts)} > {args.max_direct_reverts}")

    report: dict[str, Any] = {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "candidate": str(args.candidate),
        "online_best": str(args.online_best),
        "previous_archive": str(args.previous_archive),
        "protected_references": [str(path) for path in args.protected_reference],
        "rows": len(keys),
        "diff_total": len(diff_ids),
        "diff_seen": len(seen_diff),
        "diff_unseen": len(unseen_diff),
        "online_gain_rows_vs_previous": len(locked_ids),
        "online_gain_rows_vs_ancestors": len(ancestor_locked_ids),
        "locked_rows_union": len(all_locked_ids),
        "touched_online_gain_rows": len(touched_locked),
        "direct_reverts_to_previous": len(direct_reverts),
        "max_touched_locked": args.max_touched_locked,
        "sample_direct_reverts": direct_reverts[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

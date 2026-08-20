#!/usr/bin/env python
"""Build a tiny external-support overlay after pairwise consensus gating."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def load_prediction(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("prediction.json").decode("utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def write_zip(prediction: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction, arcname="prediction.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--seen", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-changes", type=int, default=10)
    args = parser.parse_args()

    changed = {row["image_id"]: row for row in csv.DictReader(args.changed.open("r", encoding="utf-8-sig", newline=""))}
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence_by_id = {row["image_id"]: row for row in evidence["rows"]}
    online = load_prediction(args.online)
    previous = load_prediction(args.previous)
    seen_ids = {row["image_id"] for row in csv.DictReader(args.seen.open("r", encoding="utf-8-sig", newline=""))}
    locked = {image_id for image_id in seen_ids if online.get(image_id) != previous.get(image_id)}

    eligible: list[dict[str, object]] = []
    for image_id, row in changed.items():
        ev = evidence_by_id.get(image_id)
        if not ev or image_id not in seen_ids or image_id in locked:
            continue
        if online.get(image_id) != row["old"]:
            continue
        # `slot` is zero-based in changed_rows.csv and comes from the MLP
        # top-k candidate list used to generate the original package.
        mlp_rank_new = int(row["slot"]) + 1
        if mlp_rank_new not in (1, 2):
            continue
        ext_old = ev.get("external_bioclip_old")
        ext_new = ev.get("external_bioclip_new")
        if ext_old is None or ext_new is None or float(ext_new) - float(ext_old) < 0.03:
            continue
        local_names = ["bioclip_train", "dino_train", "bioclip_text"]
        local_positive = sum(
            float(ev[f"{name}_delta_new_minus_old"]) > 0.0
            for name in local_names
            if ev.get(f"{name}_delta_new_minus_old") is not None
        )
        if local_positive < 2:
            continue
        eligible.append(
            {
                "image_id": image_id,
                "old": row["old"],
                "new": row["new"],
                "mlp_rank_new": mlp_rank_new,
                "external_delta": float(ext_new) - float(ext_old),
                "local_positive": local_positive,
                "local_deltas": [ev[f"{name}_delta_new_minus_old"] for name in local_names],
            }
        )

    eligible.sort(key=lambda item: (-int(item["local_positive"]), -float(item["external_delta"]), int(item["mlp_rank_new"])))
    selected = eligible[: args.max_changes]
    prediction = dict(online)
    for item in selected:
        prediction[str(item["image_id"])] = str(item["new"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / "prediction.json"
    zip_path = args.out_dir / "submission.zip"
    rows_path = args.out_dir / "changed_rows.csv"
    prediction_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_zip(prediction_path, zip_path)
    with rows_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "old", "new", "mlp_rank_new", "external_delta", "local_positive", "local_deltas"])
        writer.writeheader()
        for item in selected:
            writer.writerow(item)
    audit = {
        "eligible": len(eligible),
        "selected": len(selected),
        "online_gain_rows_locked": len(locked),
        "selected_rows": selected,
        "prediction_json": str(prediction_path),
        "submission_zip": str(zip_path),
        "changed_rows": str(rows_path),
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

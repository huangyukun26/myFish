#!/usr/bin/env python
"""Build consensus override submissions from multiple prediction sources."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def load_prediction(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if "prediction.json" not in names:
                raise ValueError(f"{path} has no prediction.json")
            return json.loads(zf.read("prediction.json").decode("utf-8"))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
        raise ValueError(f"{path} JSON is not a prediction dict")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        fields = rows[0].keys()
        pred_key = "prediction" if "prediction" in fields else "label" if "label" in fields else None
        if pred_key is None:
            raise ValueError(f"{path} has no prediction/label column")
        return {row["image_id"]: row[pred_key] for row in rows if row.get("image_id") and row.get(pred_key)}
    raise ValueError(f"Unsupported prediction file: {path}")


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["image_id"] for row in csv.DictReader(f)]


def read_block_ids(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                image_id = row.get("image_id")
                if image_id:
                    out.add(image_id)
    return out


def genus(label: str) -> str:
    parts = str(label or "").split()
    return parts[0] if parts else ""


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_bool_grid(value: str) -> list[bool]:
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    out: list[bool] = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Bad bool value in grid: {part}")
        out.append(mapping[key])
    return out or [False]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_submission(path: Path, prediction: dict[str, str], keys: list[str]) -> tuple[Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    ordered = {image_id: prediction[image_id] for image_id in keys}
    out_json = path / "prediction.json"
    out_zip = path / "submission.zip"
    out_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    return out_json, out_zip


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("runs/current_best_archive_20260730_seen078046/submission/prediction.json"))
    parser.add_argument("--submission-keys", type=Path, default=Path("work/full_manifests/submission_keys.csv"))
    parser.add_argument("--split-ids", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--block-csv", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-votes-grid", default="3,4,5,6")
    parser.add_argument("--vote-margin-grid", default="1,2,3")
    parser.add_argument("--same-genus-grid", default="false,true")
    parser.add_argument("--max-changed-grid", default="50,100,200,400,800")
    parser.add_argument("--protect-online-gain", action="store_true")
    parser.add_argument("--exclude-base-votes", action="store_true")
    args = parser.parse_args()

    base = load_prediction(args.base)
    previous = load_prediction(args.previous)
    keys = read_ids(args.submission_keys)
    split_ids = [image_id for image_id in read_ids(args.split_ids) if image_id in base]
    split_set = set(split_ids)
    block_ids = read_block_ids(args.block_csv)
    if args.protect_online_gain:
        block_ids |= {image_id for image_id in split_ids if base.get(image_id) != previous.get(image_id)}

    sources: list[dict[str, Any]] = []
    for path in args.source:
        pred = load_prediction(path)
        covered = sum(1 for image_id in split_ids if image_id in pred)
        changed = sum(1 for image_id in split_ids if pred.get(image_id) and pred.get(image_id) != base[image_id])
        if covered:
            sources.append({"path": str(path), "pred": pred, "covered": covered, "changed_vs_base": changed})

    candidate_rows: list[dict[str, Any]] = []
    for image_id in split_ids:
        if image_id in block_ids:
            continue
        labels = []
        for src in sources:
            label = src["pred"].get(image_id)
            if not label:
                continue
            if args.exclude_base_votes and label == base[image_id]:
                continue
            labels.append(label)
        if not labels:
            continue
        counts = Counter(labels)
        base_label = base[image_id]
        if not args.exclude_base_votes:
            counts.pop(base_label, None)
        if not counts:
            continue
        top_label, top_votes = counts.most_common(1)[0]
        second_votes = counts.most_common(2)[1][1] if len(counts) > 1 else 0
        candidate_rows.append(
            {
                "image_id": image_id,
                "base_prediction": base_label,
                "candidate_prediction": top_label,
                "candidate_votes": top_votes,
                "runner_up_votes": second_votes,
                "vote_margin": top_votes - second_votes,
                "source_coverage": len(labels),
                "same_genus": genus(base_label) == genus(top_label),
            }
        )

    candidate_rows.sort(
        key=lambda row: (
            int(row["candidate_votes"]),
            int(row["vote_margin"]),
            int(row["source_coverage"]),
            int(row["same_genus"]),
        ),
        reverse=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "candidate_rows.csv", candidate_rows)

    packages: dict[str, Any] = {}
    for min_votes in parse_ints(args.min_votes_grid):
        for vote_margin in parse_ints(args.vote_margin_grid):
            for same_genus_only in parse_bool_grid(args.same_genus_grid):
                eligible = [
                    row
                    for row in candidate_rows
                    if int(row["candidate_votes"]) >= min_votes
                    and int(row["vote_margin"]) >= vote_margin
                    and (not same_genus_only or bool(row["same_genus"]))
                ]
                for max_changed in parse_ints(args.max_changed_grid):
                    selected = eligible[:max_changed]
                    if not selected:
                        continue
                    name = f"votes{min_votes}_margin{vote_margin}_{'samegenus' if same_genus_only else 'anygenus'}_cap{max_changed}"
                    pred = dict(base)
                    changed = 0
                    for row in selected:
                        image_id = str(row["image_id"])
                        new_label = str(row["candidate_prediction"])
                        changed += int(pred[image_id] != new_label)
                        pred[image_id] = new_label
                    package_dir = args.out_dir / "packages" / name
                    out_json, out_zip = write_submission(package_dir, pred, keys)
                    changed_rows_path = package_dir / "changed_rows.csv"
                    write_csv(changed_rows_path, selected)
                    packages[name] = {
                        "min_votes": min_votes,
                        "vote_margin": vote_margin,
                        "same_genus_only": same_genus_only,
                        "cap": max_changed,
                        "eligible": len(eligible),
                        "changed": changed,
                        "prediction_json": str(out_json),
                        "zip": str(out_zip),
                        "changed_rows": str(changed_rows_path),
                    }

    summary = {
        "base": str(args.base),
        "previous": str(args.previous),
        "split_ids": str(args.split_ids),
        "split_rows": len(split_ids),
        "block_rows": len(block_ids & split_set),
        "sources": [{"path": src["path"], "covered": src["covered"], "changed_vs_base": src["changed_vs_base"]} for src in sources],
        "candidate_rows": len(candidate_rows),
        "packages": packages,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Audit a small submission overlay against independent local evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, str] | None:
    try:
        if path.suffix.lower() == ".zip":
            import zipfile

            with zipfile.ZipFile(path) as zf:
                if "prediction.json" not in zf.namelist():
                    return None
                return json.loads(zf.read("prediction.json"))
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def read_topk(path: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image_id") in ids:
                out[row["image_id"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-topk", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    changed = read_rows(args.changed)
    ids = {row["image_id"] for row in changed}
    online = read_json(args.online) or {}
    candidate = read_json(args.candidate) or {}
    topk = read_topk(args.test_topk, ids)

    # Count predictions in all existing prediction artifacts. This is evidence
    # from already-produced local runs, not a new model or external source.
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    sources: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_paths: set[Path] = set()
    for path in args.runs.rglob("prediction.json"):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        payload = read_json(path)
        if not payload:
            continue
        for image_id in ids:
            pred = payload.get(image_id)
            if pred:
                votes[image_id][pred] += 1
                sources[image_id].append((str(path), pred))

    report: list[dict[str, Any]] = []
    for row in changed:
        image_id = row["image_id"]
        old = row["old"]
        new = row["new"]
        tk = topk.get(image_id, {})
        predictions = list(tk.get("predictions", []))
        scores = [float(x) for x in tk.get("scores", [])]
        rank_new = predictions.index(new) + 1 if new in predictions else None
        rank_old = predictions.index(old) + 1 if old in predictions else None
        report.append(
            {
                "image_id": image_id,
                "old": old,
                "new": new,
                "online_old": online.get(image_id) == old,
                "candidate_new": candidate.get(image_id) == new,
                "mlp_rank_old": rank_old,
                "mlp_rank_new": rank_new,
                "mlp_score_old": scores[rank_old - 1] if rank_old else None,
                "mlp_score_new": scores[rank_new - 1] if rank_new else None,
                "mlp_top1": predictions[0] if predictions else None,
                "existing_prediction_artifacts": len(sources[image_id]),
                "artifact_votes_old": votes[image_id].get(old, 0),
                "artifact_votes_new": votes[image_id].get(new, 0),
                "artifact_votes_top": votes[image_id].most_common(8),
            }
        )

    summary = {
        "rows": len(report),
        "online_old_mismatch": sum(not x["online_old"] for x in report),
        "candidate_new_mismatch": sum(not x["candidate_new"] for x in report),
        "new_in_mlp_top20": sum(x["mlp_rank_new"] is not None for x in report),
        "old_in_mlp_top20": sum(x["mlp_rank_old"] is not None for x in report),
        "new_has_more_artifact_votes": sum(x["artifact_votes_new"] > x["artifact_votes_old"] for x in report),
        "old_has_more_artifact_votes": sum(x["artifact_votes_old"] > x["artifact_votes_new"] for x in report),
        "artifact_files_scanned": len(seen_paths),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "rows": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build small unseen probes by reranking current topK with external galleries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_external_gallery_sidecar import group_indices, load_cache, parse_floats, parse_ints, score_against_groups  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_manifest_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["image_id"] for row in csv.DictReader(f)}


def write_zip(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")


def write_package(out_dir: Path, name: str, current: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    variant = out_dir / name
    variant.mkdir(parents=True, exist_ok=True)
    pred = dict(current)
    changed = []
    for row in rows:
        image_id = row["image_id"]
        new_label = row["external_label"]
        if pred.get(image_id) == new_label:
            continue
        pred[image_id] = new_label
        changed.append(row)
    pred_path = variant / "prediction.json"
    zip_path = variant / "submission.zip"
    pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_zip(pred_path, zip_path)
    with (variant / "changed_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(changed[0].keys()) if changed else ["image_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed)
    audit = {
        "name": name,
        "changed": len(changed),
        "prediction_json": str(pred_path),
        "submission_zip": str(zip_path),
    }
    (variant / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--all-classes-json", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--current-prediction", type=Path, required=True)
    parser.add_argument("--test-unseen-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--external-mode", choices=["max", "mean"], default="max")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-base-rank", type=int, default=20)
    parser.add_argument("--min-external-score", type=float, default=-1.0)
    parser.add_argument("--min-gap", type=float, default=0.02)
    parser.add_argument("--min-margin-vs-current", type=float, default=0.0)
    parser.add_argument("--allow-current-uncovered", action="store_true")
    parser.add_argument("--package-top-counts", default="25,50,100,150,250")
    parser.add_argument("--package-gap-thresholds", default="0.04,0.06,0.08,0.10,0.12")
    args = parser.parse_args()

    query = load_cache(args.query_cache)
    external = load_cache(args.external_cache)
    all_classes = json.loads(args.all_classes_json.read_text(encoding="utf-8"))
    class_to_id = {name: idx for idx, name in enumerate(all_classes)}
    id_to_class = {idx: name for idx, name in enumerate(all_classes)}
    current = json.loads(args.current_prediction.read_text(encoding="utf-8"))
    unseen_ids = read_manifest_ids(args.test_unseen_manifest)
    topk_rows = {row["image_id"]: row for row in load_jsonl(args.topk_jsonl)}

    keep = [i for i, image_id in enumerate(query["image_ids"]) if image_id in unseen_ids and image_id in topk_rows]
    qids = [query["image_ids"][i] for i in keep]
    qfeat = query["features"][keep]

    ext_class_ids = sorted(int(x) for x in torch.unique(external["class_ids"]).tolist() if int(x) >= 0)
    ext_pos = {cid: j for j, cid in enumerate(ext_class_ids)}
    groups = group_indices(external["class_ids"], ext_class_ids)
    ext_scores = score_against_groups(
        qfeat,
        external["features"],
        groups,
        ext_class_ids,
        mode=args.external_mode,
        chunk_size=args.chunk_size,
    )

    candidates: list[dict[str, Any]] = []
    covered_topk_rows = 0
    for row_idx, image_id in enumerate(qids):
        topk = topk_rows[image_id]
        top_preds = list(topk.get("predictions", []))
        current_label = current.get(image_id, top_preds[0] if top_preds else "")
        current_cid = class_to_id.get(current_label, -1)
        candidate_cids = [class_to_id[p] for p in top_preds[: args.max_base_rank] if p in class_to_id and class_to_id[p] in ext_pos]
        if not candidate_cids:
            continue
        covered_topk_rows += 1
        cols = torch.tensor([ext_pos[cid] for cid in candidate_cids], dtype=torch.long)
        vals = ext_scores[row_idx, cols]
        order = torch.argsort(vals, descending=True)
        best_local = int(order[0])
        best_cid = candidate_cids[best_local]
        best_score = float(vals[best_local])
        second_score = float(vals[int(order[1])]) if len(order) > 1 else float("-inf")
        gap = best_score - second_score if second_score != float("-inf") else 0.0
        current_ext_score = float("-inf")
        if current_cid in ext_pos:
            current_ext_score = float(ext_scores[row_idx, ext_pos[current_cid]])
        margin_vs_current = best_score - current_ext_score if current_ext_score != float("-inf") else float("inf")
        if best_cid == current_cid:
            continue
        if current_ext_score == float("-inf") and not args.allow_current_uncovered:
            continue
        if best_score < args.min_external_score or gap < args.min_gap or margin_vs_current < args.min_margin_vs_current:
            continue
        base_rank = top_preds.index(id_to_class[best_cid]) + 1 if id_to_class[best_cid] in top_preds else 999
        candidates.append(
            {
                "image_id": image_id,
                "current_label": current_label,
                "external_label": id_to_class[best_cid],
                "external_class_id": best_cid,
                "base_rank": base_rank,
                "external_score": best_score,
                "second_external_score": second_score,
                "external_gap": gap,
                "current_external_score": current_ext_score,
                "margin_vs_current": margin_vs_current,
                "base_top1": top_preds[0] if top_preds else "",
            }
        )

    candidates.sort(key=lambda r: (r["external_gap"], r["external_score"], -r["base_rank"]), reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidate_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(candidates[0].keys()) if candidates else ["image_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    package_audits = {}
    for count in parse_ints(args.package_top_counts):
        rows = candidates[: min(count, len(candidates))]
        package_audits[f"top{count:04d}"] = write_package(args.out / "packages", f"top{count:04d}", current, rows)
    for threshold in parse_floats(args.package_gap_thresholds):
        rows = [r for r in candidates if float(r["external_gap"]) >= threshold]
        safe = str(threshold).replace("-", "m").replace(".", "p")
        package_audits[f"gap_ge_{safe}"] = write_package(args.out / "packages", f"gap_ge_{safe}", current, rows)

    summary = {
        "query_rows": len(qids),
        "external_classes": len(ext_class_ids),
        "covered_topk_rows": covered_topk_rows,
        "candidate_rows": len(candidates),
        "external_mode": args.external_mode,
        "max_base_rank": args.max_base_rank,
        "min_external_score": args.min_external_score,
        "min_gap": args.min_gap,
        "min_margin_vs_current": args.min_margin_vs_current,
        "allow_current_uncovered": args.allow_current_uncovered,
        "packages": package_audits,
    }
    (args.out / "apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

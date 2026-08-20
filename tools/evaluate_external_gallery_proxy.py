#!/usr/bin/env python
"""Evaluate an external same-species gallery on official train-only queries.

This proxy is for low-shot classes that are absent from the current validation
split. It never uses test data or validation labels for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_cache(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "features" not in obj or "class_ids" not in obj:
        raise ValueError(f"Unsupported cache format: {path}")
    obj["features"] = F.normalize(obj["features"].float(), dim=1)
    obj["class_ids"] = obj["class_ids"].long()
    return obj


def rank_of_true(scores: torch.Tensor, true_index: int) -> int:
    order = torch.argsort(scores, descending=True)
    where = (order == true_index).nonzero(as_tuple=False)
    return int(where[0, 0]) + 1


def score_by_class(query: torch.Tensor, support: torch.Tensor, support_class_ids: torch.Tensor, class_ids: list[int], mode: str) -> torch.Tensor:
    scores = []
    sims = support @ query
    for cid in class_ids:
        mask = support_class_ids == cid
        if not bool(mask.any()):
            scores.append(torch.tensor(float("-inf")))
            continue
        vals = sims[mask]
        if mode == "max":
            scores.append(vals.max())
        elif mode == "mean":
            scores.append(vals.mean())
        else:
            raise ValueError(mode)
    return torch.stack(scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train = load_cache(args.train_cache)
    external = load_cache(args.external_cache)

    ext_class_ids = sorted(int(x) for x in torch.unique(external["class_ids"]).tolist() if int(x) >= 0)
    class_names = {i: name for i, name in enumerate(train.get("classes") or [])}

    train_mask = torch.zeros_like(train["class_ids"], dtype=torch.bool)
    for cid in ext_class_ids:
        train_mask |= train["class_ids"] == cid
    query_indices = train_mask.nonzero(as_tuple=False).flatten().tolist()

    rows = []
    correct = {"internal_loo_max": 0, "external_max": 0, "external_mean": 0, "combined_max": 0}
    rank_sums = {k: 0 for k in correct}
    margins = {k: [] for k in correct}

    for qi in query_indices:
        true_cid = int(train["class_ids"][qi])
        true_pos = ext_class_ids.index(true_cid)
        query = train["features"][qi]

        internal_support_mask = train_mask.clone()
        internal_support_mask[qi] = False
        internal_support = train["features"][internal_support_mask]
        internal_support_ids = train["class_ids"][internal_support_mask]

        s_internal = score_by_class(query, internal_support, internal_support_ids, ext_class_ids, "max")
        s_external_max = score_by_class(query, external["features"], external["class_ids"], ext_class_ids, "max")
        s_external_mean = score_by_class(query, external["features"], external["class_ids"], ext_class_ids, "mean")
        s_combined = torch.maximum(s_internal, s_external_max)

        score_map = {
            "internal_loo_max": s_internal,
            "external_max": s_external_max,
            "external_mean": s_external_mean,
            "combined_max": s_combined,
        }
        row: dict[str, Any] = {
            "image_id": train["image_ids"][qi],
            "class_id": true_cid,
            "label": class_names.get(true_cid, train["labels"][qi] if train.get("labels") else ""),
        }
        for name, scores in score_map.items():
            pred_pos = int(torch.argmax(scores))
            pred_cid = ext_class_ids[pred_pos]
            rank = rank_of_true(scores, true_pos)
            sorted_scores = torch.sort(scores, descending=True).values
            margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            correct[name] += int(pred_cid == true_cid)
            rank_sums[name] += rank
            margins[name].append(margin)
            row[f"{name}_pred_class_id"] = pred_cid
            row[f"{name}_pred_label"] = class_names.get(pred_cid, "")
            row[f"{name}_true_rank"] = rank
            row[f"{name}_true_score"] = float(scores[true_pos])
            row[f"{name}_margin"] = margin
        rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.out / "proxy_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "query_rows": n,
        "classes": len(ext_class_ids),
        "class_ids": ext_class_ids,
        "class_names": {str(cid): class_names.get(cid, "") for cid in ext_class_ids},
        "metrics": {},
    }
    for name in correct:
        summary["metrics"][name] = {
            "top1": correct[name],
            "top1_acc": correct[name] / n if n else None,
            "mean_true_rank": rank_sums[name] / n if n else None,
            "mean_margin": sum(margins[name]) / n if n else None,
        }
    with (args.out / "proxy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

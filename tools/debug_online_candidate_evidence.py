#!/usr/bin/env python
"""Compare candidate labels with existing BioCLIP/DINO feature evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def groups(features: torch.Tensor, labels: list[str]) -> dict[str, torch.Tensor]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        out[str(label)].append(i)
    return {label: features[idx] for label, idx in out.items()}


def class_scores(query: torch.Tensor, by_label: dict[str, torch.Tensor]) -> dict[str, float]:
    return {label: float((features @ query).max().item()) for label, features in by_label.items()}


def rank(scores: dict[str, float], label: str) -> int | None:
    if label not in scores:
        return None
    value = scores[label]
    return 1 + sum(other > value for other in scores.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--bioclip-test", type=Path, required=True)
    parser.add_argument("--bioclip-train", type=Path, required=True)
    parser.add_argument("--dino-test", type=Path, required=True)
    parser.add_argument("--dino-train", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    changed = list(csv.DictReader(args.changed.open("r", encoding="utf-8-sig", newline="")))
    ids = [row["image_id"] for row in changed]
    id_set = set(ids)

    bio_test = load(args.bioclip_test)
    bio_train = load(args.bioclip_train)
    dino_test = load(args.dino_test)
    dino_train = load(args.dino_train)
    external = load(args.external)
    text = load(args.text)

    def index(payload: dict[str, Any]) -> dict[str, int]:
        return {str(image_id): i for i, image_id in enumerate(payload["image_ids"])}

    bio_test_f = F.normalize(bio_test["features"].float(), dim=1)
    bio_train_f = F.normalize(bio_train["features"].float(), dim=1)
    dino_test_f = F.normalize(dino_test["features"].float(), dim=1)
    dino_train_f = F.normalize(dino_train["features"].float(), dim=1)
    external_f = F.normalize(external["features"].float(), dim=1)
    text_f = F.normalize(text["features"].float(), dim=1)

    bio_train_by = groups(bio_train_f, list(bio_train["labels"]))
    dino_train_by = groups(dino_train_f, list(dino_train["labels"]))
    external_by = groups(external_f, list(external["labels"]))
    text_classes = list(text["classes"])
    text_by = {label: text_f[i] for i, label in enumerate(text_classes)}
    bio_pos = index(bio_test)
    dino_pos = index(dino_test)

    report: list[dict[str, Any]] = []
    for row in changed:
        image_id = row["image_id"]
        old = row["old"]
        new = row["new"]
        bq = bio_test_f[bio_pos[image_id]]
        dq = dino_test_f[dino_pos[image_id]]
        b_scores = class_scores(bq, bio_train_by)
        d_scores = class_scores(dq, dino_train_by)
        if text_by:
            t_scores = {label: float(text_feature @ bq) for label, text_feature in text_by.items()}
        else:
            t_scores = {}
        e_scores = class_scores(bq, external_by)

        metrics = {
            "bioclip_train": (b_scores, old, new),
            "dino_train": (d_scores, old, new),
            "bioclip_text": (t_scores, old, new),
            "external_bioclip": (e_scores, old, new),
        }
        item: dict[str, Any] = {"image_id": image_id, "old": old, "new": new}
        for name, (scores, old_label, new_label) in metrics.items():
            item[f"{name}_old"] = scores.get(old_label)
            item[f"{name}_new"] = scores.get(new_label)
            item[f"{name}_delta_new_minus_old"] = (
                scores[new_label] - scores[old_label]
                if old_label in scores and new_label in scores
                else None
            )
            item[f"{name}_rank_old"] = rank(scores, old_label)
            item[f"{name}_rank_new"] = rank(scores, new_label)
        report.append(item)

    names = ["bioclip_train", "dino_train", "bioclip_text", "external_bioclip"]
    summary = {"rows": len(report)}
    for name in names:
        deltas = [row[f"{name}_delta_new_minus_old"] for row in report if row[f"{name}_delta_new_minus_old"] is not None]
        summary[f"{name}_new_better"] = sum(delta > 0 for delta in deltas)
        summary[f"{name}_old_better"] = sum(delta < 0 for delta in deltas)
        summary[f"{name}_ties"] = sum(delta == 0 for delta in deltas)
        summary[f"{name}_mean_delta"] = sum(deltas) / len(deltas) if deltas else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "rows": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

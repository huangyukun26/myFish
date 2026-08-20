from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from train_trait_aux_rerank import TRAIT_PATTERNS, row_zscore


IMAGE_EXTRA_PATTERNS: dict[str, list[str]] = {
    "body_elongated": [r"\blong\b", r"\bthin\b"],
    "body_compressed": [r"\bside[- ]?compressed\b"],
    "body_deep": [r"\btall body\b"],
    "body_eel_like": [r"\beel\b", r"\bsnake[- ]?like\b"],
    "color_black": [r"\bnearly black\b"],
    "color_white": [r"\bpale\b", r"\bcream\b"],
    "color_silver": [r"\bmetallic\b"],
    "color_gray": [r"\bdusky\b"],
    "pattern_bars": [r"\bbarred\b", r"\bvertical dark marks\b"],
    "pattern_bands": [r"\bcrossband", r"\bcross-band"],
    "pattern_spots": [r"\bdots\b", r"\bspeckles\b", r"\bspeckled\b"],
    "pattern_blotches": [r"\bpatches\b", r"\bpatchy\b"],
    "pattern_lines": [r"\bhorizontal line", r"\blongitudinal"],
    "tail_forked": [r"\bdeeply forked\b"],
    "tail_rounded": [r"\bround tail\b"],
    "head_large_mouth": [r"\bbig mouth\b"],
    "head_barbels": [r"\bwhisker"],
}


def merged_patterns() -> dict[str, list[str]]:
    out = {key: list(value) for key, value in TRAIT_PATTERNS.items()}
    for key, values in IMAGE_EXTRA_PATTERNS.items():
        out.setdefault(key, []).extend(values)
    return out


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return " ".join(str(value).lower().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trait_hits(text: str, patterns: dict[str, list[str]]) -> list[str]:
    text = clean_text(text)
    hits = []
    for name, regexes in patterns.items():
        if any(re.search(pattern, text) for pattern in regexes):
            hits.append(name)
    return hits


def build_class_traits(classes: list[str], descriptions: dict[str, str], patterns: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    names = list(patterns)
    out: dict[str, torch.Tensor] = {}
    for class_name in classes:
        desc = clean_text(descriptions.get(class_name, ""))
        values = [1.0 if any(re.search(pattern, desc) for pattern in patterns[name]) else 0.0 for name in names]
        out[class_name] = torch.tensor(values, dtype=torch.float32)
    return out


def metrics(rows: list[dict[str, Any]], order: torch.Tensor) -> dict[str, Any]:
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    for row_idx, row in enumerate(rows):
        label = row.get("label", "")
        preds = row["predictions"]
        if not label:
            continue
        final = preds[int(order[row_idx, 0].item())]
        base = preds[0]
        base_ok = base == label
        final_ok = final == label
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        changed += int(base != final)
        try:
            rank = [preds[int(idx)] for idx in order[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks)
    return {
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
    }


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--qwen-traits-jsonl", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--weight-grid", default="0,0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    topk_by_id = {row["image_id"]: row for row in read_jsonl(args.topk_jsonl)}
    qwen_rows = [row for row in read_jsonl(args.qwen_traits_jsonl) if "error" not in row and row.get("image_id") in topk_by_id]
    rows = [topk_by_id[row["image_id"]] for row in qwen_rows]
    patterns = merged_patterns()
    pattern_names = list(patterns)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    classes = sorted({pred for row in rows for pred in row["predictions"]})
    class_traits = build_class_traits(classes, descriptions, patterns)

    image_traits = []
    image_hits = {}
    for item in qwen_rows:
        text = " ".join(
            clean_text(item.get(key, ""))
            for key in [
                "body_shape",
                "dominant_colors",
                "pattern",
                "fins_tail",
                "head_mouth_eye",
            ]
        )
        hits = trait_hits(text, patterns)
        image_hits[item["image_id"]] = hits
        values = [1.0 if name in hits else 0.0 for name in pattern_names]
        image_traits.append(values)
    image_traits_t = torch.tensor(image_traits, dtype=torch.float32)
    trait_prior = image_traits_t.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
    idf = torch.log((1.0 / trait_prior).clamp_min(1.0))
    candidate_traits = torch.stack(
        [torch.stack([class_traits[pred] for pred in row["predictions"]], dim=0) for row in rows],
        dim=0,
    )
    base_scores = torch.tensor([[float(v) for v in row["scores"]] for row in rows], dtype=torch.float32)
    q = image_traits_t[:, None, :]
    raw_trait_scores = (candidate_traits * q * idf[None, None, :]).sum(dim=2)
    denom = (candidate_traits * idf[None, None, :]).sum(dim=2).clamp_min(1.0).sqrt()
    trait_scores = raw_trait_scores / denom
    trait_scores = row_zscore(trait_scores)

    sweep = []
    best = None
    best_order = None
    for weight in parse_grid(args.weight_grid):
        final = base_scores + weight * trait_scores
        order = final.argsort(dim=1, descending=True)
        row = {"weight": weight, **metrics(rows, order)}
        sweep.append(row)
        key = (row["top1"], row["net_wins"], -row["losses"], -abs(weight))
        if best is None or key > best[0]:
            best = (key, row)
            best_order = order

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    if best_order is not None:
        with (args.out_dir / "best_predictions.csv").open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["image_id", "label", "base_prediction", "prediction", "changed", "image_traits"],
            )
            writer.writeheader()
            for row_idx, row in enumerate(rows):
                pred = row["predictions"][int(best_order[row_idx, 0].item())]
                writer.writerow(
                    {
                        "image_id": row["image_id"],
                        "label": row.get("label", ""),
                        "base_prediction": row["predictions"][0],
                        "prediction": pred,
                        "changed": pred != row["predictions"][0],
                        "image_traits": ";".join(image_hits.get(row["image_id"], [])),
                    }
                )
    summary = {
        "topk_jsonl": str(args.topk_jsonl),
        "qwen_traits_jsonl": str(args.qwen_traits_jsonl),
        "rows": len(rows),
        "valid_qwen_rows": len(qwen_rows),
        "trait_names": pattern_names,
        "best": best[1] if best else None,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

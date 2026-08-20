from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def load_topk(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidates = list(payload["candidates"])
    top_indices = payload["top_indices"].long()
    top_scores = payload["top_scores"].float()
    preds = [[candidates[int(idx)] for idx in row.tolist()] for row in top_indices]
    return {
        "image_ids": list(payload["image_ids"]),
        "labels": list(payload.get("labels", [""] * len(payload["image_ids"]))),
        "candidates": candidates,
        "top_indices": top_indices,
        "top_scores": top_scores,
        "predictions": preds,
    }


def parse_thresholds(value: str) -> list[float | None]:
    out: list[float | None] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(None if part == "all" else float(part))
    return out or [None]


def margin(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[1] < 2:
        return torch.zeros(scores.shape[0])
    return scores[:, 0] - scores[:, 1]


def evaluate(base: dict, final_pred: list[str]) -> dict:
    changed = wins = losses = known = correct = 0
    for label, base_row, pred in zip(base["labels"], base["predictions"], final_pred):
        if not label:
            continue
        known += 1
        base_pred = base_row[0]
        changed += int(pred != base_pred)
        before = base_pred == label
        after = pred == label
        correct += int(after)
        wins += int(after and not before)
        losses += int(before and not after)
    if not known:
        return {"changed": sum(p != row[0] for p, row in zip(final_pred, base["predictions"]))}
    return {
        "known": known,
        "top1": correct / known,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
    }


def write_predictions(path: Path, image_ids: list[str], predictions: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred in zip(image_ids, predictions):
            writer.writerow({"image_id": image_id, "prediction": pred})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-topk", type=Path, required=True)
    parser.add_argument("--cand-a-topk", type=Path, required=True)
    parser.add_argument("--cand-b-topk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-margin-max", default="all,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--cand-a-margin-min", default="all,0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--cand-b-margin-min", default="all,0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--require-in-base-topk", action="store_true")
    parser.add_argument("--allow-a-only", action="store_true")
    parser.add_argument("--allow-b-only", action="store_true")
    args = parser.parse_args()

    base = load_topk(args.base_topk)
    cand_a = load_topk(args.cand_a_topk)
    cand_b = load_topk(args.cand_b_topk)
    assert base["image_ids"] == cand_a["image_ids"] == cand_b["image_ids"]

    base_pred = [row[0] for row in base["predictions"]]
    a_pred = [row[0] for row in cand_a["predictions"]]
    b_pred = [row[0] for row in cand_b["predictions"]]
    base_sets = [set(row) for row in base["predictions"]]
    base_m = margin(base["top_scores"])
    a_m = margin(cand_a["top_scores"])
    b_m = margin(cand_b["top_scores"])

    rows = []
    best_row = None
    best_pred = None
    for bm in parse_thresholds(args.base_margin_max):
        for am in parse_thresholds(args.cand_a_margin_min):
            for cm in parse_thresholds(args.cand_b_margin_min):
                final = list(base_pred)
                for i in range(len(final)):
                    if bm is not None and float(base_m[i]) > bm:
                        continue
                    if am is not None and float(a_m[i]) < am:
                        continue
                    if cm is not None and float(b_m[i]) < cm:
                        continue
                    chosen = None
                    if a_pred[i] == b_pred[i] and a_pred[i] != base_pred[i]:
                        chosen = a_pred[i]
                    elif args.allow_a_only and a_pred[i] != base_pred[i] and b_pred[i] == base_pred[i]:
                        chosen = a_pred[i]
                    elif args.allow_b_only and b_pred[i] != base_pred[i] and a_pred[i] == base_pred[i]:
                        chosen = b_pred[i]
                    if chosen is None:
                        continue
                    if args.require_in_base_topk and chosen not in base_sets[i]:
                        continue
                    final[i] = chosen
                row = {
                    "base_margin_max": "all" if bm is None else bm,
                    "cand_a_margin_min": "all" if am is None else am,
                    "cand_b_margin_min": "all" if cm is None else cm,
                    "require_in_base_topk": args.require_in_base_topk,
                    "allow_a_only": args.allow_a_only,
                    "allow_b_only": args.allow_b_only,
                    **evaluate(base, final),
                }
                rows.append(row)
                key = (row.get("net_wins", 0), row.get("top1", 0.0), -row.get("losses", 0), -row.get("changed", 0))
                if best_row is None or key > (
                    best_row.get("net_wins", 0),
                    best_row.get("top1", 0.0),
                    -best_row.get("losses", 0),
                    -best_row.get("changed", 0),
                ):
                    best_row = row
                    best_pred = final

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best_pred is not None:
        write_predictions(args.out_dir / "predictions.csv", base["image_ids"], best_pred)
    summary = {
        "base_topk": str(args.base_topk),
        "cand_a_topk": str(args.cand_a_topk),
        "cand_b_topk": str(args.cand_b_topk),
        "best": best_row,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

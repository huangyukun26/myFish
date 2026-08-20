from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def parse_weight_tuples(value: str, expected_len: int) -> list[tuple[float, ...]]:
    tuples: list[tuple[float, ...]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [float(part.strip()) for part in item.split(":") if part.strip()]
        if len(parts) != expected_len:
            raise ValueError(f"Weight tuple {item!r} has {len(parts)} values, expected {expected_len}")
        tuples.append(tuple(parts))
    return tuples


def trigger_mask(
    *,
    predictions: list[list[str]],
    base_scores: torch.Tensor,
    mode: str,
    margin_threshold: float,
    genus_frac_threshold: float,
) -> torch.Tensor:
    values = []
    for row_idx, preds in enumerate(predictions):
        margin = float(base_scores[row_idx, 0] - base_scores[row_idx, 1]) if base_scores.shape[1] > 1 else 0.0
        genera = [genus(pred) for pred in preds]
        counts = Counter(genera)
        top1_frac = counts.get(genera[0], 0) / max(1, len(genera)) if genera else 0.0
        low_margin = margin <= margin_threshold
        clustered = top1_frac >= genus_frac_threshold
        if mode == "all":
            values.append(True)
        elif mode == "low_margin":
            values.append(low_margin)
        elif mode == "clustered":
            values.append(clustered)
        elif mode == "low_margin_or_clustered":
            values.append(low_margin or clustered)
        elif mode == "low_margin_and_clustered":
            values.append(low_margin and clustered)
        else:
            raise ValueError(f"Unknown trigger mode: {mode}")
    return torch.tensor(values, dtype=torch.bool)


def metrics(indices: torch.Tensor, payload: dict[str, Any]) -> dict[str, Any]:
    predictions = payload["predictions"]
    labels = payload["labels"]
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        base_pred = predictions[row_idx][0]
        final_pred = predictions[row_idx][int(indices[row_idx, 0].item())]
        base_ok = base_pred == label
        final_ok = final_pred == label
        changed += int(base_pred != final_pred)
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [predictions[row_idx][int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(predictions[row_idx]) + 1
        ranks.append(rank)
    if not ranks:
        return {}
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
        "triggered": int(getattr(indices, "triggered", 0)),
    }


def write_predictions(path: Path, payload: dict[str, Any], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "label", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, image_id in enumerate(payload["image_ids"]):
            preds = payload["predictions"][row_idx]
            pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": image_id,
                    "prediction": pred,
                    "label": payload["labels"][row_idx],
                    "base_prediction": preds[0],
                    "changed": pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-files", required=True, help="Comma-separated adapter_topk_scores.pt files.")
    parser.add_argument("--weight-grid", default="0,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument(
        "--weight-tuples",
        default="",
        help="Optional comma-separated triggered tuples, e.g. '0.02:0.005,0.01:0.01'. Overrides --weight-grid.",
    )
    parser.add_argument(
        "--always-weight-tuples",
        default="",
        help="Optional comma-separated always-on tuples added before trigger logic, e.g. '0:0.01'.",
    )
    parser.add_argument("--margin-grid", default="0.002,0.005,0.01,0.02,1.0")
    parser.add_argument("--genus-frac-grid", default="0.25,0.30,0.40,1.01")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_paths(args.score_files)
    if not paths:
        raise ValueError("--score-files is empty")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    base = payloads[0]
    for path, payload in zip(paths[1:], payloads[1:]):
        if payload["image_ids"] != base["image_ids"]:
            raise RuntimeError(f"image_ids differ in {path}")
        if payload["predictions"] != base["predictions"]:
            raise RuntimeError(f"predictions differ in {path}")
    base_scores = torch.tensor(base["base_scores"], dtype=torch.float32)
    adapter_scores = [row_zscore(payload["adapter_scores"].float()) for payload in payloads]
    if args.weight_tuples.strip():
        weight_tuples = parse_weight_tuples(args.weight_tuples, len(adapter_scores))
    else:
        grids = [parse_grid(args.weight_grid) for _ in adapter_scores]
        if len(grids) == 1:
            weight_tuples = [(value,) for value in grids[0]]
        elif len(grids) == 2:
            weight_tuples = [(a, b) for a in grids[0] for b in grids[1]]
        else:
            raise ValueError("Only one or two adapter score files are supported")
    if args.always_weight_tuples.strip():
        always_weight_tuples = parse_weight_tuples(args.always_weight_tuples, len(adapter_scores))
    else:
        always_weight_tuples = [tuple(0.0 for _ in adapter_scores)]

    rows = []
    best = None
    best_indices = None
    for always_weights in always_weight_tuples:
        always_extra = torch.zeros_like(base_scores)
        for weight, scores in zip(always_weights, adapter_scores):
            always_extra = always_extra + weight * scores
        always_scores = base_scores + always_extra
        for weights in weight_tuples:
            trigger_extra = torch.zeros_like(base_scores)
            for weight, scores in zip(weights, adapter_scores):
                trigger_extra = trigger_extra + weight * scores
            for margin_threshold in parse_grid(args.margin_grid):
                for genus_frac_threshold in parse_grid(args.genus_frac_grid):
                    for mode in [part.strip() for part in args.trigger_modes.split(",") if part.strip()]:
                        trigger = trigger_mask(
                            predictions=base["predictions"],
                            base_scores=base_scores,
                            mode=mode,
                            margin_threshold=margin_threshold,
                            genus_frac_threshold=genus_frac_threshold,
                        )
                        final_scores = torch.where(trigger[:, None], always_scores + trigger_extra, always_scores)
                        indices = final_scores.argsort(dim=1, descending=True)
                        setattr(indices, "triggered", int(trigger.sum().item()))
                        row = {
                            **{f"always_weight_{idx}": weight for idx, weight in enumerate(always_weights)},
                            **{f"weight_{idx}": weight for idx, weight in enumerate(weights)},
                            "margin_threshold": margin_threshold,
                            "genus_frac_threshold": genus_frac_threshold,
                            "trigger_mode": mode,
                            **metrics(indices, base),
                        }
                        rows.append(row)
                        penalty = sum(abs(w) for w in always_weights) + sum(abs(w) for w in weights)
                        key = (row.get("top1", 0), row.get("net_wins", 0), -row.get("losses", 0), -penalty)
                        if best is None or key > best[0]:
                            best = (key, row)
                            best_indices = indices.clone()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best_indices is not None:
        write_predictions(args.out_dir / "best_predictions.csv", base, best_indices)
    summary = {
        "score_files": [str(path) for path in paths],
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

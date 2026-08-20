from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def load_topk(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_query_features(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {image_id: payload["features"][idx].float() for idx, image_id in enumerate(payload["image_ids"])}


def evaluate_predictions(rows: List[dict], predictions: List[str]) -> dict:
    known = [idx for idx, row in enumerate(rows) if row.get("true_label")]
    if not known:
        return {}
    base_correct = []
    new_correct = []
    topk_hit = []
    for idx in known:
        row = rows[idx]
        true_label = row["true_label"]
        top_classes = row["top_classes"].split("|")
        base_correct.append(top_classes[0] == true_label)
        new_correct.append(predictions[idx] == true_label)
        topk_hit.append(true_label in top_classes)
    base = torch.tensor(base_correct, dtype=torch.bool)
    new = torch.tensor(new_correct, dtype=torch.bool)
    topk = torch.tensor(topk_hit, dtype=torch.bool)
    changed = torch.tensor([rows[idx]["top_classes"].split("|")[0] != predictions[idx] for idx in known], dtype=torch.bool)
    wins = (~base & new)
    losses = (base & ~new)
    return {
        "known": len(known),
        "base_top1": float(base.float().mean().item()),
        "new_top1": float(new.float().mean().item()),
        "topk_recall": float(topk.float().mean().item()),
        "changed": int(changed.sum().item()),
        "changed_frac": float(changed.float().mean().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net_wins": int(wins.sum().item() - losses.sum().item()),
        "win_loss_ratio": float(wins.sum().item() / max(1, losses.sum().item())),
    }


def rerank_rows(
    rows: List[dict],
    query_features: Dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    class_to_idx: Dict[str, int],
    prototype_weight: float,
    margin_threshold: float,
) -> List[str]:
    predictions: List[str] = []
    for row in rows:
        top_classes = row["top_classes"].split("|")
        base_scores = torch.tensor([float(value) for value in row["top_scores"].split("|")], dtype=torch.float32)
        margin = float(row["margin_top1_top2"])
        if margin > margin_threshold or prototype_weight == 0:
            predictions.append(top_classes[0])
            continue
        feature = query_features[row["image_id"]]
        indices = torch.tensor([class_to_idx[name] for name in top_classes], dtype=torch.long)
        proto_scores = prototypes[indices] @ feature
        final_scores = base_scores + prototype_weight * row_zscore(proto_scores)
        predictions.append(top_classes[int(final_scores.argmax().item())])
    return predictions


def write_predictions(path: Path, rows: List[dict], predictions: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"image_id": row["image_id"], "prediction": prediction})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-csv", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--prototype-weight-grid", default="0")
    parser.add_argument("--margin-threshold-grid", default="0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_topk(args.topk_csv)
    query_features = load_query_features(args.query_features)
    proto_payload = torch.load(args.prototypes, map_location="cpu", weights_only=False)
    prototypes = proto_payload["prototypes"].float()
    class_to_idx = {name: idx for idx, name in enumerate(proto_payload["classes"])}

    sweep_rows = []
    best_predictions: List[str] = []
    best_row = None
    for margin_threshold in parse_grid(args.margin_threshold_grid):
        for prototype_weight in parse_grid(args.prototype_weight_grid):
            predictions = rerank_rows(
                rows,
                query_features,
                prototypes,
                class_to_idx,
                prototype_weight=prototype_weight,
                margin_threshold=margin_threshold,
            )
            metrics = evaluate_predictions(rows, predictions)
            row = {
                "prototype_weight": prototype_weight,
                "margin_threshold": margin_threshold,
                **metrics,
            }
            sweep_rows.append(row)
            if metrics:
                key = (metrics["new_top1"], metrics["net_wins"], -metrics["losses"], -metrics["changed"])
            else:
                key = (-abs(prototype_weight), -margin_threshold)
            if best_row is None:
                best_row = row
                best_predictions = predictions
                best_key = key
            elif key > best_key:
                best_row = row
                best_predictions = predictions
                best_key = key

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    write_predictions(args.out_dir / "best_predictions.csv", rows, best_predictions)
    summary = {
        "topk_csv": str(args.topk_csv),
        "query_features": str(args.query_features),
        "prototypes": str(args.prototypes),
        "rows": len(rows),
        "best": best_row,
        "out_predictions": str(args.out_dir / "best_predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_grid(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def load_topk(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def evaluate_predictions(rows: List[dict], predictions: List[str]) -> dict:
    known = [idx for idx, row in enumerate(rows) if row.get("true_label")]
    if not known:
        return {}
    base_correct = []
    new_correct = []
    topk_hit = []
    changed_flags = []
    for idx in known:
        row = rows[idx]
        true_label = row["true_label"]
        top_classes = row["top_classes"].split("|")
        base_correct.append(top_classes[0] == true_label)
        new_correct.append(predictions[idx] == true_label)
        topk_hit.append(true_label in top_classes)
        changed_flags.append(top_classes[0] != predictions[idx])
    base = torch.tensor(base_correct, dtype=torch.bool)
    new = torch.tensor(new_correct, dtype=torch.bool)
    topk = torch.tensor(topk_hit, dtype=torch.bool)
    changed = torch.tensor(changed_flags, dtype=torch.bool)
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


def load_feature_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["features"] = F.normalize(payload["features"].float(), dim=1)
    return payload


def build_neighbor_candidate_scores(
    rows: List[dict],
    query_payload: dict,
    train_payload: dict,
    max_neighbors: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    class_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(train_payload["classes"])}
    query_by_image = {image_id: idx for idx, image_id in enumerate(query_payload["image_ids"])}
    candidate_ids = []
    query_indices = []
    for row in rows:
        query_indices.append(query_by_image[row["image_id"]])
        candidate_ids.append([class_to_idx[name] for name in row["top_classes"].split("|")])
    candidate_ids_t = torch.tensor(candidate_ids, dtype=torch.long)
    query_indices_t = torch.tensor(query_indices, dtype=torch.long)

    query_features = query_payload["features"][query_indices_t]
    train_features = train_payload["features"].to(device=device, dtype=torch.float16)
    train_class_ids = train_payload["class_ids"].long().cpu()
    max_neighbors = min(max_neighbors, train_features.shape[0])

    all_scores = torch.empty((len(rows), candidate_ids_t.shape[1]), dtype=torch.float32)
    fill_value = -1.0
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc="neighbor_scores"):
            end = min(start + batch_size, len(rows))
            q = query_features[start:end].to(device=device, dtype=torch.float16)
            sims = q @ train_features.T
            top_values, top_indices = sims.topk(max_neighbors, dim=1)
            top_values_cpu = top_values.float().cpu()
            top_classes_cpu = train_class_ids[top_indices.cpu()]
            cand_cpu = candidate_ids_t[start:end]
            for local_idx in range(end - start):
                class_to_sim: Dict[int, float] = {}
                for cls, sim in zip(top_classes_cpu[local_idx].tolist(), top_values_cpu[local_idx].tolist()):
                    if cls not in class_to_sim:
                        class_to_sim[cls] = sim
                scores = [class_to_sim.get(int(cls), fill_value) for cls in cand_cpu[local_idx].tolist()]
                all_scores[start + local_idx] = torch.tensor(scores, dtype=torch.float32)
    return all_scores


def rerank_rows(
    rows: List[dict],
    neighbor_scores: torch.Tensor,
    neighbor_weight: float,
    margin_threshold: float,
) -> List[str]:
    predictions: List[str] = []
    for idx, row in enumerate(rows):
        top_classes = row["top_classes"].split("|")
        base_scores = torch.tensor([float(value) for value in row["top_scores"].split("|")], dtype=torch.float32)
        margin = float(row["margin_top1_top2"])
        if margin > margin_threshold or neighbor_weight == 0:
            predictions.append(top_classes[0])
            continue
        final_scores = base_scores + neighbor_weight * row_zscore(neighbor_scores[idx])
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
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--neighbor-weight-grid", default="0")
    parser.add_argument("--margin-threshold-grid", default="0")
    parser.add_argument("--max-neighbors", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_topk(args.topk_csv)
    query_payload = load_feature_payload(args.query_features)
    train_payload = load_feature_payload(args.train_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neighbor_scores = build_neighbor_candidate_scores(
        rows=rows,
        query_payload=query_payload,
        train_payload=train_payload,
        max_neighbors=args.max_neighbors,
        batch_size=args.batch_size,
        device=device,
    )

    sweep_rows = []
    best_predictions: List[str] = []
    best_row = None
    best_key = None
    for margin_threshold in parse_grid(args.margin_threshold_grid):
        for neighbor_weight in parse_grid(args.neighbor_weight_grid):
            predictions = rerank_rows(
                rows,
                neighbor_scores,
                neighbor_weight=neighbor_weight,
                margin_threshold=margin_threshold,
            )
            metrics = evaluate_predictions(rows, predictions)
            row = {
                "neighbor_weight": neighbor_weight,
                "margin_threshold": margin_threshold,
                "max_neighbors": args.max_neighbors,
                **metrics,
            }
            sweep_rows.append(row)
            if metrics:
                key = (metrics["new_top1"], metrics["net_wins"], -metrics["losses"], -metrics["changed"])
            else:
                key = (-abs(neighbor_weight), -margin_threshold)
            if best_key is None or key > best_key:
                best_row = row
                best_predictions = predictions
                best_key = key

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": neighbor_scores,
            "topk_csv": str(args.topk_csv),
            "query_features": str(args.query_features),
            "train_features": str(args.train_features),
            "max_neighbors": args.max_neighbors,
        },
        args.out_dir / "neighbor_candidate_scores.pt",
    )
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    write_predictions(args.out_dir / "best_predictions.csv", rows, best_predictions)
    summary = {
        "topk_csv": str(args.topk_csv),
        "query_features": str(args.query_features),
        "train_features": str(args.train_features),
        "rows": len(rows),
        "device": str(device),
        "best": best_row,
        "out_predictions": str(args.out_dir / "best_predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

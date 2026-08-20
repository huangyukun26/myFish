from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def parse_grid(value: str, cast=float):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm(dim=1, keepdim=True).clamp_min(1e-12)


def read_topk_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            classes = row.get("top_classes", "").split("|")
            scores = [float(value) for value in row.get("top_scores", "").split("|") if value]
            rows.append(
                {
                    "image_id": row["image_id"],
                    "label": row.get("true_label", row.get("label", "")),
                    "classes": classes,
                    "scores": scores,
                    "margin": float(row.get("margin_top1_top2", scores[0] - scores[1] if len(scores) > 1 else 0.0)),
                }
            )
    return rows


def align_features(payload: dict[str, Any], image_ids: list[str]) -> torch.Tensor:
    by_id = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} image ids missing from feature payload; first={missing[:5]}")
    indices = torch.tensor([by_id[image_id] for image_id in image_ids], dtype=torch.long)
    return normalize_features(payload["features"][indices])


def compute_neighbors(features: torch.Tensor, *, max_neighbors: int, chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = normalize_features(features).to(device)
    n = features.shape[0]
    all_scores = []
    all_indices = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = features[start:end] @ features.T
        local = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), local] = -1e9
        scores, indices = sims.topk(max_neighbors, dim=1)
        all_scores.append(scores.cpu())
        all_indices.append(indices.cpu())
    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)


def make_vote_scores(
    *,
    rows: list[dict[str, Any]],
    neighbor_scores: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_count: int,
    min_neighbor_margin: float,
    sim_floor: float,
) -> torch.Tensor:
    n = len(rows)
    k = len(rows[0]["classes"])
    candidate_maps = [{name: idx for idx, name in enumerate(row["classes"])} for row in rows]
    top1_preds = [row["classes"][0] for row in rows]
    margins = torch.tensor([row["margin"] for row in rows], dtype=torch.float32)
    votes = torch.zeros((n, k), dtype=torch.float32)
    for row_idx in range(n):
        cmap = candidate_maps[row_idx]
        for rank in range(neighbor_count):
            nb = int(neighbor_indices[row_idx, rank].item())
            if margins[nb] < min_neighbor_margin:
                continue
            cls = top1_preds[nb]
            col = cmap.get(cls)
            if col is None:
                continue
            sim = float(neighbor_scores[row_idx, rank].item())
            if sim < sim_floor:
                continue
            votes[row_idx, col] += sim * (1.0 + margins[nb])
    return votes


def rank_indices(rows: list[dict[str, Any]], final_scores: torch.Tensor) -> torch.Tensor:
    return final_scores.argsort(dim=1, descending=True)


def metrics(rows: list[dict[str, Any]], indices: torch.Tensor) -> dict[str, Any]:
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    known = 0
    for row_idx, row in enumerate(rows):
        label = row.get("label", "")
        preds = row["classes"]
        base = preds[0]
        final = preds[int(indices[row_idx, 0].item())]
        changed += int(base != final)
        if not label:
            continue
        known += 1
        base_ok = base == label
        final_ok = final == label
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [preds[int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        ranks.append(rank)
    out = {"known": known, "changed": changed, "wins": wins, "losses": losses, "net_wins": wins - losses}
    if ranks:
        ranks_t = torch.tensor(ranks)
        out.update(
            {
                "top1": float((ranks_t <= 1).float().mean().item()),
                "top5": float((ranks_t <= 5).float().mean().item()),
                "top20": float((ranks_t <= 20).float().mean().item()),
                "mrr": float((1.0 / ranks_t.float()).mean().item()),
            }
        )
    return out


def write_predictions(path: Path, rows: list[dict[str, Any]], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "label", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            preds = row["classes"]
            pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "prediction": pred,
                    "label": row.get("label", ""),
                    "base_prediction": preds[0],
                    "changed": pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-csv", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--neighbor-count-grid", default="5,10,20,50,100")
    parser.add_argument("--min-neighbor-margin-grid", default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--base-margin-max-grid", default="0.002,0.005,0.01,0.02,0.05,0.1,1.0")
    parser.add_argument("--weight-grid", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--sim-floor-grid", default="-1,0,0.2,0.4")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    rows = read_topk_csv(args.topk_csv)
    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    features = align_features(payload, [row["image_id"] for row in rows])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neighbor_scores, neighbor_indices = compute_neighbors(
        features,
        max_neighbors=args.max_neighbors,
        chunk_size=args.chunk_size,
        device=device,
    )
    base_scores = torch.tensor([row["scores"] for row in rows], dtype=torch.float32)
    base_margins = torch.tensor([row["margin"] for row in rows], dtype=torch.float32)

    sweep_rows = []
    best = None
    best_indices = None
    for neighbor_count in parse_grid(args.neighbor_count_grid, int):
        for min_neighbor_margin in parse_grid(args.min_neighbor_margin_grid, float):
            for sim_floor in parse_grid(args.sim_floor_grid, float):
                votes = make_vote_scores(
                    rows=rows,
                    neighbor_scores=neighbor_scores,
                    neighbor_indices=neighbor_indices,
                    neighbor_count=neighbor_count,
                    min_neighbor_margin=min_neighbor_margin,
                    sim_floor=sim_floor,
                )
                vote_scores = row_zscore(votes)
                for base_margin_max in parse_grid(args.base_margin_max_grid, float):
                    trigger = base_margins <= base_margin_max
                    for weight in parse_grid(args.weight_grid, float):
                        final_scores = base_scores + weight * vote_scores
                        final_scores = torch.where(trigger[:, None], final_scores, base_scores)
                        indices = rank_indices(rows, final_scores)
                        row = {
                            "neighbor_count": neighbor_count,
                            "min_neighbor_margin": min_neighbor_margin,
                            "sim_floor": sim_floor,
                            "base_margin_max": base_margin_max,
                            "weight": weight,
                            "triggered": int(trigger.sum().item()),
                            **metrics(rows, indices),
                        }
                        sweep_rows.append(row)
                        key = (row.get("top1", 0), row.get("net_wins", 0), -row.get("losses", 0), -row["changed"], -abs(weight))
                        if best is None or key > best[0]:
                            best = (key, row)
                            best_indices = indices.clone()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    if best_indices is not None:
        write_predictions(args.out_dir / "best_predictions.csv", rows, best_indices)
    summary = {
        "topk_csv": str(args.topk_csv),
        "features": str(args.features),
        "rows": len(rows),
        "feature_dim": int(features.shape[1]),
        "max_neighbors": args.max_neighbors,
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

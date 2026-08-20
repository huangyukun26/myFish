from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_topk(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    candidates = list(payload["candidates"])
    top_indices = payload["top_indices"].long()
    top_scores = payload["top_scores"].float()
    return {
        "image_ids": list(payload["image_ids"]),
        "labels": list(payload.get("labels", [""] * len(payload["image_ids"]))),
        "candidates": candidates,
        "top_indices": top_indices,
        "top_scores": top_scores,
    }


def load_features(path: Path, image_ids: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    by_id = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    idx = torch.tensor([by_id[image_id] for image_id in image_ids], dtype=torch.long)
    return F.normalize(payload["features"][idx].float(), dim=1)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def build_knn_graph(features: torch.Tensor, neighbors: int, sim_power: float, device: torch.device) -> torch.Tensor:
    x = features.to(device)
    sim = x @ x.T
    sim.fill_diagonal_(-1.0)
    vals, idx = sim.topk(min(neighbors, sim.shape[0] - 1), dim=1)
    weights = vals.clamp_min(0.0).pow(sim_power)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    graph = torch.zeros_like(sim)
    graph.scatter_(1, idx, weights)
    return graph


def initial_distribution(top_indices: torch.Tensor, top_scores: torch.Tensor, class_count: int, temp: float, device: torch.device) -> torch.Tensor:
    n, k = top_indices.shape
    y = torch.zeros((n, class_count), dtype=torch.float32, device=device)
    probs = torch.softmax(row_zscore(top_scores.to(device)) / temp, dim=1)
    y.scatter_(1, top_indices.to(device), probs)
    return y


def metrics(pred_idx: torch.Tensor, topk: dict) -> dict:
    labels = topk["labels"]
    candidates = topk["candidates"]
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    base = topk["top_indices"][:, 0]
    known = correct = changed = wins = losses = 0
    for i, label in enumerate(labels):
        if not label:
            continue
        true = class_to_idx.get(label)
        if true is None:
            continue
        known += 1
        before = int(base[i]) == true
        after = int(pred_idx[i]) == true
        correct += int(after)
        changed += int(int(base[i]) != int(pred_idx[i]))
        wins += int(after and not before)
        losses += int(before and not after)
    if not known:
        return {"changed": int((pred_idx.cpu() != base).sum().item())}
    return {
        "known": known,
        "top1": correct / known,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
    }


def parse_grid(value: str, cast=float):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def write_predictions(path: Path, topk: dict, pred_idx: torch.Tensor) -> None:
    candidates = topk["candidates"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, idx in zip(topk["image_ids"], pred_idx.cpu().tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--neighbors-grid", default="10,20,50,100")
    parser.add_argument("--sim-power-grid", default="1,2,4")
    parser.add_argument("--temp-grid", default="0.5,0.75,1,1.5")
    parser.add_argument("--alpha-grid", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--iters-grid", default="1,2,4,8")
    parser.add_argument("--weight-grid", default="0,0.1,0.2,0.5,1,2,5")
    args = parser.parse_args()

    topk = load_topk(args.topk)
    features = load_features(args.features, topk["image_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_count = len(topk["candidates"])
    top_indices = topk["top_indices"].to(device)
    base_scores = topk["top_scores"].to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    best_row = None
    best_pred = None
    graph_cache: dict[tuple[int, float], torch.Tensor] = {}
    y_cache: dict[float, torch.Tensor] = {}

    for neighbors in parse_grid(args.neighbors_grid, int):
        for sim_power in parse_grid(args.sim_power_grid, float):
            graph_key = (neighbors, sim_power)
            if graph_key not in graph_cache:
                graph_cache[graph_key] = build_knn_graph(features, neighbors, sim_power, device)
            graph = graph_cache[graph_key]
            for temp in parse_grid(args.temp_grid, float):
                if temp not in y_cache:
                    y_cache[temp] = initial_distribution(top_indices.cpu(), topk["top_scores"], class_count, temp, device)
                y0 = y_cache[temp]
                for alpha in parse_grid(args.alpha_grid, float):
                    f = y0
                    states: dict[int, torch.Tensor] = {}
                    max_iters = max(parse_grid(args.iters_grid, int))
                    for it in range(1, max_iters + 1):
                        f = (1.0 - alpha) * y0 + alpha * (graph @ f)
                        if it in parse_grid(args.iters_grid, int):
                            states[it] = f
                    for iters, propagated in states.items():
                        prop_topk = propagated.gather(1, top_indices)
                        for weight in parse_grid(args.weight_grid, float):
                            final = row_zscore(base_scores) + weight * row_zscore(prop_topk)
                            pos = final.argmax(dim=1)
                            pred_idx = top_indices[torch.arange(top_indices.shape[0], device=device), pos].cpu()
                            row = {
                                "neighbors": neighbors,
                                "sim_power": sim_power,
                                "temp": temp,
                                "alpha": alpha,
                                "iters": iters,
                                "weight": weight,
                                **metrics(pred_idx, topk),
                            }
                            sweep_rows.append(row)
                            key = (row.get("net_wins", 0), row.get("top1", 0.0), -row.get("losses", 0), -row.get("changed", 0))
                            if best_row is None or key > (
                                best_row.get("net_wins", 0),
                                best_row.get("top1", 0.0),
                                -best_row.get("losses", 0),
                                -best_row.get("changed", 0),
                            ):
                                best_row = row
                                best_pred = pred_idx

    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    if best_pred is not None:
        write_predictions(args.out_dir / "predictions.csv", topk, best_pred)
    summary = {
        "topk": str(args.topk),
        "features": str(args.features),
        "best": best_row,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

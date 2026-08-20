from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def read_topk_jsonl(path: Path) -> dict[str, Any]:
    image_ids: list[str] = []
    predictions: list[list[str]] = []
    scores: list[list[float]] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_ids.append(row["image_id"])
            predictions.append(list(row["predictions"]))
            scores.append([float(v) for v in row["scores"]])
            labels.append(row.get("label", ""))
    return {"image_ids": image_ids, "predictions": predictions, "base_scores": scores, "labels": labels}


def load_topk(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "top_indices" in payload and "top_scores" in payload:
            candidates = list(payload["candidates"])
            top_indices = payload["top_indices"].long()
            return {
                "image_ids": list(payload["image_ids"]),
                "predictions": [[candidates[int(idx)] for idx in row.tolist()] for row in top_indices],
                "base_scores": payload["top_scores"].float().tolist(),
                "labels": payload.get("labels", [""] * len(payload["image_ids"])),
            }
        return {
            "image_ids": list(payload["image_ids"]),
            "predictions": payload["predictions"],
            "base_scores": payload["base_scores"],
            "labels": payload.get("labels", [""] * len(payload["image_ids"])),
        }
    return read_topk_jsonl(path)


def load_features(path: Path, image_ids: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    by_id = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} feature rows missing; first={missing[:5]}")
    indices = torch.tensor([by_id[image_id] for image_id in image_ids], dtype=torch.long)
    return F.normalize(payload["features"][indices].float(), dim=1)


def knn(features: torch.Tensor, *, neighbors: int, chunk_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = features.to(device)
    all_values = []
    all_indices = []
    n = x.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sim = x[start:end] @ x.T
        row = torch.arange(start, end, device=device)
        sim[torch.arange(end - start, device=device), row] = -1.0
        values, indices = sim.topk(neighbors, dim=1)
        all_values.append(values.cpu())
        all_indices.append(indices.cpu())
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)


def build_neighbor_scores(
    *,
    predictions: list[list[str]],
    base_scores: torch.Tensor,
    nn_values: torch.Tensor,
    nn_indices: torch.Tensor,
    sim_threshold: float,
    neighbor_topm: int,
    confidence_scale: float,
) -> torch.Tensor:
    n, k = base_scores.shape
    out = torch.zeros((n, k), dtype=torch.float32)
    margins = (base_scores[:, 0] - base_scores[:, 1]).clamp_min(0.0) if k > 1 else torch.zeros(n)
    pred_maps = [{name: idx for idx, name in enumerate(row)} for row in predictions]
    for i in range(n):
        cand_to_col = pred_maps[i]
        for sim, j_tensor in zip(nn_values[i].tolist(), nn_indices[i].tolist()):
            if sim < sim_threshold:
                continue
            j = int(j_tensor)
            vote_weight = (sim - sim_threshold) * (1.0 + confidence_scale * float(margins[j]))
            if vote_weight <= 0:
                continue
            for rank, cls in enumerate(predictions[j][:neighbor_topm]):
                col = cand_to_col.get(cls)
                if col is not None:
                    out[i, col] += vote_weight / float(rank + 1)
    return out


def metrics(indices: torch.Tensor, payload: dict[str, Any]) -> dict[str, Any]:
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    for row_idx, label in enumerate(payload["labels"]):
        if not label:
            continue
        preds = payload["predictions"][row_idx]
        base_pred = preds[0]
        final_pred = preds[int(indices[row_idx, 0].item())]
        changed += int(base_pred != final_pred)
        base_ok = base_pred == label
        final_ok = final_pred == label
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [preds[int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
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
    }


def parse_grid(value: str, cast=float) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def write_predictions(path: Path, payload: dict[str, Any], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, image_id in enumerate(payload["image_ids"]):
            preds = payload["predictions"][row_idx]
            pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": image_id,
                    "prediction": pred,
                    "base_prediction": preds[0],
                    "changed": pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=Path, required=True, help=".jsonl topK or .pt adapter score dump")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--sim-threshold-grid", default="0.85,0.90,0.93,0.95")
    parser.add_argument("--neighbor-topm-grid", default="1,3")
    parser.add_argument("--weight-grid", default="0,0.01,0.02,0.05,0.1")
    parser.add_argument("--confidence-scale-grid", default="0,1")
    parser.add_argument("--apply-sim-threshold", type=float, default=None)
    parser.add_argument("--apply-neighbor-topm", type=int, default=None)
    parser.add_argument("--apply-weight", type=float, default=None)
    parser.add_argument("--apply-confidence-scale", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = load_topk(args.topk)
    base_scores = torch.tensor(payload["base_scores"], dtype=torch.float32)
    features = load_features(args.features, payload["image_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_values, nn_indices = knn(features, neighbors=args.neighbors, chunk_size=args.chunk_size, device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sim_quantiles = {
        f"q{int(q * 100):02d}": float(torch.quantile(nn_values[:, 0], q).item())
        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    }
    sweep_rows = []
    best = None
    best_indices = None
    if args.apply_sim_threshold is not None and args.apply_weight is not None and args.apply_neighbor_topm is not None:
        sweep = [
            (
                args.apply_sim_threshold,
                args.apply_neighbor_topm,
                args.apply_weight,
                args.apply_confidence_scale,
            )
        ]
    else:
        sweep = [
            (sim_thr, topm, weight, conf)
            for sim_thr in parse_grid(args.sim_threshold_grid, float)
            for topm in parse_grid(args.neighbor_topm_grid, int)
            for weight in parse_grid(args.weight_grid, float)
            for conf in parse_grid(args.confidence_scale_grid, float)
        ]
    cache: dict[tuple[float, int, float], torch.Tensor] = {}
    for sim_thr, topm, weight, conf in sweep:
        key = (sim_thr, topm, conf)
        if key not in cache:
            cache[key] = build_neighbor_scores(
                predictions=payload["predictions"],
                base_scores=base_scores,
                nn_values=nn_values,
                nn_indices=nn_indices,
                sim_threshold=sim_thr,
                neighbor_topm=topm,
                confidence_scale=conf,
            )
        neighbor_scores = cache[key]
        final = base_scores + weight * row_zscore(neighbor_scores)
        indices = final.argsort(dim=1, descending=True)
        row = {
            "sim_threshold": sim_thr,
            "neighbor_topm": topm,
            "weight": weight,
            "confidence_scale": conf,
            "nonzero_rows": int((neighbor_scores.abs().sum(dim=1) > 0).sum().item()),
            **metrics(indices, payload),
        }
        if not row.get("top1"):
            row.update({"changed": int((indices[:, 0] != 0).sum().item())})
        sweep_rows.append(row)
        key_score = (row.get("top1", 0.0), row.get("net_wins", 0), -row.get("losses", 0), -row["changed"])
        if best is None or key_score > best[0]:
            best = (key_score, row)
            best_indices = indices.clone()

    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    assert best is not None and best_indices is not None
    write_predictions(args.out_dir / "predictions.csv", payload, best_indices)
    torch.save(
        {
            "image_ids": payload["image_ids"],
            "predictions": payload["predictions"],
            "base_scores": base_scores,
            "neighbor_values": nn_values,
            "neighbor_indices": nn_indices,
            "labels": payload["labels"],
            "best": best[1],
        },
        args.out_dir / "neighbor_topk_scores.pt",
    )
    summary = {
        "topk": str(args.topk),
        "features": str(args.features),
        "rows": len(payload["image_ids"]),
        "neighbors": args.neighbors,
        "sim_quantiles": sim_quantiles,
        "best": best[1],
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

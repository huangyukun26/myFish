from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def genus_of(name: str) -> str:
    parts = str(name).strip().split()
    return parts[0] if parts else ""


def load_excluded(path: Path | None, mode: str) -> set[str]:
    if path is None:
        return set()
    classes = json.loads(path.read_text(encoding="utf-8"))
    if mode == "species":
        return set(classes)
    if mode == "genus":
        return {genus_of(name) for name in classes}
    raise ValueError(f"unknown exclude mode: {mode}")


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def parse_thresholds(value: str) -> list[float | None]:
    out: list[float | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(None if item == "all" else float(item))
    return out or [None]


def topk_metrics(pred_indices: torch.Tensor, labels: list[str], candidates: list[str], top_indices: torch.Tensor | None = None) -> dict:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    ranks: list[int] = []
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        if int(pred_indices[row_idx]) == true_idx:
            ranks.append(1)
        elif top_indices is not None:
            row = top_indices[row_idx].tolist()
            ranks.append(row.index(true_idx) + 1 if true_idx in row else len(row) + 1)
        else:
            ranks.append(2)
    if not ranks:
        return {}
    r = torch.tensor(ranks)
    return {
        "known": int(r.numel()),
        "top1": float((r <= 1).float().mean().item()),
        "top5": float((r <= 5).float().mean().item()),
        "top20": float((r <= 20).float().mean().item()),
        "mrr": float((1.0 / r.float()).mean().item()),
        "mean_rank": float(r.float().mean().item()),
        "median_rank": float(r.float().median().item()),
    }


def build_genus_prototypes(train_payload: dict, exclude: set[str], exclude_mode: str) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    features = F.normalize(train_payload["features"].float(), dim=1)
    labels = list(train_payload["labels"])

    genus_to_feats: dict[str, list[torch.Tensor]] = {}
    for idx, label in enumerate(labels):
        g = genus_of(label)
        if not g:
            continue
        if exclude_mode == "species" and label in exclude:
            continue
        if exclude_mode == "genus" and g in exclude:
            continue
        genus_to_feats.setdefault(g, []).append(features[idx])

    genera = sorted(genus_to_feats)
    proto = torch.stack([torch.stack(genus_to_feats[g]).mean(dim=0) for g in genera])
    proto = F.normalize(proto, dim=1)
    return proto, genera, {g: i for i, g in enumerate(genera)}


def write_predictions(path: Path, image_ids: list[str], predictions: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred in zip(image_ids, predictions):
            writer.writerow({"image_id": image_id, "prediction": pred})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--base-topk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-classes", type=Path, default=None)
    parser.add_argument("--exclude-mode", choices=["species", "genus"], default="species")
    parser.add_argument("--weights", default="0,0.25,0.5,0.75,1,1.5,2,3,5")
    parser.add_argument("--base-margin-max", default="all")
    parser.add_argument("--genus-margin-min", default="all")
    parser.add_argument("--top-genus-k", type=int, default=0)
    args = parser.parse_args()

    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    topk_payload = torch.load(args.base_topk, map_location="cpu", weights_only=False)

    excluded = load_excluded(args.exclude_classes, args.exclude_mode)
    proto, genera, genus_to_idx = build_genus_prototypes(train_payload, excluded, args.exclude_mode)

    query_features = F.normalize(query_payload["features"].float(), dim=1)
    genus_logits = query_features @ proto.T
    genus_top_scores, genus_top_idx = genus_logits.topk(min(10, genus_logits.shape[1]), dim=1)
    genus_margin = genus_top_scores[:, 0] - genus_top_scores[:, 1] if genus_top_scores.shape[1] > 1 else torch.zeros(genus_logits.shape[0])

    image_ids = list(topk_payload["image_ids"])
    labels = list(topk_payload.get("labels", [""] * len(image_ids)))
    candidates = list(topk_payload["candidates"])
    top_indices = topk_payload["top_indices"].long()
    top_scores = topk_payload["top_scores"].float()
    base_margin = top_scores[:, 0] - top_scores[:, 1]
    base_pred = top_indices[:, 0]

    cand_genus_ids = torch.full_like(top_indices, -1)
    for row_pos in range(top_indices.shape[1]):
        for row_idx, class_idx in enumerate(top_indices[:, row_pos].tolist()):
            g = genus_of(candidates[class_idx])
            cand_genus_ids[row_idx, row_pos] = genus_to_idx.get(g, -1)

    genus_for_topk = torch.zeros_like(top_scores)
    known_mask = cand_genus_ids >= 0
    rows = torch.arange(top_indices.shape[0]).unsqueeze(1).expand_as(top_indices)
    genus_for_topk[known_mask] = genus_logits[rows[known_mask], cand_genus_ids[known_mask]]
    genus_for_topk_z = row_zscore(genus_for_topk)

    if args.top_genus_k > 0:
        allowed = torch.zeros_like(known_mask)
        top_allowed = genus_top_idx[:, : args.top_genus_k]
        for pos in range(top_indices.shape[1]):
            allowed[:, pos] = (cand_genus_ids[:, pos:pos + 1] == top_allowed).any(dim=1)
        genus_for_topk_z = torch.where(allowed, genus_for_topk_z, torch.full_like(genus_for_topk_z, -3.0))

    base_thresholds = parse_thresholds(args.base_margin_max)
    genus_thresholds = parse_thresholds(args.genus_margin_min)
    weights = [float(x) for x in args.weights.split(",") if x.strip()]

    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    base_top1 = topk_metrics(base_pred, labels, candidates, top_indices).get("top1")
    rows_out: list[dict] = []
    best: dict | None = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for base_thr in base_thresholds:
        for genus_thr in genus_thresholds:
            trigger = torch.ones(top_indices.shape[0], dtype=torch.bool)
            if base_thr is not None:
                trigger &= base_margin <= base_thr
            if genus_thr is not None:
                trigger &= genus_margin >= genus_thr

            for weight in weights:
                final_scores = row_zscore(top_scores) + weight * genus_for_topk_z
                rerank_pos = final_scores.argmax(dim=1)
                pred = top_indices[torch.arange(top_indices.shape[0]), rerank_pos]
                pred = torch.where(trigger, pred, base_pred)
                metrics = topk_metrics(pred, labels, candidates, top_indices)

                changed = int((pred != base_pred).sum().item())
                wins = losses = 0
                if metrics:
                    for i, label in enumerate(labels):
                        true_idx = class_to_idx.get(label)
                        if true_idx is None:
                            continue
                        before = int(base_pred[i]) == true_idx
                        after = int(pred[i]) == true_idx
                        wins += int(after and not before)
                        losses += int(before and not after)

                row = {
                    "weight": weight,
                    "base_margin_max": "all" if base_thr is None else base_thr,
                    "genus_margin_min": "all" if genus_thr is None else genus_thr,
                    "triggered": int(trigger.sum().item()),
                    "changed": changed,
                    "wins": wins,
                    "losses": losses,
                    "net_wins": wins - losses,
                    "base_top1": base_top1,
                    **metrics,
                }
                rows_out.append(row)
                if metrics and (best is None or (row["net_wins"], row["top1"], -row["changed"]) > (best["net_wins"], best["top1"], -best["changed"])):
                    best = row | {"pred_indices": pred}

    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [k for k in rows_out[0].keys() if k != "pred_indices"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: v for k, v in row.items() if k != "pred_indices"})

    if best is not None:
        pred_indices = best.pop("pred_indices")
        predictions = [candidates[int(idx)] for idx in pred_indices.tolist()]
        write_predictions(args.out_dir / "predictions.csv", image_ids, predictions)

    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "base_topk": str(args.base_topk),
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_mode": args.exclude_mode,
        "genera": len(genera),
        "rows": len(image_ids),
        "top_genus_k": args.top_genus_k,
        "best": best,
        "sweep_csv": str(sweep_path),
        "predictions_csv": str(args.out_dir / "predictions.csv") if best is not None else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

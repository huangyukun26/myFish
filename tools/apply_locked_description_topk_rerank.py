"""Apply a pre-registered BioCLIP description rerank inside a fixed Top-K.

This script does not tune parameters: it materializes the base and reranked
label, plus the base margin, so a rule selected on one proxy split can be
tested unchanged on independent pseudo-unseen splits or the public queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.keys()) if isinstance(data, dict) else list(data)


def load_features(path: Path, candidates: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    lookup = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in lookup]
    if missing:
        raise RuntimeError(f"{len(missing)} classes missing from {path}; first={missing[:3]}")
    indices = torch.tensor([lookup[name] for name in candidates], dtype=torch.long)
    return F.normalize(payload["features"][indices].float(), dim=1)


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(1e-6)


def metrics(base: torch.Tensor, reranked: torch.Tensor, labels: list[str], candidates: list[str]) -> dict[str, int | float]:
    idx = {name: pos for pos, name in enumerate(candidates)}
    known = correct_base = correct_rerank = wins = losses = changed = 0
    for row, label in enumerate(labels):
        truth = idx.get(label)
        if truth is None:
            continue
        known += 1
        before = int(base[row])
        after = int(reranked[row])
        base_ok = before == truth
        rerank_ok = after == truth
        correct_base += base_ok
        correct_rerank += rerank_ok
        changed += before != after
        wins += (not base_ok) and rerank_ok
        losses += base_ok and (not rerank_ok)
    return {
        "known": known,
        "base_top1": correct_base / known if known else 0.0,
        "rerank_top1": correct_rerank / known if known else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--fish-text", type=Path, required=True)
    parser.add_argument("--taxon-text", type=Path, required=True)
    parser.add_argument("--description-text", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--taxon-weight", type=float, default=0.85)
    parser.add_argument("--rerank-weight", type=float, default=-0.0075)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes)
    fish = load_features(args.fish_text, candidates)
    taxon = load_features(args.taxon_text, candidates)
    description = load_features(args.description_text, candidates)
    images = F.normalize(image_payload["features"].float(), dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fish, taxon, description = fish.to(device), taxon.to(device), description.to(device)

    base_indices: list[torch.Tensor] = []
    rerank_indices: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for start in range(0, len(images), args.batch_size):
        image = images[start : start + args.batch_size].to(device)
        base_logits = (1.0 - args.taxon_weight) * (image @ fish.T) + args.taxon_weight * (image @ taxon.T)
        values, indices = base_logits.topk(min(args.topk, len(candidates)), dim=1)
        description_scores = torch.gather(image @ description.T, 1, indices)
        final = values + args.rerank_weight * zscore(description_scores)
        base_indices.append(indices[:, 0].cpu())
        rerank_indices.append(torch.gather(indices, 1, final.argmax(dim=1, keepdim=True)).squeeze(1).cpu())
        margins.append((values[:, 0] - values[:, 1]).cpu())

    base = torch.cat(base_indices)
    reranked = torch.cat(rerank_indices)
    margin = torch.cat(margins)
    payload = {
        "image_ids": list(image_payload["image_ids"]),
        "labels": list(image_payload.get("labels", [""] * len(base))),
        "candidates": candidates,
        "base": base,
        "reranked": reranked,
        "base_margin": margin,
        "config": {
            "taxon_weight": args.taxon_weight,
            "rerank_weight": args.rerank_weight,
            "topk": args.topk,
            "image_features": str(args.image_features),
        },
    }
    payload["metrics_if_labeled"] = metrics(base, reranked, payload["labels"], candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(json.dumps({"out": str(args.out), **payload["metrics_if_labeled"], "rows": len(base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

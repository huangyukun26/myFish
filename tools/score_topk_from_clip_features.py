from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_topk(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def metrics(indices: torch.Tensor, labels: Sequence[str], predictions: Sequence[Sequence[str]]) -> dict[str, float | int]:
    labeled = 0
    top1 = 0
    top5 = 0
    top20 = 0
    changed = 0
    wins = 0
    losses = 0
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        labeled += 1
        preds = list(predictions[row_idx])
        base_pred = preds[0]
        final_pred = preds[int(indices[row_idx, 0].item())]
        base_ok = base_pred == label
        final_ok = final_pred == label
        changed += int(base_pred != final_pred)
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        ranked_preds = [preds[int(idx)] for idx in indices[row_idx].tolist()]
        try:
            rank = ranked_preds.index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
        top20 += int(rank <= 20)
    return {
        "labeled": labeled,
        "top1": top1 / labeled if labeled else 0.0,
        "top5": top5 / labeled if labeled else 0.0,
        "top20": top20 / labeled if labeled else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weight-grid", default="-0.5,-0.2,-0.1,-0.05,-0.02,0,0.02,0.05,0.1,0.2,0.5,1.0")
    args = parser.parse_args()

    rows = load_topk(args.topk_jsonl)
    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    image_ids = [row["image_id"] for row in rows]
    labels = [row.get("label", "") for row in rows]
    predictions = [list(row["predictions"]) for row in rows]
    base_scores = torch.tensor([row["scores"] for row in rows], dtype=torch.float32)

    image_to_idx = {image_id: idx for idx, image_id in enumerate(image_payload["image_ids"])}
    class_to_idx = {name: idx for idx, name in enumerate(text_payload["classes"])}
    image_features = normalize(image_payload["features"].float())
    text_features = normalize(text_payload["features"].float())

    missing_images = [image_id for image_id in image_ids if image_id not in image_to_idx]
    if missing_images:
        raise RuntimeError(f"{len(missing_images)} topK images missing from {args.image_features}; first={missing_images[:5]}")
    missing_classes = sorted({pred for preds in predictions for pred in preds if pred not in class_to_idx})
    if missing_classes:
        raise RuntimeError(f"{len(missing_classes)} topK classes missing from {args.text_features}; first={missing_classes[:5]}")

    adapter_scores = torch.zeros_like(base_scores)
    for row_idx, (image_id, preds) in enumerate(zip(image_ids, predictions)):
        image_feature = image_features[image_to_idx[image_id]]
        pred_indices = torch.tensor([class_to_idx[pred] for pred in preds], dtype=torch.long)
        adapter_scores[row_idx] = image_feature @ text_features[pred_indices].T

    base_indices = base_scores.argsort(dim=1, descending=True)
    sweep = []
    for weight in [float(part.strip()) for part in args.weight_grid.split(",") if part.strip()]:
        final = base_scores + weight * row_zscore(adapter_scores)
        indices = final.argsort(dim=1, descending=True)
        sweep.append({"weight": weight, **metrics(indices, labels, predictions)})
    best = max(sweep, key=lambda item: (item["top1"], item["net"], -item["changed"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "adapter_topk_scores.pt"
    torch.save(
        {
            "image_ids": image_ids,
            "predictions": predictions,
            "base_scores": base_scores,
            "adapter_scores": adapter_scores,
            "labels": labels,
        },
        score_path,
    )
    summary = {
        "topk_jsonl": str(args.topk_jsonl),
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "rows": len(rows),
        "base": metrics(base_indices, labels, predictions),
        "best": best,
        "sweep": sweep,
        "score_file": str(score_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

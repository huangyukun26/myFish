from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    classes = list(checkpoint["classes"])
    state = checkpoint["state_dict"]
    arch = checkpoint.get("arch", {})
    if arch.get("type") == "mlp":
        model = MLPClassifier(
            int(arch["in_dim"]),
            int(arch["hidden_dim"]),
            len(classes),
            float(arch.get("dropout", 0.0)),
        )
    else:
        weight = state["weight"]
        model = nn.Linear(weight.shape[1], weight.shape[0])
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    features = normalize(payload["features"])
    image_ids = list(payload["image_ids"])
    labels = list(payload.get("labels", [""] * len(image_ids)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    topk_path = args.out_dir / "topk.jsonl"
    pred_path = args.out_dir / "predictions.csv"
    ranks = []
    with topk_path.open("w", encoding="utf-8") as topk_fp, pred_path.open("w", encoding="utf-8", newline="") as pred_fp:
        writer = csv.DictWriter(pred_fp, fieldnames=["image_id", "prediction", "label"])
        writer.writeheader()
        with torch.inference_mode():
            for start in range(0, features.shape[0], args.batch_size):
                xb = features[start : start + args.batch_size].to(device)
                logits = model(xb)
                scores, indices = logits.topk(min(args.topk, logits.shape[1]), dim=1)
                for local_idx, image_id in enumerate(image_ids[start : start + args.batch_size]):
                    row_idx = start + local_idx
                    preds = [classes[int(idx)] for idx in indices[local_idx].cpu().tolist()]
                    label = labels[row_idx] if row_idx < len(labels) else ""
                    if label:
                        try:
                            ranks.append(preds.index(label) + 1)
                        except ValueError:
                            ranks.append(args.topk + 1)
                    topk_fp.write(
                        json.dumps(
                            {
                                "image_id": image_id,
                                "label": label,
                                "predictions": preds,
                                "scores": [float(v) for v in scores[local_idx].cpu().tolist()],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    writer.writerow({"image_id": image_id, "prediction": preds[0], "label": label})
    summary = {
        "model": str(args.model),
        "cache": str(args.cache),
        "rows": len(image_ids),
        "topk_jsonl": str(topk_path),
        "predictions_csv": str(pred_path),
    }
    if ranks:
        ranks_t = torch.tensor(ranks)
        summary.update(
            {
                "top1": float((ranks_t <= 1).float().mean().item()),
                "top5": float((ranks_t <= 5).float().mean().item()),
                "top20": float((ranks_t <= 20).float().mean().item()),
            }
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

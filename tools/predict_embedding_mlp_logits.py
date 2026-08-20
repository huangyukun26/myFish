from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

if os.name == "nt":
    # Cloud checkpoints may contain pathlib.PosixPath objects even on Windows.
    pathlib.PosixPath = pathlib.WindowsPath


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


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def collect_logits(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            chunks.append(model(xb).half().cpu())
    return torch.cat(chunks, dim=0)


def write_topk(path: Path, logits: torch.Tensor, image_ids: list[str], classes: list[str], topk: int) -> None:
    scores, indices = logits.float().topk(min(topk, logits.shape[1]), dim=1)
    with path.open("w", encoding="utf-8") as fp:
        for i, image_id in enumerate(image_ids):
            row = {
                "image_id": image_id,
                "predictions": [classes[int(idx)] for idx in indices[i].tolist()],
                "scores": [float(v) for v in scores[i].tolist()],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, logits: torch.Tensor, image_ids: list[str], classes: list[str]) -> None:
    pred = logits.float().argmax(dim=1).tolist()
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, idx in zip(image_ids, pred):
            writer.writerow({"image_id": image_id, "prediction": classes[int(idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-logits", type=Path, required=True)
    parser.add_argument("--out-topk", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cache = load_payload(args.cache)
    arch = ckpt["arch"]
    classes = list(ckpt["classes"])
    x = F.normalize(cache["features"].float(), dim=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        int(arch["in_dim"]),
        int(arch["hidden_dim"]),
        len(classes),
        float(arch["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    logits = collect_logits(model, x, args.batch_size, device)

    args.out_logits.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": logits,
            "image_ids": list(cache["image_ids"]),
            "classes": classes,
            "source_cache": str(args.cache),
            "checkpoint": str(args.checkpoint),
        },
        args.out_logits,
    )
    if args.out_topk:
        args.out_topk.parent.mkdir(parents=True, exist_ok=True)
        write_topk(args.out_topk, logits, list(cache["image_ids"]), classes, args.topk)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.out_csv, logits, list(cache["image_ids"]), classes)
    print(json.dumps({"rows": int(logits.shape[0]), "classes": len(classes), "out_logits": str(args.out_logits)}, indent=2))


if __name__ == "__main__":
    main()

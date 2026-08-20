from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import Any

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


def load(path: Path) -> dict[str, Any]:
    # Cloud checkpoints may contain PosixPath objects even on Windows.
    pathlib.PosixPath = pathlib.WindowsPath
    return torch.load(path, map_location="cpu", weights_only=False)


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    top5 = logits.topk(5, dim=1).indices
    top1_correct = top5[:, 0].eq(labels)
    top5_correct = top5.eq(labels[:, None]).any(dim=1)
    return {
        "rows": len(labels),
        "top1_correct": int(top1_correct.sum()),
        "top5_correct": int(top5_correct.sum()),
        "top1": float(top1_correct.float().mean()),
        "top5": float(top5_correct.float().mean()),
        "oracle_complement": int((~top1_correct & top5_correct).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    cache = load(args.joint_cache)
    checkpoint = load(args.checkpoint)
    reference = load(args.reference_logits)
    if list(cache["image_ids"]) != list(reference["image_ids"]):
        raise RuntimeError("joint cache and reference image_ids are not aligned")
    if list(cache["classes"]) != list(reference["classes"]):
        raise RuntimeError("joint cache and reference class orders differ")
    labels = torch.as_tensor(cache["class_ids"], dtype=torch.long)
    if not torch.equal(labels, torch.as_tensor(reference["class_ids"], dtype=torch.long)):
        raise RuntimeError("joint cache and reference class_ids differ")
    arch = checkpoint["arch"]
    model = MLPClassifier(
        int(arch["in_dim"]),
        int(arch["hidden_dim"]),
        len(checkpoint["classes"]),
        float(arch["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = F.normalize(cache["features"].float(), dim=1)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            chunks.append(model(features[start : start + args.batch_size].to(device)).half().cpu())
    reconstructed = torch.cat(chunks, dim=0)
    reference_values = reference["logits"].half()
    reconstructed_top5 = reconstructed.topk(5, dim=1).indices
    reference_top5 = reference_values.topk(5, dim=1).indices
    reconstructed_sets = reconstructed_top5.sort(dim=1).values
    reference_sets = reference_top5.sort(dim=1).values
    difference = (reconstructed.float() - reference_values.float()).abs()
    summary = {
        "joint_cache": str(args.joint_cache.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "reference_logits": str(args.reference_logits.resolve()),
        "reconstructed": metrics(reconstructed.float(), labels),
        "reference": metrics(reference_values.float(), labels),
        "top1_prediction_agreement": float(
            reconstructed_top5[:, 0].eq(reference_top5[:, 0]).float().mean()
        ),
        "top1_prediction_disagreements": int(
            reconstructed_top5[:, 0].ne(reference_top5[:, 0]).sum()
        ),
        "top5_set_agreement": float(reconstructed_sets.eq(reference_sets).all(dim=1).float().mean()),
        "logit_abs_mean": float(difference.mean()),
        "logit_abs_max": float(difference.max()),
        "metric_reproduced": metrics(reconstructed.float(), labels)["top1_correct"]
        == metrics(reference_values.float(), labels)["top1_correct"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": reconstructed,
            "class_ids": labels,
            "labels": list(cache["labels"]),
            "image_ids": list(cache["image_ids"]),
            "classes": list(cache["classes"]),
            "source_cache": str(args.joint_cache.resolve()),
        },
        args.out_dir / "reconstructed_val_logits.pt",
    )
    (args.out_dir / "verification.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Read-only calibration of official-data BioCLIP nearest-neighbour retrieval.

It reports whether an override by the closest official training image is safer
than the exact current local validation baseline.  No prediction files are
written, and it deliberately uses no external gallery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_CACHE = Path("work/cloud_20260713/artifacts/bioclip25_letterbox_full")
DEFAULT_LOGITS = Path(
    "runs/local_20260803_strong_oof_rebuild/"
    "joint_reconstruction_exact_verification/reconstructed_val_logits.pt"
)


def load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def nearest_labels(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_class_ids: torch.Tensor,
    device: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gallery = F.normalize(gallery_features.float().to(device), dim=1)
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for start in range(0, len(query_features), batch_size):
        query = F.normalize(query_features[start : start + batch_size].float().to(device), dim=1)
        scores, indices = (query @ gallery.T).max(dim=1)
        all_indices.append(indices.cpu())
        all_scores.append(scores.cpu())
    indices = torch.cat(all_indices)
    return gallery_class_ids[indices].long(), torch.cat(all_scores)


def evaluate(
    truth: torch.Tensor,
    baseline: torch.Tensor,
    nearest: torch.Tensor,
    similarity: torch.Tensor,
) -> None:
    print(f"baseline_correct={int((baseline == truth).sum())}/{len(truth)}")
    print(f"nearest_correct={int((nearest == truth).sum())}/{len(truth)}")
    print(
        "similarity="
        f"min:{similarity.min():.5f} p50:{similarity.median():.5f} max:{similarity.max():.5f}"
    )
    print("threshold,n,nn_correct,nn_accuracy,base_accuracy,wins,losses,net")
    for threshold in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
        selected = similarity >= threshold
        count = int(selected.sum())
        nn_correct = int((nearest[selected] == truth[selected]).sum())
        base_correct = int((baseline[selected] == truth[selected]).sum())
        wins = int(((nearest == truth) & (baseline != truth) & selected).sum())
        losses = int(((nearest != truth) & (baseline == truth) & selected).sum())
        print(
            f"{threshold:.2f},{count},{nn_correct},{nn_correct / max(count, 1):.5f},"
            f"{base_correct / max(count, 1):.5f},{wins},{losses},{wins - losses}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train = load_cache(args.cache_dir / "train_letterbox.pt")
    val = load_cache(args.cache_dir / "val_letterbox.pt")
    logits = load_cache(args.logits)
    if val["image_ids"] != logits["image_ids"]:
        raise RuntimeError("Validation cache and exact baseline logits have different image order.")
    # Feature caches can omit class names for classes absent in a particular
    # split, while their integer ids retain the shared global ordering.  The
    # exact validation artifact is the authoritative mapping for this check.
    if val["labels"] != logits["labels"]:
        raise RuntimeError("Validation class mappings differ; refusing to compare labels.")

    nearest, similarity = nearest_labels(
        val["features"], train["features"], train["class_ids"], args.device, args.batch_size
    )
    evaluate(val["class_ids"].long(), logits["logits"].argmax(dim=1).long(), nearest, similarity)


if __name__ == "__main__":
    main()

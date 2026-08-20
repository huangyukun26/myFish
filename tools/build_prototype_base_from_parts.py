from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parts", type=Path, required=True)
    parser.add_argument("--val-parts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = torch.load(args.train_parts, map_location="cpu", weights_only=False)
    val = torch.load(args.val_parts, map_location="cpu", weights_only=False)
    classes = list(train["classes"])
    num_classes = len(classes)
    train_features = F.normalize(train["cls"].float(), dim=1)
    val_features = F.normalize(val["cls"].float(), dim=1)
    prototypes = torch.zeros((num_classes, train_features.shape[1]), dtype=torch.float32)
    prototypes.index_add_(0, train["class_ids"].long(), train_features)
    counts = torch.bincount(train["class_ids"].long(), minlength=num_classes).float()
    prototypes = F.normalize(prototypes / counts[:, None].clamp_min(1), dim=1)
    logits = val_features.matmul(prototypes.T)

    common = {"classes": classes, "source": "ROI CLS reused as public full-image CLS"}
    torch.save(
        {
            "features": train_features,
            "image_ids": train["image_ids"],
            "labels": train["labels"],
            "class_ids": train["class_ids"].long(),
            **common,
        },
        args.out_dir / "train_full_cls.pt",
    )
    torch.save(
        {
            "features": val_features,
            "image_ids": val["image_ids"],
            "labels": val["labels"],
            "class_ids": val["class_ids"].long(),
            **common,
        },
        args.out_dir / "val_full_cls.pt",
    )
    torch.save(
        {
            "logits": logits.half(),
            "image_ids": val["image_ids"],
            "labels": val["labels"],
            "class_ids": val["class_ids"].long(),
            "classes": classes,
            "source": "train-only normalized class prototypes",
        },
        args.out_dir / "val_base_logits.pt",
    )
    prediction = logits.argmax(dim=1)
    top5 = logits.topk(min(5, num_classes), dim=1).indices
    labels = val["class_ids"].long()
    print(
        {
            "rows": len(labels),
            "classes": num_classes,
            "base_correct": int(prediction.eq(labels).sum()),
            "top5_correct": int(top5.eq(labels[:, None]).any(dim=1).sum()),
        }
    )


if __name__ == "__main__":
    main()

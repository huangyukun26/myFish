from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fishnet.env import environment_report, gpu_snapshot


class ManifestDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: int, augment: bool):
        self.root = root
        self.image_size = image_size
        self.augment = augment
        with (root / "manifest.csv").open("r", encoding="utf-8", newline="") as fp:
            rows = list(csv.DictReader(fp))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise RuntimeError(f"No rows for split={split} in {root / 'manifest.csv'}")

    def __len__(self) -> int:
        return len(self.rows)

    def _transform(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            w, h = image.size
            scale = random.uniform(0.75, 1.0)
            crop_w = max(1, int(w * scale))
            crop_h = max(1, int(h * scale))
            if crop_w < w and crop_h < h:
                left = random.randint(0, w - crop_w)
                top = random.randint(0, h - crop_h)
                image = image.crop((left, top, left + crop_w, top + crop_h))

        image = ImageOps.contain(image, (self.image_size, self.image_size))
        canvas = Image.new("RGB", (self.image_size, self.image_size), (124, 124, 124))
        canvas.paste(image, ((self.image_size - image.width) // 2, (self.image_size - image.height) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        with Image.open(image_path) as image:
            x = self._transform(image)
        y = int(row["class_id"])
        return x, y


class TinyFishCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def read_num_classes(root: Path) -> int:
    data = json.loads((root / "class_to_idx.json").read_text(encoding="utf-8"))
    return len(data)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total += y.numel()
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("work/smoke_subset"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.run_root / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(environment_report(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    num_classes = read_num_classes(args.data_root)
    train_ds = ManifestDataset(args.data_root, "train", args.image_size, augment=True)
    val_ds = ManifestDataset(args.data_root, "val", args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyFishCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    metrics_path = run_dir / "metrics.jsonl"
    start = time.time()
    best_val_acc = -1.0
    history: List[Dict] = []
    print(f"run_dir={run_dir}")
    print(f"device={device} num_classes={num_classes} train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        gpu = gpu_snapshot()
        row: Dict = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "epoch_sec": round(time.time() - epoch_start, 3),
            "gpu": gpu,
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            "val_loss={val_loss:.4f} val_acc={val_acc:.3f} sec={epoch_sec}".format(**row)
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save({"model": model.state_dict(), "num_classes": num_classes, "args": vars(args)}, run_dir / "best.pt")

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "num_classes": num_classes,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "epochs": args.epochs,
        "best_val_acc": best_val_acc,
        "total_sec": round(time.time() - start, 3),
        "last": history[-1] if history else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from fishnet.env import environment_report, gpu_snapshot

ImageFile.LOAD_TRUNCATED_IMAGES = True


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_cuda(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def jsonable_args(args: argparse.Namespace) -> Dict:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


class CsvImageDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        image_root: Path,
        image_size: int,
        train: bool,
        max_samples: int = 0,
        seed: int = 2026,
        randaugment: bool = False,
        random_erasing: float = 0.0,
    ):
        self.manifest = manifest
        self.image_root = image_root
        self.train = train
        self.randaugment = randaugment
        self.random_erasing = random_erasing
        with manifest.open("r", encoding="utf-8", newline="") as fp:
            self.rows = list(csv.DictReader(fp))
        if max_samples and len(self.rows) > max_samples:
            rng = random.Random(seed)
            self.rows = rng.sample(self.rows, max_samples)
        if not self.rows:
            raise RuntimeError(f"No rows in {manifest}")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            train_transforms = [
                transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.6, 1.8)),
                transforms.RandomHorizontalFlip(),
            ]
            if getattr(self, "randaugment", False):
                train_transforms.append(transforms.RandAugment(num_ops=2, magnitude=7))
            train_transforms.extend(
                [
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
            if getattr(self, "random_erasing", 0.0) > 0:
                train_transforms.append(transforms.RandomErasing(p=self.random_erasing, scale=(0.02, 0.18), ratio=(0.3, 3.3)))
            self.transform = transforms.Compose(train_transforms)
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.15)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.image_root / row["image_id"]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        y = int(row["class_id"])
        return x, y


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if name.startswith("timm:"):
        import timm

        return timm.create_model(name.split(":", 1)[1], pretrained=pretrained, num_classes=num_classes)
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = models.convnext_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "convnext_base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = models.convnext_base(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model {name}")


def class_weights_from_manifest(manifest: Path) -> List[float]:
    counts: Dict[int, int] = {}
    labels: List[int] = []
    with manifest.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            label = int(row["class_id"])
            labels.append(label)
            counts[label] = counts.get(label, 0) + 1
    return [1.0 / counts[label] for label in labels]


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    max_k = min(k, logits.shape[1])
    pred = logits.topk(max_k, dim=1).indices
    return int((pred == targets[:, None]).any(dim=1).sum().item())


def run_epoch(
    model,
    loader,
    criterion,
    device,
    scaler,
    optimizer=None,
    accum_steps: int = 1,
    amp: bool = True,
    epoch: int = 0,
    phase: str = "train",
    log_every: int = 0,
    step_metrics_path: Optional[Path] = None,
    save_every_steps: int = 0,
    checkpoint_dir: Optional[Path] = None,
    scheduler=None,
    args_json: Optional[Dict] = None,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total = 0
    optimizer_steps = 0
    window_loss = 0.0
    window_top1 = 0
    window_total = 0
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with autocast_cuda(enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
                loss_for_backward = loss / accum_steps
            if is_train:
                scaler.scale(loss_for_backward).backward()
                if step % accum_steps == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
        batch = y.numel()
        total += batch
        total_loss += float(loss.item()) * batch
        batch_top1 = topk_correct(logits.detach(), y, 1)
        batch_top5 = topk_correct(logits.detach(), y, 5)
        total_top1 += batch_top1
        total_top5 += batch_top5
        window_loss += float(loss.item()) * batch
        window_top1 += batch_top1
        window_total += batch
        if log_every and (step % log_every == 0 or step == len(loader)):
            row = {
                "epoch": epoch,
                "phase": phase,
                "step": step,
                "steps": len(loader),
                "loss": window_loss / max(1, window_total),
                "top1": window_top1 / max(1, window_total),
                "samples_seen": total,
                "optimizer_steps": optimizer_steps,
            }
            if step_metrics_path is not None:
                with step_metrics_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                "epoch={epoch} phase={phase} step={step}/{steps} loss={loss:.4f} top1={top1:.4f} samples={samples_seen}".format(
                    **row
                ),
                flush=True,
            )
            window_loss = 0.0
            window_top1 = 0
            window_total = 0
        if (
            is_train
            and save_every_steps
            and checkpoint_dir is not None
            and step % save_every_steps == 0
        ):
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch - 1,
                "partial_epoch": epoch,
                "partial_step": step,
                "args": args_json or {},
                "note": "Periodic step checkpoint. Use --init-checkpoint to continue approximately from these weights.",
            }
            torch.save(state, checkpoint_dir / "step_latest.pt")
            torch.save(state, checkpoint_dir / f"step_epoch{epoch}_step{step}.pt")
    return {
        "loss": total_loss / max(1, total),
        "top1": total_top1 / max(1, total),
        "top5": total_top5 / max(1, total),
        "samples": total,
        "optimizer_steps": optimizer_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, default=Path("work/supervised_splits/train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("work/supervised_splits/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--model",
        default="resnet18",
    )
    parser.add_argument("--num-classes", type=int, default=5795)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=0,
        help="During training, periodically save step_latest.pt and step_epoch*_step*.pt.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume from a train_supervised.py last.pt/best.pt checkpoint. Epochs is the final target epoch.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from a checkpoint but start a fresh optimizer/scheduler run.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    missing = [path for path in [args.train_manifest, args.val_manifest, args.image_root] if not path.exists()]
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        missing.append(args.resume_checkpoint)
    if args.init_checkpoint is not None and not args.init_checkpoint.exists():
        missing.append(args.init_checkpoint)
    if args.resume_checkpoint is not None and args.init_checkpoint is not None:
        raise ValueError("Use only one of --resume-checkpoint or --init-checkpoint")
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume_checkpoint is not None:
        run_dir = args.resume_checkpoint.parent
    else:
        run_dir = args.run_root / f"supervised_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args_json = jsonable_args(args)
    args_path = run_dir / ("resume_args.json" if args.resume_checkpoint is not None else "args.json")
    args_path.write_text(json.dumps(args_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    env_path = run_dir / ("resume_env.json" if args.resume_checkpoint is not None else "env.json")
    env_path.write_text(json.dumps(environment_report(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    train_ds = CsvImageDataset(
        args.train_manifest,
        args.image_root,
        args.image_size,
        train=True,
        max_samples=args.max_train_samples,
        seed=args.seed,
        randaugment=args.randaugment,
        random_erasing=args.random_erasing,
    )
    val_ds = CsvImageDataset(
        args.val_manifest,
        args.image_root,
        args.image_size,
        train=False,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    sampler: Optional[WeightedRandomSampler] = None
    shuffle = True
    if args.weighted_sampler:
        weights = class_weights_from_manifest(args.train_manifest)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model, args.num_classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=(not args.no_amp) and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    start_epoch = 0

    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        print(f"resumed_from={args.resume_checkpoint} completed_epoch={start_epoch}", flush=True)
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"initialized_model_from={args.init_checkpoint}", flush=True)

    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} classes={args.num_classes}", flush=True)
    print(f"steps_per_epoch={len(train_loader)} batch_size={args.batch_size} accum_steps={args.accum_steps}", flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    step_metrics_path = run_dir / "step_metrics.jsonl"
    best_top1 = -math.inf
    history = []
    if args.resume_checkpoint is not None and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                row = json.loads(line)
                history.append(row)
                best_top1 = max(best_top1, float(row.get("val_top1", -math.inf)))
    if start_epoch >= args.epochs:
        print(f"checkpoint already reached target epochs: completed={start_epoch} target={args.epochs}", flush=True)
        return
    started = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            scaler,
            optimizer=optimizer,
            accum_steps=args.accum_steps,
            amp=not args.no_amp,
            epoch=epoch,
            phase="train",
            log_every=args.log_every,
            step_metrics_path=step_metrics_path,
            save_every_steps=args.save_every_steps,
            checkpoint_dir=run_dir,
            scheduler=scheduler,
            args_json=args_json,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            scaler,
            optimizer=None,
            amp=not args.no_amp,
            epoch=epoch,
            phase="val",
            log_every=max(args.log_every, len(val_loader)),
            step_metrics_path=step_metrics_path,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_sec": round(time.time() - epoch_start, 3),
            "gpu": gpu_snapshot(),
        }
        history.append(row)
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            "epoch={epoch} lr={lr:.6g} train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
            "val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f} sec={epoch_sec}".format(**row),
            flush=True,
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": args_json,
            "metrics": row,
        }
        torch.save(state, run_dir / "last.pt")
        if row["val_top1"] > best_top1:
            best_top1 = row["val_top1"]
            torch.save(state, run_dir / "best.pt")

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "best_val_top1": best_top1,
        "last": history[-1] if history else None,
        "total_sec": round(time.time() - started, 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

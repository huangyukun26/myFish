from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

if os.name == "nt":
    # Cloud checkpoints contain pathlib.PosixPath objects.
    pathlib.PosixPath = pathlib.WindowsPath


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_classes(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
    return list(data)


def letterbox_with_mask(image: Image.Image, fill: tuple[int, int, int]) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    mask = Image.new("L", (side, side), 0)
    offset = ((side - width) // 2, (side - height) // 2)
    canvas.paste(image, offset)
    mask.paste(Image.new("L", (width, height), 255), offset)
    return canvas, mask


def patch_mask_from_image(mask: Image.Image, image_size: int, patch_size: int) -> torch.Tensor:
    mask_t = TF.to_tensor(mask)
    pooled = F.avg_pool2d(mask_t.unsqueeze(0), kernel_size=patch_size, stride=patch_size).flatten()
    return pooled > 0.5


class FishImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: Path,
        label_to_idx: dict[str, int],
        image_size: int,
        patch_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        train: bool,
        hflip_p: float,
        translate: float,
        scale_min: float,
        scale_max: float,
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.train = train
        self.hflip_p = hflip_p
        self.translate = translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.fill = tuple(round(float(v) * 255) for v in mean)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.image_root / row["image_id"]) as image:
            image = image.convert("RGB")
            image, mask = letterbox_with_mask(image, self.fill)
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        if self.train:
            if random.random() < self.hflip_p:
                image = ImageOps.mirror(image)
                mask = ImageOps.mirror(mask)
            if self.translate > 0 or self.scale_min != 1.0 or self.scale_max != 1.0:
                max_dx = self.translate * self.image_size
                max_dy = self.translate * self.image_size
                tx = int(round(random.uniform(-max_dx, max_dx)))
                ty = int(round(random.uniform(-max_dy, max_dy)))
                scale = random.uniform(self.scale_min, self.scale_max)
                image = TF.affine(
                    image,
                    angle=0.0,
                    translate=[tx, ty],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BICUBIC,
                    fill=self.fill,
                )
                mask = TF.affine(
                    mask,
                    angle=0.0,
                    translate=[tx, ty],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=0,
                )
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=self.mean, std=self.std)
        patch_mask = patch_mask_from_image(mask, self.image_size, self.patch_size)
        class_text = row.get("class_id", "")
        if class_text not in {"", None}:
            y = int(class_text)
        else:
            y = self.label_to_idx[row["label"]]
        return x, patch_mask, y, row["image_id"], row.get("label", "")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class MLPHead(nn.Module):
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


def set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, module)


def inject_lora(model: nn.Module, first_block: int, rank: int, alpha: float, dropout: float) -> list[str]:
    targets = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not (name.endswith(".attn.qkv") or name.endswith(".attn.proj")):
            continue
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "blocks":
            continue
        block_idx = int(parts[1])
        if block_idx < first_block:
            continue
        set_module(model, name, LoRALinear(module, rank, alpha, dropout))
        targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules were matched")
    return targets


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k or k.startswith("norm.")}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_unexpected = [k for k in unexpected if "lora_" in k or k.startswith("norm.")]
    if bad_unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {bad_unexpected[:5]}")
    missing_lora = [k for k in missing if "lora_" in k or k.startswith("norm.")]
    if missing_lora:
        raise RuntimeError(f"Missing LoRA keys: {missing_lora[:5]}")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if callable(value):
        return str(value)
    return value


def pooled_features(model: nn.Module, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
    tokens = model.forward_features(x)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise RuntimeError("Expected forward_features to return [B, tokens, dim]")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    patches = tokens[:, prefix_tokens:]
    if patches.shape[1] != patch_mask.shape[1]:
        raise RuntimeError(f"patch token/mask mismatch: {patches.shape} vs {patch_mask.shape}")
    mask = patch_mask.to(device=patches.device, dtype=patches.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    valid_mean = (patches * mask).sum(dim=1) / denom
    feat = torch.cat([tokens[:, 0], valid_mean], dim=1)
    return F.normalize(feat.float(), dim=1)


def evaluate(model: nn.Module, head: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    head.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for x, patch_mask, y, _image_id, _label in tqdm(loader, desc="eval", leave=False):
            x = x.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = head(pooled_features(model, x, patch_mask))
                loss = F.cross_entropy(logits.float(), y)
            total += int(y.numel())
            loss_sum += float(loss.detach().cpu()) * int(y.numel())
            pred = logits.float().topk(min(5, logits.shape[1]), dim=1).indices
            correct1 += int(pred[:, 0].eq(y).sum().item())
            correct5 += int(pred.eq(y[:, None]).any(dim=1).sum().item())
    return {"loss": loss_sum / max(1, total), "top1": correct1 / max(1, total), "top5": correct5 / max(1, total)}


def collect_logits(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    head.eval()
    logits_out = []
    class_ids = []
    image_ids: list[str] = []
    labels: list[str] = []
    with torch.inference_mode():
        for x, patch_mask, y, batch_image_ids, batch_labels in tqdm(loader, desc="logits", leave=False):
            x = x.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = head(pooled_features(model, x, patch_mask))
            logits_out.append(logits.float().half().cpu())
            class_ids.extend(int(v) for v in y.tolist())
            image_ids.extend(batch_image_ids)
            labels.extend(batch_labels)
    return {
        "logits": torch.cat(logits_out, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.tensor(class_ids, dtype=torch.long),
        "classes": classes,
    }


def make_loaders(args: argparse.Namespace, data_config: dict[str, Any], classes: list[str]) -> tuple[DataLoader, DataLoader]:
    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    mean = tuple(float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))
    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
    train_ds = FishImageDataset(
        train_rows,
        args.image_root,
        label_to_idx,
        args.image_size,
        args.patch_size,
        mean,
        std,
        True,
        args.hflip_p,
        args.translate,
        args.scale_min,
        args.scale_max,
    )
    val_ds = FishImageDataset(
        val_rows,
        args.image_root,
        label_to_idx,
        args.image_size,
        args.patch_size,
        mean,
        std,
        False,
        0.0,
        0.0,
        1.0,
        1.0,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace):
    import timm
    from timm.data import resolve_model_data_config

    create_kwargs = {
        "pretrained": True,
        "num_classes": 0,
        "img_size": args.image_size,
    }
    if args.pretrained_file is not None:
        create_kwargs["pretrained_cfg_overlay"] = {"file": str(args.pretrained_file)}
    model = timm.create_model(args.model, **create_kwargs)
    data_config = resolve_model_data_config(model)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = inject_lora(model, args.lora_first_block, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad_(True)
    return model, data_config, targets


def command_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    classes = load_classes(args.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config, targets = build_model(args)
    train_loader, val_loader = make_loaders(args, data_config, classes)
    model.to(device)
    head = MLPHead(args.feature_dim, args.hidden_dim, len(classes), args.head_dropout).to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": args.lora_weight_decay},
            {"params": head_params, "lr": args.head_lr, "weight_decay": args.head_weight_decay},
        ]
    )
    total_updates = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_updates = max(1, int(total_updates * args.warmup_frac))

    def lr_scale(step: int) -> float:
        if step < warmup_updates:
            return float(step + 1) / float(warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_key = None
    best_row = None
    global_step = 0
    rows = []
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "train_rows": len(train_loader.dataset),
                "val_rows": len(val_loader.dataset),
                "classes": len(classes),
                "lora_targets": targets,
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_seen = 0
        for step, (x, patch_mask, y, _image_ids, _labels) in enumerate(tqdm(train_loader, desc=f"epoch{epoch}", leave=False), start=1):
            if args.max_steps_per_epoch and step > args.max_steps_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = head(pooled_features(model, x, patch_mask))
                loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * args.grad_accum * int(y.numel())
            total_seen += int(y.numel())
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        val = evaluate(model, head, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_seen),
            **val,
            "global_step": global_step,
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        key = (val["top1"], val["top5"], -val["loss"])
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
            torch.save(
                {
                    "model_trainable_state": trainable_state(model),
                    "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "classes": classes,
                    "args": jsonable(vars(args)),
                    "data_config": data_config,
                    "lora_targets": targets,
                    "best": best_row,
                    "feature_dim": args.feature_dim,
                },
                args.out_dir / "best_lora.pt",
            )
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ckpt = torch.load(args.out_dir / "best_lora.pt", map_location="cpu", weights_only=False)
    model, _data_config, _targets = build_model(args)
    load_trainable_state(model, ckpt["model_trainable_state"])
    head.load_state_dict(ckpt["head_state"])
    model.to(device)
    head.to(device)
    val_logits = collect_logits(model, head, val_loader, classes, device)
    val_logits["checkpoint"] = str(args.out_dir / "best_lora.pt")
    torch.save(val_logits, args.out_dir / "val_logits.pt")
    summary = {
        "best": best_row,
        "checkpoint": str(args.out_dir / "best_lora.pt"),
        "val_logits": str(args.out_dir / "val_logits.pt"),
        "metrics_csv": str(args.out_dir / "metrics.csv"),
        "lora_targets": targets,
        "args": jsonable(vars(args)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def shard_paths(shard_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard_{shard_index:05d}"
    return shard_dir / f"{stem}.pt", shard_dir / f"{stem}.summary.json"


def command_cache(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config, _targets = build_model(args)
    load_trainable_state(model, ckpt["model_trainable_state"])
    model.to(device).eval()
    label_to_idx = {name: idx for idx, name in enumerate(classes)}
    mean = tuple(float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))
    rows = read_manifest(args.manifest)
    if args.max_samples:
        rows = rows[: args.max_samples]
    ds = FishImageDataset(
        rows,
        args.image_root,
        label_to_idx,
        args.image_size,
        args.patch_size,
        mean,
        std,
        False,
        0.0,
        0.0,
        1.0,
        1.0,
    )
    shard_count = math.ceil(len(rows) / args.shard_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    features = []
    image_ids: list[str] = []
    labels: list[str] = []
    class_ids = []
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        end = min(start + args.shard_size, len(rows))
        shard_path, summary_path = shard_paths(args.shard_dir, shard_index)
        if args.resume and summary_path.exists() and shard_path.exists():
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        else:
            subset = torch.utils.data.Subset(ds, list(range(start, end)))
            loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            shard_features = []
            shard_ids: list[str] = []
            shard_labels: list[str] = []
            shard_class_ids = []
            with torch.inference_mode():
                for x, patch_mask, y, batch_image_ids, batch_labels in tqdm(loader, desc=f"cache{shard_index:05d}", leave=False):
                    x = x.to(device, non_blocking=True)
                    patch_mask = patch_mask.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        feat = pooled_features(model, x, patch_mask)
                    shard_features.append(feat.float().cpu())
                    shard_ids.extend(batch_image_ids)
                    shard_labels.extend(batch_labels)
                    shard_class_ids.extend(int(v) for v in y.tolist())
            payload = {
                "features": torch.cat(shard_features, dim=0),
                "image_ids": shard_ids,
                "labels": shard_labels,
                "class_ids": torch.tensor(shard_class_ids, dtype=torch.long),
            }
            torch.save(payload, shard_path)
            summary_path.write_text(
                json.dumps({"shard_index": shard_index, "rows": len(shard_ids), "dim": int(payload["features"].shape[1])}, indent=2) + "\n",
                encoding="utf-8",
            )
        features.append(payload["features"].float())
        image_ids.extend(payload["image_ids"])
        labels.extend(payload["labels"])
        class_ids.append(payload["class_ids"].long())
    output = {
        "features": torch.cat(features, dim=0),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": torch.cat(class_ids, dim=0),
        "classes": classes,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "image_size": args.image_size,
        "preprocess_mode": "letterbox_valid_patch",
        "feature_pool": "cls_valid_patch_mean",
        "shard_dir": str(args.shard_dir),
    }
    torch.save(output, args.out)
    summary = {"out": str(args.out), "rows": len(image_ids), "dim": int(output["features"].shape[1]), "shards": shard_count}
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    ap.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    ap.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--feature-dim", type=int, default=2048)
    ap.add_argument("--lora-first-block", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pretrained-file", type=Path, default=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--val-manifest", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--hidden-dim", type=int, default=2048)
    train.add_argument("--head-dropout", type=float, default=0.2)
    train.add_argument("--lora-lr", type=float, default=1e-4)
    train.add_argument("--head-lr", type=float, default=5e-4)
    train.add_argument("--lora-weight-decay", type=float, default=0.01)
    train.add_argument("--head-weight-decay", type=float, default=1e-3)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--warmup-frac", type=float, default=0.05)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--hflip-p", type=float, default=0.5)
    train.add_argument("--translate", type=float, default=0.03)
    train.add_argument("--scale-min", type=float, default=0.95)
    train.add_argument("--scale-max", type=float, default=1.05)
    train.add_argument("--seed", type=int, default=2031)
    train.add_argument("--max-train-samples", type=int, default=0)
    train.add_argument("--max-val-samples", type=int, default=0)
    train.add_argument("--max-steps-per-epoch", type=int, default=0)
    train.set_defaults(func=command_train)

    cache = sub.add_parser("cache")
    add_common(cache)
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--out", type=Path, required=True)
    cache.add_argument("--shard-dir", type=Path, required=True)
    cache.add_argument("--shard-size", type=int, default=1000)
    cache.add_argument("--max-samples", type=int, default=0)
    cache.add_argument("--resume", action="store_true")
    cache.set_defaults(func=command_cache)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

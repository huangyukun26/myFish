from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from train_dinov3_lora_seen import (
    MLPHead,
    build_model,
    load_classes,
    load_trainable_state,
    pooled_features,
    trainable_state,
)


if __import__("os").name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_flags(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in read_csv(path):
        for category in row["categories"].split("|"):
            if category:
                result[row["image_id"]].add(category)
    return dict(result)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def letterbox(image: Image.Image, fill: tuple[int, int, int]) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), fill)
    mask = Image.new("L", (side, side), 0)
    offset = ((side - width) // 2, (side - height) // 2)
    canvas.paste(image, offset)
    mask.paste(Image.new("L", (width, height), 255), offset)
    return canvas, mask


def render_view(
    source: Image.Image,
    image_size: int,
    patch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    categories: set[str],
    strong: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    fill = tuple(round(value * 255) for value in mean)
    image, mask = letterbox(source, fill)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)

    scale_low, scale_high = (0.97, 1.05)
    translate = 0.025
    if strong:
        if "目标过小" in categories:
            scale_low, scale_high, translate = 1.18, 1.42, 0.06
        elif "鱼的一部分" in categories:
            scale_low, scale_high, translate = 1.08, 1.25, 0.08
        elif categories & {"复杂背景", "异物干扰", "保护色"}:
            scale_low, scale_high, translate = 1.04, 1.18, 0.05
    scale = random.uniform(scale_low, scale_high)
    max_shift = translate * image_size
    shift = [int(round(random.uniform(-max_shift, max_shift))) for _ in range(2)]
    image = TF.affine(
        image,
        angle=0.0,
        translate=shift,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=TF.InterpolationMode.BICUBIC,
        fill=fill,
    )
    mask = TF.affine(
        mask,
        angle=0.0,
        translate=shift,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=TF.InterpolationMode.NEAREST,
        fill=0,
    )
    if strong and "模糊" in categories:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.35, 1.15)))
    if strong and categories & {"颜色偏移", "保护色"}:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.72, 1.28))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.12))

    x = TF.normalize(TF.to_tensor(image), mean=mean, std=std)
    mask_tensor = TF.to_tensor(mask)
    patch_mask = F.avg_pool2d(
        mask_tensor.unsqueeze(0), kernel_size=patch_size, stride=patch_size
    ).flatten() > 0.5
    return x, patch_mask


class QualityEpisodeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        flags: dict[str, set[str]],
        image_root: Path,
        cached_features: dict[str, torch.Tensor],
        hard_classes: dict[int, list[int]],
        genus_to_idx: dict[str, int],
        image_size: int,
        patch_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        max_episodes: int,
    ) -> None:
        self.rows = rows
        self.flags = flags
        self.image_root = image_root
        self.cached_features = cached_features
        self.hard_classes = hard_classes
        self.genus_to_idx = genus_to_idx
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_class[int(row["class_id"])].append(row)
        self.by_class = dict(by_class)
        episodes = [row for row in rows if row["image_id"] in flags]
        episodes.sort(key=lambda row: (stable_hash(row["image_id"]), row["image_id"]))
        if max_episodes:
            episodes = episodes[:max_episodes]
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.episodes[index]
        class_id = int(anchor["class_id"])
        positives = [row for row in self.by_class[class_id] if row["image_id"] != anchor["image_id"]]
        if not positives:
            raise RuntimeError(f"No distinct positive for class {class_id}")
        positive = positives[random.randrange(len(positives))]
        categories = self.flags[anchor["image_id"]]
        with Image.open(self.image_root / anchor["image_id"]) as handle:
            anchor_image = handle.convert("RGB")
        with Image.open(self.image_root / positive["image_id"]) as handle:
            positive_image = handle.convert("RGB")
        a_full, a_full_mask = render_view(
            anchor_image, self.image_size, self.patch_size, self.mean, self.std, categories, False
        )
        a_strong, a_strong_mask = render_view(
            anchor_image, self.image_size, self.patch_size, self.mean, self.std, categories, True
        )
        positive_view, positive_mask = render_view(
            positive_image, self.image_size, self.patch_size, self.mean, self.std, set(), False
        )
        genus = anchor["label"].split()[0]
        return {
            "images": torch.stack([a_full, a_strong, positive_view]),
            "patch_masks": torch.stack([a_full_mask, a_strong_mask, positive_mask]),
            "class_id": class_id,
            "genus_id": self.genus_to_idx[genus],
            "anchor_target": self.cached_features[anchor["image_id"]],
            "positive_target": self.cached_features[positive["image_id"]],
            "hard_class_ids": torch.tensor(self.hard_classes[class_id], dtype=torch.long),
            "anchor_id": anchor["image_id"],
            "positive_id": positive["image_id"],
            "categories": "|".join(sorted(categories)),
        }


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values.float()), dim=-1)


class JointHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def build_prototypes(cache: dict[str, Any], class_count: int) -> torch.Tensor:
    features = F.normalize(cache["features"].float(), dim=1)
    labels = cache["class_ids"].long()
    sums = torch.zeros(class_count, features.shape[1], dtype=torch.float32)
    sums.index_add_(0, labels, features)
    counts = torch.bincount(labels, minlength=class_count).float().clamp_min(1.0)
    return F.normalize(sums / counts[:, None], dim=1)


def build_hard_classes(
    prototypes: torch.Tensor,
    classes: list[str],
    query_classes: list[int],
    count: int,
) -> dict[int, list[int]]:
    query = prototypes[torch.tensor(query_classes)]
    similarities = query @ prototypes.T
    output: dict[int, list[int]] = {}
    genera = [label.split()[0] for label in classes]
    for row, class_id in enumerate(query_classes):
        scores = similarities[row].clone()
        scores[class_id] = -torch.inf
        order = scores.topk(min(len(classes) - 1, max(count * 8, 64))).indices.tolist()
        same_genus = [candidate for candidate in order if genera[candidate] == genera[class_id]]
        chosen = same_genus[: max(1, count // 2)]
        chosen.extend(candidate for candidate in order if candidate not in chosen)
        output[class_id] = chosen[:count]
    return output


def supcon_with_prototypes(
    episode_features: torch.Tensor,
    negative_features: torch.Tensor,
    projection: nn.Module,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = episode_features.shape[0]
    projected = projection(episode_features.reshape(-1, episode_features.shape[-1])).view(batch, 3, -1)
    negative_projected = projection(negative_features.reshape(-1, negative_features.shape[-1])).view(
        batch, negative_features.shape[1], -1
    )
    losses = []
    margins = []
    for index in range(batch):
        z = projected[index]
        positive_scores = z @ z.T / temperature
        negative_scores = z @ negative_projected[index].T / temperature
        eye = torch.eye(3, dtype=torch.bool, device=z.device)
        for query_index in range(3):
            positives = positive_scores[query_index][~eye[query_index]]
            all_scores = torch.cat([positives, negative_scores[query_index]])
            losses.append(-(torch.logsumexp(positives, dim=0) - torch.logsumexp(all_scores, dim=0)))
            positive_similarity = (z[query_index] @ z[~eye[query_index]].T).mean()
            negative_similarity = (z[query_index] @ negative_projected[index].T).max()
            margins.append(F.relu(margin + negative_similarity - positive_similarity))
    return torch.stack(losses).mean(), torch.stack(margins).mean()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if callable(value):
        return str(value)
    return value


def prepare_training(args: argparse.Namespace):
    classes = load_classes(args.class_map)
    init = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    train_cache = torch.load(args.adapted_train_cache, map_location="cpu", weights_only=False)
    if list(init["classes"]) != classes or list(train_cache["classes"]) != classes:
        raise RuntimeError("Class order mismatch")
    train_rows = read_csv(args.train_manifest)
    flags = read_flags(args.flags_train)
    train_ids = {row["image_id"] for row in train_rows}
    if not set(flags).issubset(train_ids):
        raise RuntimeError("flags_train contains IDs outside the train manifest")
    cache_by_id = {
        image_id: train_cache["features"][index].float()
        for index, image_id in enumerate(train_cache["image_ids"])
    }
    if not train_ids.issubset(cache_by_id):
        raise RuntimeError("Adapted train cache does not cover the train manifest")
    prototypes = build_prototypes(train_cache, len(classes))
    flagged_classes = sorted({int(row["class_id"]) for row in train_rows if row["image_id"] in flags})
    hard_classes = build_hard_classes(prototypes, classes, flagged_classes, args.hard_negative_count)
    genera = sorted({label.split()[0] for label in classes})
    genus_to_idx = {name: index for index, name in enumerate(genera)}
    return classes, init, train_rows, flags, cache_by_id, prototypes, hard_classes, genera, genus_to_idx


def command_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    (
        classes,
        init,
        train_rows,
        flags,
        cache_by_id,
        prototypes,
        hard_classes,
        genera,
        genus_to_idx,
    ) = prepare_training(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config, targets = build_model(args)
    load_trainable_state(model, init["model_trainable_state"])
    if args.gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    head = MLPHead(args.feature_dim, args.hidden_dim, len(classes), args.head_dropout)
    head.load_state_dict(init["head_state"])
    projection = ProjectionHead(args.feature_dim, args.projection_dim)
    genus_head = nn.Linear(args.feature_dim, len(genera))
    mean = tuple(float(value) for value in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in data_config.get("std", (0.229, 0.224, 0.225)))
    dataset = QualityEpisodeDataset(
        train_rows,
        flags,
        args.image_root,
        cache_by_id,
        hard_classes,
        genus_to_idx,
        args.image_size,
        args.patch_size,
        mean,
        std,
        args.max_episodes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model.to(device)
    head.to(device)
    projection.to(device)
    genus_head.to(device)
    prototypes = prototypes.to(device)
    lora_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_lr, "weight_decay": args.lora_weight_decay},
            {"params": head.parameters(), "lr": args.head_lr, "weight_decay": args.head_weight_decay},
            {"params": list(projection.parameters()) + list(genus_head.parameters()), "lr": args.aux_lr},
        ]
    )
    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    print(
        json.dumps(
            {
                "event": "start",
                "device": str(device),
                "episodes": len(dataset),
                "classes": len(classes),
                "genera": len(genera),
                "lora_targets": targets,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        projection.train()
        genus_head.train()
        totals = defaultdict(float)
        rows_seen = 0
        for step, batch in enumerate(tqdm(loader, desc=f"quality-epoch{epoch}"), start=1):
            images = batch["images"].to(device, non_blocking=True)
            patch_masks = batch["patch_masks"].to(device, non_blocking=True)
            batch_size = images.shape[0]
            images = images.view(-1, *images.shape[2:])
            patch_masks = patch_masks.view(-1, patch_masks.shape[-1])
            class_ids = batch["class_id"].to(device)
            genus_ids = batch["genus_id"].to(device)
            repeated_classes = class_ids[:, None].expand(-1, 3).reshape(-1)
            repeated_genera = genus_ids[:, None].expand(-1, 3).reshape(-1)
            negative_features = prototypes[batch["hard_class_ids"].to(device)]
            anchor_target = F.normalize(batch["anchor_target"].to(device).float(), dim=1)
            positive_target = F.normalize(batch["positive_target"].to(device).float(), dim=1)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                features = pooled_features(model, images, patch_masks).view(batch_size, 3, -1)
                species_logits = head(features.reshape(-1, args.feature_dim))
                genus_logits = genus_head(features.reshape(-1, args.feature_dim).float())
                ce = F.cross_entropy(species_logits.float(), repeated_classes, label_smoothing=args.label_smoothing)
                genus_ce = F.cross_entropy(genus_logits.float(), repeated_genera, label_smoothing=0.02)
                supcon, margin = supcon_with_prototypes(
                    features, negative_features, projection, args.temperature, args.triplet_margin
                )
                distill = (
                    1.0 - F.cosine_similarity(features[:, 0].float(), anchor_target, dim=1)
                ).mean() + (
                    1.0 - F.cosine_similarity(features[:, 2].float(), positive_target, dim=1)
                ).mean()
                loss_unscaled = (
                    args.ce_weight * ce
                    + args.supcon_weight * supcon
                    + args.hard_margin_weight * margin
                    + args.genus_weight * genus_ce
                    + args.distill_weight * distill
                )
                loss = loss_unscaled / args.grad_accum
            scaler.scale(loss).backward()
            rows_seen += batch_size
            for name, value in {
                "loss": loss_unscaled,
                "ce": ce,
                "supcon": supcon,
                "margin": margin,
                "genus_ce": genus_ce,
                "distill": distill,
            }.items():
                totals[name] += float(value.detach().cpu()) * batch_size
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    lora_parameters + list(head.parameters()) + list(projection.parameters()) + list(genus_head.parameters()),
                    args.clip_grad,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch, "episodes": rows_seen, "global_step": global_step}
        row.update({name: value / max(1, rows_seen) for name, value in totals.items()})
        metrics.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    checkpoint = {
        "model_trainable_state": trainable_state(model),
        "head_state": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "projection_state": {key: value.detach().cpu() for key, value in projection.state_dict().items()},
        "genus_head_state": {key: value.detach().cpu() for key, value in genus_head.state_dict().items()},
        "classes": classes,
        "genera": genera,
        "args": jsonable(vars(args)),
        "data_config": data_config,
        "lora_targets": targets,
        "metrics": metrics,
        "init_checkpoint": str(args.init_checkpoint),
    }
    torch.save(checkpoint, args.out_dir / "quality_lora_final.pt")
    summary = {
        "checkpoint": str(args.out_dir / "quality_lora_final.pt"),
        "episodes": len(dataset),
        "metrics": metrics,
        "args": jsonable(vars(args)),
        "train_flag_categories": {
            category: sum(category in values for values in flags.values())
            for category in sorted({category for values in flags.values() for category in values})
        },
        "test_flags_used": False,
    }
    (args.out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


class EvalDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: Path,
        image_size: int,
        patch_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.rows = rows
        self.image_root = image_root
        self.image_size = image_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.image_root / row["image_id"]) as handle:
            image = handle.convert("RGB")
        fill = tuple(round(value * 255) for value in self.mean)
        image, mask = letterbox(image, fill)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        x = TF.normalize(TF.to_tensor(image), mean=self.mean, std=self.std)
        patch_mask = F.avg_pool2d(
            TF.to_tensor(mask).unsqueeze(0), kernel_size=self.patch_size, stride=self.patch_size
        ).flatten() > 0.5
        return x, patch_mask, int(row["class_id"]), row["image_id"], row["label"]


def paired_stats(reference: torch.Tensor, candidate: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, int]:
    ref_ok = reference.eq(labels)
    candidate_ok = candidate.eq(labels)
    return {
        "rows": int(mask.sum()),
        "reference_correct": int((mask & ref_ok).sum()),
        "candidate_correct": int((mask & candidate_ok).sum()),
        "net": int((mask & ~ref_ok & candidate_ok).sum() - (mask & ref_ok & ~candidate_ok).sum()),
        "wins": int((mask & ~ref_ok & candidate_ok).sum()),
        "losses": int((mask & ref_ok & ~candidate_ok).sum()),
        "changed": int((mask & reference.ne(candidate)).sum()),
    }


def split_dev_sealed(image_ids: list[str], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, labels.tolist())):
        groups[class_id].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        for position, (_digest, index) in enumerate(sorted(rows)):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def command_evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = load_classes(args.class_map)
    if list(checkpoint["classes"]) != classes:
        raise RuntimeError("Class order mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data_config, _targets = build_model(args)
    load_trainable_state(model, checkpoint["model_trainable_state"])
    model.to(device).eval()
    joint_checkpoint = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    arch = joint_checkpoint["arch"]
    joint_head = JointHead(
        int(arch["in_dim"]), int(arch["hidden_dim"]), len(classes), float(arch["dropout"])
    )
    joint_head.load_state_dict(joint_checkpoint["state_dict"])
    joint_head.to(device).eval()
    frozen = torch.load(args.frozen_joint_val_cache, map_location="cpu", weights_only=False)
    reference = torch.load(args.reference_logits, map_location="cpu", weights_only=False)
    rows = read_csv(args.val_manifest)
    if args.max_val_samples:
        rows = rows[: args.max_val_samples]
    expected_ids = [row["image_id"] for row in rows]
    if list(frozen["image_ids"][: len(rows)]) != expected_ids or list(reference["image_ids"][: len(rows)]) != expected_ids:
        raise RuntimeError("Validation order mismatch")
    mean = tuple(float(value) for value in data_config.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(value) for value in data_config.get("std", (0.229, 0.224, 0.225)))
    dataset = EvalDataset(rows, args.image_root, args.image_size, args.patch_size, mean, std)
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    logits_parts = []
    feature_parts = []
    cursor = 0
    with torch.inference_mode():
        for x, patch_mask, _labels, _image_ids, _names in tqdm(loader, desc="quality-eval"):
            x = x.to(device, non_blocking=True)
            patch_mask = patch_mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                adapted = pooled_features(model, x, patch_mask)
                batch_frozen = F.normalize(
                    frozen["features"][cursor : cursor + len(x)].float().to(device), dim=1
                )
                joint = torch.cat([batch_frozen, F.normalize(adapted.float(), dim=1)], dim=1) / math.sqrt(2.0)
                logits = joint_head(F.normalize(joint, dim=1))
            feature_parts.append(adapted.half().cpu())
            logits_parts.append(logits.half().cpu())
            cursor += len(x)
    logits = torch.cat(logits_parts).float()
    adapted_features = torch.cat(feature_parts)
    labels = torch.tensor([int(row["class_id"]) for row in rows], dtype=torch.long)
    image_ids = expected_ids
    reference_logits = reference["logits"][: len(rows)].float()
    reference_pred = reference_logits.argmax(dim=1)
    candidate_pred = logits.argmax(dim=1)
    dev, sealed = split_dev_sealed(image_ids, labels)
    flags = read_flags(args.flags_val)
    masks = {
        "all": torch.ones(len(rows), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
        "any_quality_flag": torch.tensor([image_id in flags for image_id in image_ids]),
    }
    for category in sorted({category for values in flags.values() for category in values}):
        masks[f"flag:{category}"] = torch.tensor(
            [category in flags.get(image_id, set()) for image_id in image_ids]
        )
    paired = {
        name: paired_stats(reference_pred, candidate_pred, labels, mask)
        for name, mask in masks.items()
    }
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": logits.half(),
            "class_ids": labels,
            "labels": [row["label"] for row in rows],
            "image_ids": image_ids,
            "classes": classes,
            "checkpoint": str(args.checkpoint),
        },
        output_dir / "joint_val_logits.pt",
    )
    torch.save(
        {
            "features": adapted_features,
            "class_ids": labels,
            "labels": [row["label"] for row in rows],
            "image_ids": image_ids,
            "classes": classes,
            "checkpoint": str(args.checkpoint),
        },
        output_dir / "adapted_val_features.pt",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "rows": len(rows),
        "reference_correct": int(reference_pred.eq(labels).sum()),
        "candidate_correct": int(candidate_pred.eq(labels).sum()),
        "paired": paired,
        "test_flags_used": False,
        "checkpoint_selection": "fixed final epoch; validation never selected the checkpoint",
    }
    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--pretrained-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--class-map", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--lora-first-block", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    add_model_args(train)
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--flags-train", type=Path, required=True)
    train.add_argument("--adapted-train-cache", type=Path, required=True)
    train.add_argument("--init-checkpoint", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-episodes", type=int, default=0)
    train.add_argument("--max-steps", type=int, default=0)
    train.add_argument("--grad-accum", type=int, default=4)
    train.add_argument("--hidden-dim", type=int, default=2048)
    train.add_argument("--head-dropout", type=float, default=0.2)
    train.add_argument("--projection-dim", type=int, default=256)
    train.add_argument("--hard-negative-count", type=int, default=8)
    train.add_argument("--lora-lr", type=float, default=2e-5)
    train.add_argument("--head-lr", type=float, default=2e-5)
    train.add_argument("--aux-lr", type=float, default=1e-4)
    train.add_argument("--lora-weight-decay", type=float, default=0.01)
    train.add_argument("--head-weight-decay", type=float, default=1e-3)
    train.add_argument("--label-smoothing", type=float, default=0.05)
    train.add_argument("--temperature", type=float, default=0.10)
    train.add_argument("--triplet-margin", type=float, default=0.08)
    train.add_argument("--ce-weight", type=float, default=1.0)
    train.add_argument("--supcon-weight", type=float, default=0.08)
    train.add_argument("--hard-margin-weight", type=float, default=0.04)
    train.add_argument("--genus-weight", type=float, default=0.04)
    train.add_argument("--distill-weight", type=float, default=0.20)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--gradient-checkpointing", action="store_true")
    train.add_argument("--seed", type=int, default=2031)
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate")
    add_model_args(evaluate)
    evaluate.set_defaults(image_size=512, batch_size=1)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--val-manifest", type=Path, required=True)
    evaluate.add_argument("--flags-val", type=Path, required=True)
    evaluate.add_argument("--frozen-joint-val-cache", type=Path, required=True)
    evaluate.add_argument("--joint-checkpoint", type=Path, required=True)
    evaluate.add_argument("--reference-logits", type=Path, required=True)
    evaluate.add_argument("--out-dir", type=Path, required=True)
    evaluate.add_argument("--eval-batch-size", type=int, default=1)
    evaluate.add_argument("--max-val-samples", type=int, default=0)
    evaluate.set_defaults(func=command_evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

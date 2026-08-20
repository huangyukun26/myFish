from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], class_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, class_ids.tolist())):
        groups[int(class_id)].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for position, (_digest, index) in enumerate(rows):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def zscore_rows(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def paired_stats(
    base_prediction: torch.Tensor,
    prediction: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    base_correct = base_prediction.eq(labels)
    candidate_correct = prediction.eq(labels)
    changed = mask & prediction.ne(base_prediction)
    wins = changed & ~base_correct & candidate_correct
    losses = changed & base_correct & ~candidate_correct
    return {
        "rows": int(mask.sum()),
        "base_correct": int((mask & base_correct).sum()),
        "candidate_correct": int((mask & candidate_correct).sum()),
        "net": int(wins.sum() - losses.sum()),
        "changed": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "efficiency": float((wins.sum() - losses.sum()) / max(1, int(changed.sum()))),
    }


class DensePairComparator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        grid_rows: int,
        grid_cols: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.tokens = 3 + grid_rows * grid_cols
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.token_embedding = nn.Parameter(torch.zeros(self.tokens, projection_dim))
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        interaction_dim = self.tokens * self.tokens + 4 * self.tokens + 5
        self.head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        flip_order = [0, 1, 2]
        for row in range(grid_rows):
            start = 3 + row * grid_cols
            flip_order.extend(reversed(range(start, start + grid_cols)))
        self.register_buffer("flip_order", torch.tensor(flip_order, dtype=torch.long))

    def forward(self, query: torch.Tensor, exemplar: torch.Tensor) -> torch.Tensor:
        # query [B,T,D], exemplar [B,K,T,D]
        query_raw = F.normalize(query.float(), dim=-1)
        exemplar_raw = F.normalize(exemplar.float(), dim=-1)
        query_projected = self.projection(query_raw) + self.token_embedding[None]
        exemplar_projected = self.projection(exemplar_raw) + self.token_embedding[None, None]
        query_projected = F.normalize(query_projected, dim=-1)
        exemplar_projected = F.normalize(exemplar_projected, dim=-1)
        cross = torch.einsum("btd,bksd->bkts", query_projected, exemplar_projected)
        direct = (query_projected[:, None] * exemplar_projected).sum(dim=-1)
        flipped = (
            query_projected[:, None]
            * exemplar_projected.index_select(2, self.flip_order)
        ).sum(dim=-1)
        raw_direct = (query_raw[:, None] * exemplar_raw).sum(dim=-1)
        features = torch.cat(
            [
                cross.flatten(2),
                cross.max(dim=3).values,
                cross.max(dim=2).values,
                direct,
                flipped,
                cross.mean(dim=(2, 3))[..., None],
                cross.amax(dim=(2, 3))[..., None],
                raw_direct[..., 0:1],
                raw_direct[..., 1:2],
                raw_direct[..., 2:3],
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


class BaseAwareReranker(nn.Module):
    def __init__(self, pair: DensePairComparator) -> None:
        super().__init__()
        self.pair = pair
        self.gate = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # softplus(0.5413) is approximately 1.0.
        self.log_scale = nn.Parameter(torch.tensor(0.54132485))

    @staticmethod
    def aggregate(pair_scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = pair_scores.masked_fill(~valid, float("-inf"))
        counts = valid.sum(dim=2).clamp_min(1)
        return torch.logsumexp(masked, dim=2) - counts.float().log()

    def forward(
        self,
        query: torch.Tensor,
        exemplars: torch.Tensor,
        exemplar_valid: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, candidates, top_r, tokens, dim = exemplars.shape
        flat = exemplars.reshape(batch, candidates * top_r, tokens, dim)
        pair_scores = self.pair(query, flat).reshape(batch, candidates, top_r)
        local_scores = self.aggregate(pair_scores, exemplar_valid)
        local_z = zscore_rows(local_scores)
        base_z = zscore_rows(base_scores.float())
        base_probability = base_z.softmax(dim=1)
        base_entropy = -(base_probability * base_probability.clamp_min(1e-8).log()).sum(dim=1)
        local_sorted = local_z.sort(dim=1, descending=True).values
        local_prediction = local_z.argmax(dim=1)
        gate_features = torch.stack(
            [
                base_z[:, 0],
                base_z[:, 0] - base_z[:, 1],
                base_entropy,
                local_z[:, 0],
                local_sorted[:, 0],
                local_sorted[:, 0] - local_sorted[:, 1],
                local_prediction.eq(0).float(),
            ],
            dim=1,
        )
        gate_logits = self.gate(gate_features).squeeze(1)
        gate = torch.sigmoid(gate_logits)
        scale = F.softplus(self.log_scale)
        adjusted = base_z + scale * gate[:, None] * local_z
        return {
            "adjusted": adjusted,
            "base_z": base_z,
            "local_scores": local_scores,
            "local_z": local_z,
            "pair_scores": pair_scores,
            "gate": gate,
            "gate_logits": gate_logits,
            "scale": scale,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_batch(full: dict[str, Any], parts: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            full["features"].index_select(0, indices).float()[:, None],
            parts["cls"].index_select(0, indices).float()[:, None],
            parts["valid_mean"].index_select(0, indices).float()[:, None],
            parts["parts"].index_select(0, indices).float(),
        ],
        dim=1,
    )


def build_members(class_ids: torch.Tensor, num_classes: int) -> list[list[int]]:
    members: list[list[int]] = [[] for _ in range(num_classes)]
    for row, class_id in enumerate(class_ids.tolist()):
        members[int(class_id)].append(row)
    counts = [len(rows) for rows in members]
    if min(counts) < 2:
        raise RuntimeError("self-excluded exemplar retrieval needs at least two rows per class")
    return members


def retrieve_exemplars(
    candidate_ids: torch.Tensor,
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    members: list[list[int]],
    top_r: int,
    device: torch.device,
    query_train_rows: torch.Tensor | None,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_total, candidates = candidate_ids.shape
    result = torch.zeros((rows_total, candidates, top_r), dtype=torch.long)
    valid = torch.zeros_like(result, dtype=torch.bool)
    occurrences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in range(rows_total):
        for slot, class_id in enumerate(candidate_ids[row].tolist()):
            occurrences[int(class_id)].append((row, slot))
    train_gpu = train_features.to(device)
    query_gpu = query_features.to(device)
    with torch.inference_mode():
        for progress, (class_id, pairs) in enumerate(occurrences.items(), 1):
            gallery = torch.tensor(members[class_id], dtype=torch.long, device=device)
            take = min(top_r, gallery.numel())
            for chunk_start in range(0, len(pairs), 2048):
                chunk = pairs[chunk_start : chunk_start + 2048]
                query_rows = torch.tensor([item[0] for item in chunk], dtype=torch.long, device=device)
                similarities = query_gpu.index_select(0, query_rows).matmul(
                    train_gpu.index_select(0, gallery).T
                )
                if query_train_rows is not None:
                    source_rows = query_train_rows.index_select(0, query_rows.cpu()).to(device)
                    similarities = similarities.masked_fill(source_rows[:, None].eq(gallery[None]), -10.0)
                values, positions = similarities.topk(take, dim=1)
                selected = gallery[positions].cpu()
                selected_valid = values.gt(-2.0).cpu()
                for local_index, (row, slot) in enumerate(chunk):
                    result[row, slot, :take] = selected[local_index]
                    valid[row, slot, :take] = selected_valid[local_index]
            if log_every and (progress % log_every == 0 or progress == len(occurrences)):
                print(
                    json.dumps(
                        {"stage": "retrieve", "classes": progress, "total_classes": len(occurrences)}
                    ),
                    flush=True,
                )
    if bool(valid.sum(dim=2).eq(0).any()):
        raise RuntimeError("one or more candidate classes have no valid exemplar")
    del train_gpu, query_gpu
    return result, valid


def extract_topk(payload: dict[str, Any], topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    if "top_scores" in payload and "top_indices" in payload:
        return payload["top_scores"][:, :topk].float(), payload["top_indices"][:, :topk].long()
    if "logits" not in payload:
        raise KeyError("base payload requires top_scores/top_indices or logits")
    return payload["logits"].float().topk(topk, dim=1)


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    data = {
        "train_parts": torch.load(args.train_parts, map_location="cpu", weights_only=False),
        "val_parts": torch.load(args.val_parts, map_location="cpu", weights_only=False),
        "train_full": torch.load(args.train_full_cls, map_location="cpu", weights_only=False),
        "val_full": torch.load(args.val_full_cls, map_location="cpu", weights_only=False),
        "train_base": torch.load(args.train_oof_topk, map_location="cpu", weights_only=False),
        "val_base": torch.load(args.val_base_logits, map_location="cpu", weights_only=False),
    }
    for split in ("train", "val"):
        parts = data[f"{split}_parts"]
        full = data[f"{split}_full"]
        base = data[f"{split}_base"]
        if list(parts["image_ids"]) != list(full["image_ids"]):
            raise RuntimeError(f"{split} part/full image_ids are not aligned")
        if list(parts["image_ids"]) != list(base["image_ids"]):
            raise RuntimeError(f"{split} cache/base image_ids are not aligned")
        if not torch.equal(parts["class_ids"].long(), base["class_ids"].long()):
            raise RuntimeError(f"{split} cache/base labels are not aligned")
    classes = list(data["train_parts"]["classes"])
    if classes != list(data["train_base"]["classes"]):
        raise RuntimeError("train parts and OOF class orders differ")
    if classes != list(data["val_base"]["classes"]):
        raise RuntimeError("train parts and val base class orders differ")
    if not bool(data["train_base"].get("strict_oof", False)):
        raise RuntimeError("train base predictions are not marked strict_oof")
    for split in ("train", "val"):
        data[f"{split}_full"]["features"] = F.normalize(
            data[f"{split}_full"]["features"].float(), dim=1
        )
    data["classes"] = classes
    data["train_scores"], data["train_candidates"] = extract_topk(data["train_base"], args.topk)
    data["val_scores"], data["val_candidates"] = extract_topk(data["val_base"], args.topk)
    return data


def target_info(candidates: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    matches = candidates.eq(labels[:, None])
    has_true = matches.any(dim=1)
    true_slot = matches.float().argmax(dim=1).long()
    target_slot = torch.where(has_true, true_slot, torch.zeros_like(true_slot))
    base_correct = candidates[:, 0].eq(labels)
    need_change = ~base_correct & has_true
    return {
        "has_true": has_true,
        "true_slot": true_slot,
        "target_slot": target_slot,
        "base_correct": base_correct,
        "need_change": need_change,
    }


def load_or_retrieve(
    path: Path,
    candidate_ids: torch.Tensor,
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    members: list[list[int]],
    top_r: int,
    device: torch.device,
    query_train_rows: torch.Tensor | None,
    image_ids: list[str],
    resume: bool,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if resume and path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            list(payload.get("image_ids", [])) == image_ids
            and torch.equal(payload.get("candidate_ids", torch.empty(0)).long(), candidate_ids.long())
            and int(payload.get("top_r", -1)) == top_r
        ):
            return payload["indices"].long(), payload["valid"].bool()
    indices, valid = retrieve_exemplars(
        candidate_ids,
        query_features,
        train_features,
        members,
        top_r,
        device,
        query_train_rows,
        log_every,
    )
    torch.save(
        {
            "indices": indices,
            "valid": valid,
            "candidate_ids": candidate_ids,
            "image_ids": image_ids,
            "top_r": top_r,
            "self_excluded": query_train_rows is not None,
        },
        path,
    )
    return indices, valid


def make_batch_tokens(
    query_full: dict[str, Any],
    query_parts: dict[str, Any],
    train_full: dict[str, Any],
    train_parts: dict[str, Any],
    rows: torch.Tensor,
    exemplar_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = token_batch(query_full, query_parts, rows)
    selected = exemplar_indices.index_select(0, rows)
    batch, candidates, top_r = selected.shape
    exemplars = token_batch(train_full, train_parts, selected.flatten())
    exemplars = exemplars.reshape(batch, candidates, top_r, query.shape[1], query.shape[2])
    return query, exemplars


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def train_model(
    args: argparse.Namespace,
    model: BaseAwareReranker,
    data: dict[str, Any],
    train_exemplars: torch.Tensor,
    train_valid: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    labels = data["train_parts"]["class_ids"].long()
    info = target_info(data["train_candidates"], labels)
    sample_weights = torch.ones(len(labels), dtype=torch.float32)
    sample_weights[info["need_change"]] = args.recovery_weight
    sample_weights[~info["has_true"]] = args.unrecoverable_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    history: list[dict[str, Any]] = []
    checkpoint_path = args.out_dir / "last.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint and int(checkpoint.get("epoch", 0)) < args.epochs:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            history = list(checkpoint.get("history", []))
        elif int(checkpoint.get("epoch", 0)) >= args.epochs:
            return list(checkpoint.get("history", []))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch * 1009)
        order = torch.randperm(len(labels), generator=generator)
        accum = defaultdict(float)
        seen = 0
        started = time.time()
        steps = math.ceil(len(labels) / args.batch_size)
        for step in range(steps):
            rows = order[step * args.batch_size : (step + 1) * args.batch_size]
            query, exemplars = make_batch_tokens(
                data["train_full"],
                data["train_parts"],
                data["train_full"],
                data["train_parts"],
                rows,
                train_exemplars,
            )
            query = query.to(device, non_blocking=True)
            exemplars = exemplars.to(device, non_blocking=True)
            valid = train_valid.index_select(0, rows).to(device)
            base_scores = data["train_scores"].index_select(0, rows).to(device)
            targets = info["target_slot"].index_select(0, rows).to(device)
            has_true = info["has_true"].index_select(0, rows).to(device)
            base_correct = info["base_correct"].index_select(0, rows).to(device)
            need_change = info["need_change"].index_select(0, rows).to(device)
            weights = sample_weights.index_select(0, rows).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(query, exemplars, valid, base_scores)
                adjusted_ce = weighted_mean(
                    F.cross_entropy(output["adjusted"], targets, reduction="none"), weights
                )
                local_losses = F.cross_entropy(output["local_z"], targets, reduction="none")
                local_mask = has_true.float()
                local_ce = weighted_mean(local_losses, weights * local_mask)
                gate_target = need_change.float()
                gate_weights = torch.where(
                    need_change,
                    torch.full_like(gate_target, args.gate_positive_weight),
                    torch.ones_like(gate_target),
                )
                gate_bce = weighted_mean(
                    F.binary_cross_entropy_with_logits(
                        output["gate_logits"], gate_target, reduction="none"
                    ),
                    gate_weights,
                )
                other_adjusted = output["adjusted"].clone()
                other_adjusted[:, 0] = float("-inf")
                preserve_gap = output["adjusted"][:, 0] - other_adjusted.max(dim=1).values
                preserve_loss = (
                    F.relu(args.preserve_margin - preserve_gap)[base_correct].mean()
                    if bool(base_correct.any())
                    else output["adjusted"].sum() * 0
                )
                row_ids = torch.arange(len(rows), device=device)
                correction_gap = output["adjusted"][row_ids, targets] - output["adjusted"][:, 0]
                correction_loss = (
                    F.relu(args.correction_margin - correction_gap)[need_change].mean()
                    if bool(need_change.any())
                    else output["adjusted"].sum() * 0
                )
                loss = (
                    adjusted_ce
                    + args.local_loss_weight * local_ce
                    + args.gate_loss_weight * gate_bce
                    + args.preserve_loss_weight * preserve_loss
                    + args.correction_loss_weight * correction_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(rows)
            seen += batch_size
            accum["loss"] += float(loss.detach()) * batch_size
            accum["adjusted_ce"] += float(adjusted_ce.detach()) * batch_size
            accum["local_ce"] += float(local_ce.detach()) * batch_size
            accum["gate_bce"] += float(gate_bce.detach()) * batch_size
            accum["preserve_loss"] += float(preserve_loss.detach()) * batch_size
            accum["correction_loss"] += float(correction_loss.detach()) * batch_size
            accum["adjusted_correct"] += int(output["adjusted"].detach().argmax(dim=1).eq(targets).sum())
            accum["local_correct"] += int((output["local_z"].detach().argmax(dim=1).eq(targets) & has_true).sum())
            accum["has_true"] += int(has_true.sum())
            if args.log_every and ((step + 1) % args.log_every == 0 or step + 1 == steps):
                print(
                    json.dumps(
                        {
                            "stage": "train",
                            "epoch": epoch,
                            "step": step + 1,
                            "steps": steps,
                            "loss": accum["loss"] / seen,
                            "adjusted_target_acc": accum["adjusted_correct"] / seen,
                            "local_true_acc": accum["local_correct"] / max(1, accum["has_true"]),
                            "scale": float(output["scale"].detach()),
                            "seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        row = {
            "epoch": epoch,
            "rows": seen,
            "seconds": time.time() - started,
            **{key: value / seen for key, value in accum.items() if key not in {"has_true", "local_correct"}},
            "local_true_acc": accum["local_correct"] / max(1, accum["has_true"]),
            "scale": float(F.softplus(model.log_scale).detach().cpu()),
        }
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "history": history,
                "arch": model_arch(args, data),
            },
            checkpoint_path,
        )
    return history


def model_arch(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_dim": int(data["train_parts"]["parts"].shape[-1]),
        "projection_dim": args.projection_dim,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "dropout": args.dropout,
        "tokens": 3 + args.grid_rows * args.grid_cols,
    }


def score_split(
    args: argparse.Namespace,
    model: BaseAwareReranker,
    query_full: dict[str, Any],
    query_parts: dict[str, Any],
    train_full: dict[str, Any],
    train_parts: dict[str, Any],
    base_scores: torch.Tensor,
    exemplar_indices: torch.Tensor,
    exemplar_valid: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(base_scores), args.eval_batch_size):
            stop = min(len(base_scores), start + args.eval_batch_size)
            rows = torch.arange(start, stop, dtype=torch.long)
            query, exemplars = make_batch_tokens(
                query_full, query_parts, train_full, train_parts, rows, exemplar_indices
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(
                    query.to(device),
                    exemplars.to(device),
                    exemplar_valid[start:stop].to(device),
                    base_scores[start:stop].to(device),
                )
            for key in ("adjusted", "base_z", "local_scores", "local_z", "gate"):
                outputs[key].append(output[key].float().cpu())
            if args.log_every and (stop == len(base_scores) or stop // args.eval_batch_size % args.log_every == 0):
                print(json.dumps({"stage": "score", "rows": stop, "total": len(base_scores)}), flush=True)
    return {key: torch.cat(chunks, dim=0) for key, chunks in outputs.items()}


def evaluate(
    args: argparse.Namespace,
    scored: dict[str, torch.Tensor],
    candidates: torch.Tensor,
    labels: torch.Tensor,
    image_ids: list[str],
) -> dict[str, Any]:
    base_prediction = candidates[:, 0]
    base_correct = base_prediction.eq(labels)
    true_in_topk = candidates.eq(labels[:, None]).any(dim=1)
    local_prediction = candidates.gather(1, scored["local_z"].argmax(dim=1, keepdim=True)).squeeze(1)
    dev, sealed = split_dev_sealed(image_ids, labels)
    masks = {"all": torch.ones(len(labels), dtype=torch.bool), "dev": dev, "sealed": sealed}
    standalone = {
        name: {
            **paired_stats(base_prediction, local_prediction, labels, mask),
            "oracle_complement": int((mask & ~base_correct & local_prediction.eq(labels)).sum()),
        }
        for name, mask in masks.items()
    }
    adjusted_slot = scored["adjusted"].argmax(dim=1)
    trials: list[dict[str, Any]] = []
    for threshold in args.gate_thresholds:
        slot = adjusted_slot.clone()
        slot[scored["gate"] < threshold] = 0
        prediction = candidates.gather(1, slot[:, None]).squeeze(1)
        trials.append(
            {
                "gate_threshold": threshold,
                **{
                    name: paired_stats(base_prediction, prediction, labels, mask)
                    for name, mask in masks.items()
                },
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            row["dev"]["net"],
            -row["dev"]["changed"],
            row["gate_threshold"],
        ),
    )
    default = min(trials, key=lambda row: abs(row["gate_threshold"] - args.default_gate_threshold))
    return {
        "base": {
            "rows": len(labels),
            "correct": int(base_correct.sum()),
            "errors": int((~base_correct).sum()),
            "topk_correct": int(true_in_topk.sum()),
            "candidate_oracle_complement": int((~base_correct & true_in_topk).sum()),
            "dev_rows": int(dev.sum()),
            "sealed_rows": int(sealed.sum()),
        },
        "standalone": standalone,
        "default": default,
        "selected_by_dev": selected,
        "trials": trials,
        "gate_mean": float(scored["gate"].mean()),
        "gate_mean_base_correct": float(scored["gate"][base_correct].mean()),
        "gate_mean_recoverable_error": float(scored["gate"][~base_correct & true_in_topk].mean()),
    }


def markdown_report(result: dict[str, Any]) -> str:
    evaluation = result["validation"]
    base = evaluation["base"]
    standalone = evaluation["standalone"]
    default = evaluation["default"]
    selected = evaluation["selected_by_dev"]
    passed = result["decision"]["passed"]
    lines = [
        "# Base-aware Dense Query–Exemplar Comparator",
        "",
        "## Outcome",
        "",
        f"- Strong reference: {base['correct']}/{base['rows']} ({base['correct']/base['rows']:.9f}).",
        f"- Frozen top-{result['config']['topk']} candidate ceiling: {base['topk_correct']}/{base['rows']}; candidate oracle complement {base['candidate_oracle_complement']}.",
        f"- Learned standalone comparator oracle complement: {standalone['all']['oracle_complement']}.",
        f"- Learned default gate ({default['gate_threshold']}): all {default['all']['net']:+d}, dev {default['dev']['net']:+d}, sealed {default['sealed']['net']:+d}.",
        f"- Dev-selected threshold ({selected['gate_threshold']}): all {selected['all']['net']:+d}, dev {selected['dev']['net']:+d}, sealed {selected['sealed']['net']:+d}.",
        f"- Continuation gate: {'PASS' if passed else 'STOP'}.",
        "",
        "## Locked decision gates",
        "",
        f"- Standalone oracle >= {result['decision']['oracle_required']}: {'yes' if result['decision']['oracle_pass'] else 'no'}.",
        f"- Dev-selected net >= {result['decision']['dev_net_required']}: {'yes' if result['decision']['dev_pass'] else 'no'}.",
        f"- Sealed net > 0: {'yes' if result['decision']['sealed_pass'] else 'no'}.",
        "",
        "## Protocol",
        "",
        "- Train candidates come from strict 5-fold OOF predictions of the reconstructed strong 6144-D joint head.",
        "- Validation candidates remain frozen to the untouched cloud strong-reference top-5.",
        "- Every train query excludes itself from exemplar retrieval.",
        f"- Each image uses full-image, ROI-global, ROI-valid and {result['config']['grid_rows']}×{result['config']['grid_cols']} dense ROI tokens.",
        "- The objective upweights recoverable OOF errors, preserves correct top-1 margins, and learns an explicit reject gate.",
        "- No class-count/global-assignment constraint, test_seen inference, or submission is used.",
        "",
        "## Training",
        "",
        "| Epoch | Loss | Adjusted target acc | Local true acc | Scale | Seconds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["training"]:
        lines.append(
            f"| {row['epoch']} | {row['loss']:.5f} | {row['adjusted_correct']:.5f} | "
            f"{row['local_true_acc']:.5f} | {row['scale']:.4f} | {row['seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parts", type=Path, required=True)
    parser.add_argument("--val-parts", type=Path, required=True)
    parser.add_argument("--train-full-cls", type=Path, required=True)
    parser.add_argument("--val-full-cls", type=Path, required=True)
    parser.add_argument("--train-oof-topk", type=Path, required=True)
    parser.add_argument("--val-base-logits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--recovery-weight", type=float, default=5.0)
    parser.add_argument("--unrecoverable-weight", type=float, default=0.5)
    parser.add_argument("--gate-positive-weight", type=float, default=5.0)
    parser.add_argument("--local-loss-weight", type=float, default=0.5)
    parser.add_argument("--gate-loss-weight", type=float, default=0.2)
    parser.add_argument("--preserve-loss-weight", type=float, default=0.75)
    parser.add_argument("--correction-loss-weight", type=float, default=0.75)
    parser.add_argument("--preserve-margin", type=float, default=0.35)
    parser.add_argument("--correction-margin", type=float, default=0.20)
    parser.add_argument("--gate-thresholds", type=float, nargs="+", default=[0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
    parser.add_argument("--default-gate-threshold", type=float, default=0.5)
    parser.add_argument("--oracle-required", type=int, default=250)
    parser.add_argument("--dev-net-required", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    data = load_inputs(args)
    part_shape = data["train_parts"]["parts"].shape
    if part_shape[1] != args.grid_rows * args.grid_cols:
        raise RuntimeError(
            f"train parts contain {part_shape[1]} tokens, expected {args.grid_rows}x{args.grid_cols}"
        )
    members = build_members(data["train_parts"]["class_ids"].long(), len(data["classes"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_exemplars, train_valid = load_or_retrieve(
        args.out_dir / "train_exemplars.pt",
        data["train_candidates"],
        data["train_full"]["features"],
        data["train_full"]["features"],
        members,
        args.top_r,
        device,
        torch.arange(len(data["train_candidates"]), dtype=torch.long),
        list(data["train_parts"]["image_ids"]),
        args.resume,
        args.log_every,
    )
    val_exemplars, val_valid = load_or_retrieve(
        args.out_dir / "val_exemplars.pt",
        data["val_candidates"],
        data["val_full"]["features"],
        data["train_full"]["features"],
        members,
        args.top_r,
        device,
        None,
        list(data["val_parts"]["image_ids"]),
        args.resume,
        args.log_every,
    )
    pair = DensePairComparator(
        input_dim=int(part_shape[-1]),
        projection_dim=args.projection_dim,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        dropout=args.dropout,
    )
    model = BaseAwareReranker(pair).to(device)
    training = train_model(args, model, data, train_exemplars, train_valid, device)
    train_scored = score_split(
        args,
        model,
        data["train_full"],
        data["train_parts"],
        data["train_full"],
        data["train_parts"],
        data["train_scores"],
        train_exemplars,
        train_valid,
        device,
    )
    val_scored = score_split(
        args,
        model,
        data["val_full"],
        data["val_parts"],
        data["train_full"],
        data["train_parts"],
        data["val_scores"],
        val_exemplars,
        val_valid,
        device,
    )
    train_eval = evaluate(
        args,
        train_scored,
        data["train_candidates"],
        data["train_parts"]["class_ids"].long(),
        list(data["train_parts"]["image_ids"]),
    )
    val_eval = evaluate(
        args,
        val_scored,
        data["val_candidates"],
        data["val_parts"]["class_ids"].long(),
        list(data["val_parts"]["image_ids"]),
    )
    selected = val_eval["selected_by_dev"]
    oracle_pass = val_eval["standalone"]["all"]["oracle_complement"] >= args.oracle_required
    dev_pass = selected["dev"]["net"] >= args.dev_net_required
    sealed_pass = selected["sealed"]["net"] > 0
    result = {
        "config": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "topk": args.topk,
            "top_r": args.top_r,
            "epochs": args.epochs,
            "seed": args.seed,
            "train_oof_topk": str(args.train_oof_topk.resolve()),
            "val_base_logits": str(args.val_base_logits.resolve()),
        },
        "training": training,
        "train_oof_evaluation": train_eval,
        "validation": val_eval,
        "decision": {
            "oracle_required": args.oracle_required,
            "dev_net_required": args.dev_net_required,
            "oracle_pass": oracle_pass,
            "dev_pass": dev_pass,
            "sealed_pass": sealed_pass,
            "passed": oracle_pass and dev_pass and sealed_pass,
            "test_seen_allowed": oracle_pass and dev_pass and sealed_pass,
        },
    }
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "arch": model_arch(args, data),
            "training": training,
        },
        args.out_dir / "model_final.pt",
    )
    torch.save(
        {
            **val_scored,
            "top_indices": data["val_candidates"],
            "top_values": data["val_scores"],
            "class_ids": data["val_parts"]["class_ids"].long(),
            "image_ids": list(data["val_parts"]["image_ids"]),
            "classes": data["classes"],
        },
        args.out_dir / "val_scores.pt",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "EXPERIMENT_REPORT.md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

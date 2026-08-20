from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids, class_ids):
    groups = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, class_ids.tolist())):
        groups[int(class_id)].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for position, (_digest, index) in enumerate(rows):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def zscore_rows(values):
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def paired_stats(base_prediction, prediction, labels, mask):
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
        "efficiency": float((wins.sum() - losses.sum()) / max(1, changed.sum())),
    }


class PairComparator(nn.Module):
    """Class-agnostic learned interaction over full, ROI, and ROI-part tokens."""

    def __init__(self, input_dim=768, projection_dim=96, tokens=9, dropout=0.10):
        super().__init__()
        self.tokens = tokens
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.token_embedding = nn.Parameter(torch.zeros(tokens, projection_dim))
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        # cross matrix + row/column maxima + direct and flip diagonals + 5 stats
        interaction_dim = tokens * tokens + 4 * tokens + 5
        self.head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )
        # full, roi-cls, roi-valid-mean, then a horizontally flipped 2x3 grid.
        self.register_buffer(
            "flip_order",
            torch.tensor([0, 1, 2, 5, 4, 3, 8, 7, 6], dtype=torch.long),
        )

    def forward(self, query, exemplar):
        # query: [B,T,D], exemplar: [B,K,T,D]
        query_raw = F.normalize(query.float(), dim=-1)
        exemplar_raw = F.normalize(exemplar.float(), dim=-1)
        query = self.projection(query_raw) + self.token_embedding[None]
        exemplar = self.projection(exemplar_raw) + self.token_embedding[None, None]
        query = F.normalize(query, dim=-1)
        exemplar = F.normalize(exemplar, dim=-1)
        cross = torch.einsum("btd,bksd->bkts", query, exemplar)
        direct = (query[:, None] * exemplar).sum(dim=-1)
        flipped = (
            query[:, None] * exemplar.index_select(2, self.flip_order)
        ).sum(dim=-1)
        raw_direct = (query_raw[:, None] * exemplar_raw).sum(dim=-1)
        features = torch.cat(
            [
                cross.flatten(2),
                cross.max(dim=3).values,
                cross.max(dim=2).values,
                direct,
                flipped,
                cross.mean(dim=(2, 3), keepdim=False)[..., None],
                cross.amax(dim=(2, 3), keepdim=False)[..., None],
                raw_direct[..., 0:1],
                raw_direct[..., 1:2],
                raw_direct[..., 2:3],
            ],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


def load_inputs(args):
    train_parts = torch.load(args.train_parts, map_location="cpu", weights_only=False)
    val_parts = torch.load(args.val_parts, map_location="cpu", weights_only=False)
    train_full = torch.load(args.train_full_cls, map_location="cpu", weights_only=False)
    val_full = torch.load(args.val_full_cls, map_location="cpu", weights_only=False)
    base = torch.load(args.base_logits, map_location="cpu", weights_only=False)
    if list(train_parts["image_ids"]) != list(train_full["image_ids"]):
        raise RuntimeError("train part/full caches are not aligned")
    if list(val_parts["image_ids"]) != list(val_full["image_ids"]):
        raise RuntimeError("val part/full caches are not aligned")
    if list(val_parts["image_ids"]) != list(base["image_ids"]):
        raise RuntimeError("val cache/base logits are not aligned")
    if not torch.equal(val_parts["class_ids"].long(), base["class_ids"].long()):
        raise RuntimeError("val labels/base labels are not aligned")
    if list(train_parts["classes"]) != list(base["classes"]):
        raise RuntimeError("class order differs between train cache and base logits")
    train_full["features"] = F.normalize(train_full["features"].float(), dim=1)
    val_full["features"] = F.normalize(val_full["features"].float(), dim=1)
    return train_parts, val_parts, train_full, val_full, base


def build_member_table(class_ids, num_classes):
    members = [[] for _ in range(num_classes)]
    position = torch.empty(len(class_ids), dtype=torch.long)
    for row, class_id in enumerate(class_ids.tolist()):
        position[row] = len(members[class_id])
        members[class_id].append(row)
    counts = torch.tensor([len(rows) for rows in members], dtype=torch.long)
    if int(counts.min()) < 2:
        raise RuntimeError("leave-one-out training needs at least two rows per class")
    table = torch.full((num_classes, int(counts.max())), -1, dtype=torch.long)
    for class_id, rows in enumerate(members):
        table[class_id, : len(rows)] = torch.tensor(rows, dtype=torch.long)
    return members, table, counts, position


def build_class_prototypes(features, class_ids, num_classes):
    prototypes = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
    prototypes.index_add_(0, class_ids, features.float())
    counts = torch.bincount(class_ids, minlength=num_classes).float().clamp_min(1)
    return F.normalize(prototypes / counts[:, None], dim=1)


def build_hard_class_pools(prototypes, classes, visual_pool, device):
    num_classes = len(classes)
    genera = defaultdict(list)
    for class_id, name in enumerate(classes):
        genera[name.split(maxsplit=1)[0] if name else ""].append(class_id)
    same_lists = []
    for class_id, name in enumerate(classes):
        peers = [
            peer
            for peer in genera[name.split(maxsplit=1)[0] if name else ""]
            if peer != class_id
        ]
        if peers:
            peer_tensor = torch.tensor(peers, dtype=torch.long)
            similarities = prototypes.index_select(0, peer_tensor).matmul(
                prototypes[class_id]
            )
            peers = peer_tensor[similarities.argsort(descending=True)].tolist()
        same_lists.append(peers)

    visual = torch.empty((num_classes, visual_pool), dtype=torch.long)
    proto_gpu = prototypes.to(device)
    with torch.inference_mode():
        for start in range(0, num_classes, 256):
            stop = min(num_classes, start + 256)
            similarities = proto_gpu[start:stop].matmul(proto_gpu.T)
            row_ids = torch.arange(start, stop, device=device)
            similarities[torch.arange(stop - start, device=device), row_ids] = -2
            visual[start:stop] = similarities.topk(visual_pool, dim=1).indices.cpu()
    del proto_gpu

    max_same = max(1, max(map(len, same_lists)))
    same = torch.full((num_classes, max_same), -1, dtype=torch.long)
    same_count = torch.zeros(num_classes, dtype=torch.long)
    for class_id, peers in enumerate(same_lists):
        same_count[class_id] = len(peers)
        if peers:
            same[class_id, : len(peers)] = torch.tensor(peers, dtype=torch.long)
    return same, same_count, visual


def token_batch(full, parts, indices):
    return torch.cat(
        [
            full["features"].index_select(0, indices)[:, None],
            parts["cls"].index_select(0, indices).float()[:, None],
            parts["valid_mean"].index_select(0, indices).float()[:, None],
            parts["parts"].index_select(0, indices).float(),
        ],
        dim=1,
    )


def candidate_indices_for_batch(
    rows,
    epoch,
    class_ids,
    member_table,
    counts,
    positions,
    same,
    same_count,
    visual,
    negatives,
):
    labels = class_ids.index_select(0, rows)
    row_hash = rows * 104729 + (epoch + 1) * 13007
    positive_offset = row_hash.remainder(counts.index_select(0, labels) - 1) + 1
    positive_position = (
        positions.index_select(0, rows) + positive_offset
    ).remainder(counts.index_select(0, labels))
    positive = member_table[labels, positive_position]
    negative_rows = []
    for slot in range(negatives):
        use_same = slot < math.ceil(negatives / 2)
        hash_value = row_hash + (slot + 1) * 8191
        if use_same:
            available = same_count.index_select(0, labels)
            safe_available = available.clamp_min(1)
            selected_class = same[
                labels, hash_value.remainder(safe_available)
            ]
            fallback = available.eq(0)
            if fallback.any():
                selected_class[fallback] = visual[
                    labels[fallback],
                    hash_value[fallback].remainder(visual.shape[1]),
                ]
        else:
            selected_class = visual[
                labels, hash_value.remainder(visual.shape[1])
            ]
        selected_count = counts.index_select(0, selected_class)
        selected_position = (hash_value * 17 + slot * 31).remainder(selected_count)
        negative_rows.append(member_table[selected_class, selected_position])
    return torch.stack([positive] + negative_rows, dim=1)


def train_model(
    args,
    model,
    train_parts,
    train_full,
    member_table,
    counts,
    positions,
    same,
    same_count,
    visual,
    device,
):
    class_ids = train_parts["class_ids"].long()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(args.epochs):
        model.train()
        generator = torch.Generator().manual_seed(args.seed + epoch * 997)
        order = torch.randperm(len(class_ids), generator=generator)
        loss_sum = 0.0
        correct = 0
        seen = 0
        started = time.time()
        steps = math.ceil(len(order) / args.batch_size)
        if args.max_steps_per_epoch:
            steps = min(steps, args.max_steps_per_epoch)
        for step in range(steps):
            rows = order[step * args.batch_size : (step + 1) * args.batch_size]
            candidates = candidate_indices_for_batch(
                rows,
                epoch,
                class_ids,
                member_table,
                counts,
                positions,
                same,
                same_count,
                visual,
                args.negatives,
            )
            query = token_batch(train_full, train_parts, rows).to(
                device, non_blocking=True
            )
            flat_candidates = candidates.flatten()
            exemplar = token_batch(train_full, train_parts, flat_candidates)
            exemplar = exemplar.reshape(
                len(rows), args.negatives + 1, model.tokens, -1
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                scores = model(query, exemplar)
                targets = torch.zeros(len(rows), dtype=torch.long, device=device)
                loss = F.cross_entropy(
                    scores, targets, label_smoothing=args.label_smoothing
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(rows)
            correct += int(scores.detach().argmax(dim=1).eq(0).sum())
            seen += len(rows)
            if args.log_every and ((step + 1) % args.log_every == 0 or step + 1 == steps):
                print(
                    json.dumps(
                        {
                            "stage": "train",
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "steps": steps,
                            "loss": loss_sum / seen,
                            "pair_top1": correct / seen,
                            "seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / seen,
                "pair_top1": correct / seen,
                "rows": seen,
                "seconds": time.time() - started,
            }
        )
    return history


def retrieve_eval_exemplars(
    top_indices,
    train_features,
    val_features,
    members,
    top_r,
    device,
    max_eval_rows=0,
):
    rows_total = len(top_indices) if not max_eval_rows else min(len(top_indices), max_eval_rows)
    top_indices = top_indices[:rows_total]
    result = torch.zeros((rows_total, top_indices.shape[1], top_r), dtype=torch.long)
    valid = torch.zeros_like(result, dtype=torch.bool)
    occurrences = defaultdict(list)
    for row in range(rows_total):
        for slot, class_id in enumerate(top_indices[row].tolist()):
            occurrences[int(class_id)].append((row, slot))
    train_gpu = train_features.to(device)
    val_gpu = val_features[:rows_total].to(device)
    with torch.inference_mode():
        for progress, (class_id, pairs) in enumerate(occurrences.items(), 1):
            query_rows = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
            gallery = torch.tensor(members[class_id], dtype=torch.long, device=device)
            similarities = val_gpu.index_select(0, query_rows).matmul(
                train_gpu.index_select(0, gallery).T
            )
            take = min(top_r, len(gallery))
            selected = gallery[similarities.topk(take, dim=1).indices].cpu()
            for pair_index, (row, slot) in enumerate(pairs):
                result[row, slot, :take] = selected[pair_index]
                valid[row, slot, :take] = True
            if progress % 1000 == 0 or progress == len(occurrences):
                print(
                    json.dumps(
                        {
                            "stage": "retrieve",
                            "classes": progress,
                            "total_classes": len(occurrences),
                        }
                    ),
                    flush=True,
                )
    del train_gpu, val_gpu
    return result, valid


def score_eval(
    args,
    model,
    exemplar_indices,
    exemplar_valid,
    val_parts,
    val_full,
    train_parts,
    train_full,
    device,
):
    model.eval()
    rows_total, topk, top_r = exemplar_indices.shape
    raw = torch.empty((rows_total, topk, top_r), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, rows_total, args.eval_batch_size):
            stop = min(rows_total, start + args.eval_batch_size)
            query_rows = torch.arange(start, stop, dtype=torch.long)
            query = token_batch(val_full, val_parts, query_rows).to(device)
            indices = exemplar_indices[start:stop].flatten()
            exemplar = token_batch(train_full, train_parts, indices)
            exemplar = exemplar.reshape(stop - start, topk * top_r, model.tokens, -1).to(
                device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                scores = model(query, exemplar)
            raw[start:stop] = scores.float().cpu().reshape(stop - start, topk, top_r)
            if args.log_every and (
                stop == rows_total or stop // args.eval_batch_size % args.log_every == 0
            ):
                print(
                    json.dumps(
                        {"stage": "score", "rows": stop, "total_rows": rows_total}
                    ),
                    flush=True,
                )
    masked = raw.masked_fill(~exemplar_valid, float("-inf"))
    score_max = masked.max(dim=2).values
    score_mean = raw.masked_fill(~exemplar_valid, 0).sum(dim=2) / exemplar_valid.sum(
        dim=2
    ).clamp_min(1)
    score_lse = torch.logsumexp(masked, dim=2) - exemplar_valid.sum(dim=2).float().log()
    families = {"max": score_max, "mean": score_mean, "logmeanexp": score_lse}
    for take in (2, 3, 5):
        if top_r >= take:
            top_values = masked.topk(take, dim=2).values
            top_valid = torch.isfinite(top_values)
            families[f"top{take}_mean"] = top_values.masked_fill(
                ~top_valid, 0
            ).sum(dim=2) / top_valid.sum(dim=2).clamp_min(1)
    return families, raw


def evaluate(args, score_families, top_values, top_indices, labels, image_ids):
    rows_total = len(labels)
    dev, sealed = split_dev_sealed(image_ids[:rows_total], labels)
    masks = {
        "all": torch.ones(rows_total, dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    base_prediction = top_indices[:, 0]
    base_z = zscore_rows(top_values.float())
    base_correct = base_prediction.eq(labels)
    family_results = []
    candidates = []
    for name, local_scores in score_families.items():
        local_z = zscore_rows(local_scores.float())
        local_prediction = top_indices.gather(
            1, local_scores.argmax(dim=1, keepdim=True)
        ).squeeze(1)
        standalone = {
            split_name: {
                **paired_stats(base_prediction, local_prediction, labels, mask),
                "oracle_complement": int(
                    (mask & ~base_correct & local_prediction.eq(labels)).sum()
                ),
            }
            for split_name, mask in masks.items()
        }
        trials = []
        for alpha in args.alphas:
            prediction = top_indices.gather(
                1, (base_z + alpha * local_z).argmax(dim=1, keepdim=True)
            ).squeeze(1)
            stats = {
                split_name: paired_stats(
                    base_prediction, prediction, labels, mask
                )
                for split_name, mask in masks.items()
            }
            trial = {"family": name, "alpha": alpha, **stats}
            trials.append(trial)
            candidates.append(trial)
        family_results.append(
            {"name": name, "standalone": standalone, "trials": trials}
        )
    selected = max(
        candidates,
        key=lambda row: (row["dev"]["net"], -row["dev"]["changed"], -row["alpha"]),
    )
    return {
        "base": {
            "rows": rows_total,
            "correct": int(base_correct.sum()),
            "errors": int((~base_correct).sum()),
            "top5_correct": int(top_indices.eq(labels[:, None]).any(dim=1).sum()),
            "true_in_top5_errors": int(
                (~base_correct & top_indices.eq(labels[:, None]).any(dim=1)).sum()
            ),
            "dev_rows": int(dev.sum()),
            "sealed_rows": int(sealed.sum()),
            "dev_correct": int((dev & base_correct).sum()),
            "sealed_correct": int((sealed & base_correct).sum()),
        },
        "families": family_results,
        "selected_by_dev": selected,
    }


def markdown_report(result):
    base = result["evaluation"]["base"]
    selected = result["evaluation"]["selected_by_dev"]
    lines = [
        "# Learned Query–Exemplar Comparator Scout",
        "",
        "## Outcome",
        "",
        f"- Strong-base accuracy: {base['correct']}/{base['rows']} ({base['correct']/base['rows']:.6f}).",
        f"- Base top-5 ceiling: {base['top5_correct']}/{base['rows']}; recoverable base errors: {base['true_in_top5_errors']}.",
        f"- Dev-selected rule: `{selected['family']}`, alpha `{selected['alpha']}`.",
        f"- All: net {selected['all']['net']:+d} ({selected['all']['wins']} wins / {selected['all']['losses']} losses).",
        f"- Locked dev: net {selected['dev']['net']:+d} ({selected['dev']['wins']} wins / {selected['dev']['losses']} losses).",
        f"- Sealed: net {selected['sealed']['net']:+d} ({selected['sealed']['wins']} wins / {selected['sealed']['losses']} losses).",
        "",
        "## Standalone comparator families",
        "",
        "| Aggregation | All net | Oracle complement | Dev net | Sealed net |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in result["evaluation"]["families"]:
        standalone = family["standalone"]
        lines.append(
            f"| {family['name']} | {standalone['all']['net']:+d} | "
            f"{standalone['all']['oracle_complement']} | {standalone['dev']['net']:+d} | "
            f"{standalone['sealed']['net']:+d} |"
        )
    lines += [
        "",
        "## Protocol",
        "",
        "- Candidate scope is frozen to the strong baseline top-5.",
        "- Training positives are same-class, different-image leave-one-out exemplars.",
        "- Half the negatives are same-genus when available; the rest are train-only visual-nearest classes.",
        "- The comparator is class-agnostic and consumes 512px full-image, ROI-global, and six ROI-part tokens.",
        "- Exemplar retrieval uses only train images. Alpha and aggregation are selected on deterministic per-class dev; sealed is reported after selection.",
        "- No test_seen predictions or submission were produced.",
        "",
        "## Training",
        "",
        "| Epoch | Pair loss | Pair top-1 | Seconds |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["training"]:
        lines.append(
            f"| {row['epoch']} | {row['loss']:.5f} | {row['pair_top1']:.5f} | {row['seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parts", type=Path, required=True)
    parser.add_argument("--val-parts", type=Path, required=True)
    parser.add_argument("--train-full-cls", type=Path, required=True)
    parser.add_argument("--val-full-cls", type=Path, required=True)
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--visual-pool", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00],
    )
    args = parser.parse_args()
    if args.load_checkpoint and args.init_checkpoint:
        parser.error("--load-checkpoint and --init-checkpoint are mutually exclusive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"stage": "load", "device": str(device)}), flush=True)
    train_parts, val_parts, train_full, val_full, base = load_inputs(args)
    num_classes = len(train_parts["classes"])
    members, member_table, counts, positions = build_member_table(
        train_parts["class_ids"].long(), num_classes
    )
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        model = PairComparator(**config).to(device)
        model.load_state_dict(checkpoint["model"])
        training = []
        print(
            json.dumps(
                {"stage": "checkpoint", "path": str(args.load_checkpoint.resolve())}
            ),
            flush=True,
        )
    else:
        prototypes = build_class_prototypes(
            train_full["features"], train_parts["class_ids"].long(), num_classes
        )
        same, same_count, visual = build_hard_class_pools(
            prototypes, train_parts["classes"], args.visual_pool, device
        )
        print(
            json.dumps(
                {
                    "stage": "pools",
                    "classes": num_classes,
                    "classes_with_same_genus_negative": int(same_count.gt(0).sum()),
                    "visual_pool": args.visual_pool,
                }
            ),
            flush=True,
        )
        if args.init_checkpoint:
            checkpoint = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=False
            )
            config = checkpoint["model_config"]
            model = PairComparator(**config).to(device)
            model.load_state_dict(checkpoint["model"])
            print(
                json.dumps(
                    {
                        "stage": "initialization",
                        "path": str(args.init_checkpoint.resolve()),
                    }
                ),
                flush=True,
            )
        else:
            model = PairComparator(
                projection_dim=args.projection_dim, dropout=args.dropout
            ).to(device)
        training = train_model(
            args,
            model,
            train_parts,
            train_full,
            member_table,
            counts,
            positions,
            same,
            same_count,
            visual,
            device,
        )
    logits = base["logits"]
    top_values, top_indices = logits.topk(args.topk, dim=1)
    del logits, base["logits"]
    rows_total = len(top_indices)
    if args.max_eval_rows:
        rows_total = min(rows_total, args.max_eval_rows)
        top_values = top_values[:rows_total]
        top_indices = top_indices[:rows_total]
    exemplar_indices, exemplar_valid = retrieve_eval_exemplars(
        top_indices,
        train_full["features"],
        val_full["features"],
        members,
        args.top_r,
        device,
        max_eval_rows=rows_total,
    )
    score_families, raw_scores = score_eval(
        args,
        model,
        exemplar_indices,
        exemplar_valid,
        val_parts,
        val_full,
        train_parts,
        train_full,
        device,
    )
    labels = val_parts["class_ids"][:rows_total].long()
    evaluation = evaluate(
        args,
        score_families,
        top_values,
        top_indices,
        labels,
        val_parts["image_ids"][:rows_total],
    )
    result = {
        "protocol": {
            "train_parts": str(args.train_parts.resolve()),
            "val_parts": str(args.val_parts.resolve()),
            "train_full_cls": str(args.train_full_cls.resolve()),
            "val_full_cls": str(args.val_full_cls.resolve()),
            "base_logits": str(args.base_logits.resolve()),
            "candidate_scope": "frozen strong-base top-5",
            "positive": "same class, different train image, leave-one-out",
            "negatives": "same-genus plus train-only visual-nearest classes",
            "tokens": "512px full CLS + ROI CLS + ROI valid mean + 2x3 ROI parts",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "negatives_per_query": args.negatives,
            "top_r": args.top_r,
            "alphas": args.alphas,
            "device": str(device),
            "loaded_checkpoint": str(args.load_checkpoint.resolve())
            if args.load_checkpoint
            else None,
            "initialized_from": str(args.init_checkpoint.resolve())
            if args.init_checkpoint
            else None,
            "test_seen_touched": False,
        },
        "training": training,
        "evaluation": evaluation,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": {
                "input_dim": 768,
                "projection_dim": args.projection_dim,
                "tokens": model.tokens,
                "dropout": args.dropout,
            },
            "protocol": result["protocol"],
        },
        args.out_dir / "comparator.pt",
    )
    torch.save(
        {
            "image_ids": val_parts["image_ids"][:rows_total],
            "labels": labels,
            "top_indices": top_indices,
            "top_values": top_values.half(),
            "exemplar_indices": exemplar_indices,
            "exemplar_valid": exemplar_valid,
            "raw_scores": raw_scores.half(),
            "score_families": {k: v.half() for k, v in score_families.items()},
        },
        args.out_dir / "val_scores.pt",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "EXPERIMENT_REPORT.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    print(json.dumps({"stage": "done", **evaluation["selected_by_dev"]}), flush=True)


if __name__ == "__main__":
    main()

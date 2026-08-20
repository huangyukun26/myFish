from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        "efficiency": float((wins.sum() - losses.sum()) / max(1, int(changed.sum()))),
    }


def zscore_rows(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def topk_target(top_indices: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    match = top_indices.eq(labels[:, None])
    in_topk = match.any(dim=1)
    target = torch.zeros(len(labels), dtype=torch.long)
    target[in_topk] = match.float().argmax(dim=1)[in_topk].long()
    return target, in_topk


def build_members(class_ids: torch.Tensor, num_classes: int):
    members = [[] for _ in range(num_classes)]
    position = torch.empty(len(class_ids), dtype=torch.long)
    for row, class_id in enumerate(class_ids.tolist()):
        position[row] = len(members[class_id])
        members[class_id].append(row)
    return [torch.tensor(rows, dtype=torch.long) for rows in members], position


def normalize_views(features: torch.Tensor, component_dims: list[int]) -> dict[str, torch.Tensor]:
    first = int(component_dims[0]) if component_dims else features.shape[1] // 2
    return {
        "full": F.normalize(features.float(), dim=1),
        "bio": F.normalize(features[:, :first].float(), dim=1),
        "dino": F.normalize(features[:, first:].float(), dim=1),
    }


def build_prototypes(
    train_views: dict[str, torch.Tensor], class_ids: torch.Tensor, num_classes: int
) -> dict[str, torch.Tensor]:
    counts = torch.bincount(class_ids, minlength=num_classes).float().clamp_min(1)
    output = {}
    for name, features in train_views.items():
        proto = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
        proto.index_add_(0, class_ids, features)
        output[name] = F.normalize(proto / counts[:, None], dim=1)
    return output


def compute_support_stats_for_view(
    query_features: torch.Tensor,
    support_features: torch.Tensor,
    candidate_indices: torch.Tensor,
    members: list[torch.Tensor],
    *,
    query_labels: torch.Tensor | None,
    support_position: torch.Tensor | None,
    device: str,
    label: str,
) -> torch.Tensor:
    """Return [N, K, 4] max/top2/top3/mean support similarities.

    For training rows, when the candidate class equals the query label, the query
    image itself is excluded from the support set.
    """

    start_time = time.time()
    num_rows, topk = candidate_indices.shape
    stats = torch.full((num_rows, topk, 4), -2.0, dtype=torch.float32)
    unique_classes = torch.unique(candidate_indices.cpu()).tolist()
    support_on_device_cache: dict[int, torch.Tensor] = {}

    for offset, class_id in enumerate(unique_classes, start=1):
        locations = candidate_indices.eq(int(class_id)).nonzero(as_tuple=False)
        if locations.numel() == 0:
            continue
        rows = locations[:, 0].long()
        slots = locations[:, 1].long()
        support_rows = members[int(class_id)]
        if support_rows.numel() == 0:
            continue

        query = query_features.index_select(0, rows).to(device, non_blocking=False)
        support = support_features.index_select(0, support_rows).to(
            device, non_blocking=False
        )
        sims = query.matmul(support.T)
        valid = torch.ones_like(sims, dtype=torch.bool)

        if query_labels is not None and support_position is not None:
            same = query_labels.index_select(0, rows).eq(int(class_id))
            if bool(same.any()):
                same_positions = support_position.index_select(0, rows[same]).to(device)
                same_rows = same.nonzero(as_tuple=False).flatten().to(device)
                valid[same_rows, same_positions] = False

        sims_for_rank = sims.masked_fill(~valid, -2.0)
        valid_counts = valid.sum(dim=1).clamp_min(1)
        max_values = sims_for_rank.max(dim=1).values
        k2 = min(2, sims_for_rank.shape[1])
        k3 = min(3, sims_for_rank.shape[1])
        top2 = sims_for_rank.topk(k2, dim=1).values
        top3 = sims_for_rank.topk(k3, dim=1).values
        top2_mean = top2.sum(dim=1) / torch.minimum(
            valid_counts, torch.full_like(valid_counts, k2)
        ).float()
        top3_mean = top3.sum(dim=1) / torch.minimum(
            valid_counts, torch.full_like(valid_counts, k3)
        ).float()
        mean = sims.masked_fill(~valid, 0.0).sum(dim=1) / valid_counts.float()

        values = torch.stack([max_values, top2_mean, top3_mean, mean], dim=1).cpu()
        stats[rows, slots] = values

        if offset % 500 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{label}] {offset}/{len(unique_classes)} classes, "
                f"{elapsed:.1f}s",
                flush=True,
            )

    support_on_device_cache.clear()
    return stats


def compute_prototype_stats(
    query_views: dict[str, torch.Tensor],
    prototypes: dict[str, torch.Tensor],
    candidate_indices: torch.Tensor,
    device: str,
) -> torch.Tensor:
    out = []
    rows = torch.arange(candidate_indices.shape[0], dtype=torch.long)
    for name in ["full", "bio", "dino"]:
        query = query_views[name].to(device)
        proto = prototypes[name].to(device)
        values = torch.empty(candidate_indices.shape, dtype=torch.float32)
        with torch.inference_mode():
            for start in range(0, len(rows), 512):
                stop = min(len(rows), start + 512)
                q = query[start:stop]
                p = proto.index_select(0, candidate_indices[start:stop].reshape(-1).to(device))
                p = p.reshape(stop - start, candidate_indices.shape[1], -1)
                sim = (q[:, None] * p).sum(dim=2).cpu()
                values[start:stop] = sim
        out.append(values[:, :, None])
        del query, proto
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(out, dim=2)


def base_slot_features(
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    classes: list[str],
    class_counts: torch.Tensor,
    error_gate: torch.Tensor | None,
) -> tuple[torch.Tensor, list[str]]:
    top_scores = top_scores.float()
    num_rows, topk = top_scores.shape
    score_z = zscore_rows(top_scores)
    top1 = top_scores[:, :1]
    delta_top = top_scores - top1
    soft = torch.softmax(top_scores, dim=1)
    entropy = -(soft * soft.clamp_min(1e-8).log()).sum(dim=1)
    margin12 = top_scores[:, 0] - top_scores[:, 1]
    margin15 = top_scores[:, 0] - top_scores[:, -1]
    ranks = torch.arange(topk).float()[None].expand(num_rows, topk)
    is_top1 = torch.zeros_like(top_scores)
    is_top1[:, 0] = 1.0

    genera = []
    for name in classes:
        genera.append(name.split(maxsplit=1)[0] if name else "")
    top1_genus = [genera[int(x)] for x in top_indices[:, 0].tolist()]
    same_genus = torch.zeros((num_rows, topk), dtype=torch.float32)
    for row in range(num_rows):
        genus = top1_genus[row]
        for slot in range(topk):
            same_genus[row, slot] = 1.0 if genera[int(top_indices[row, slot])] == genus else 0.0

    count_log = torch.log1p(class_counts.index_select(0, top_indices.reshape(-1)).float())
    count_log = count_log.reshape(num_rows, topk)

    repeated = [
        top_scores[:, :, None],
        score_z[:, :, None],
        delta_top[:, :, None],
        ranks[:, :, None] / max(1, topk - 1),
        is_top1[:, :, None],
        count_log[:, :, None],
        same_genus[:, :, None],
        margin12[:, None, None].expand(num_rows, topk, 1),
        margin15[:, None, None].expand(num_rows, topk, 1),
        entropy[:, None, None].expand(num_rows, topk, 1),
    ]
    names = [
        "base_score",
        "base_score_z",
        "delta_from_top1",
        "rank_norm",
        "is_top1",
        "class_count_log",
        "same_genus_as_top1",
        "row_margin12",
        "row_margin15",
        "row_entropy5",
    ]
    if error_gate is not None:
        repeated.append(error_gate.float()[:, None, None].expand(num_rows, topk, 1))
        names.append("row_error_gate")
    return torch.cat(repeated, dim=2), names


def assemble_slot_features(
    *,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    classes: list[str],
    class_counts: torch.Tensor,
    support_stats: dict[str, torch.Tensor],
    prototype_stats: torch.Tensor,
    error_gate: torch.Tensor | None,
) -> tuple[torch.Tensor, list[str]]:
    parts, names = base_slot_features(
        top_scores, top_indices, classes, class_counts, error_gate
    )
    all_parts = [parts]
    all_names = list(names)
    for view_name in ["full", "bio", "dino"]:
        values = support_stats[view_name]
        all_parts.append(values)
        all_names.extend(
            [
                f"{view_name}_support_max",
                f"{view_name}_support_top2mean",
                f"{view_name}_support_top3mean",
                f"{view_name}_support_mean",
            ]
        )
    all_parts.append(prototype_stats)
    all_names.extend(["full_proto", "bio_proto", "dino_proto"])

    features = torch.cat(all_parts, dim=2).float()
    return features, all_names


class SlotModel(nn.Module):
    def __init__(self, feature_dim: int, architecture: str, hidden_dim: int, dropout: float):
        super().__init__()
        if architecture == "linear":
            self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))
        elif architecture == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def train_slot_model(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_in_topk: torch.Tensor,
    *,
    architecture: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    error_weight: float,
    preserve_weight: float,
    miss_weight: float,
    device: str,
) -> SlotModel:
    torch.manual_seed(seed)
    model = SlotModel(train_features.shape[2], architecture, hidden_dim, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = torch.full((len(train_targets),), float(miss_weight), dtype=torch.float32)
    weights[train_in_topk & train_targets.eq(0)] = float(preserve_weight)
    weights[train_targets.gt(0)] = float(error_weight)
    batch_size = 4096
    n = len(train_targets)
    train_features = train_features.float()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(n, generator=generator)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x = train_features.index_select(0, idx).to(device)
            y = train_targets.index_select(0, idx).to(device)
            w = weights.index_select(0, idx).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
            loss = (loss * w).sum() / w.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * float(w.sum().cpu())
            total_weight += float(w.sum().cpu())
        if (epoch + 1) % max(1, epochs // 4) == 0:
            print(
                f"  {architecture} seed={seed} epoch={epoch+1}/{epochs} "
                f"loss={total_loss / max(1.0, total_weight):.5f}",
                flush=True,
            )
    return model.cpu()


def predict_logits(model: SlotModel, features: torch.Tensor, device: str) -> torch.Tensor:
    model = model.to(device)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(features), 4096):
            stop = min(len(features), start + 4096)
            chunks.append(model(features[start:stop].float().to(device)).cpu())
    model.cpu()
    return torch.cat(chunks, dim=0)


def choose_by_grid(
    slot_logits: torch.Tensor,
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    error_gate: torch.Tensor,
    *,
    name: str,
    base_prediction: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    best_slot = slot_logits.argmax(dim=1)
    best_delta = slot_logits.gather(1, best_slot[:, None]).squeeze(1) - slot_logits[:, 0]
    candidate_prediction = top_indices.gather(1, best_slot[:, None]).squeeze(1)

    changed_mask = best_slot.ne(0)
    if bool(changed_mask.any()):
        delta_values = best_delta[changed_mask]
        delta_thresholds = torch.quantile(
            delta_values.float(),
            torch.tensor([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
        ).unique().tolist()
    else:
        delta_thresholds = [0.0]
    gate_thresholds = torch.quantile(
        error_gate.float(),
        torch.tensor([0.00, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]),
    ).unique().tolist()
    gate_thresholds = [0.0] + [float(x) for x in gate_thresholds]

    best = None
    best_prediction = base_prediction.clone()
    for gate_threshold in gate_thresholds:
        for delta_threshold in delta_thresholds:
            use = changed_mask & error_gate.float().ge(float(gate_threshold)) & best_delta.ge(
                float(delta_threshold)
            )
            prediction = base_prediction.clone()
            prediction[use] = candidate_prediction[use]
            dev_stats = paired_stats(base_prediction, prediction, labels, dev)
            record = {
                "name": name,
                "gate_threshold": float(gate_threshold),
                "delta_threshold": float(delta_threshold),
                "dev_net": dev_stats["net"],
                "dev_changed": dev_stats["changed"],
                "dev_wins": dev_stats["wins"],
                "dev_losses": dev_stats["losses"],
            }
            # Dev net is the selection metric. Prefer fewer changes on ties.
            key = (record["dev_net"], -record["dev_changed"])
            if best is None or key > best[0]:
                best = (key, record, prediction)
                best_prediction = prediction

    assert best is not None
    record = dict(best[1])
    record["all"] = paired_stats(base_prediction, best_prediction, labels, torch.ones_like(dev))
    record["dev"] = paired_stats(base_prediction, best_prediction, labels, dev)
    record["sealed"] = paired_stats(base_prediction, best_prediction, labels, sealed)
    return best_prediction, record


def raw_metric_sweeps(
    val_features: torch.Tensor,
    feature_names: list[str],
    top_indices: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    error_gate: torch.Tensor,
    base_prediction: torch.Tensor,
) -> list[dict]:
    records = []
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    candidate_metric_names = [
        name
        for name in feature_names
        if name.endswith("_support_max")
        or name.endswith("_support_top2mean")
        or name.endswith("_support_top3mean")
        or name.endswith("_proto")
    ]
    for metric_name in candidate_metric_names:
        metric = val_features[:, :, name_to_index[metric_name]]
        slot_logits = metric
        prediction, record = choose_by_grid(
            slot_logits,
            top_indices,
            labels,
            dev,
            sealed,
            error_gate,
            name=f"raw_{metric_name}",
            base_prediction=base_prediction,
        )
        records.append(record)
    records.sort(key=lambda item: (item["dev"]["net"], item["sealed"]["net"]), reverse=True)
    return records


def combine_predictions(
    base_prediction: torch.Tensor,
    labels: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> list[dict]:
    records = []
    keys = list(predictions.keys())
    for first in keys:
        for second in keys:
            if first == second:
                continue
            pred = base_prediction.clone()
            first_changed = predictions[first].ne(base_prediction)
            second_changed = predictions[second].ne(base_prediction)
            pred[first_changed] = predictions[first][first_changed]
            fill = ~first_changed & second_changed
            pred[fill] = predictions[second][fill]
            records.append(
                {
                    "name": f"{first}_then_{second}",
                    "all": paired_stats(base_prediction, pred, labels, torch.ones_like(dev)),
                    "dev": paired_stats(base_prediction, pred, labels, dev),
                    "sealed": paired_stats(base_prediction, pred, labels, sealed),
                }
            )
            agree = first_changed & second_changed & predictions[first].eq(predictions[second])
            pred_agree = base_prediction.clone()
            pred_agree[agree] = predictions[first][agree]
            records.append(
                {
                    "name": f"{first}_agree_{second}",
                    "all": paired_stats(base_prediction, pred_agree, labels, torch.ones_like(dev)),
                    "dev": paired_stats(base_prediction, pred_agree, labels, dev),
                    "sealed": paired_stats(base_prediction, pred_agree, labels, sealed),
                }
            )
    records.sort(key=lambda item: (item["dev"]["net"], item["sealed"]["net"]), reverse=True)
    return records


def build_or_load_features(args, train, val, train_oof, bank, gate, out_dir: Path):
    cache_path = out_dir / "exemplar_slot_features.pt"
    if cache_path.exists() and not args.recompute_features:
        print(f"Loading cached slot features: {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    classes = list(train["classes"])
    num_classes = len(classes)
    if list(train["image_ids"]) != list(train_oof["image_ids"]):
        raise RuntimeError("train features and OOF topk image order differ")
    if list(val["image_ids"]) != list(bank["image_ids"]):
        raise RuntimeError("val features and candidate bank image order differ")
    if classes != list(train_oof["classes"]) or classes != list(val["classes"]):
        raise RuntimeError("class order differs")

    train_class_ids = train["class_ids"].long()
    class_counts = torch.bincount(train_class_ids, minlength=num_classes).long()
    members, support_position = build_members(train_class_ids, num_classes)

    component_dims = train.get("component_dims", [])
    train_views = normalize_views(train["features"], component_dims)
    val_views = normalize_views(val["features"], component_dims)
    prototypes = build_prototypes(train_views, train_class_ids, num_classes)

    train_top_indices = train_oof["top_indices"].long()
    val_top_indices = bank["top_indices"].long()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing exemplar slot features on device={device}", flush=True)

    train_support_stats = {}
    val_support_stats = {}
    for view_name in ["full", "bio", "dino"]:
        train_support_stats[view_name] = compute_support_stats_for_view(
            train_views[view_name],
            train_views[view_name],
            train_top_indices,
            members,
            query_labels=train_class_ids,
            support_position=support_position,
            device=device,
            label=f"train/{view_name}",
        )
        val_support_stats[view_name] = compute_support_stats_for_view(
            val_views[view_name],
            train_views[view_name],
            val_top_indices,
            members,
            query_labels=None,
            support_position=None,
            device=device,
            label=f"val/{view_name}",
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("Computing prototype stats", flush=True)
    train_proto_stats = compute_prototype_stats(train_views, prototypes, train_top_indices, device)
    val_proto_stats = compute_prototype_stats(val_views, prototypes, val_top_indices, device)

    train_features, feature_names = assemble_slot_features(
        top_scores=train_oof["top_scores"].float(),
        top_indices=train_top_indices,
        classes=classes,
        class_counts=class_counts,
        support_stats=train_support_stats,
        prototype_stats=train_proto_stats,
        error_gate=gate.get("train_error_gate"),
    )
    val_features, val_feature_names = assemble_slot_features(
        top_scores=bank["top_values"].float(),
        top_indices=val_top_indices,
        classes=classes,
        class_counts=class_counts,
        support_stats=val_support_stats,
        prototype_stats=val_proto_stats,
        error_gate=gate.get("val_error_gate"),
    )
    if feature_names != val_feature_names:
        raise RuntimeError("train/val feature names differ")

    output = {
        "train_slot_features": train_features,
        "val_slot_features": val_features,
        "feature_names": feature_names,
        "train_top_indices": train_top_indices,
        "val_top_indices": val_top_indices,
        "train_labels": train_class_ids,
        "val_labels": bank["labels"].long(),
        "image_ids": list(bank["image_ids"]),
        "classes": classes,
    }
    torch.save(output, cache_path)
    print(f"Saved slot feature cache: {cache_path}", flush=True)
    return output


def write_report(path: Path, results: dict):
    lines = [
        "# 2026-08-07 Exemplar Slot Reranker",
        "",
        "## Result",
        "",
        f"- Best trained selector: `{results['best_model_name']}`.",
        (
            f"- Best trained net: all `{results['best_model']['all']['net']:+d}`, "
            f"dev `{results['best_model']['dev']['net']:+d}`, "
            f"sealed `{results['best_model']['sealed']['net']:+d}`, "
            f"changed `{results['best_model']['all']['changed']}`."
        ),
        (
            f"- Prior error-gated crossfit reference: all "
            f"`{results['crossfit_gate']['all']['net']:+d}`, dev "
            f"`{results['crossfit_gate']['dev']['net']:+d}`, sealed "
            f"`{results['crossfit_gate']['sealed']['net']:+d}`."
        ),
        "",
        "No `test_seen` inference or submission was produced.",
        "",
        "## What was tested",
        "",
        "This round used train-set exemplar information as a new candidate source:",
        "",
        "- for each strong-base top-5 candidate class, compute max/top-2/top-3/mean similarity to train exemplars;",
        "- compute the same for full 6144-D, BioCLIP component, and DINO component;",
        "- add class-prototype similarities, class frequency, genus consistency, base top-5 score shape, and the reusable error gate;",
        "- train top-5 slot selectors on strict train OOF predictions, then select only thresholds on validation-dev and evaluate sealed.",
        "",
        "## Model results",
        "",
        "| Model | Dev | Sealed | All | Changed | Wins | Losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results["model_records"]:
        lines.append(
            f"| {item['name']} | {item['dev']['net']:+d} | {item['sealed']['net']:+d} | "
            f"{item['all']['net']:+d} | {item['all']['changed']} | "
            f"{item['all']['wins']} | {item['all']['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Best raw exemplar metrics",
            "",
            "| Raw rule | Dev | Sealed | All | Changed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in results["raw_records"][:10]:
        lines.append(
            f"| {item['name']} | {item['dev']['net']:+d} | {item['sealed']['net']:+d} | "
            f"{item['all']['net']:+d} | {item['all']['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Combination diagnostics",
            "",
            "| Rule | Dev | Sealed | All | Changed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in results["combination_records"][:12]:
        lines.append(
            f"| {item['name']} | {item['dev']['net']:+d} | {item['sealed']['net']:+d} | "
            f"{item['all']['net']:+d} | {item['all']['changed']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            results["decision"],
            "",
            "## Artifacts",
            "",
            f"- Results JSON: `{results['results_json']}`",
            f"- Feature cache: `{results['feature_cache']}`",
            f"- Model predictions: `{results['prediction_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="runs/local_20260803_strong_oof_rebuild/joint_frozen_adapted_exact/train.pt")
    parser.add_argument("--val-features", default="runs/local_20260803_strong_oof_rebuild/joint_frozen_adapted_exact/val.pt")
    parser.add_argument("--train-oof", default="runs/local_20260803_strong_oof_rebuild/oof_joint_mlp_5fold_seed2031/oof_topk.pt")
    parser.add_argument("--base-logits", default="runs/local_20260803_strong_oof_rebuild/joint_reconstruction_exact_verification/reconstructed_val_logits.pt")
    parser.add_argument("--candidate-bank", default="runs/local_20260807_seen_candidate_bank_fusion/candidate_bank_scores.pt")
    parser.add_argument("--gate-outputs", default="runs/local_20260807_error_quality_gate_scout/gate_outputs.pt")
    parser.add_argument("--out-dir", default="runs/local_20260807_exemplar_slot_reranker")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--architectures", nargs="+", default=["linear", "mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2031, 2032])
    parser.add_argument("--error-weights", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--preserve-weights", nargs="+", type=float, default=[1.0, 2.0])
    parser.add_argument("--miss-weight", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = torch.load(args.train_features, map_location="cpu", weights_only=False)
    val = torch.load(args.val_features, map_location="cpu", weights_only=False)
    train_oof = torch.load(args.train_oof, map_location="cpu", weights_only=False)
    bank = torch.load(args.candidate_bank, map_location="cpu", weights_only=False)
    gate = torch.load(args.gate_outputs, map_location="cpu", weights_only=False)

    features = build_or_load_features(args, train, val, train_oof, bank, gate, out_dir)
    train_slot_features = features["train_slot_features"]
    val_slot_features = features["val_slot_features"]
    feature_names = features["feature_names"]

    train_top_indices = features["train_top_indices"].long()
    val_top_indices = features["val_top_indices"].long()
    train_labels = features["train_labels"].long()
    val_labels = features["val_labels"].long()
    dev = bank["dev"].bool()
    sealed = bank["sealed"].bool()
    val_error_gate = gate["val_error_gate"].float()
    base_prediction = val_top_indices[:, 0].long()
    train_targets, train_in_topk = topk_target(train_top_indices, train_labels)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_records = raw_metric_sweeps(
        val_slot_features,
        feature_names,
        val_top_indices,
        val_labels,
        dev,
        sealed,
        val_error_gate,
        base_prediction,
    )
    print("Best raw exemplar rules:", flush=True)
    for item in raw_records[:5]:
        print(
            item["name"],
            item["dev"]["net"],
            item["sealed"]["net"],
            item["all"]["net"],
            item["all"]["changed"],
            flush=True,
        )

    model_records = []
    model_predictions = {}
    for architecture in args.architectures:
        for seed in args.seeds:
            for error_weight in args.error_weights:
                for preserve_weight in args.preserve_weights:
                    run_name = (
                        f"{architecture}_seed{seed}_ew{error_weight:g}_pw{preserve_weight:g}"
                    )
                    print(f"Training {run_name}", flush=True)
                    model = train_slot_model(
                        train_slot_features,
                        train_targets,
                        train_in_topk,
                        architecture=architecture,
                        seed=seed,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        hidden_dim=args.hidden_dim,
                        dropout=args.dropout,
                        error_weight=error_weight,
                        preserve_weight=preserve_weight,
                        miss_weight=args.miss_weight,
                        device=device,
                    )
                    logits = predict_logits(model, val_slot_features, device)
                    prediction, record = choose_by_grid(
                        logits,
                        val_top_indices,
                        val_labels,
                        dev,
                        sealed,
                        val_error_gate,
                        name=run_name,
                        base_prediction=base_prediction,
                    )
                    model_records.append(record)
                    model_predictions[run_name] = prediction
                    print(
                        f"{run_name}: dev {record['dev']['net']:+d}, "
                        f"sealed {record['sealed']['net']:+d}, "
                        f"all {record['all']['net']:+d}, changed {record['all']['changed']}",
                        flush=True,
                    )

    model_records.sort(
        key=lambda item: (item["dev"]["net"], item["sealed"]["net"], item["all"]["net"]),
        reverse=True,
    )
    best_model_name = model_records[0]["name"] if model_records else "none"
    predictions_for_combo = {}
    if best_model_name in model_predictions:
        predictions_for_combo["exemplar_selector"] = model_predictions[best_model_name]
    predictions_for_combo["crossfit_gate"] = gate["crossfit_gated_prediction"].long()
    predictions_for_combo["consensus_gate"] = gate["consensus_gated_prediction"].long()
    combination_records = combine_predictions(
        base_prediction, val_labels, dev, sealed, predictions_for_combo
    )

    crossfit_gate_stats = {
        "all": paired_stats(
            base_prediction,
            gate["crossfit_gated_prediction"].long(),
            val_labels,
            torch.ones_like(dev),
        ),
        "dev": paired_stats(
            base_prediction, gate["crossfit_gated_prediction"].long(), val_labels, dev
        ),
        "sealed": paired_stats(
            base_prediction, gate["crossfit_gated_prediction"].long(), val_labels, sealed
        ),
    }

    best_model = model_records[0] if model_records else {
        "all": paired_stats(base_prediction, base_prediction, val_labels, torch.ones_like(dev)),
        "dev": paired_stats(base_prediction, base_prediction, val_labels, dev),
        "sealed": paired_stats(base_prediction, base_prediction, val_labels, sealed),
    }
    best_combo = combination_records[0] if combination_records else None
    if best_model["all"]["net"] >= 120 and best_model["sealed"]["net"] > 0:
        decision = "Continue: the train-exemplar selector passed the local continuation gate."
    elif best_combo is not None and best_combo["all"]["net"] >= 120 and best_combo["sealed"]["net"] > 0:
        decision = "Continue only after validating the combination more strictly; the standalone selector did not pass by itself."
    else:
        decision = (
            "Stop this branch for test inference. Train-exemplar top-5 reranking did not "
            "reach the +120 continuation gate; treat any combination as diagnostic unless "
            "it shows a large sealed-positive gain."
        )

    prediction_path = out_dir / "model_predictions.pt"
    torch.save(
        {
            "model_predictions": model_predictions,
            "best_model_name": best_model_name,
            "crossfit_gate": gate["crossfit_gated_prediction"].long(),
            "consensus_gate": gate["consensus_gated_prediction"].long(),
            "base_prediction": base_prediction,
            "labels": val_labels,
            "dev": dev,
            "sealed": sealed,
        },
        prediction_path,
    )

    results = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "crossfit_gate": crossfit_gate_stats,
        "model_records": model_records,
        "raw_records": raw_records,
        "combination_records": combination_records,
        "decision": decision,
        "feature_names": feature_names,
        "results_json": str(out_dir / "results.json"),
        "feature_cache": str(out_dir / "exemplar_slot_features.pt"),
        "prediction_path": str(prediction_path),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(out_dir / "EXPERIMENT_REPORT.md", results)
    print(json.dumps({
        "best_model_name": best_model_name,
        "best_model_all": best_model["all"]["net"],
        "best_model_dev": best_model["dev"]["net"],
        "best_model_sealed": best_model["sealed"]["net"],
        "best_combo": best_combo,
        "decision": decision,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

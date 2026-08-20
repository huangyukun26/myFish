from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CURRENT_BASE_CORRECT = {
    "all": 9823,
    "dev": 6662,
    "sealed": 3161,
}


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (image_id, cls) in enumerate(zip(image_ids, y.tolist())):
        groups[int(cls)].append((stable_hash(image_id), idx))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for j, (_digest, idx) in enumerate(rows):
            if j % 5 in {0, 1, 2}:
                dev[idx] = True
    return dev, ~dev


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalized_features(payload: dict[str, Any]) -> torch.Tensor:
    return F.normalize(payload["features"].float(), dim=1)


def load_aligned_views(
    hflip_path: Path,
    letterbox_path: Path,
    dino_path: Path,
    *,
    canonical_classes: list[str] | None = None,
) -> dict[str, Any]:
    hflip = load_payload(hflip_path)
    letterbox = load_payload(letterbox_path)
    dino = load_payload(dino_path)
    image_ids = list(hflip["image_ids"])
    labels = list(hflip["labels"])
    class_ids = hflip["class_ids"].long()
    for payload, path in ((letterbox, letterbox_path), (dino, dino_path)):
        if list(payload["image_ids"]) != image_ids:
            raise RuntimeError(f"image_id mismatch: {path}")
        if list(payload["labels"]) != labels:
            raise RuntimeError(f"label mismatch: {path}")
        if not torch.equal(payload["class_ids"].long(), class_ids):
            raise RuntimeError(f"class_id mismatch: {path}")
    classes = list(hflip["classes"]) if canonical_classes is None else canonical_classes
    if canonical_classes is None and any(not name for name in classes):
        raise RuntimeError("Training cache has blank class metadata")
    hflip_features = normalized_features(hflip)
    letterbox_features = normalized_features(letterbox)
    dino_features = normalized_features(dino)
    fused = F.normalize(
        torch.cat([hflip_features, letterbox_features, dino_features], dim=1),
        dim=1,
    )
    return {
        "hflip": hflip_features,
        "letterbox": letterbox_features,
        "dino": dino_features,
        "fused": fused,
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": class_ids,
        "classes": classes,
        "component_dims": [
            hflip_features.shape[1],
            letterbox_features.shape[1],
            dino_features.shape[1],
        ],
        "sources": [str(hflip_path), str(letterbox_path), str(dino_path)],
    }


def build_partner_indices(y: torch.Tensor, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, cls in enumerate(y.tolist()):
        groups[int(cls)].append(idx)
    partner = torch.full((len(y),), -1, dtype=torch.long)
    for rows in groups.values():
        if len(rows) < 2:
            continue
        rng.shuffle(rows)
        for position, idx in enumerate(rows):
            partner[idx] = rows[(position + 1) % len(rows)]
    return partner


def class_prototypes(features: torch.Tensor, y: torch.Tensor, class_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros((class_count, features.shape[1]), dtype=torch.float32)
    sums.index_add_(0, y, features.float())
    counts = torch.bincount(y, minlength=class_count).float()
    prototypes = F.normalize(sums / counts.clamp_min(1)[:, None], dim=1)
    return prototypes, counts.long()


def nearest_class_neighbors(
    prototypes: torch.Tensor,
    topk: int,
    chunk_size: int,
) -> torch.Tensor:
    neighbors = []
    for start in range(0, len(prototypes), chunk_size):
        end = min(start + chunk_size, len(prototypes))
        similarities = prototypes[start:end] @ prototypes.T
        row_indices = torch.arange(start, end)
        similarities[torch.arange(end - start), row_indices] = -float("inf")
        neighbors.append(similarities.topk(topk, dim=1).indices.cpu())
    return torch.cat(neighbors, dim=0)


class ContrastiveMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        class_count: int,
        projection_dim: int,
        dropout: float,
        text_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, class_count)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_adapter = (
            nn.Linear(hidden_dim, text_dim, bias=False)
            if text_dim is not None
            else None
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(hidden), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(features))

    def align_text(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.text_adapter is None:
            raise RuntimeError("Text adapter is unavailable for this arm")
        return F.normalize(self.text_adapter(hidden), dim=1)


def make_single_view_inputs(
    hflip: torch.Tensor,
    letterbox: torch.Tensor,
    dino_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = hflip.shape[0]
    h_dim = hflip.shape[1]
    l_dim = letterbox.shape[1]
    zeros_h = torch.zeros((batch, h_dim), dtype=hflip.dtype, device=hflip.device)
    zeros_l = torch.zeros((batch, l_dim), dtype=letterbox.dtype, device=letterbox.device)
    zeros_d = torch.zeros((batch, dino_dim), dtype=hflip.dtype, device=hflip.device)
    view_h = torch.cat([hflip, zeros_l, zeros_d], dim=1)
    view_l = torch.cat([zeros_h, letterbox, zeros_d], dim=1)
    return view_h, view_l


def masked_instance_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = left @ right.T / temperature
    same_class = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    logits = logits.masked_fill(same_class & ~diagonal, -1e4)
    target = torch.arange(len(labels), device=labels.device)
    return 0.5 * (
        F.cross_entropy(logits, target)
        + F.cross_entropy(logits.T, target)
    )


def multi_positive_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    left_labels: torch.Tensor,
    right_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = left @ right.T / temperature
    positive = left_labels[:, None].eq(right_labels[None, :])
    log_denominator = torch.logsumexp(logits, dim=1)
    log_numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    forward = (log_denominator - log_numerator).mean()
    reverse_logits = logits.T
    reverse_positive = positive.T
    reverse_denominator = torch.logsumexp(reverse_logits, dim=1)
    reverse_numerator = torch.logsumexp(
        reverse_logits.masked_fill(~reverse_positive, -1e4),
        dim=1,
    )
    reverse = (reverse_denominator - reverse_numerator).mean()
    return 0.5 * (forward + reverse)


def evaluate_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> dict[str, Any]:
    prediction = logits.argmax(dim=1)
    top5 = logits.topk(5, dim=1).indices
    result: dict[str, Any] = {}
    masks = {
        "all": torch.ones(len(y), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    for name, mask in masks.items():
        rows = int(mask.sum())
        correct = int(prediction[mask].eq(y[mask]).sum())
        top5_correct = int(
            top5[mask].eq(y[mask, None]).any(dim=1).sum()
        )
        result[name] = {
            "rows": rows,
            "correct": correct,
            "top1": correct / max(1, rows),
            "top5": top5_correct / max(1, rows),
            "raw_net_vs_current_base_count": correct - CURRENT_BASE_CORRECT[name],
        }
    return result


def collect_logits(
    model: ContrastiveMLP,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            chunks.append(
                model(features[start : start + batch_size].to(device)).half().cpu()
            )
    return torch.cat(chunks, dim=0)


def paired_stats(
    name: str,
    mask: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, Any]:
    reference_prediction = reference_logits.argmax(dim=1)
    candidate_prediction = candidate_logits.argmax(dim=1)
    reference_correct = reference_prediction.eq(y)
    candidate_correct = candidate_prediction.eq(y)
    changed = mask & reference_prediction.ne(candidate_prediction)
    wins = changed & ~reference_correct & candidate_correct
    losses = changed & reference_correct & ~candidate_correct
    rows = int(mask.sum())
    return {
        "name": name,
        "rows": rows,
        "reference_correct": int((reference_correct & mask).sum()),
        "candidate_correct": int((candidate_correct & mask).sum()),
        "raw_net": int((candidate_correct & mask).sum() - (reference_correct & mask).sum()),
        "changed": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "efficiency": float((wins.sum() - losses.sum()) / max(1, int(changed.sum()))),
        "oracle_complement": int((~reference_correct & candidate_correct & mask).sum()),
    }


def row_standardize(logits: torch.Tensor) -> torch.Tensor:
    centered = logits.float() - logits.float().mean(dim=1, keepdim=True)
    return centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)


def add_candidate_scores(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    alpha: float,
    topk: int,
) -> torch.Tensor:
    if topk <= 0:
        return reference + alpha * candidate
    mixed = reference.clone()
    indices = reference.topk(topk, dim=1).indices
    mixed.scatter_add_(1, indices, alpha * candidate.gather(1, indices))
    return mixed


def scan_reference_ensembles(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    y: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
) -> dict[str, Any]:
    reference = row_standardize(reference_logits)
    candidate = row_standardize(candidate_logits)
    trials = []
    alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    topks = [5, 10, 20, 0]
    all_mask = torch.ones(len(y), dtype=torch.bool)
    for topk in topks:
        for alpha in alphas:
            mixed = add_candidate_scores(reference, candidate, alpha, topk)
            prediction = mixed.argmax(dim=1)
            row = {
                "alpha": alpha,
                "topk": topk,
                "dev_correct": int(prediction[dev].eq(y[dev]).sum()),
                "dev_rows": int(dev.sum()),
            }
            trials.append(row)
    best = max(
        trials,
        key=lambda row: (
            row["dev_correct"],
            -row["alpha"],
            -(row["topk"] if row["topk"] > 0 else 10_000),
        ),
    )
    locked = add_candidate_scores(reference, candidate, best["alpha"], best["topk"])
    return {
        "selection": "alpha and topk selected on deterministic dev only",
        "best_by_dev": best,
        "locked_vs_reference": {
            "dev": paired_stats("dev", dev, reference_logits, locked, y),
            "sealed": paired_stats("sealed", sealed, reference_logits, locked, y),
            "all": paired_stats("all", all_mask, reference_logits, locked, y),
        },
        "trials": trials,
        "locked_logits": locked.half(),
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "ce_loss": row["ce_loss"],
                "view_loss": row["view_loss"],
                "species_loss": row["species_loss"],
                "hard_loss": row["hard_loss"],
                "taxon_loss": row.get("taxon_loss", 0.0),
                "genus_loss": row.get("genus_loss", 0.0),
                "all_top1": row["validation"]["all"]["top1"],
                "dev_top1": row["validation"]["dev"]["top1"],
                "sealed_top1": row["validation"]["sealed"]["top1"],
                "all_net_vs_current_base_count": row["validation"]["all"][
                    "raw_net_vs_current_base_count"
                ],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def arm_weights(arm: str) -> tuple[float, float, float, float, float]:
    return {
        "ce": (0.0, 0.0, 0.0, 0.0, 0.0),
        "view": (0.05, 0.0, 0.0, 0.0, 0.0),
        "species": (0.05, 0.05, 0.0, 0.0, 0.0),
        "hard": (0.05, 0.05, 0.10, 0.0, 0.0),
        "taxtext": (0.0, 0.0, 0.0, 0.02, 0.02),
    }[arm]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hflip", type=Path, required=True)
    parser.add_argument("--train-letterbox", type=Path, required=True)
    parser.add_argument("--train-dino", type=Path, required=True)
    parser.add_argument("--val-hflip", type=Path, required=True)
    parser.add_argument("--val-letterbox", type=Path, required=True)
    parser.add_argument("--val-dino", type=Path, required=True)
    parser.add_argument("--taxon-text", type=Path, default=None)
    parser.add_argument("--genus-text", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=["ce", "view", "species", "hard", "taxtext"],
        required=True,
    )
    parser.add_argument("--reference-logits", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-margin", type=float, default=0.10)
    parser.add_argument("--hard-neighbor-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train = load_aligned_views(
        args.train_hflip,
        args.train_letterbox,
        args.train_dino,
    )
    val = load_aligned_views(
        args.val_hflip,
        args.val_letterbox,
        args.val_dino,
        canonical_classes=train["classes"],
    )
    if train["component_dims"] != val["component_dims"]:
        raise RuntimeError("Train and validation component dimensions differ")
    class_count = len(train["classes"])
    if class_count != 5795:
        raise RuntimeError(f"Expected 5795 seen classes, found {class_count}")
    train_y = train["class_ids"]
    val_y = val["class_ids"]
    counts = torch.bincount(train_y, minlength=class_count)
    if int(counts.min()) < 2:
        raise RuntimeError("Standard training split must retain at least two images per class")

    taxon_text = None
    genus_text = None
    class_to_genus = None
    if args.arm == "taxtext":
        if args.taxon_text is None or args.genus_text is None:
            raise RuntimeError("taxtext arm requires --taxon-text and --genus-text")
        taxon_payload = load_payload(args.taxon_text)
        genus_payload = load_payload(args.genus_text)
        for payload, path in (
            (taxon_payload, args.taxon_text),
            (genus_payload, args.genus_text),
        ):
            if list(payload["classes"]) != train["classes"]:
                raise RuntimeError(f"Text class order mismatch: {path}")
        taxon_text = F.normalize(taxon_payload["features"].float(), dim=1)
        genus_per_class = F.normalize(genus_payload["features"].float(), dim=1)
        genus_names = [name.split(maxsplit=1)[0] for name in train["classes"]]
        unique_genera = sorted(set(genus_names))
        genus_to_id = {name: index for index, name in enumerate(unique_genera)}
        class_to_genus = torch.tensor(
            [genus_to_id[name] for name in genus_names],
            dtype=torch.long,
        )
        genus_sums = torch.zeros((len(unique_genera), genus_per_class.shape[1]))
        genus_sums.index_add_(0, class_to_genus, genus_per_class)
        genus_counts = torch.bincount(
            class_to_genus,
            minlength=len(unique_genera),
        ).float()
        genus_text = F.normalize(
            genus_sums / genus_counts[:, None].clamp_min(1),
            dim=1,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    (args.out_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "arm": args.arm,
                "epochs": args.epochs,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prototypes, class_counts = class_prototypes(train["fused"], train_y, class_count)
    neighbors = nearest_class_neighbors(
        prototypes,
        args.hard_neighbor_count,
        chunk_size=256,
    )
    genera = [name.split(maxsplit=1)[0] for name in train["classes"]]
    same_genus_top1 = sum(
        genera[index] == genera[int(neighbors[index, 0])]
        for index in range(class_count)
    )
    torch.save(
        {
            "prototypes": prototypes.half(),
            "neighbors": neighbors,
            "class_counts": class_counts,
            "classes": train["classes"],
            "same_genus_top1": same_genus_top1,
        },
        args.out_dir / "visual_hard_neighbors.pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveMLP(
        train["fused"].shape[1],
        args.hidden_dim,
        class_count,
        args.projection_dim,
        args.dropout,
        text_dim=(taxon_text.shape[1] if taxon_text is not None else None),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_y)), train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev, sealed = split_dev_sealed(val["image_ids"], val_y)
    if int(dev.sum()) != 7444 or int(sealed.sum()) != 3346:
        raise RuntimeError("Deterministic PE dev/sealed split does not match historical row counts")
    (
        view_weight,
        species_weight,
        hard_weight,
        taxon_weight,
        genus_weight,
    ) = arm_weights(args.arm)
    if taxon_text is not None:
        taxon_text = taxon_text.to(device)
        genus_text = genus_text.to(device)
        class_to_genus = class_to_genus.to(device)
    history = []
    dino_dim = train["component_dims"][2]

    for epoch in range(1, args.epochs + 1):
        model.train()
        partner_indices = build_partner_indices(train_y, args.seed + epoch)
        total_rows = 0
        totals = {
            "loss": 0.0,
            "ce": 0.0,
            "view": 0.0,
            "species": 0.0,
            "hard": 0.0,
            "taxon": 0.0,
            "genus": 0.0,
        }
        for batch_indices, batch_labels in loader:
            batch_indices = batch_indices.long()
            batch_labels = batch_labels.to(device)
            fused = train["fused"][batch_indices].to(device)
            hidden = model.encode(fused)
            logits = model.classifier(hidden)
            ce_loss = F.cross_entropy(logits, batch_labels)
            loss = ce_loss
            view_loss = torch.zeros((), device=device)
            species_loss = torch.zeros((), device=device)
            hard_loss = torch.zeros((), device=device)
            taxon_loss = torch.zeros((), device=device)
            genus_loss = torch.zeros((), device=device)
            projected = None
            positive_projected = None

            if view_weight > 0:
                hflip = train["hflip"][batch_indices].to(device)
                letterbox = train["letterbox"][batch_indices].to(device)
                view_h, view_l = make_single_view_inputs(hflip, letterbox, dino_dim)
                projected_h = model.project(model.encode(view_h))
                projected_l = model.project(model.encode(view_l))
                view_loss = masked_instance_nce(
                    projected_h,
                    projected_l,
                    batch_labels,
                    args.temperature,
                )
                loss = loss + view_weight * view_loss

            if species_weight > 0 or hard_weight > 0:
                projected = model.project(hidden)
                batch_partners = partner_indices[batch_indices]
                if bool(batch_partners.lt(0).any()):
                    raise RuntimeError("A class without a distinct positive partner entered training")
                positive_features = train["fused"][batch_partners].to(device)
                positive_hidden = model.encode(positive_features)
                positive_projected = model.project(positive_hidden)

            if species_weight > 0:
                positive_labels = train_y[batch_partners].to(device)
                species_loss = multi_positive_nce(
                    projected,
                    positive_projected,
                    batch_labels,
                    positive_labels,
                    args.temperature,
                )
                loss = loss + species_weight * species_loss

            if hard_weight > 0:
                rank = (epoch - 1) % args.hard_neighbor_count
                hard_classes = neighbors[batch_labels.cpu(), rank]
                hard_features = prototypes[hard_classes].to(device)
                hard_projected = model.project(model.encode(hard_features))
                positive_similarity = (projected * positive_projected).sum(dim=1)
                negative_similarity = (projected * hard_projected).sum(dim=1)
                hard_loss = (
                    F.softplus(
                        (negative_similarity - positive_similarity + args.hard_margin)
                        / args.temperature
                    )
                    * args.temperature
                ).mean()
                loss = loss + hard_weight * hard_loss

            if taxon_weight > 0 or genus_weight > 0:
                text_aligned = model.align_text(hidden)
                if taxon_weight > 0:
                    taxon_logits = text_aligned @ taxon_text.T / args.temperature
                    taxon_loss = F.cross_entropy(taxon_logits, batch_labels)
                    loss = loss + taxon_weight * taxon_loss
                if genus_weight > 0:
                    genus_logits = text_aligned @ genus_text.T / args.temperature
                    genus_targets = class_to_genus[batch_labels]
                    genus_loss = F.cross_entropy(genus_logits, genus_targets)
                    loss = loss + genus_weight * genus_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_rows = len(batch_indices)
            total_rows += batch_rows
            totals["loss"] += float(loss.detach()) * batch_rows
            totals["ce"] += float(ce_loss.detach()) * batch_rows
            totals["view"] += float(view_loss.detach()) * batch_rows
            totals["species"] += float(species_loss.detach()) * batch_rows
            totals["hard"] += float(hard_loss.detach()) * batch_rows
            totals["taxon"] += float(taxon_loss.detach()) * batch_rows
            totals["genus"] += float(genus_loss.detach()) * batch_rows

        val_logits = collect_logits(
            model,
            val["fused"],
            args.eval_batch_size,
            device,
        )
        validation = evaluate_logits(val_logits.float(), val_y, dev, sealed)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / total_rows,
            "ce_loss": totals["ce"] / total_rows,
            "view_loss": totals["view"] / total_rows,
            "species_loss": totals["species"] / total_rows,
            "hard_loss": totals["hard"] / total_rows,
            "taxon_loss": totals["taxon"] / total_rows,
            "genus_loss": totals["genus"] / total_rows,
            "validation": validation,
        }
        history.append(row)
        with progress_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    final_logits = collect_logits(
        model,
        val["fused"],
        args.eval_batch_size,
        device,
    )
    final_validation = evaluate_logits(final_logits.float(), val_y, dev, sealed)
    val_logits_path = args.out_dir / "val_logits.pt"
    torch.save(
        {
            "logits": final_logits,
            "class_ids": val_y,
            "labels": val["labels"],
            "image_ids": val["image_ids"],
            "classes": train["classes"],
            "arm": args.arm,
            "fixed_epoch": args.epochs,
            "sources": val["sources"],
        },
        val_logits_path,
    )
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "classes": train["classes"],
            "args": vars(args),
            "fixed_epoch": args.epochs,
            "arch": {
                "input_dim": train["fused"].shape[1],
                "hidden_dim": args.hidden_dim,
                "projection_dim": args.projection_dim,
                "class_count": class_count,
                "dropout": args.dropout,
                "text_dim": (
                    int(taxon_text.shape[1])
                    if taxon_text is not None
                    else None
                ),
            },
        },
        args.out_dir / "final_model.pt",
    )
    write_history(args.out_dir / "metrics.csv", history)

    summary: dict[str, Any] = {
        "arm": args.arm,
        "device": str(device),
        "selection_policy": (
            f"fixed epoch {args.epochs}; validation and sealed never select checkpoint"
        ),
        "train_rows": len(train_y),
        "val_rows": len(val_y),
        "classes": class_count,
        "class_count_min": int(class_counts.min()),
        "class_count_median": float(class_counts.float().median()),
        "component_dims": train["component_dims"],
        "loss_weights": {
            "view": view_weight,
            "species": species_weight,
            "hard": hard_weight,
            "taxon": taxon_weight,
            "genus": genus_weight,
        },
        "visual_hard_neighbors": {
            "topk": args.hard_neighbor_count,
            "same_genus_top1": same_genus_top1,
            "same_genus_top1_fraction": same_genus_top1 / class_count,
        },
        "final_validation": final_validation,
        "history": history,
        "val_logits": str(val_logits_path),
    }

    if args.reference_logits is not None:
        reference = load_payload(args.reference_logits)
        if list(reference["image_ids"]) != val["image_ids"]:
            raise RuntimeError("Reference logits image order does not match validation cache")
        if not torch.equal(reference["class_ids"].long(), val_y):
            raise RuntimeError("Reference logits targets do not match validation cache")
        reference_logits = reference["logits"].float()
        all_mask = torch.ones(len(val_y), dtype=torch.bool)
        summary["paired_vs_reference"] = {
            "dev": paired_stats("dev", dev, reference_logits, final_logits.float(), val_y),
            "sealed": paired_stats(
                "sealed",
                sealed,
                reference_logits,
                final_logits.float(),
                val_y,
            ),
            "all": paired_stats(
                "all",
                all_mask,
                reference_logits,
                final_logits.float(),
                val_y,
            ),
        }
        ensemble = scan_reference_ensembles(
            reference_logits,
            final_logits.float(),
            val_y,
            dev,
            sealed,
        )
        locked_logits = ensemble.pop("locked_logits")
        summary["reference_ensemble"] = ensemble
        torch.save(
            {
                "logits": locked_logits,
                "class_ids": val_y,
                "labels": val["labels"],
                "image_ids": val["image_ids"],
                "classes": train["classes"],
                "source_reference": str(args.reference_logits),
                "source_candidate": str(val_logits_path),
                "selection": ensemble["best_by_dev"],
            },
            args.out_dir / "locked_reference_ensemble_logits.pt",
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "arm": args.arm,
                "epochs": args.epochs,
                "seed": args.seed,
                "summary": str(summary_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()

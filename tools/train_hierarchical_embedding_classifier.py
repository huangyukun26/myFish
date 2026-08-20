from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1)


def genus(class_name: str) -> str:
    parts = str(class_name).split()
    return parts[0] if parts else class_name


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


class HierarchicalClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        species_count: int,
        genus_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.species_head = nn.Linear(hidden_dim, species_count)
        self.genus_head = nn.Linear(hidden_dim, genus_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.species_head(hidden), self.genus_head(hidden)


def collect_scores(
    model: nn.Module,
    features: torch.Tensor,
    class_to_genus: torch.Tensor,
    genus_score_weights: list[float],
    batch_size: int,
    device: torch.device,
) -> dict[float, torch.Tensor]:
    chunks = {weight: [] for weight in genus_score_weights}
    model.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            species_logits, genus_logits = model(features[start : start + batch_size].to(device))
            mapped_genus = genus_logits[:, class_to_genus.to(device)]
            for weight in genus_score_weights:
                chunks[weight].append((species_logits + weight * mapped_genus).half().cpu())
    return {weight: torch.cat(rows, dim=0) for weight, rows in chunks.items()}


def metrics(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = scores.topk(min(20, scores.shape[1]), dim=1).indices
    return {
        "top1": float((top[:, 0] == targets).float().mean().item()),
        "top5": float((top[:, :5] == targets[:, None]).any(dim=1).float().mean().item()),
        "top20": float((top == targets[:, None]).any(dim=1).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--genus-loss-weight", type=float, default=0.25)
    parser.add_argument("--genus-score-weights", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_payload = load_payload(args.train_cache)
    val_payload = load_payload(args.val_cache)
    classes = list(train_payload["classes"])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    genera = sorted({genus(name) for name in classes})
    genus_to_idx = {name: idx for idx, name in enumerate(genera)}
    class_to_genus = torch.tensor(
        [genus_to_idx[genus(name)] for name in classes], dtype=torch.long
    )

    x_train = normalize(train_payload["features"])
    y_train = torch.tensor([class_to_idx[name] for name in train_payload["labels"]], dtype=torch.long)
    g_train = class_to_genus[y_train]
    x_val = normalize(val_payload["features"])
    y_val = torch.tensor([class_to_idx[name] for name in val_payload["labels"]], dtype=torch.long)
    genus_score_weights = parse_floats(args.genus_score_weights)
    if 0.0 not in genus_score_weights:
        genus_score_weights.insert(0, 0.0)

    species_counts = torch.bincount(y_train, minlength=len(classes)).float()
    genus_counts = torch.bincount(g_train, minlength=len(genera)).float()
    species_offset = species_counts.clamp_min(1).log()
    genus_offset = genus_counts.clamp_min(1).log()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalClassifier(
        x_train.shape[1],
        args.hidden_dim,
        len(classes),
        len(genera),
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(x_train, y_train, g_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    species_offset = species_offset.to(device)
    genus_offset = genus_offset.to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = None
    best_state = None
    best_row = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        rows = 0
        for features, species_targets, genus_targets in loader:
            features = features.to(device)
            species_targets = species_targets.to(device)
            genus_targets = genus_targets.to(device)
            species_logits, genus_logits = model(features)
            species_loss = F.cross_entropy(species_logits + species_offset, species_targets)
            genus_loss = F.cross_entropy(genus_logits + genus_offset, genus_targets)
            loss = species_loss + args.genus_loss_weight * genus_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * features.shape[0]
            rows += features.shape[0]

        score_sets = collect_scores(
            model,
            x_val,
            class_to_genus,
            genus_score_weights,
            args.eval_batch_size,
            device,
        )
        evaluations = {str(weight): metrics(scores, y_val) for weight, scores in score_sets.items()}
        epoch_best_weight, epoch_best_metrics = max(
            ((weight, evaluations[str(weight)]) for weight in genus_score_weights),
            key=lambda item: (item[1]["top1"], item[1]["top5"], item[1]["top20"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, rows),
            "best_genus_score_weight": epoch_best_weight,
            **epoch_best_metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (row["top1"], row["top5"], row["top20"])
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}

    assert best_state is not None and best_row is not None
    model.load_state_dict(best_state)
    model.to(device)
    best_weight = float(best_row["best_genus_score_weight"])
    val_logits = collect_scores(
        model,
        x_val,
        class_to_genus,
        [best_weight],
        args.eval_batch_size,
        device,
    )[best_weight]
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "genera": genera,
            "class_to_genus": class_to_genus,
            "args": vars(args),
            "best": best_row,
        },
        args.out_dir / "best_model.pt",
    )
    torch.save(
        {
            "logits": val_logits,
            "class_ids": y_val,
            "labels": list(val_payload["labels"]),
            "image_ids": list(val_payload["image_ids"]),
            "classes": classes,
            "full_class_counts": val_payload.get("full_class_counts"),
            "genus_score_weight": best_weight,
            "source_cache": str(args.val_cache),
        },
        args.out_dir / "val_logits.pt",
    )
    summary = {
        "train_cache": str(args.train_cache),
        "val_cache": str(args.val_cache),
        "train_rows": x_train.shape[0],
        "val_rows": x_val.shape[0],
        "classes": len(classes),
        "genera": len(genera),
        "best": best_row,
        "val_logits": str(args.out_dir / "val_logits.pt"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

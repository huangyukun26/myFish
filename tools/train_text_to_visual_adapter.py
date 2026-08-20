from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def load_class_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def load_exclusions(path: Path | None, exclude_genera: bool) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    classes = set(load_class_list(path))
    genera = {genus(name) for name in classes} if exclude_genera else set()
    return classes, genera


def build_visual_prototypes(
    image_payload: dict,
    text_classes: list[str],
    *,
    exclude_classes: set[str],
    exclude_genera: set[str],
    min_count: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    class_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    features = normalize(image_payload["features"])
    sums = torch.zeros((len(text_classes), features.shape[1]), dtype=torch.float32)
    counts = torch.zeros(len(text_classes), dtype=torch.long)
    for row_idx, label in enumerate(image_payload["labels"]):
        if not label or label in exclude_classes or genus(label) in exclude_genera:
            continue
        class_idx = class_to_idx.get(label)
        if class_idx is None:
            continue
        sums[class_idx] += features[row_idx]
        counts[class_idx] += 1
    keep = counts >= min_count
    prototypes = torch.zeros_like(sums)
    prototypes[keep] = normalize(sums[keep])
    classes = [name for idx, name in enumerate(text_classes) if bool(keep[idx])]
    return classes, prototypes[keep], counts[keep]


class ResidualTextAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, residual_scale: float, dropout: float) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        if hidden_dim <= 0:
            self.net = nn.Linear(dim, dim, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.residual_scale * self.net(x), dim=1)


def train_model(
    text_x: torch.Tensor,
    target_proto: torch.Tensor,
    *,
    hidden_dim: int,
    residual_scale: float,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    contrastive_weight: float,
    device: torch.device,
) -> tuple[ResidualTextAdapter, list[float]]:
    model = ResidualTextAdapter(text_x.shape[1], hidden_dim, residual_scale, dropout).to(device)
    text_x = text_x.to(device)
    target_proto = target_proto.to(device)
    labels = torch.arange(text_x.shape[0], dtype=torch.long)
    loader = DataLoader(TensorDataset(text_x, target_proto, labels), batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, idxb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            idxb = idxb.to(device)
            pred = model(xb)
            cosine_loss = 1.0 - (pred * yb).sum(dim=1).mean()
            if contrastive_weight:
                logits = 20.0 * (pred @ target_proto.T)
                contrastive_loss = F.cross_entropy(logits, idxb)
                loss = cosine_loss + contrastive_weight * contrastive_loss
            else:
                loss = cosine_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        losses.append(total / max(1, count))
    model.eval()
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-classes", type=Path, default=None)
    parser.add_argument("--exclude-genera", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--blend-original", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    text_classes = list(text_payload["classes"])
    text_features = normalize(text_payload["features"])
    exclude_classes, exclude_genera = load_exclusions(args.exclude_classes, args.exclude_genera)
    train_classes, target_proto, counts = build_visual_prototypes(
        image_payload,
        text_classes,
        exclude_classes=exclude_classes,
        exclude_genera=exclude_genera,
        min_count=args.min_count,
    )
    text_idx = torch.tensor([text_classes.index(name) for name in train_classes], dtype=torch.long)
    train_text = text_features[text_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = train_model(
        train_text,
        target_proto,
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        contrastive_weight=args.contrastive_weight,
        device=device,
    )
    with torch.inference_mode():
        adapted = model(text_features.to(device)).cpu()
    if args.blend_original > 0:
        adapted = normalize(args.blend_original * text_features + (1.0 - args.blend_original) * adapted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classes": text_classes,
            "features": adapted,
            "source_text_features": str(args.text_features),
            "source_image_features": str(args.image_features),
            "train_classes": train_classes,
            "train_counts": counts,
            "losses": losses,
            "config": vars(args),
        },
        args.out,
    )
    summary = {
        "out": str(args.out),
        "source_text_features": str(args.text_features),
        "source_image_features": str(args.image_features),
        "train_classes": len(train_classes),
        "min_count": args.min_count,
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_genera": args.exclude_genera,
        "hidden_dim": args.hidden_dim,
        "residual_scale": args.residual_scale,
        "blend_original": args.blend_original,
        "epochs": args.epochs,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

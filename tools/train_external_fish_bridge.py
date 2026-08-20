from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class IdentityAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(3.5))
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1.0, 100.0)

    def identity_loss(self) -> torch.Tensor:
        eye = torch.eye(self.proj.weight.shape[0], device=self.proj.weight.device)
        return (self.proj.weight - eye).pow(2).mean()


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def load_classes(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [name for name, _idx in sorted(payload.items(), key=lambda item: int(item[1]))]
    return list(payload)


def load_text(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["classes"]), F.normalize(payload["features"].float(), dim=1)


def load_external(paths: list[Path], text_classes: list[str], selected_classes: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    class_to_idx = {name: idx for idx, name in enumerate(selected_classes)}
    features: list[torch.Tensor] = []
    labels: list[int] = []
    raw_labels: list[str] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row_idx, label in enumerate(payload.get("labels", [])):
            if label not in class_to_idx:
                continue
            features.append(F.normalize(payload["features"][row_idx].float(), dim=0))
            labels.append(class_to_idx[label])
            raw_labels.append(label)
    if not features:
        raise RuntimeError("No external rows matched the selected class scope")
    return torch.stack(features), torch.tensor(labels, dtype=torch.long), raw_labels


def load_anchor(path: Path | None, selected_classes: list[str], count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if path is None or count <= 0:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(selected_classes)}
    keep = [idx for idx, label in enumerate(payload.get("labels", [])) if label in class_to_idx]
    if not keep:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    if len(keep) > count:
        order = torch.randperm(len(keep), generator=generator)[:count].tolist()
        keep = [keep[idx] for idx in order]
    x = F.normalize(payload["features"][torch.tensor(keep)].float(), dim=1)
    y = torch.tensor([class_to_idx[payload["labels"][idx]] for idx in keep], dtype=torch.long)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--external-caches", required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--class-scope", choices=["seen", "external", "all"], required=True)
    parser.add_argument("--class-json", type=Path, default=None)
    parser.add_argument("--anchor-cache", type=Path, default=None)
    parser.add_argument("--anchor-count", type=int, default=0)
    parser.add_argument("--external-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--identity-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=2041)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    text_classes, text_features = load_text(args.text_features)
    external_paths = parse_paths(args.external_caches)
    external_labels = []
    for path in external_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        external_labels.extend(str(label) for label in payload.get("labels", []))
    if args.class_scope == "seen":
        selected_classes = load_classes(args.class_json, text_classes)
    elif args.class_scope == "all":
        selected_classes = text_classes
    elif args.class_scope == "external":
        selected_classes = sorted(set(external_labels), key=lambda name: text_classes.index(name) if name in text_classes else 10**9)
    else:
        raise AssertionError(args.class_scope)
    text_index = {name: idx for idx, name in enumerate(text_classes)}
    missing = [name for name in selected_classes if name not in text_index]
    if missing:
        raise RuntimeError(f"{len(missing)} selected classes missing from text features; first={missing[:5]}")
    selected_text = text_features[torch.tensor([text_index[name] for name in selected_classes], dtype=torch.long)]

    external_x, external_y, raw_labels = load_external(external_paths, text_classes, selected_classes)
    anchor_x, anchor_y = load_anchor(args.anchor_cache, selected_classes, args.anchor_count, args.seed)
    if anchor_x.numel():
        x = torch.cat([external_x, anchor_x], dim=0)
        y = torch.cat([external_y, anchor_y], dim=0)
        weights = torch.cat(
            [
                torch.full((external_x.shape[0],), args.external_weight),
                torch.full((anchor_x.shape[0],), args.anchor_weight),
            ]
        )
    else:
        x, y = external_x, external_y
        weights = torch.full((external_x.shape[0],), args.external_weight)

    # Older remote checkpoints carry PosixPath objects in args; make that
    # metadata portable before loading the trusted local file on Windows.
    posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(args.base_adapter, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = posix_path
    model = IdentityAdapter(int(x.shape[1]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    selected_text = selected_text.to(device)
    loader = DataLoader(TensorDataset(x, y, weights), batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        correct = 0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = model.scale() * (model(xb) @ selected_text.T)
            per_row = F.cross_entropy(logits, yb, reduction="none")
            loss = (per_row * wb).mean() + args.identity_weight * model.identity_loss()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * xb.shape[0]
            total_rows += xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().item())
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_rows),
            "training_top1": correct / max(1, total_rows),
            "scale": float(model.scale().detach().cpu()),
            "identity_loss": float(model.identity_loss().detach().cpu()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, args.out_dir / "adapter.pt")
    summary = {
        "base_adapter": str(args.base_adapter),
        "external_caches": [str(path) for path in external_paths],
        "text_features": str(args.text_features),
        "class_scope": args.class_scope,
        "selected_classes": len(selected_classes),
        "external_rows": int(external_x.shape[0]),
        "anchor_rows": int(anchor_x.shape[0]),
        "unique_external_labels": len(set(raw_labels)),
        "device": str(device),
        "history": history,
        "out_adapter": str(args.out_dir / "adapter.pt"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

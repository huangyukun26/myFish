from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def load_text(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["classes"]), normalize(payload["features"])


def class_order(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


class IdentityAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(3.5))
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize(self.proj(x))

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1.0, 100.0)

    def identity_loss(self) -> torch.Tensor:
        eye = torch.eye(self.proj.weight.shape[0], device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        return (self.proj.weight - eye).pow(2).mean()


def make_train_tensors(feature_path: Path, text_classes: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    keep = []
    labels = []
    for row_idx, label in enumerate(payload["labels"]):
        idx = class_to_idx.get(label)
        if idx is None:
            continue
        keep.append(row_idx)
        labels.append(idx)
    if not keep:
        raise RuntimeError("No train rows matched text classes")
    indices = torch.tensor(keep, dtype=torch.long)
    return normalize(payload["features"][indices]), torch.tensor(labels, dtype=torch.long), list(payload["labels"])


def adapt_feature_file(model: IdentityAdapter, in_path: Path, out_path: Path, batch_size: int, device: torch.device) -> dict:
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    features = normalize(payload["features"])
    outs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            outs.append(model(features[start:end].to(device)).cpu())
    out = dict(payload)
    out["features"] = torch.cat(outs, dim=0)
    out["adapter_source"] = str(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {"input": str(in_path), "output": str(out_path), "rows": int(features.shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--seen-classes", type=Path, default=Path("work/full_manifests/seen_class_to_idx.json"))
    parser.add_argument("--eval-features", default="")
    parser.add_argument("--eval-outs", default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-weight", type=float, default=20.0)
    args = parser.parse_args()

    all_text_classes, all_text_features = load_text(args.text_features)
    seen_classes = class_order(args.seen_classes, all_text_classes)
    all_class_to_idx = {name: idx for idx, name in enumerate(all_text_classes)}
    seen_indices = torch.tensor([all_class_to_idx[name] for name in seen_classes if name in all_class_to_idx], dtype=torch.long)
    if len(seen_indices) != len(seen_classes):
        raise RuntimeError("Some seen classes are missing from text features")
    seen_text = all_text_features[seen_indices]
    train_x, train_y_all, _labels = make_train_tensors(args.train_features, all_text_classes)
    all_to_seen = {int(all_idx): seen_idx for seen_idx, all_idx in enumerate(seen_indices.tolist())}
    keep = [idx for idx, label in enumerate(train_y_all.tolist()) if int(label) in all_to_seen]
    y = torch.tensor([all_to_seen[int(train_y_all[idx])] for idx in keep], dtype=torch.long)
    x = train_x[torch.tensor(keep, dtype=torch.long)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IdentityAdapter(x.shape[1]).to(device)
    seen_text = seen_text.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, num_workers=0)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        correct = 0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z = model(xb)
            logits = model.scale() * (z @ seen_text.T)
            ce = F.cross_entropy(logits, yb)
            loss = ce + args.identity_weight * model.identity_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            count += xb.shape[0]
        row = {
            "epoch": epoch,
            "loss": total / max(1, count),
            "train_top1": correct / max(1, count),
            "scale": float(model.scale().detach().cpu()),
            "identity_loss": float(model.identity_loss().detach().cpu()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, args.out_dir / "adapter.pt")
    adapted = []
    eval_paths = [Path(part.strip()) for part in args.eval_features.split(",") if part.strip()]
    eval_outs = [Path(part.strip()) for part in args.eval_outs.split(",") if part.strip()]
    if eval_paths:
        if len(eval_paths) != len(eval_outs):
            raise ValueError("--eval-features and --eval-outs must have the same count")
        for in_path, out_path in zip(eval_paths, eval_outs):
            adapted.append(adapt_feature_file(model, in_path, out_path, args.batch_size, device))
    summary = {
        "train_features": str(args.train_features),
        "text_features": str(args.text_features),
        "seen_classes": str(args.seen_classes),
        "train_rows": int(x.shape[0]),
        "history": history,
        "adapted": adapted,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

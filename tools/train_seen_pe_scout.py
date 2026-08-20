from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids: list[str], y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (image_id, cls) in enumerate(zip(image_ids, y.tolist())):
        groups[int(cls)].append((stable_hash(image_id), idx))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for j, (_h, idx) in enumerate(rows):
            if j % 5 in {0, 1, 2}:
                dev[idx] = True
    return dev, ~dev


def align_features(paths: list[Path]) -> dict[str, Any]:
    payloads = [load_payload(path) for path in paths]
    base = payloads[0]
    image_ids = list(base["image_ids"])
    labels = list(base["labels"])
    class_ids = base["class_ids"].long()
    classes = list(base["classes"])
    feats = [normalize(base["features"])]
    for payload, path in zip(payloads[1:], paths[1:]):
        if list(payload["image_ids"]) != image_ids:
            raise RuntimeError(f"image_id mismatch for {path}")
        if "labels" in payload and list(payload["labels"]) != labels:
            raise RuntimeError(f"label mismatch for {path}")
        if "class_ids" in payload and not torch.equal(payload["class_ids"].long(), class_ids):
            raise RuntimeError(f"class_id mismatch for {path}")
        # Some historical concatenated feature caches carry stale or partially
        # blank `classes` metadata on val, while labels/class_ids are aligned.
        # Feature concatenation only needs row alignment; output class order is
        # taken from the train cache and checked against current base logits.
        feats.append(normalize(payload["features"]))
    return {
        "features": torch.cat(feats, dim=1),
        "image_ids": image_ids,
        "labels": labels,
        "class_ids": class_ids,
        "classes": classes,
        "sources": [str(path) for path in paths],
    }


def label_ids(labels: list[str], classes: list[str]) -> torch.Tensor:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    return torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def top1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    rows = int(mask.sum().item())
    return float(logits.argmax(dim=1)[mask].eq(y[mask]).float().mean().item()) if rows else 0.0


def collect_logits(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(model(x[start : start + batch_size].to(device)).half().cpu())
    return torch.cat(chunks, dim=0)


def paired_stats(name: str, mask: torch.Tensor, base_logits: torch.Tensor, cand_logits: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    bpred = base_logits.argmax(dim=1)
    cpred = cand_logits.argmax(dim=1)
    bc = bpred.eq(y)
    cc = cpred.eq(y)
    changed = mask & cpred.ne(bpred)
    wins = changed & (~bc) & cc
    losses = changed & bc & (~cc)
    rows = int(mask.sum().item())
    return {
        "name": name,
        "rows": rows,
        "base_acc": float((bc & mask).sum().item() / max(1, rows)),
        "cand_acc": float((cc & mask).sum().item() / max(1, rows)),
        "raw_net": int((cc & mask).sum().item() - (bc & mask).sum().item()),
        "changed": int(changed.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net_changed": int(wins.sum().item() - losses.sum().item()),
        "efficiency": float((wins.sum().item() - losses.sum().item()) / max(1, int(changed.sum().item()))),
        "oracle_complement": int(((~bc) & cc & mask).sum().item()),
        "oracle_complement_pp": float(((~bc) & cc & mask).sum().item() / max(1, rows)),
    }


def scan_alpha(
    *,
    base_logits: torch.Tensor,
    cand_logits: torch.Tensor,
    y: torch.Tensor,
    dev: torch.Tensor,
    sealed: torch.Tensor,
    alphas: list[float],
) -> dict[str, Any]:
    trials = []
    base = base_logits.float()
    cand = cand_logits.float()
    # Per-row centering plus std matching keeps alpha values comparable across heads.
    cand = cand - cand.mean(dim=1, keepdim=True)
    cand = cand / cand.std(dim=1, keepdim=True).clamp_min(1e-6)
    base_std = base.std(dim=1, keepdim=True).median().item()
    cand = cand * float(base_std)
    for alpha in alphas:
        logits = base + float(alpha) * cand
        trials.append(
            {
                "alpha": float(alpha),
                "dev": paired_stats("dev", dev, base, logits, y),
                "sealed": paired_stats("sealed", sealed, base, logits, y),
                "all": paired_stats("all", torch.ones_like(dev), base, logits, y),
            }
        )
    trials.sort(key=lambda t: (t["dev"]["raw_net"], t["dev"]["efficiency"], t["dev"]["changed"]), reverse=True)
    return {"best_by_dev": trials[0], "trials": trials}


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--val-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-val-logits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2037)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train = align_features(args.train_cache)
    val = align_features(args.val_cache)
    base_val = load_payload(args.base_val_logits)
    if list(base_val["image_ids"]) != val["image_ids"]:
        raise RuntimeError("base val logits not aligned to val cache")
    classes = train["classes"]
    if list(base_val["classes"]) != classes:
        raise RuntimeError("class order mismatch between base logits and train cache")
    x_train = train["features"].float()
    y_train = label_ids(train["labels"], classes)
    x_val = val["features"].float()
    y_val = val["class_ids"].long()
    dev, sealed = split_dev_sealed(val["image_ids"], y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(x_train.shape[1], args.hidden_dim, len(classes), args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_key = None
    best_state = None
    best_row = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        val_logits = collect_logits(model, x_val, args.eval_batch_size, device)
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, count),
            "dev_top1": top1(val_logits, y_val, dev),
            "sealed_top1": top1(val_logits, y_val, sealed),
            "all_top1": top1(val_logits, y_val, torch.ones_like(dev)),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        key = (row["dev_top1"], row["all_top1"], -row["train_loss"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_row = row
    assert best_state is not None and best_row is not None
    model.load_state_dict(best_state)
    model.to(device)
    val_logits = collect_logits(model, x_val, args.eval_batch_size, device)
    base_logits = base_val["logits"].float()
    raw = {
        "dev": paired_stats("dev", dev, base_logits, val_logits, y_val),
        "sealed": paired_stats("sealed", sealed, base_logits, val_logits, y_val),
        "all": paired_stats("all", torch.ones_like(dev), base_logits, val_logits, y_val),
    }
    alpha_scan = scan_alpha(
        base_logits=base_logits,
        cand_logits=val_logits,
        y=y_val,
        dev=dev,
        sealed=sealed,
        alphas=[0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0],
    )
    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "args": vars(args),
            "best": best_row,
            "arch": {"type": "mlp", "in_dim": x_train.shape[1], "hidden_dim": args.hidden_dim, "dropout": args.dropout},
        },
        args.out_dir / "best_model.pt",
    )
    torch.save(
        {
            "logits": val_logits,
            "image_ids": val["image_ids"],
            "labels": val["labels"],
            "class_ids": y_val,
            "classes": classes,
            "sources": val["sources"],
        },
        args.out_dir / "val_logits.pt",
    )
    write_metrics(args.out_dir / "metrics.csv", rows)
    summary = {
        "train_sources": train["sources"],
        "val_sources": val["sources"],
        "base_val_logits": str(args.base_val_logits),
        "best_epoch_by_dev": best_row,
        "dev_rows": int(dev.sum().item()),
        "sealed_rows": int(sealed.sum().item()),
        "raw_vs_current078": raw,
        "alpha_scan_vs_current078": alpha_scan,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["raw_vs_current078"], indent=2, ensure_ascii=False), flush=True)
    print(json.dumps(summary["alpha_scan_vs_current078"]["best_by_dev"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

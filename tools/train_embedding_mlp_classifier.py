from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def label_to_ids(labels: list[str], label_to_idx: dict[str, int]) -> torch.Tensor:
    ids = []
    missing = []
    for label in labels:
        if label not in label_to_idx:
            missing.append(label)
            ids.append(-1)
        else:
            ids.append(label_to_idx[label])
    if missing:
        raise RuntimeError(f"{len(missing)} labels missing from train classes; first={missing[:5]}")
    return torch.tensor(ids, dtype=torch.long)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    ranks = []
    losses = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            losses.append(float(F.cross_entropy(logits, yb).detach().cpu()) * xb.shape[0])
            order = logits.argsort(dim=1, descending=True)
            rank = (order == yb[:, None]).nonzero(as_tuple=False)[:, 1] + 1
            ranks.append(rank.cpu())
    ranks_t = torch.cat(ranks)
    return {
        "loss": sum(losses) / max(1, x.shape[0]),
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
    }


def write_topk(
    path: Path,
    model: nn.Module,
    x: torch.Tensor,
    image_ids: list[str],
    classes: list[str],
    batch_size: int,
    device: torch.device,
    topk: int,
) -> None:
    model.eval()
    with path.open("w", encoding="utf-8") as fp:
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                xb = x[start : start + batch_size].to(device)
                logits = model(xb)
                scores, indices = logits.topk(min(topk, logits.shape[1]), dim=1)
                for local_idx, image_id in enumerate(image_ids[start : start + batch_size]):
                    row = {
                        "image_id": image_id,
                        "predictions": [classes[int(idx)] for idx in indices[local_idx].cpu().tolist()],
                        "scores": [float(v) for v in scores[local_idx].cpu().tolist()],
                    }
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_predictions_csv(path: Path, topk_jsonl: Path) -> None:
    with topk_jsonl.open("r", encoding="utf-8") as in_fp, path.open("w", encoding="utf-8", newline="") as out_fp:
        writer = csv.DictWriter(out_fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for line in in_fp:
            row = json.loads(line)
            writer.writerow({"image_id": row["image_id"], "prediction": row["predictions"][0]})


def collect_logits(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(model(x[start : start + batch_size].to(device)).half().cpu())
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=["none", "balanced_sqrt", "balanced"], default="balanced_sqrt")
    parser.add_argument("--loss-mode", choices=["cross_entropy", "balanced_softmax"], default="cross_entropy")
    parser.add_argument("--sampler", choices=["shuffle", "class_balanced"], default="shuffle")
    parser.add_argument("--save-val-logits", action="store_true")
    parser.add_argument("--train-on-val-too", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_payload = load_payload(args.train_cache)
    val_payload = load_payload(args.val_cache)
    classes = list(train_payload["classes"])
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    x_train = normalize(train_payload["features"])
    y_train = label_to_ids(list(train_payload["labels"]), label_to_idx)
    x_val = normalize(val_payload["features"])
    y_val = label_to_ids(list(val_payload["labels"]), label_to_idx)
    if args.train_on_val_too:
        x_train = torch.cat([x_train, x_val], dim=0)
        y_train = torch.cat([y_train, y_val], dim=0)

    if args.loss_mode == "balanced_softmax" and args.class_weight != "none":
        raise ValueError("balanced_softmax requires --class-weight none")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(x_train.shape[1], args.hidden_dim, len(classes), args.dropout).to(device)
    class_weight = None
    counts = torch.bincount(y_train, minlength=len(classes)).float()
    if args.class_weight != "none":
        weights = x_train.shape[0] / counts.clamp_min(1.0)
        if args.class_weight == "balanced_sqrt":
            weights = weights.sqrt()
        class_weight = (weights / weights.mean()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(x_train, y_train)
    sampler = None
    if args.sampler == "class_balanced":
        sample_weights = counts.clamp_min(1.0).reciprocal()[y_train]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=x_train.shape[0],
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    balanced_softmax_offset = counts.clamp_min(1.0).log().to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss_logits = logits
            if args.loss_mode == "balanced_softmax":
                loss_logits = logits + balanced_softmax_offset
            loss = F.cross_entropy(
                loss_logits,
                yb,
                weight=class_weight,
                label_smoothing=args.label_smoothing,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        val = evaluate(model, x_val, y_val, args.eval_batch_size, device)
        row = {"epoch": epoch, "train_loss": total / max(1, count), **val}
        metrics_rows.append(row)
        print(json.dumps(row), flush=True)
        key = (val["top1"], val["top5"], -val["loss"])
        if best is None or key > best[0]:
            best = (key, row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None and best is not None
    model.load_state_dict(best_state)
    model.to(device)
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    torch.save(
        {
            "state_dict": best_state,
            "classes": classes,
            "args": vars(args),
            "best": best[1],
            "arch": {"type": "mlp", "in_dim": x_train.shape[1], "hidden_dim": args.hidden_dim, "dropout": args.dropout},
        },
        args.out_dir / "best_model.pt",
    )
    val_logits_path = None
    if args.save_val_logits:
        val_logits_path = args.out_dir / "val_logits.pt"
        torch.save(
            {
                "logits": collect_logits(model, x_val, args.eval_batch_size, device),
                "class_ids": y_val,
                "labels": list(val_payload["labels"]),
                "image_ids": list(val_payload["image_ids"]),
                "classes": classes,
                "full_class_counts": val_payload.get("full_class_counts"),
                "source_cache": str(args.val_cache),
            },
            val_logits_path,
        )
    if args.test_cache is not None:
        test_payload = load_payload(args.test_cache)
        x_test = normalize(test_payload["features"])
        topk_jsonl = args.out_dir / "test_topk.jsonl"
        write_topk(topk_jsonl, model, x_test, list(test_payload["image_ids"]), classes, args.eval_batch_size, device, args.topk)
        write_predictions_csv(args.out_dir / "test_predictions.csv", topk_jsonl)
    summary = {
        "train_cache": str(args.train_cache),
        "val_cache": str(args.val_cache),
        "test_cache": str(args.test_cache) if args.test_cache else None,
        "train_rows": int(x_train.shape[0]),
        "val_rows": int(x_val.shape[0]),
        "classes": len(classes),
        "best": best[1],
        "metrics_csv": str(args.out_dir / "metrics.csv"),
        "val_logits": str(val_logits_path) if val_logits_path else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


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


def read_topk_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_text_payload(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["classes"]), normalize_features(payload["features"])


def load_candidates(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def load_candidate_features(path: Path, candidates: list[str]) -> torch.Tensor:
    classes, features = load_text_payload(path)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidates missing from {path}; first={missing[:5]}")
    indices = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return features[indices]


def filter_train(
    payload: dict[str, Any],
    *,
    text_class_to_idx: dict[str, int],
    exclude_classes: set[str],
    exclude_genera: set[str],
) -> tuple[torch.Tensor, list[str]]:
    keep = []
    labels = []
    for idx, label in enumerate(payload["labels"]):
        if label not in text_class_to_idx:
            continue
        if label in exclude_classes:
            continue
        if exclude_genera and genus(label) in exclude_genera:
            continue
        keep.append(idx)
        labels.append(label)
    if not keep:
        raise RuntimeError("No train rows remain after filtering")
    indices = torch.tensor(keep, dtype=torch.long)
    return normalize_features(payload["features"][indices]), labels


def reorder_query(payload: dict[str, Any], image_ids: list[str]) -> torch.Tensor:
    by_image = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in by_image]
    if missing:
        raise RuntimeError(f"{len(missing)} query image ids missing; first={missing[:5]}")
    indices = torch.tensor([by_image[image_id] for image_id in image_ids], dtype=torch.long)
    return normalize_features(payload["features"][indices])


def reorder_labels(payload: dict[str, Any], image_ids: list[str]) -> list[str]:
    labels = list(payload.get("labels", [""] * len(payload["image_ids"])))
    by_image = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    return [labels[by_image[image_id]] for image_id in image_ids]


class VisualTextAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1.0, 100.0)


def train_adapter(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_text_features: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[VisualTextAdapter, list[float]]:
    model = VisualTextAdapter(x_train.shape[1], train_text_features.shape[1]).to(device)
    train_text_features = train_text_features.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    losses = []
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z = model(xb)
            logits = model.scale() * (z @ train_text_features.T)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        losses.append(total / max(1, count))
    model.eval()
    return model, losses


def trigger_mask(
    *,
    rows: list[dict[str, Any]],
    mode: str,
    margin_threshold: float,
    genus_frac_threshold: float,
) -> torch.Tensor:
    values = []
    for row in rows:
        scores = [float(v) for v in row["scores"]]
        margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        genera = [genus(pred) for pred in row["predictions"]]
        counts = Counter(genera)
        top1_frac = counts.get(genera[0], 0) / max(1, len(genera)) if genera else 0.0
        low_margin = margin <= margin_threshold
        clustered = top1_frac >= genus_frac_threshold
        if mode == "all":
            values.append(True)
        elif mode == "low_margin":
            values.append(low_margin)
        elif mode == "clustered":
            values.append(clustered)
        elif mode == "low_margin_or_clustered":
            values.append(low_margin or clustered)
        elif mode == "low_margin_and_clustered":
            values.append(low_margin and clustered)
        else:
            raise ValueError(f"Unknown trigger mode: {mode}")
    return torch.tensor(values, dtype=torch.bool)


def candidate_text_matrix(
    rows: list[dict[str, Any]],
    text_features: torch.Tensor,
    text_class_to_idx: dict[str, int],
) -> torch.Tensor:
    k = len(rows[0]["predictions"])
    out = torch.zeros((len(rows), k, text_features.shape[1]), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, class_name in enumerate(row["predictions"]):
            idx = text_class_to_idx.get(class_name)
            if idx is not None:
                out[row_idx, col_idx] = text_features[idx]
    return out


def rerank_indices(
    *,
    rows: list[dict[str, Any]],
    adapter_scores: torch.Tensor,
    weight: float,
    trigger: torch.Tensor,
) -> torch.Tensor:
    base_scores = torch.tensor([[float(v) for v in row["scores"]] for row in rows], dtype=torch.float32)
    adapter_scores = row_zscore(adapter_scores)
    final_scores = base_scores + weight * adapter_scores
    final_scores = torch.where(trigger[:, None], final_scores, base_scores)
    indices = final_scores.argsort(dim=1, descending=True)
    setattr(indices, "triggered", int(trigger.sum().item()))
    return indices


def metrics(indices: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = []
    wins = 0
    losses = 0
    changed = 0
    for row_idx, row in enumerate(rows):
        label = row.get("label", "")
        if not label:
            continue
        preds = row["predictions"]
        base_pred = preds[0]
        final_pred = preds[int(indices[row_idx, 0].item())]
        base_ok = base_pred == label
        final_ok = final_pred == label
        changed += int(base_pred != final_pred)
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
        try:
            rank = [preds[int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            rank = len(preds) + 1
        ranks.append(rank)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    return {
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "triggered": int(getattr(indices, "triggered", 0)),
    }


def global_metrics(logits: torch.Tensor, labels: list[str], candidates: list[str], topk: int = 20) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    ranks = []
    for row_idx, label in enumerate(labels):
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        true_score = logits[row_idx, true_idx]
        rank = int((logits[row_idx] > true_score).sum().item()) + 1
        ranks.append(rank)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    return {
        "rank_known": len(ranks),
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= topk).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "median_rank": float(ranks_t.float().median().item()),
        "mean_rank": float(ranks_t.float().mean().item()),
    }


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def write_predictions(path: Path, rows: list[dict[str, Any]], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "label", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            preds = row["predictions"]
            pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "prediction": pred,
                    "label": row.get("label", ""),
                    "base_prediction": preds[0],
                    "changed": pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--exclude-classes", type=Path, default=None)
    parser.add_argument("--exclude-genera", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rerank-weight-grid", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--margin-grid", default="0.002,0.005,0.01,0.02,1.0")
    parser.add_argument("--genus-frac-grid", default="0.25,0.30,0.40,1.01")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered",
    )
    parser.add_argument("--global-base-image-features", type=Path, default=None)
    parser.add_argument("--global-base-text-features", type=Path, default=None)
    parser.add_argument("--global-extra-text-features", type=Path, default=None)
    parser.add_argument("--global-extra-weight", type=float, default=0.0)
    parser.add_argument("--global-candidate-classes", type=Path, default=None)
    parser.add_argument("--global-score-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--global-adapter-weight-grid", default="0,0.002,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    text_classes, text_features = load_text_payload(args.text_features)
    text_class_to_idx = {name: idx for idx, name in enumerate(text_classes)}
    exclude_classes, exclude_genera = load_exclusions(args.exclude_classes, args.exclude_genera)
    x_train, train_labels = filter_train(
        train_payload,
        text_class_to_idx=text_class_to_idx,
        exclude_classes=exclude_classes,
        exclude_genera=exclude_genera,
    )
    train_classes = sorted(set(train_labels))
    train_class_to_local = {name: idx for idx, name in enumerate(train_classes)}
    y_train = torch.tensor([train_class_to_local[label] for label in train_labels], dtype=torch.long)
    text_indices = torch.tensor([text_class_to_idx[name] for name in train_classes], dtype=torch.long)
    train_text_features = text_features[text_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = train_adapter(
        x_train,
        y_train,
        train_text_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    rows = read_topk_jsonl(args.topk_jsonl)
    image_ids = [row["image_id"] for row in rows]
    x_query = reorder_query(query_payload, image_ids)
    with torch.inference_mode():
        z_query = model(x_query.to(device)).cpu()
    cand_text = candidate_text_matrix(rows, text_features, text_class_to_idx)
    adapter_scores = (cand_text * z_query[:, None, :]).sum(dim=2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": image_ids,
            "predictions": [row["predictions"] for row in rows],
            "base_scores": [[float(value) for value in row["scores"]] for row in rows],
            "labels": [row.get("label", "") for row in rows],
            "adapter_scores": adapter_scores,
            "text_features": str(args.text_features),
            "train_features": str(args.train_features),
            "query_features": str(args.query_features),
        },
        args.out_dir / "adapter_topk_scores.pt",
    )

    sweep_rows = []
    best = None
    best_indices = None
    for weight in parse_grid(args.rerank_weight_grid):
        for margin_threshold in parse_grid(args.margin_grid):
            for genus_frac_threshold in parse_grid(args.genus_frac_grid):
                for mode in [part.strip() for part in args.trigger_modes.split(",") if part.strip()]:
                    trigger = trigger_mask(
                        rows=rows,
                        mode=mode,
                        margin_threshold=margin_threshold,
                        genus_frac_threshold=genus_frac_threshold,
                    )
                    indices = rerank_indices(rows=rows, adapter_scores=adapter_scores, weight=weight, trigger=trigger)
                    row = {
                        "weight": weight,
                        "margin_threshold": margin_threshold,
                        "genus_frac_threshold": genus_frac_threshold,
                        "trigger_mode": mode,
                        **metrics(indices, rows),
                    }
                    sweep_rows.append(row)
                    key = (row.get("top1", 0), row.get("net_wins", 0), -row.get("losses", 0), -abs(weight))
                    if best is None or key > best[0]:
                        best = (key, row)
                        best_indices = indices.clone()

    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    if best_indices is not None:
        write_predictions(args.out_dir / "best_predictions.csv", rows, best_indices)
    global_summary = None
    if args.global_base_image_features is not None:
        if args.global_base_text_features is None:
            raise ValueError("--global-base-text-features is required with --global-base-image-features")
        base_payload = torch.load(args.global_base_image_features, map_location="cpu", weights_only=False)
        base_image_features = reorder_query(base_payload, image_ids)
        global_labels = reorder_labels(base_payload, image_ids)
        base_text_classes, _ = load_text_payload(args.global_base_text_features)
        candidates = load_candidates(args.global_candidate_classes, base_text_classes)
        base_text_features = load_candidate_features(args.global_base_text_features, candidates)
        base_logits = base_image_features @ base_text_features.T
        if args.global_score_normalization == "zscore":
            base_logits = row_zscore(base_logits)
        if args.global_extra_text_features is not None:
            extra_text_features = load_candidate_features(args.global_extra_text_features, candidates)
            extra_logits = base_image_features @ extra_text_features.T
            if args.global_score_normalization == "zscore":
                extra_logits = row_zscore(extra_logits)
            base_logits = base_logits * (1.0 - args.global_extra_weight) + extra_logits * args.global_extra_weight
        adapter_candidate_text = load_candidate_features(args.text_features, candidates)
        adapter_logits = z_query @ adapter_candidate_text.T
        adapter_logits = row_zscore(adapter_logits)
        global_rows = []
        global_best = None
        for adapter_weight in parse_grid(args.global_adapter_weight_grid):
            combined = base_logits + adapter_weight * adapter_logits
            row = {
                "adapter_weight": adapter_weight,
                **global_metrics(combined, global_labels, candidates, topk=20),
            }
            global_rows.append(row)
            key = (row.get("top1", 0), row.get("top5", 0), row.get("mrr", 0), -abs(adapter_weight))
            if global_best is None or key > global_best[0]:
                global_best = (key, row)
        global_sweep_path = args.out_dir / "global_sweep.csv"
        with global_sweep_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(global_rows[0].keys()))
            writer.writeheader()
            writer.writerows(global_rows)
        global_summary = {
            "base_image_features": str(args.global_base_image_features),
            "base_text_features": str(args.global_base_text_features),
            "extra_text_features": str(args.global_extra_text_features) if args.global_extra_text_features else None,
            "extra_weight": args.global_extra_weight,
            "candidate_classes": str(args.global_candidate_classes) if args.global_candidate_classes else None,
            "candidate_count": len(candidates),
            "score_normalization": args.global_score_normalization,
            "best": global_best[1] if global_best else None,
            "sweep_csv": str(global_sweep_path),
        }
    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "topk_jsonl": str(args.topk_jsonl),
        "text_features": str(args.text_features),
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_genera": args.exclude_genera,
        "train_rows": len(train_labels),
        "train_classes": len(train_classes),
        "query_rows": len(rows),
        "losses": losses,
        "logit_scale": float(model.scale().detach().cpu().item()),
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
        "global": global_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

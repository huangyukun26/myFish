from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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


def load_species_text_payload(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["classes"]), normalize_features(payload["features"])


def build_genus_text_features(species_classes: list[str], species_features: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, class_name in enumerate(species_classes):
        groups[genus(class_name)].append(idx)
    genus_classes = sorted(key for key in groups if key)
    features = []
    for name in genus_classes:
        idx = torch.tensor(groups[name], dtype=torch.long)
        features.append(species_features[idx].mean(dim=0))
    return genus_classes, F.normalize(torch.stack(features).float(), dim=1)


def filter_train(
    payload: dict[str, Any],
    *,
    genus_to_idx: dict[str, int],
    exclude_classes: set[str],
    exclude_genera: set[str],
) -> tuple[torch.Tensor, list[str]]:
    keep = []
    labels = []
    for idx, label in enumerate(payload["labels"]):
        g = genus(label)
        if g not in genus_to_idx:
            continue
        if label in exclude_classes:
            continue
        if exclude_genera and g in exclude_genera:
            continue
        keep.append(idx)
        labels.append(g)
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


def candidate_genus_text_matrix(
    rows: list[dict[str, Any]],
    genus_features: torch.Tensor,
    genus_to_idx: dict[str, int],
) -> torch.Tensor:
    k = len(rows[0]["predictions"])
    out = torch.zeros((len(rows), k, genus_features.shape[1]), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, class_name in enumerate(row["predictions"]):
            idx = genus_to_idx.get(genus(class_name))
            if idx is not None:
                out[row_idx, col_idx] = genus_features[idx]
    return out


def trigger_mask(
    *,
    rows: list[dict[str, Any]],
    mode: str,
    margin_threshold: float,
    genus_frac_threshold: float,
    min_distinct_genera: int,
) -> torch.Tensor:
    values = []
    for row in rows:
        scores = [float(v) for v in row["scores"]]
        margin = scores[0] - scores[1] if len(scores) >= 2 else 0.0
        genera = [genus(pred) for pred in row["predictions"]]
        counts = Counter(genera)
        top1_frac = counts.get(genera[0], 0) / max(1, len(genera)) if genera else 0.0
        distinct_genera = len(counts)
        low_margin = margin <= margin_threshold
        clustered = top1_frac >= genus_frac_threshold
        enough_genera = distinct_genera >= min_distinct_genera
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
        elif mode == "low_margin_and_multigenus":
            values.append(low_margin and enough_genera)
        elif mode == "low_margin_or_multigenus":
            values.append(low_margin or enough_genera)
        else:
            raise ValueError(f"Unknown trigger mode: {mode}")
    return torch.tensor(values, dtype=torch.bool)


def rerank_indices(
    *,
    rows: list[dict[str, Any]],
    adapter_scores: torch.Tensor,
    weight: float,
    trigger: torch.Tensor,
    score_norm: str,
) -> torch.Tensor:
    base_scores = torch.tensor([[float(v) for v in row["scores"]] for row in rows], dtype=torch.float32)
    adapter = adapter_scores
    if score_norm == "zscore":
        adapter = row_zscore(adapter)
    elif score_norm == "center":
        adapter = adapter - adapter.mean(dim=1, keepdim=True)
    elif score_norm != "none":
        raise ValueError(f"Unknown score norm: {score_norm}")
    final_scores = base_scores + weight * adapter
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
    parser.add_argument("--species-text-features", type=Path, required=True)
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
    parser.add_argument("--min-distinct-genera-grid", default="2")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered,low_margin_and_multigenus,low_margin_or_multigenus",
    )
    parser.add_argument("--score-norm", choices=["zscore", "center", "none"], default="zscore")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    species_classes, species_text_features = load_species_text_payload(args.species_text_features)
    genus_classes, genus_text_features = build_genus_text_features(species_classes, species_text_features)
    genus_to_idx = {name: idx for idx, name in enumerate(genus_classes)}

    exclude_classes, exclude_genera = load_exclusions(args.exclude_classes, args.exclude_genera)
    x_train, train_genera = filter_train(
        train_payload,
        genus_to_idx=genus_to_idx,
        exclude_classes=exclude_classes,
        exclude_genera=exclude_genera,
    )
    train_classes = sorted(set(train_genera))
    train_class_to_local = {name: idx for idx, name in enumerate(train_classes)}
    y_train = torch.tensor([train_class_to_local[name] for name in train_genera], dtype=torch.long)
    text_indices = torch.tensor([genus_to_idx[name] for name in train_classes], dtype=torch.long)
    train_text_features = genus_text_features[text_indices]

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
    cand_text = candidate_genus_text_matrix(rows, genus_text_features, genus_to_idx)
    adapter_scores = (cand_text * z_query[:, None, :]).sum(dim=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": image_ids,
            "predictions": [row["predictions"] for row in rows],
            "base_scores": [[float(value) for value in row["scores"]] for row in rows],
            "labels": [row.get("label", "") for row in rows],
            "adapter_scores": adapter_scores,
            "species_text_features": str(args.species_text_features),
            "train_features": str(args.train_features),
            "query_features": str(args.query_features),
            "adapter_kind": "genus_text",
            "score_norm": args.score_norm,
        },
        args.out_dir / "adapter_topk_scores.pt",
    )

    sweep_rows = []
    best = None
    best_indices = None
    trigger_modes = [part.strip() for part in args.trigger_modes.split(",") if part.strip()]
    for weight in parse_grid(args.rerank_weight_grid):
        for margin_threshold in parse_grid(args.margin_grid):
            for genus_frac_threshold in parse_grid(args.genus_frac_grid):
                for min_distinct_genera in [int(v) for v in parse_grid(args.min_distinct_genera_grid)]:
                    for mode in trigger_modes:
                        trigger = trigger_mask(
                            rows=rows,
                            mode=mode,
                            margin_threshold=margin_threshold,
                            genus_frac_threshold=genus_frac_threshold,
                            min_distinct_genera=min_distinct_genera,
                        )
                        indices = rerank_indices(
                            rows=rows,
                            adapter_scores=adapter_scores,
                            weight=weight,
                            trigger=trigger,
                            score_norm=args.score_norm,
                        )
                        row = {
                            "weight": weight,
                            "margin_threshold": margin_threshold,
                            "genus_frac_threshold": genus_frac_threshold,
                            "min_distinct_genera": min_distinct_genera,
                            "trigger_mode": mode,
                            "score_norm": args.score_norm,
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
    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "topk_jsonl": str(args.topk_jsonl),
        "species_text_features": str(args.species_text_features),
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_genera": args.exclude_genera,
        "train_rows": len(train_genera),
        "train_genera": len(train_classes),
        "all_text_genera": len(genus_classes),
        "query_rows": len(rows),
        "losses": losses,
        "logit_scale": float(model.scale().detach().cpu().item()),
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

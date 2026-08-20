from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIT_PATTERNS: dict[str, list[str]] = {
    "body_elongated": [r"\belongated\b", r"\blong bod", r"\bslender\b"],
    "body_fusiform": [r"\bfusiform\b", r"\bstreamlined\b"],
    "body_compressed": [r"\bcompressed\b", r"\blaterally compressed\b"],
    "body_deep": [r"\bdeep[- ]bod", r"\bdeep body\b", r"\bshort, deep\b"],
    "body_flattened": [r"\bflattened\b", r"\bdepressed\b", r"\bdorsoventrally\b"],
    "body_eel_like": [r"\beel[- ]like\b", r"\banguilliform\b"],
    "body_oval": [r"\boval\b"],
    "body_robust": [r"\brobust\b", r"\bstout\b"],
    "color_black": [r"\bblack\b", r"\bdark\b"],
    "color_white": [r"\bwhite\b", r"\bwhitish\b"],
    "color_silver": [r"\bsilver\b", r"\bsilvery\b"],
    "color_blue": [r"\bblue\b", r"\bbluish\b"],
    "color_green": [r"\bgreen\b", r"\bgreenish\b"],
    "color_red": [r"\bred\b", r"\breddish\b"],
    "color_orange": [r"\borange\b"],
    "color_yellow": [r"\byellow\b", r"\byellowish\b"],
    "color_brown": [r"\bbrown\b", r"\bbrownish\b"],
    "color_gray": [r"\bgray\b", r"\bgrey\b", r"\bgreyish\b", r"\bgrayish\b"],
    "color_olive": [r"\bolive\b"],
    "pattern_stripes": [r"\bstripe", r"\bstriped\b"],
    "pattern_bars": [r"\bbar\b", r"\bbars\b", r"\bbarred\b", r"\bvertical bars\b"],
    "pattern_bands": [r"\bband\b", r"\bbands\b", r"\bbanded\b"],
    "pattern_spots": [r"\bspot", r"\bspotted\b"],
    "pattern_blotches": [r"\bblotch", r"\bblotched\b"],
    "pattern_mottled": [r"\bmottled\b", r"\bmarbled\b"],
    "pattern_saddles": [r"\bsaddle", r"\bsaddles\b"],
    "pattern_ocellus": [r"\bocellus\b", r"\bocelli\b", r"\beyespot\b"],
    "pattern_lines": [r"\bline\b", r"\blines\b", r"\blined\b", r"\blateral stripe\b"],
    "tail_forked": [r"\bforked caudal\b", r"\bcaudal fin is forked\b", r"\bforked tail\b"],
    "tail_truncate": [r"\btruncate caudal\b", r"\bcaudal fin is truncate\b", r"\btruncate tail\b"],
    "tail_rounded": [r"\brounded caudal\b", r"\bcaudal fin is rounded\b", r"\brounded tail\b"],
    "tail_emarginate": [r"\bemarginate\b"],
    "tail_lunate": [r"\blunate\b"],
    "fin_filament": [r"\bfilament", r"\bfilamentous\b"],
    "fin_pointed": [r"\bpointed fins\b", r"\bpointed fin\b"],
    "head_terminal_mouth": [r"\bterminal mouth\b"],
    "head_subterminal_mouth": [r"\bsub[- ]terminal mouth\b"],
    "head_inferior_mouth": [r"\binferior mouth\b"],
    "head_superior_mouth": [r"\bsuperior mouth\b"],
    "head_large_mouth": [r"\blarge mouth\b", r"\bwide mouth\b"],
    "head_small_mouth": [r"\bsmall mouth\b"],
    "head_long_snout": [r"\blong snout\b", r"\bprominent snout\b", r"\btapering snout\b"],
    "head_short_snout": [r"\bshort snout\b"],
    "head_concave_forehead": [r"\bconcave forehead\b"],
    "head_large_eye": [r"\blarge eye\b", r"\blarge eyes\b"],
    "head_barbels": [r"\bbarbel", r"\bmustache", r"\bmoustache"],
    "head_spine": [r"\bspine\b", r"\bspines\b"],
}


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def genus(name: str) -> str:
    parts = str(name or "").split()
    return parts[0] if parts else ""


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


def clean_description(text: str) -> str:
    text = " ".join((text or "").lower().split())
    return text.replace("*", "")


def build_trait_table(classes: list[str], descriptions: dict[str, str]) -> tuple[torch.Tensor, list[str], dict[str, list[str]]]:
    names = list(TRAIT_PATTERNS)
    rows = []
    hits_by_class: dict[str, list[str]] = {}
    for class_name in classes:
        desc = clean_description(descriptions.get(class_name, ""))
        hits = []
        values = []
        for trait_name in names:
            found = any(re.search(pattern, desc) for pattern in TRAIT_PATTERNS[trait_name])
            values.append(1.0 if found else 0.0)
            if found:
                hits.append(trait_name)
        rows.append(values)
        hits_by_class[class_name] = hits
    return torch.tensor(rows, dtype=torch.float32), names, hits_by_class


def filter_train_payload(payload: dict[str, Any], exclude_classes: set[str], exclude_genera: set[str]) -> tuple[torch.Tensor, list[str]]:
    keep = []
    labels = []
    for idx, label in enumerate(payload["labels"]):
        if label in exclude_classes:
            continue
        if exclude_genera and genus(label) in exclude_genera:
            continue
        keep.append(idx)
        labels.append(label)
    if not keep:
        raise RuntimeError("No train rows remain after exclusions")
    indices = torch.tensor(keep, dtype=torch.long)
    return normalize_features(payload["features"][indices]), labels


def train_head(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[nn.Linear, list[float]]:
    model = nn.Linear(x.shape[1], y.shape[1]).to(device)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1)).clamp(1.0, 20.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    losses = []
    model.train()
    for _epoch in range(epochs):
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        losses.append(total / max(1, count))
    model.eval()
    return model, losses


def read_topk_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reorder_features(payload: dict[str, Any], image_ids: list[str]) -> torch.Tensor:
    by_id = {image_id: idx for idx, image_id in enumerate(payload["image_ids"])}
    missing = [image_id for image_id in image_ids if image_id not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} query image ids missing; first={missing[:5]}")
    indices = torch.tensor([by_id[image_id] for image_id in image_ids], dtype=torch.long)
    return normalize_features(payload["features"][indices])


def class_trait_matrix_for_topk(
    rows: list[dict[str, Any]],
    trait_by_class: dict[str, torch.Tensor],
    trait_dim: int,
) -> torch.Tensor:
    out = torch.zeros((len(rows), len(rows[0]["predictions"]), trait_dim), dtype=torch.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, class_name in enumerate(row["predictions"]):
            trait = trait_by_class.get(class_name)
            if trait is not None:
                out[row_idx, col_idx] = trait
    return out


def topk_metrics(indices: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = []
    base_correct = []
    final_correct = []
    wins = 0
    losses = 0
    changed = 0
    triggered = 0
    for row_idx, row in enumerate(rows):
        label = row.get("label", "")
        if not label:
            continue
        preds = row["predictions"]
        base = preds[0]
        final = preds[int(indices[row_idx, 0].item())]
        base_ok = base == label
        final_ok = final == label
        base_correct.append(base_ok)
        final_correct.append(final_ok)
        if final != base:
            changed += 1
        if (not base_ok) and final_ok:
            wins += 1
        if base_ok and (not final_ok):
            losses += 1
        try:
            final_rank = [preds[int(idx)] for idx in indices[row_idx].tolist()].index(label) + 1
        except ValueError:
            final_rank = len(preds) + 1
        ranks.append(final_rank)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    triggered = int(getattr(indices, "triggered", 0))
    return {
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "triggered": triggered,
    }


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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


def rerank_indices(
    *,
    rows: list[dict[str, Any]],
    trait_probs: torch.Tensor,
    candidate_traits: torch.Tensor,
    trait_prior: torch.Tensor,
    trait_weights: torch.Tensor,
    score_mode: str,
    weight: float,
    trigger: torch.Tensor,
) -> torch.Tensor:
    base_scores = torch.tensor([[float(v) for v in row["scores"]] for row in rows], dtype=torch.float32)
    eps = 1e-6
    if score_mode == "positive":
        pos_counts = candidate_traits.sum(dim=2).clamp_min(1.0)
        trait_scores = (candidate_traits * trait_probs[:, None, :]).sum(dim=2) / pos_counts.sqrt()
    elif score_mode == "idf_positive":
        weighted_traits = candidate_traits * trait_weights[None, None, :]
        denom = weighted_traits.sum(dim=2).clamp_min(1.0).sqrt()
        trait_scores = (weighted_traits * trait_probs[:, None, :]).sum(dim=2) / denom
    elif score_mode == "centered_positive":
        weighted_traits = candidate_traits * trait_weights[None, None, :]
        denom = weighted_traits.sum(dim=2).clamp_min(1.0).sqrt()
        trait_scores = (weighted_traits * (trait_probs - trait_prior)[:, None, :]).sum(dim=2) / denom
    elif score_mode == "signed_centered":
        signed_traits = (candidate_traits * 2.0 - 1.0) * trait_weights[None, None, :]
        trait_scores = (signed_traits * (trait_probs - trait_prior)[:, None, :]).sum(dim=2) / math.sqrt(
            max(1, candidate_traits.shape[2])
        )
    elif score_mode == "agreement":
        log_p = torch.log(trait_probs.clamp(eps, 1.0 - eps))
        log_not_p = torch.log((1.0 - trait_probs).clamp(eps, 1.0 - eps))
        trait_scores = (candidate_traits * log_p[:, None, :] + (1.0 - candidate_traits) * log_not_p[:, None, :]).mean(dim=2)
    else:
        raise ValueError(f"Unknown trait score mode: {score_mode}")
    trait_scores = row_zscore(trait_scores)
    final_scores = base_scores + weight * trait_scores
    final_scores = torch.where(trigger[:, None], final_scores, base_scores)
    indices = final_scores.argsort(dim=1, descending=True)
    setattr(indices, "triggered", int(trigger.sum().item()))
    return indices


def write_best_predictions(path: Path, rows: list[dict[str, Any]], indices: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction", "label", "base_prediction", "changed"])
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            preds = row["predictions"]
            final_pred = preds[int(indices[row_idx, 0].item())]
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "prediction": final_pred,
                    "label": row.get("label", ""),
                    "base_prediction": preds[0],
                    "changed": final_pred != preds[0],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, default=Path("dataset/descriptions.json"))
    parser.add_argument("--exclude-classes", type=Path, default=None)
    parser.add_argument("--exclude-genera", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rerank-weight-grid", default="0,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument(
        "--trait-score-modes",
        default="positive,idf_positive,centered_positive,signed_centered,agreement",
    )
    parser.add_argument("--margin-grid", default="0.002,0.005,0.01,0.02,1.0")
    parser.add_argument("--genus-frac-grid", default="0.25,0.30,0.40,1.01")
    parser.add_argument(
        "--trigger-modes",
        default="all,low_margin,clustered,low_margin_or_clustered,low_margin_and_clustered",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    train_payload = torch.load(args.train_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_features, map_location="cpu", weights_only=False)
    rows = read_topk_jsonl(args.topk_jsonl)
    image_ids = [row["image_id"] for row in rows]

    all_classes = sorted(set(train_payload["labels"]) | {pred for row in rows for pred in row["predictions"]})
    trait_table, trait_names, hits_by_class = build_trait_table(all_classes, descriptions)
    class_to_trait = {name: trait_table[idx] for idx, name in enumerate(all_classes)}
    exclude_classes, exclude_genera = load_exclusions(args.exclude_classes, args.exclude_genera)
    x_train, train_labels = filter_train_payload(train_payload, exclude_classes, exclude_genera)
    y_train = torch.stack([class_to_trait[label] for label in train_labels])
    keep_trait = y_train.sum(dim=0) > 0
    x_train = normalize_features(x_train)
    y_train = y_train[:, keep_trait]
    trait_names = [name for name, keep in zip(trait_names, keep_trait.tolist()) if keep]
    class_to_trait = {name: trait[keep_trait] for name, trait in class_to_trait.items()}
    trait_prior = y_train.mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
    trait_weights = torch.log((1.0 / trait_prior).clamp_min(1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = train_head(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    x_query = reorder_features(query_payload, image_ids)
    with torch.inference_mode():
        trait_probs = torch.sigmoid(model(x_query.to(device))).cpu()
    candidate_traits = class_trait_matrix_for_topk(rows, class_to_trait, len(trait_names))

    sweep_rows = []
    best = None
    best_indices = None
    for score_mode in [part.strip() for part in args.trait_score_modes.split(",") if part.strip()]:
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
                        indices = rerank_indices(
                            rows=rows,
                            trait_probs=trait_probs,
                            candidate_traits=candidate_traits,
                            trait_prior=trait_prior,
                            trait_weights=trait_weights,
                            score_mode=score_mode,
                            weight=weight,
                            trigger=trigger,
                        )
                        metrics = topk_metrics(indices, rows)
                        row = {
                            "score_mode": score_mode,
                            "weight": weight,
                            "margin_threshold": margin_threshold,
                            "genus_frac_threshold": genus_frac_threshold,
                            "trigger_mode": mode,
                            **metrics,
                        }
                        sweep_rows.append(row)
                        key = (row.get("top1", 0), row.get("net_wins", 0), -row.get("losses", 0), -abs(weight))
                        if best is None or key > best[0]:
                            best = (key, row)
                            best_indices = indices.clone()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.out_dir / "sweep.csv"
    with sweep_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    if best_indices is not None:
        write_best_predictions(args.out_dir / "best_predictions.csv", rows, best_indices)
    trait_density = {
        name: float(y_train[:, idx].mean().item())
        for idx, name in enumerate(trait_names)
    }
    summary = {
        "train_features": str(args.train_features),
        "query_features": str(args.query_features),
        "topk_jsonl": str(args.topk_jsonl),
        "exclude_classes": str(args.exclude_classes) if args.exclude_classes else None,
        "exclude_genera": args.exclude_genera,
        "train_rows": len(train_labels),
        "query_rows": len(rows),
        "traits": trait_names,
        "trait_density": trait_density,
        "losses": losses,
        "best": best[1] if best else None,
        "sweep_csv": str(sweep_path),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def row_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)


def load_classes(path: Path | None, fallback: list[str]) -> list[str]:
    if path is None:
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def load_text_features(path: Path, candidates: list[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} missing text classes in {path}; first={missing[:5]}")
    indices = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize(payload["features"][indices])


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def parse_weights(value: str, count: int) -> list[float]:
    if not value.strip():
        return [1.0 / count for _ in range(count)]
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(weights) != count:
        raise ValueError(f"Expected {count} text weights, got {len(weights)}")
    total = sum(weights)
    if total == 0:
        raise ValueError("Text weights must not sum to zero")
    return [w / total for w in weights]


def compute_logits(image_features: torch.Tensor, text_features: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    text = text_features.to(device)
    chunks: list[torch.Tensor] = []
    for start in range(0, image_features.shape[0], batch_size):
        end = min(start + batch_size, image_features.shape[0])
        image = image_features[start:end].to(device)
        chunks.append((image @ text.T).cpu())
    return torch.cat(chunks, dim=0)


def compute_ensemble_logits(
    image_features: torch.Tensor,
    text_feature_sets: list[torch.Tensor],
    weights: list[float],
    batch_size: int,
    device: torch.device,
    normalization: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    text_sets = [features.to(device) for features in text_feature_sets]
    for start in range(0, image_features.shape[0], batch_size):
        end = min(start + batch_size, image_features.shape[0])
        image = image_features[start:end].to(device)
        combined = None
        for weight, text_features in zip(weights, text_sets):
            part = image @ text_features.T
            if normalization == "zscore":
                part = row_zscore(part)
            elif normalization != "none":
                raise ValueError(f"Unknown logit normalization: {normalization}")
            combined = part.mul(weight) if combined is None else combined + part.mul(weight)
        if combined is None:
            raise RuntimeError("No text feature sets provided")
        chunks.append(combined.cpu())
    return torch.cat(chunks, dim=0)


def topk_metrics(scores: torch.Tensor, labels: list[str], candidates: list[str], topk: int = 20) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    k = min(topk, scores.shape[1])
    indices = scores.topk(k, dim=1).indices
    ranks = []
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        true_score = scores[row_idx, true_idx]
        ranks.append(int((scores[row_idx] > true_score).sum().item()) + 1)
    if not ranks:
        return {}
    ranks_t = torch.tensor(ranks)
    return {
        "known": len(ranks),
        "top1": float((ranks_t <= 1).float().mean().item()),
        "top5": float((ranks_t <= 5).float().mean().item()),
        "top20": float((ranks_t <= 20).float().mean().item()),
        "mrr": float((1.0 / ranks_t.float()).mean().item()),
        "median_rank": float(ranks_t.float().median().item()),
        "mean_rank": float(ranks_t.float().mean().item()),
    }


def pred_metrics(pred_indices: torch.Tensor, labels: list[str], candidates: list[str], base_pred_indices: torch.Tensor) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    known = 0
    correct = 0
    base_correct = 0
    wins = 0
    losses = 0
    changed = 0
    for row_idx, label in enumerate(labels):
        if not label:
            continue
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        known += 1
        pred = int(pred_indices[row_idx].item())
        base_pred = int(base_pred_indices[row_idx].item())
        ok = pred == true_idx
        base_ok = base_pred == true_idx
        correct += int(ok)
        base_correct += int(base_ok)
        changed += int(pred != base_pred)
        wins += int((not base_ok) and ok)
        losses += int(base_ok and (not ok))
    return {
        "known": known,
        "top1": correct / known if known else 0.0,
        "base_top1": base_correct / known if known else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def active_indices(logits: torch.Tensor, mode: str, count: int, union_topk: int) -> torch.Tensor:
    n, c = logits.shape
    count = min(count, c)
    selected: set[int] = set()
    if union_topk > 0:
        top = logits.topk(min(union_topk, c), dim=1).indices.flatten().tolist()
        selected.update(int(idx) for idx in top)
    remaining = count - len(selected)
    if remaining > 0:
        if mode == "max":
            strength = logits.max(dim=0).values
        elif mode == "mean_top5":
            strength = logits.topk(min(5, n), dim=0).values.mean(dim=0)
        elif mode == "logsumexp":
            strength = torch.logsumexp(logits * 10.0, dim=0) / 10.0
        else:
            raise ValueError(f"Unknown active mode: {mode}")
        if selected:
            used = torch.tensor(list(selected), dtype=torch.long)
            strength[used] = -1e9
        extra = strength.topk(remaining).indices.tolist()
        selected.update(int(idx) for idx in extra)
    values = sorted(selected)
    if len(values) > count:
        # If per-image topK union is larger than requested, keep the globally strongest classes from that union.
        if mode == "max":
            strength = logits.max(dim=0).values
        elif mode == "mean_top5":
            strength = logits.topk(min(5, n), dim=0).values.mean(dim=0)
        else:
            strength = torch.logsumexp(logits * 10.0, dim=0) / 10.0
        selected_t = torch.tensor(values, dtype=torch.long)
        keep = strength[selected_t].topk(count).indices
        selected_t = selected_t[keep]
        values = sorted(int(v) for v in selected_t.tolist())
    return torch.tensor(values, dtype=torch.long)


def confidence_row_weights(
    logits: torch.Tensor,
    mode: str,
    floor: float,
    power: float,
    clean_fraction: float,
) -> torch.Tensor:
    n = logits.shape[0]
    if mode == "none":
        return torch.ones(n, dtype=logits.dtype, device=logits.device)
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"row weight floor must be in (0, 1], got {floor}")
    if power <= 0.0:
        raise ValueError(f"row weight power must be positive, got {power}")

    top2 = logits.topk(2, dim=1).values
    if mode in {"margin_rank", "margin_top"}:
        confidence = top2[:, 0] - top2[:, 1]
    elif mode == "top1_rank":
        confidence = top2[:, 0]
    else:
        raise ValueError(f"Unknown row weight mode: {mode}")

    if mode == "margin_top":
        fraction = min(1.0, max(0.0, clean_fraction))
        clean_count = max(1, min(n, round(n * fraction)))
        weights = torch.full_like(confidence, floor)
        weights[confidence.topk(clean_count).indices] = 1.0
    else:
        order = confidence.argsort()
        ranks = torch.empty_like(confidence)
        ranks[order] = torch.linspace(0.0, 1.0, n, dtype=logits.dtype, device=logits.device)
        weights = floor + (1.0 - floor) * ranks.pow(power)
    return weights / weights.mean().clamp_min(1e-12)


def class_prior(
    logits: torch.Tensor,
    mode: str,
    alpha: float,
    uniform_mix: float,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    k = logits.shape[1]
    uniform = torch.full((k,), 1.0 / k, dtype=logits.dtype, device=logits.device)
    if mode == "uniform":
        return uniform
    evidence = logits
    if row_weights is not None:
        weights = row_weights.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
        evidence = logits + weights.log()[:, None] / 10.0
    if mode == "max":
        strength = evidence.max(dim=0).values
    elif mode == "mean_top5":
        strength = evidence.topk(min(5, logits.shape[0]), dim=0).values.mean(dim=0)
    elif mode == "logsumexp":
        strength = torch.logsumexp(evidence * 10.0, dim=0) / 10.0
    else:
        raise ValueError(f"Unknown prior mode: {mode}")
    strength = (strength - strength.mean()) / strength.std().clamp_min(1e-6)
    prior = torch.softmax(alpha * strength, dim=0)
    mix = min(1.0, max(0.0, uniform_mix))
    return (1.0 - mix) * prior + mix * uniform


def sinkhorn(
    logits: torch.Tensor,
    tau: float,
    iters: int,
    prior: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    # SwAV-style balancing. Output shape is [images, classes] with each image row summing to 1.
    q = torch.exp((logits / tau).T)
    q = q / q.sum().clamp_min(1e-12)
    k, n = q.shape
    if prior is None:
        prior = torch.full((k,), 1.0 / k, dtype=q.dtype, device=q.device)
    else:
        prior = prior.to(device=q.device, dtype=q.dtype)
        prior = prior / prior.sum().clamp_min(1e-12)
    if row_weights is None:
        row_target = torch.full((n,), 1.0 / n, dtype=q.dtype, device=q.device)
    else:
        row_target = row_weights.to(device=q.device, dtype=q.dtype).clamp_min(1e-6)
        row_target = row_target / row_target.sum().clamp_min(1e-12)
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q * prior[:, None]
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q * row_target[None, :]
    return (q / row_target[None, :].clamp_min(1e-12)).T


def write_predictions(path: Path, image_ids: list[str], pred_indices: torch.Tensor, candidates: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred_idx in zip(image_ids, pred_indices.tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(pred_idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", default="", help="Comma-separated additional text feature .pt files.")
    parser.add_argument("--text-weights", default="", help="Comma-separated weights for base plus extra text features.")
    parser.add_argument("--logit-normalization", choices=["none", "zscore"], default="none")
    parser.add_argument("--candidate-classes", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--active-count-grid", default="500,750,1000,1250,1500,2000,3000")
    parser.add_argument("--active-mode-grid", default="max,mean_top5")
    parser.add_argument("--union-topk-grid", default="0,1,2,5")
    parser.add_argument("--tau-grid", default="0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--blend-grid", default="0.1,0.2,0.5,1.0,2.0,5.0")
    parser.add_argument("--prior-mode-grid", default="uniform")
    parser.add_argument("--prior-alpha-grid", default="1.0")
    parser.add_argument("--prior-uniform-mix-grid", default="0.0")
    parser.add_argument("--row-weight-mode-grid", default="none")
    parser.add_argument("--row-weight-floor-grid", default="0.1")
    parser.add_argument("--row-weight-power-grid", default="1.0")
    parser.add_argument("--row-weight-clean-fraction-grid", default="0.5")
    parser.add_argument("--row-weight-scope-grid", default="both", choices=None)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--apply-active-count", type=int, default=None)
    parser.add_argument("--apply-active-mode", default=None)
    parser.add_argument("--apply-union-topk", type=int, default=None)
    parser.add_argument("--apply-tau", type=float, default=None)
    parser.add_argument("--apply-blend", type=float, default=None)
    parser.add_argument("--apply-prior-mode", default="uniform")
    parser.add_argument("--apply-prior-alpha", type=float, default=1.0)
    parser.add_argument("--apply-prior-uniform-mix", type=float, default=0.0)
    parser.add_argument("--apply-row-weight-mode", default="none")
    parser.add_argument("--apply-row-weight-floor", type=float, default=0.1)
    parser.add_argument("--apply-row-weight-power", type=float, default=1.0)
    parser.add_argument("--apply-row-weight-clean-fraction", type=float, default=0.5)
    parser.add_argument("--apply-row-weight-scope", choices=["none", "prior", "columns", "both"], default="none")
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))
    image_features = normalize(image_payload["features"])
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_paths = [args.text_features] + parse_paths(args.extra_text_features)
    text_weights = parse_weights(args.text_weights, len(text_paths))
    text_features = [load_text_features(path, candidates) for path in text_paths]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(text_features) == 1:
        logits = compute_logits(image_features, text_features[0], args.score_batch_size, device)
    else:
        logits = compute_ensemble_logits(
            image_features,
            text_features,
            text_weights,
            args.score_batch_size,
            device,
            args.logit_normalization,
        )
    base_pred = logits.argmax(dim=1)
    base_summary = topk_metrics(logits, labels, candidates)

    if args.apply_active_count is not None:
        grid = [
            (
                args.apply_active_count,
                args.apply_active_mode or "max",
                args.apply_union_topk or 0,
                args.apply_tau or 0.03,
                args.apply_blend or 1.0,
                args.apply_prior_mode,
                args.apply_prior_alpha,
                args.apply_prior_uniform_mix,
                args.apply_row_weight_mode,
                args.apply_row_weight_floor,
                args.apply_row_weight_power,
                args.apply_row_weight_clean_fraction,
                args.apply_row_weight_scope,
            )
        ]
    else:
        row_weight_configs = []
        for row_mode in args.row_weight_mode_grid.split(","):
            row_mode = row_mode.strip()
            if not row_mode:
                continue
            if row_mode == "none":
                row_weight_configs.append(("none", 1.0, 1.0, 0.5, "none"))
                continue
            powers = ["1.0"] if row_mode == "margin_top" else args.row_weight_power_grid.split(",")
            clean_fractions = (
                args.row_weight_clean_fraction_grid.split(",") if row_mode == "margin_top" else ["0.5"]
            )
            row_weight_configs.extend(
                (
                    row_mode,
                    float(row_floor),
                    float(row_power),
                    float(clean_fraction),
                    row_scope.strip(),
                )
                for row_floor in args.row_weight_floor_grid.split(",")
                if row_floor.strip()
                for row_power in powers
                if row_power.strip()
                for clean_fraction in clean_fractions
                if clean_fraction.strip()
                for row_scope in args.row_weight_scope_grid.split(",")
                if row_scope.strip()
            )
        grid = [
            (
                int(active_count),
                active_mode.strip(),
                int(union_topk),
                float(tau),
                float(blend),
                prior_mode.strip(),
                float(prior_alpha),
                float(prior_mix),
                row_mode,
                row_floor,
                row_power,
                clean_fraction,
                row_scope,
            )
            for active_count in args.active_count_grid.split(",")
            if active_count.strip()
            for active_mode in args.active_mode_grid.split(",")
            if active_mode.strip()
            for union_topk in args.union_topk_grid.split(",")
            if union_topk.strip()
            for tau in args.tau_grid.split(",")
            if tau.strip()
            for blend in args.blend_grid.split(",")
            if blend.strip()
            for prior_mode in args.prior_mode_grid.split(",")
            if prior_mode.strip()
            for prior_alpha in args.prior_alpha_grid.split(",")
            if prior_alpha.strip()
            for prior_mix in args.prior_uniform_mix_grid.split(",")
            if prior_mix.strip()
            for row_mode, row_floor, row_power, clean_fraction, row_scope in row_weight_configs
        ]

    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, int, int], dict[str, Any], torch.Tensor] | None = None
    active_cache: dict[tuple[int, str, int], torch.Tensor] = {}
    row_weight_cache: dict[tuple[str, float, float, float], torch.Tensor] = {}
    for (
        active_count,
        active_mode,
        union_topk,
        tau,
        blend,
        prior_mode,
        prior_alpha,
        prior_mix,
        row_weight_mode,
        row_weight_floor,
        row_weight_power,
        row_weight_clean_fraction,
        row_weight_scope,
    ) in grid:
        active_key = (active_count, active_mode, union_topk)
        if active_key not in active_cache:
            active_cache[active_key] = active_indices(logits.clone(), active_mode, active_count, union_topk)
        active = active_cache[active_key]
        active_logits = logits[:, active].to(device)
        row_weight_key = (
            row_weight_mode,
            row_weight_floor,
            row_weight_power,
            row_weight_clean_fraction,
        )
        if row_weight_key not in row_weight_cache:
            row_weight_cache[row_weight_key] = confidence_row_weights(
                logits,
                row_weight_mode,
                row_weight_floor,
                row_weight_power,
                row_weight_clean_fraction,
            )
        row_weights = row_weight_cache[row_weight_key].to(device)
        prior_weights = row_weights if row_weight_scope in {"prior", "both"} else None
        column_weights = row_weights if row_weight_scope in {"columns", "both"} else None
        prior = class_prior(active_logits, prior_mode, prior_alpha, prior_mix, prior_weights)
        balanced = sinkhorn(
            active_logits,
            tau=tau,
            iters=args.sinkhorn_iters,
            prior=prior,
            row_weights=column_weights,
        )
        final = row_zscore(active_logits) + blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))
        pred_in_active = final.argmax(dim=1).cpu()
        pred = active[pred_in_active]
        row = {
            "active_count": int(active_count),
            "active_actual": int(len(active)),
            "active_mode": active_mode,
            "union_topk": int(union_topk),
            "tau": float(tau),
            "blend": float(blend),
            "prior_mode": prior_mode,
            "prior_alpha": float(prior_alpha),
            "prior_uniform_mix": float(prior_mix),
            "row_weight_mode": row_weight_mode,
            "row_weight_floor": float(row_weight_floor),
            "row_weight_power": float(row_weight_power),
            "row_weight_clean_fraction": float(row_weight_clean_fraction),
            "row_weight_scope": row_weight_scope,
            **pred_metrics(pred, labels, candidates, base_pred),
        }
        if any(labels):
            true_indices = [candidates.index(label) for label in labels if label in candidates]
            active_set = set(int(v) for v in active.tolist())
            row["active_label_coverage"] = sum(int(idx in active_set) for idx in true_indices) / max(1, len(true_indices))
        rows.append(row)
        key = (row.get("top1", 0.0), row.get("net", 0), -row.get("losses", 0))
        if best is None or key > best[0]:
            best = (key, row, pred)

    assert best is not None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_predictions(args.out_dir / "predictions.csv", image_ids, best[2], candidates)
    torch.save(
        {
            "image_ids": image_ids,
            "labels": labels,
            "candidates": candidates,
            "base_logits_top1": base_pred,
            "best_pred_indices": best[2],
            "best": best[1],
        },
        args.out_dir / "transductive_predictions.pt",
    )
    summary = {
        "image_features": str(args.image_features),
        "text_features": [str(path) for path in text_paths],
        "text_weights": text_weights,
        "logit_normalization": args.logit_normalization,
        "candidate_classes": str(args.candidate_classes) if args.candidate_classes else None,
        "rows": len(image_ids),
        "candidates": len(candidates),
        "base": base_summary,
        "best": best[1],
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

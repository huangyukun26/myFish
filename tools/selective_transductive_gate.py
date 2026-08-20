from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from transductive_active_sinkhorn import (
    active_indices,
    class_prior,
    compute_logits,
    load_classes,
    load_text_features,
    normalize,
    row_zscore,
    sinkhorn,
)


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def config_scores(
    logits: torch.Tensor,
    active: torch.Tensor,
    *,
    tau: float,
    blend: float,
    prior_mode: str,
    prior_alpha: float,
    prior_mix: float,
    iters: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_logits = logits[:, active].to(device)
    prior = class_prior(active_logits, prior_mode, prior_alpha, prior_mix)
    balanced = sinkhorn(active_logits, tau=tau, iters=iters, prior=prior)
    final = row_zscore(active_logits) + blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))
    top2 = final.topk(2, dim=1)
    pred = active[top2.indices[:, 0].cpu()]
    margin = (top2.values[:, 0] - top2.values[:, 1]).detach().cpu()
    top_score = top2.values[:, 0].detach().cpu()
    return pred, margin, top_score


def load_labels(path: Path) -> tuple[list[str], list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    image_ids = list(payload["image_ids"])
    labels = list(payload.get("labels", [""] * len(image_ids)))
    features = normalize(payload["features"])
    return image_ids, labels, features


def eval_preds(pred: torch.Tensor, reference: torch.Tensor, labels: list[str], candidates: list[str]) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    known = correct = ref_correct = wins = losses = changed = 0
    for row_idx, label in enumerate(labels):
        true_idx = class_to_idx.get(label)
        if true_idx is None:
            continue
        known += 1
        p = int(pred[row_idx])
        r = int(reference[row_idx])
        ok = p == true_idx
        ref_ok = r == true_idx
        correct += int(ok)
        ref_correct += int(ref_ok)
        wins += int((not ref_ok) and ok)
        losses += int(ref_ok and (not ok))
        changed += int(p != r)
    return {
        "known": known,
        "top1": correct / known if known else 0.0,
        "reference_top1": ref_correct / known if known else 0.0,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def write_predictions(path: Path, image_ids: list[str], pred: torch.Tensor, candidates: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, pred_idx in zip(image_ids, pred.tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(pred_idx)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--active-count", type=int, default=11598)
    parser.add_argument("--active-mode", default="max")
    parser.add_argument("--union-topk", type=int, default=0)
    parser.add_argument("--base-tau", type=float, default=0.02)
    parser.add_argument("--base-blend", type=float, default=5.0)
    parser.add_argument("--base-prior-alpha", type=float, default=0.5)
    parser.add_argument("--base-prior-mix", type=float, default=0.95)
    parser.add_argument("--cand-tau", type=float, default=0.04)
    parser.add_argument("--cand-blend", type=float, default=9.0)
    parser.add_argument("--cand-prior-alpha", type=float, default=1.0)
    parser.add_argument("--cand-prior-mix", type=float, default=0.98)
    parser.add_argument("--prior-mode", default="logsumexp")
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--cand-margin-quantiles", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--base-margin-quantiles", default="1.0,0.9,0.8,0.7,0.6,0.5")
    parser.add_argument("--delta-grid", default="-999,-0.5,0,0.5,1.0,1.5,2.0")
    args = parser.parse_args()

    image_ids, labels, image_features = load_labels(args.image_features)
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_features = load_text_features(args.text_features, candidates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = compute_logits(image_features, text_features, args.score_batch_size, device)
    active = active_indices(logits.clone(), args.active_mode, args.active_count, args.union_topk)

    base_pred, base_margin, base_top_score = config_scores(
        logits,
        active,
        tau=args.base_tau,
        blend=args.base_blend,
        prior_mode=args.prior_mode,
        prior_alpha=args.base_prior_alpha,
        prior_mix=args.base_prior_mix,
        iters=args.sinkhorn_iters,
        device=device,
    )
    cand_pred, cand_margin, cand_top_score = config_scores(
        logits,
        active,
        tau=args.cand_tau,
        blend=args.cand_blend,
        prior_mode=args.prior_mode,
        prior_alpha=args.cand_prior_alpha,
        prior_mix=args.cand_prior_mix,
        iters=args.sinkhorn_iters,
        device=device,
    )

    cand_qs = parse_grid(args.cand_margin_quantiles)
    base_qs = parse_grid(args.base_margin_quantiles)
    deltas = parse_grid(args.delta_grid)
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, int, int], dict[str, Any], torch.Tensor] | None = None
    disagree = cand_pred != base_pred
    score_delta = cand_top_score - base_top_score
    for cand_q in cand_qs:
        cand_thr = float(torch.quantile(cand_margin, cand_q).item())
        for base_q in base_qs:
            base_thr = float(torch.quantile(base_margin, base_q).item())
            for delta in deltas:
                mask = disagree & (cand_margin >= cand_thr) & (base_margin <= base_thr) & (score_delta >= delta)
                pred = torch.where(mask, cand_pred, base_pred)
                row = {
                    "cand_margin_q": cand_q,
                    "cand_margin_thr": cand_thr,
                    "base_margin_q": base_q,
                    "base_margin_thr": base_thr,
                    "delta": delta,
                    "override_count": int(mask.sum().item()),
                    **eval_preds(pred, base_pred, labels, candidates),
                }
                rows.append(row)
                key = (row["top1"], row["net"], -row["losses"])
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
            "base_pred_indices": base_pred,
            "cand_pred_indices": cand_pred,
            "best_pred_indices": best[2],
            "base_margin": base_margin,
            "cand_margin": cand_margin,
            "base_top_score": base_top_score,
            "cand_top_score": cand_top_score,
            "best": best[1],
        },
        args.out_dir / "selective_gate.pt",
    )
    summary = {
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "candidate_classes": str(args.candidate_classes),
        "rows": len(image_ids),
        "candidates": len(candidates),
        "base_config": {
            "tau": args.base_tau,
            "blend": args.base_blend,
            "prior_alpha": args.base_prior_alpha,
            "prior_mix": args.base_prior_mix,
        },
        "candidate_config": {
            "tau": args.cand_tau,
            "blend": args.cand_blend,
            "prior_alpha": args.cand_prior_alpha,
            "prior_mix": args.cand_prior_mix,
        },
        "best": best[1],
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

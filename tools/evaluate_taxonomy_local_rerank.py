from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from transductive_active_sinkhorn import (
    class_prior,
    compute_logits,
    load_classes,
    load_text_features,
    normalize,
    pred_metrics,
    row_zscore,
    sinkhorn,
)


def genus(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def build_known_genera(
    known_classes_path: Path,
    holdout_classes_path: Path | None,
    exclude_holdout_genera: bool,
) -> set[str]:
    known_classes = load_classes(known_classes_path, [])
    holdout_classes = set(load_classes(holdout_classes_path, [])) if holdout_classes_path else set()
    heldout_genera = {genus(name) for name in holdout_classes} if exclude_holdout_genera else set()
    train_classes = [
        name
        for name in known_classes
        if name not in holdout_classes and genus(name) not in heldout_genera
    ]
    return {genus(name) for name in train_classes}


def apply_novelty_gate(
    proposed: torch.Tensor,
    base: torch.Tensor,
    candidate_is_novel: torch.Tensor,
    gate: str,
) -> tuple[torch.Tensor, int]:
    if gate == "none":
        eligible = torch.ones_like(base, dtype=torch.bool)
    elif gate == "rerank_novel":
        eligible = candidate_is_novel[proposed]
    elif gate == "move_to_novel_genus":
        eligible = (~candidate_is_novel[base]) & candidate_is_novel[proposed]
    else:
        raise ValueError(f"Unknown novelty gate: {gate}")
    pred = base.clone()
    pred[eligible] = proposed[eligible]
    return pred, int(eligible.sum().item())


def transductive_scores(
    logits: torch.Tensor,
    device: torch.device,
    *,
    tau: float,
    blend: float,
    prior_alpha: float,
    prior_uniform_mix: float,
    sinkhorn_iters: int,
) -> torch.Tensor:
    work = logits.to(device)
    prior = class_prior(work, "logsumexp", prior_alpha, prior_uniform_mix)
    balanced = sinkhorn(work, tau=tau, iters=sinkhorn_iters, prior=prior)
    return (row_zscore(work) + blend * row_zscore(torch.log(balanced.clamp_min(1e-12)))).cpu()


def choose_predictions(
    top_indices: torch.Tensor,
    species_scores: torch.Tensor,
    branch_global_pred: torch.Tensor,
    candidate_genus_ids: torch.Tensor,
    mode: str,
    support_weight: float,
    support_temperature: float,
) -> torch.Tensor:
    predictions = top_indices[:, 0].clone()
    for row_idx in range(top_indices.shape[0]):
        indices = top_indices[row_idx]
        scores = species_scores[row_idx]
        genera = candidate_genus_ids[indices]
        base_genus = int(genera[0].item())
        branch_genus = int(candidate_genus_ids[int(branch_global_pred[row_idx])].item())

        if mode == "global":
            selected_position = int(scores.argmax().item())
        elif mode == "base_genus":
            allowed = genera == base_genus
            selected_position = int(scores.masked_fill(~allowed, -torch.inf).argmax().item())
        elif mode == "agree_genus":
            if base_genus != branch_genus:
                continue
            allowed = genera == base_genus
            selected_position = int(scores.masked_fill(~allowed, -torch.inf).argmax().item())
        elif mode == "branch_genus":
            allowed = genera == branch_genus
            if not bool(allowed.any()):
                continue
            selected_position = int(scores.masked_fill(~allowed, -torch.inf).argmax().item())
        elif mode == "pooled_genus":
            best_group_score = -float("inf")
            selected_position = 0
            for genus_id in genera.unique().tolist():
                positions = (genera == int(genus_id)).nonzero(as_tuple=False).flatten()
                group_scores = scores[positions]
                group_max, local_position = group_scores.max(dim=0)
                centered = (group_scores - group_max) / support_temperature
                support = support_temperature * torch.logsumexp(centered, dim=0)
                group_score = float((group_max + support_weight * support).item())
                if group_score > best_group_score:
                    best_group_score = group_score
                    selected_position = int(positions[int(local_position.item())].item())
        else:
            raise ValueError(f"Unknown mode: {mode}")
        predictions[row_idx] = indices[selected_position]
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--adapter-text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--topk-grid", default="10,20,50")
    parser.add_argument("--branch-source-grid", default="adapter_logits,adapter_sinkhorn")
    parser.add_argument("--species-weight-grid", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--mode-grid", default="global,base_genus,agree_genus,branch_genus,pooled_genus")
    parser.add_argument("--support-weight-grid", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--support-temperature-grid", default="0.5,1.0")
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--blend", type=float, default=5.0)
    parser.add_argument("--prior-alpha", type=float, default=0.5)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.95)
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--known-classes", type=Path, default=None)
    parser.add_argument("--holdout-classes", type=Path, default=None)
    parser.add_argument("--exclude-holdout-genera", action="store_true")
    parser.add_argument("--novelty-gate-grid", default="none")
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_features = normalize(image_payload["features"])
    image_ids = list(image_payload["image_ids"])
    labels = list(image_payload.get("labels", [""] * len(image_ids)))
    base_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(base_payload["classes"]))
    base_text = load_text_features(args.base_text_features, candidates)
    adapter_text = load_text_features(args.adapter_text_features, candidates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_logits = compute_logits(image_features, base_text, args.score_batch_size, device)
    adapter_logits = compute_logits(image_features, adapter_text, args.score_batch_size, device)
    base_scores = transductive_scores(
        base_logits,
        device,
        tau=args.tau,
        blend=args.blend,
        prior_alpha=args.prior_alpha,
        prior_uniform_mix=args.prior_uniform_mix,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    adapter_sinkhorn = transductive_scores(
        adapter_logits,
        device,
        tau=args.tau,
        blend=args.blend,
        prior_alpha=args.prior_alpha,
        prior_uniform_mix=args.prior_uniform_mix,
        sinkhorn_iters=args.sinkhorn_iters,
    )
    branch_scores = {
        "adapter_logits": adapter_logits,
        "adapter_sinkhorn": adapter_sinkhorn,
    }
    base_pred = base_scores.argmax(dim=1)
    genus_names = {name: idx for idx, name in enumerate(sorted({genus(name) for name in candidates}))}
    candidate_genus_ids = torch.tensor([genus_names[genus(name)] for name in candidates], dtype=torch.long)
    novelty_gates = [value.strip() for value in args.novelty_gate_grid.split(",") if value.strip()]
    if any(gate != "none" for gate in novelty_gates):
        if args.known_classes is None:
            raise ValueError("--known-classes is required for novelty gating")
        known_genera = build_known_genera(
            args.known_classes,
            args.holdout_classes,
            args.exclude_holdout_genera,
        )
        candidate_is_novel = torch.tensor(
            [genus(name) not in known_genera for name in candidates],
            dtype=torch.bool,
        )
    else:
        known_genera = set()
        candidate_is_novel = torch.zeros(len(candidates), dtype=torch.bool)

    rows: list[dict[str, Any]] = []
    predictions: dict[str, torch.Tensor] = {}
    for topk in [int(value) for value in args.topk_grid.split(",") if value.strip()]:
        topk = min(topk, len(candidates))
        base_top_values, top_indices = base_scores.topk(topk, dim=1)
        base_local = row_zscore(base_top_values)
        for branch_source in [value.strip() for value in args.branch_source_grid.split(",") if value.strip()]:
            branch = branch_scores[branch_source]
            branch_local = row_zscore(branch.gather(1, top_indices))
            branch_global_pred = branch.argmax(dim=1)
            for species_weight in [float(value) for value in args.species_weight_grid.split(",") if value.strip()]:
                species_scores = base_local + species_weight * branch_local
                for mode in [value.strip() for value in args.mode_grid.split(",") if value.strip()]:
                    support_pairs = (
                        [
                            (float(weight), float(temperature))
                            for weight in args.support_weight_grid.split(",")
                            if weight.strip()
                            for temperature in args.support_temperature_grid.split(",")
                            if temperature.strip()
                        ]
                        if mode == "pooled_genus"
                        else [(0.0, 1.0)]
                    )
                    for support_weight, support_temperature in support_pairs:
                        proposed = choose_predictions(
                            top_indices,
                            species_scores,
                            branch_global_pred,
                            candidate_genus_ids,
                            mode,
                            support_weight,
                            support_temperature,
                        )
                        for novelty_gate in novelty_gates:
                            pred, eligible_rows = apply_novelty_gate(
                                proposed,
                                base_pred,
                                candidate_is_novel,
                                novelty_gate,
                            )
                            config_id = (
                                f"k{topk}_{branch_source}_sw{species_weight:g}_{mode}"
                                f"_gw{support_weight:g}_gt{support_temperature:g}"
                                f"_ng{novelty_gate}"
                            )
                            row = {
                                "config_id": config_id,
                                "topk": topk,
                                "branch_source": branch_source,
                                "species_weight": species_weight,
                                "mode": mode,
                                "support_weight": support_weight,
                                "support_temperature": support_temperature,
                                "novelty_gate": novelty_gate,
                                "known_genera": len(known_genera),
                                "eligible_rows": eligible_rows,
                                **pred_metrics(pred, labels, candidates, base_pred),
                            }
                            rows.append(row)
                            predictions[config_id] = pred

    rows.sort(key=lambda row: (row["top1"], row["net"], -row["losses"]), reverse=True)
    best = rows[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "image_ids": image_ids,
            "labels": labels,
            "candidates": candidates,
            "base_pred_indices": base_pred,
            "best_pred_indices": predictions[best["config_id"]],
            "best": best,
        },
        args.out_dir / "predictions.pt",
    )
    summary = {
        "image_features": str(args.image_features),
        "base_text_features": str(args.base_text_features),
        "adapter_text_features": str(args.adapter_text_features),
        "candidate_classes": str(args.candidate_classes),
        "rows": len(image_ids),
        "candidates": len(candidates),
        "configs": len(rows),
        "base_top1": pred_metrics(base_pred, labels, candidates, base_pred)["top1"],
        "best": best,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

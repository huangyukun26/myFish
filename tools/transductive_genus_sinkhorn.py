from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from transductive_active_sinkhorn import (
    class_prior,
    compute_logits,
    load_classes,
    load_text_features,
    normalize,
    row_zscore,
    sinkhorn,
)


def genus_name(class_name: str) -> str:
    return class_name.split()[0]


def genus_logits_from_species(logits: torch.Tensor, candidates: list[str], mode: str) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(candidates):
        groups[genus_name(name)].append(idx)
    genera = sorted(groups)
    species_to_genus = torch.empty(len(candidates), dtype=torch.long)
    parts = []
    for genus_idx, genus in enumerate(genera):
        idx = torch.tensor(groups[genus], dtype=torch.long)
        species_to_genus[idx] = genus_idx
        values = logits[:, idx]
        if mode == "max":
            parts.append(values.max(dim=1).values)
        elif mode == "mean_top3":
            parts.append(values.topk(min(3, values.shape[1]), dim=1).values.mean(dim=1))
        elif mode == "logsumexp":
            parts.append(torch.logsumexp(values * 10.0, dim=1) / 10.0)
        else:
            raise ValueError(f"Unknown genus mode: {mode}")
    return torch.stack(parts, dim=1), genera, species_to_genus


def pred_metrics(pred: torch.Tensor, labels: list[str], candidates: list[str], reference: torch.Tensor) -> dict[str, Any]:
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
        for image_id, idx in zip(image_ids, pred.tolist()):
            writer.writerow({"image_id": image_id, "prediction": candidates[int(idx)]})


def parse_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--species-tau", type=float, default=0.04)
    parser.add_argument("--species-blend", type=float, default=9.0)
    parser.add_argument("--species-prior-alpha", type=float, default=1.0)
    parser.add_argument("--species-prior-mix", type=float, default=0.98)
    parser.add_argument("--genus-mode-grid", default="max,mean_top3,logsumexp")
    parser.add_argument("--genus-tau-grid", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--genus-blend-grid", default="0.5,1,2,3,5")
    parser.add_argument("--genus-prior-alpha-grid", default="0.25,0.5,1.0")
    parser.add_argument("--genus-prior-mix-grid", default="0.9,0.98")
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    args = parser.parse_args()

    payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_ids = list(payload["image_ids"])
    labels = list(payload.get("labels", [""] * len(image_ids)))
    image_features = normalize(payload["features"])
    text_payload = torch.load(args.text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(text_payload["classes"]))
    text_features = load_text_features(args.text_features, candidates)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = compute_logits(image_features, text_features, args.score_batch_size, device)

    species_prior = class_prior(logits.to(device), "logsumexp", args.species_prior_alpha, args.species_prior_mix)
    species_balanced = sinkhorn(logits.to(device), tau=args.species_tau, iters=args.sinkhorn_iters, prior=species_prior)
    species_final = row_zscore(logits.to(device)) + args.species_blend * row_zscore(torch.log(species_balanced.clamp_min(1e-12)))
    reference = species_final.argmax(dim=1).cpu()

    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, int, int], dict[str, Any], torch.Tensor] | None = None
    for genus_mode in [part.strip() for part in args.genus_mode_grid.split(",") if part.strip()]:
        genus_logits, genera, species_to_genus = genus_logits_from_species(logits, candidates, genus_mode)
        genus_logits_dev = genus_logits.to(device)
        species_to_genus_dev = species_to_genus.to(device)
        for genus_tau in parse_grid(args.genus_tau_grid):
            for genus_alpha in parse_grid(args.genus_prior_alpha_grid):
                for genus_mix in parse_grid(args.genus_prior_mix_grid):
                    prior = class_prior(genus_logits_dev, "logsumexp", genus_alpha, genus_mix)
                    balanced = sinkhorn(genus_logits_dev, tau=genus_tau, iters=args.sinkhorn_iters, prior=prior)
                    genus_score = row_zscore(genus_logits_dev) + row_zscore(torch.log(balanced.clamp_min(1e-12)))
                    species_genus_score = genus_score[:, species_to_genus_dev]
                    for genus_blend in parse_grid(args.genus_blend_grid):
                        final = species_final + genus_blend * row_zscore(species_genus_score)
                        pred = final.argmax(dim=1).cpu()
                        row = {
                            "species_tau": args.species_tau,
                            "species_blend": args.species_blend,
                            "species_prior_alpha": args.species_prior_alpha,
                            "species_prior_mix": args.species_prior_mix,
                            "genus_mode": genus_mode,
                            "genus_tau": genus_tau,
                            "genus_blend": genus_blend,
                            "genus_prior_alpha": genus_alpha,
                            "genus_prior_mix": genus_mix,
                            "num_genera": len(genera),
                            **pred_metrics(pred, labels, candidates, reference),
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
    summary = {
        "image_features": str(args.image_features),
        "text_features": str(args.text_features),
        "candidate_classes": str(args.candidate_classes),
        "rows": len(image_ids),
        "candidates": len(candidates),
        "best": best[1],
        "sweep_csv": str(args.out_dir / "sweep.csv"),
        "predictions_csv": str(args.out_dir / "predictions.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

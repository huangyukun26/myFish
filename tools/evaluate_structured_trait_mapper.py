from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from evaluate_crossdim_text_visual_oof import train_mapper
from train_text_to_visual_adapter import build_visual_prototypes, load_exclusions
from transductive_active_sinkhorn import class_prior, pred_metrics, row_zscore, sinkhorn


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"Expected name=path, got {value}")
    return name.strip(), Path(path.strip())


def load_class_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if not isinstance(payload, dict) else list(payload.keys())


def reorder_features(payload: dict[str, Any], classes: list[str]) -> torch.Tensor:
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in classes if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} classes missing from feature payload; first={missing[:5]}")
    indices = torch.tensor([class_to_idx[name] for name in classes], dtype=torch.long)
    return normalize(payload["features"][indices])


def compute_logits(
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    classes = class_features.to(device)
    chunks = []
    for start in range(0, len(image_features), batch_size):
        image = image_features[start : start + batch_size].to(device)
        chunks.append((image @ classes.T).cpu())
    return torch.cat(chunks)


def h8192_prediction(logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    work = logits.to(device)
    prior = class_prior(work, "logsumexp", alpha=0.5, uniform_mix=0.95)
    balanced = sinkhorn(work, tau=0.02, iters=5, prior=prior)
    scores = row_zscore(work) + 5.0 * row_zscore(torch.log(balanced.clamp_min(1e-12)))
    return scores.argmax(dim=1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-image-features", type=Path, required=True)
    parser.add_argument("--query-image-features", type=Path, required=True)
    parser.add_argument("--trait-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--holdout-classes", type=Path, required=True)
    parser.add_argument("--exclude-genera", action="store_true")
    parser.add_argument("--candidate-classes", action="append", required=True, help="name=path")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-dims", default="0,1024")
    parser.add_argument("--seeds", default="2027,2028")
    parser.add_argument("--blend-grid", default="0.05,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    args = parser.parse_args()

    train_payload = torch.load(args.train_image_features, map_location="cpu", weights_only=False)
    query_payload = torch.load(args.query_image_features, map_location="cpu", weights_only=False)
    trait_payload = torch.load(args.trait_features, map_location="cpu", weights_only=False)
    base_text_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    all_classes = list(trait_payload["classes"])
    all_to_idx = {name: idx for idx, name in enumerate(all_classes)}
    trait_features = normalize(trait_payload["features"])
    base_text_features = reorder_features(base_text_payload, all_classes)
    query_features = normalize(query_payload["features"])
    labels = list(query_payload.get("labels", [""] * len(query_features)))
    holdout_classes = load_class_list(args.holdout_classes)
    if set(labels) != set(holdout_classes):
        raise RuntimeError("Query labels and holdout class list differ")

    exclude_classes, exclude_genera = load_exclusions(args.holdout_classes, args.exclude_genera)
    train_classes, target_prototypes, train_counts = build_visual_prototypes(
        train_payload,
        all_classes,
        exclude_classes=exclude_classes,
        exclude_genera=exclude_genera,
        min_count=1,
    )
    train_indices = torch.tensor([all_to_idx[name] for name in train_classes], dtype=torch.long)
    train_traits = trait_features[train_indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_sets = [(name, load_class_list(path), path) for name, path in map(parse_named_path, args.candidate_classes)]
    hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    blend_grid = [float(value) for value in args.blend_grid.split(",") if value.strip()]

    rows: list[dict[str, Any]] = []
    model_summaries = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for hidden_dim in hidden_dims:
        for seed in seeds:
            model, losses = train_mapper(
                train_traits,
                target_prototypes,
                hidden_dim=hidden_dim,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                contrastive_weight=args.contrastive_weight,
                seed=seed,
                device=device,
            )
            model.eval()
            mapped_chunks = []
            with torch.inference_mode():
                for start in range(0, len(trait_features), 1024):
                    mapped_chunks.append(model(trait_features[start : start + 1024].to(device)).cpu())
            mapped_features = torch.cat(mapped_chunks)
            torch.save(
                {
                    "classes": all_classes,
                    "features": mapped_features,
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                    "losses": losses,
                    "train_classes": train_classes,
                    "train_counts": train_counts,
                },
                args.out_dir / f"mapped_h{hidden_dim}_seed{seed}.pt",
            )
            model_summaries.append(
                {
                    "hidden_dim": hidden_dim,
                    "seed": seed,
                    "initial_loss": losses[0],
                    "final_loss": losses[-1],
                }
            )

            for split_name, candidates, candidate_path in candidate_sets:
                candidate_indices = torch.tensor([all_to_idx[name] for name in candidates], dtype=torch.long)
                base_candidate_features = base_text_features[candidate_indices]
                mapped_candidate_features = mapped_features[candidate_indices]
                base_logits = compute_logits(
                    query_features,
                    base_candidate_features,
                    device,
                    args.score_batch_size,
                )
                base_independent_pred = base_logits.argmax(dim=1)
                base_h8192_pred = h8192_prediction(base_logits, device)
                for blend in blend_grid:
                    combined_features = normalize(
                        (1.0 - blend) * base_candidate_features + blend * mapped_candidate_features
                    )
                    logits = compute_logits(query_features, combined_features, device, args.score_batch_size)
                    independent_pred = logits.argmax(dim=1)
                    h8192_pred = h8192_prediction(logits, device)
                    independent = pred_metrics(
                        independent_pred,
                        labels,
                        candidates,
                        base_independent_pred,
                    )
                    transductive = pred_metrics(
                        h8192_pred,
                        labels,
                        candidates,
                        base_h8192_pred,
                    )
                    rows.append(
                        {
                            "split": split_name,
                            "candidate_classes": str(candidate_path),
                            "hidden_dim": hidden_dim,
                            "seed": seed,
                            "blend": blend,
                            "independent_top1": independent["top1"],
                            "independent_base_top1": independent["base_top1"],
                            "independent_changed": independent["changed"],
                            "independent_wins": independent["wins"],
                            "independent_losses": independent["losses"],
                            "independent_net": independent["net"],
                            "h8192_top1": transductive["top1"],
                            "h8192_base_top1": transductive["base_top1"],
                            "h8192_changed": transductive["changed"],
                            "h8192_wins": transductive["wins"],
                            "h8192_losses": transductive["losses"],
                            "h8192_net": transductive["net"],
                        }
                    )

    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "train_image_features": str(args.train_image_features),
        "query_image_features": str(args.query_image_features),
        "trait_features": str(args.trait_features),
        "base_text_features": str(args.base_text_features),
        "holdout_classes": str(args.holdout_classes),
        "exclude_genera": args.exclude_genera,
        "train_classes": len(train_classes),
        "query_rows": len(labels),
        "candidate_sets": [name for name, _, _ in candidate_sets],
        "models": model_summaries,
        "rows": len(rows),
        "sweep_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

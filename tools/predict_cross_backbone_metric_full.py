from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from evaluate_cross_backbone_metric_oof import prepare_pair, train_metric


def parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def aligned_features(base: dict[str, Any], alternate: dict[str, Any], source: str) -> torch.Tensor:
    if list(base["image_ids"]) != list(alternate["image_ids"]):
        raise RuntimeError(f"Image order differs for {source}")
    if list(base.get("labels", [])) != list(alternate.get("labels", [])):
        raise RuntimeError(f"Labels differ for {source}")
    return prepare_pair(base, alternate)


def load_labeled_pairs(
    base_paths: list[Path],
    alternate_paths: list[Path],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    if len(base_paths) != len(alternate_paths):
        raise ValueError("Labeled base and alternate cache counts differ")
    features = []
    class_ids = []
    image_ids: list[str] = []
    labels: list[str] = []
    classes: list[str] | None = None
    for base_path, alternate_path in zip(base_paths, alternate_paths):
        base = load_cache(base_path)
        alternate = load_cache(alternate_path)
        if classes is None:
            classes = list(base["classes"])
        if list(base["classes"]) != classes or list(alternate["classes"]) != classes:
            raise RuntimeError("Class orders differ across labeled caches")
        features.append(aligned_features(base, alternate, str(base_path)))
        class_ids.append(base["class_ids"].long())
        image_ids.extend(base["image_ids"])
        labels.extend(base["labels"])
    if classes is None:
        raise RuntimeError("No labeled caches")
    return torch.cat(features), torch.cat(class_ids), image_ids, labels, classes


@torch.inference_mode()
def transform_batches(model, features: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(features), batch_size):
        chunks.append(model(features[start : start + batch_size].to(device)).cpu())
    return torch.cat(chunks)


def class_prototypes(
    features: torch.Tensor,
    class_ids: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros((class_count, features.shape[1]), dtype=torch.float32)
    sums.index_add_(0, class_ids, features.float())
    counts = torch.bincount(class_ids, minlength=class_count)
    if bool((counts == 0).any()):
        missing = torch.where(counts == 0)[0].tolist()
        raise RuntimeError(f"{len(missing)} classes have no labeled prototype; first={missing[:5]}")
    return F.normalize(sums / counts[:, None], dim=1), counts


@torch.inference_mode()
def score_topk(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    batch_size: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = prototypes.to(device)
    values = []
    indices = []
    for start in range(0, len(query), batch_size):
        scores = query[start : start + batch_size].to(device) @ prototypes.T
        batch_values, batch_indices = scores.topk(min(topk, prototypes.shape[0]), dim=1)
        values.append(batch_values.cpu())
        indices.append(batch_indices.cpu())
    return torch.cat(values), torch.cat(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-base-support", type=Path, required=True)
    parser.add_argument("--pair-base-query", type=Path, required=True)
    parser.add_argument("--pair-alt-support", type=Path, required=True)
    parser.add_argument("--pair-alt-query", type=Path, required=True)
    parser.add_argument("--labeled-base-caches", required=True)
    parser.add_argument("--labeled-alt-caches", required=True)
    parser.add_argument("--test-base-cache", type=Path, required=True)
    parser.add_argument("--test-alt-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="2027,2028")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    pair_base_support = load_cache(args.pair_base_support)
    pair_base_query = load_cache(args.pair_base_query)
    pair_alt_support = load_cache(args.pair_alt_support)
    pair_alt_query = load_cache(args.pair_alt_query)
    if pair_base_support["classes"] != pair_alt_support["classes"]:
        raise RuntimeError("Pair support class orders differ")
    pair_support = aligned_features(pair_base_support, pair_alt_support, "pair support")
    pair_query = aligned_features(pair_base_query, pair_alt_query, "pair query")
    if pair_base_support["class_ids"].long().tolist() != pair_base_query["class_ids"].long().tolist():
        raise RuntimeError("Pair support/query class IDs differ")
    pair_labels = list(pair_base_query["labels"])
    train_indices = torch.arange(len(pair_labels))

    labeled_features, labeled_class_ids, _labeled_ids, _labels, classes = load_labeled_pairs(
        parse_paths(args.labeled_base_caches),
        parse_paths(args.labeled_alt_caches),
    )
    test_base = load_cache(args.test_base_cache)
    test_alternate = load_cache(args.test_alt_cache)
    if list(test_base["classes"]) != classes or list(test_alternate["classes"]) != classes:
        raise RuntimeError("Test cache class order differs from labeled caches")
    test_features = aligned_features(test_base, test_alternate, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        model, losses = train_metric(
            pair_support,
            pair_query,
            pair_labels,
            train_indices,
            rank=args.rank,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            seed=seed,
            device=device,
        )
        transformed_labeled = transform_batches(model, labeled_features, device, args.batch_size)
        prototypes, counts = class_prototypes(transformed_labeled, labeled_class_ids, len(classes))
        transformed_test = transform_batches(model, test_features, device, args.batch_size)
        topk_values, topk_indices = score_topk(
            transformed_test,
            prototypes,
            device,
            args.batch_size,
            args.topk,
        )
        prediction_path = args.out_dir / f"test_seen_metric_seed{seed}_topk.pt"
        torch.save(
            {
                "topk_indices": topk_indices,
                "topk_values": topk_values,
                "image_ids": list(test_base["image_ids"]),
                "labels": list(test_base.get("labels", [])),
                "classes": classes,
                "prototype_counts": counts,
                "seed": seed,
            },
            prediction_path,
        )
        model_path = args.out_dir / f"metric_seed{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(pair_support.shape[1]),
                "rank": args.rank,
                "dropout": args.dropout,
                "seed": seed,
                "losses": losses,
            },
            model_path,
        )
        summaries.append(
            {
                "seed": seed,
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "prediction_path": str(prediction_path),
                "model_path": str(model_path),
            }
        )

    summary = {
        "device": str(device),
        "pair_rows": len(pair_labels),
        "labeled_rows": len(labeled_features),
        "test_rows": len(test_features),
        "classes": len(classes),
        "input_dim": int(pair_support.shape[1]),
        "runs": summaries,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

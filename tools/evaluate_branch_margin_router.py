from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def genus_fold(label: str, folds: int) -> int:
    genus = label.split(maxsplit=1)[0]
    digest = hashlib.sha1(genus.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % folds


def metrics(prediction: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> dict:
    correct = prediction.eq(target)
    base_correct = base.eq(target)
    return {
        "top1": float(correct.float().mean().item()),
        "changed": int(prediction.ne(base).sum().item()),
        "wins": int((correct & ~base_correct).sum().item()),
        "losses": int((~correct & base_correct).sum().item()),
        "net": int((correct.sum() - base_correct.sum()).item()),
    }


def route(
    base_prediction: torch.Tensor,
    alt_prediction: torch.Tensor,
    base_margin: torch.Tensor,
    alt_margin: torch.Tensor,
    base_topk: torch.Tensor,
    frequency_mask: torch.Tensor,
    *,
    base_threshold: float,
    alt_threshold: float,
    require_base_topk: int,
) -> torch.Tensor:
    mask = (
        base_margin.le(base_threshold)
        & alt_margin.ge(alt_threshold)
        & alt_prediction.ne(base_prediction)
        & frequency_mask
    )
    if require_base_topk > 0:
        mask &= base_topk[:, :require_base_topk].eq(alt_prediction[:, None]).any(dim=1)
    return torch.where(mask, alt_prediction, base_prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--alternate-topk", type=Path, required=True)
    parser.add_argument("--alternate-topk-2", type=Path, default=None)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    base_payload = torch.load(args.base_logits, map_location="cpu", weights_only=False)
    alt_payload = torch.load(args.alternate_topk, map_location="cpu", weights_only=False)
    if base_payload["image_ids"] != alt_payload["image_ids"]:
        raise RuntimeError("Base and alternate image order differs")
    if base_payload["classes"] != alt_payload["classes"]:
        raise RuntimeError("Base and alternate class order differs")

    logits = base_payload["logits"].float()
    target = base_payload["class_ids"].long()
    base_values, base_topk = logits.topk(20, dim=1)
    base_prediction = base_topk[:, 0]
    base_margin = base_values[:, 0] - base_values[:, 1]
    alt_topk = alt_payload["topk_indices"].long()
    alt_values = alt_payload["topk_values"].float()
    alt_prediction = alt_topk[:, 0]
    alt_margin = alt_values[:, 0] - alt_values[:, 1]
    agreement_mask = torch.ones_like(alt_prediction, dtype=torch.bool)
    if args.alternate_topk_2 is not None:
        alt_payload_2 = torch.load(args.alternate_topk_2, map_location="cpu", weights_only=False)
        if base_payload["image_ids"] != alt_payload_2["image_ids"]:
            raise RuntimeError("Second alternate branch has different image order")
        if base_payload["classes"] != alt_payload_2["classes"]:
            raise RuntimeError("Second alternate branch has different class order")
        alt_prediction_2 = alt_payload_2["topk_indices"].long()[:, 0]
        alt_values_2 = alt_payload_2["topk_values"].float()
        alt_margin_2 = alt_values_2[:, 0] - alt_values_2[:, 1]
        agreement_mask = alt_prediction.eq(alt_prediction_2)
        alt_margin = torch.minimum(alt_margin, alt_margin_2)
    labels = list(base_payload["labels"])
    folds = torch.tensor([genus_fold(label, args.folds) for label in labels], dtype=torch.long)
    class_counts = None
    if args.train_cache is not None:
        train_payload = torch.load(args.train_cache, map_location="cpu", weights_only=False)
        class_counts = torch.bincount(
            train_payload["class_ids"].long(), minlength=len(base_payload["classes"])
        ) + 1

    def make_frequency_mask(mode: str) -> torch.Tensor:
        if mode == "all":
            return torch.ones_like(base_prediction, dtype=torch.bool)
        if class_counts is None:
            raise RuntimeError("Frequency-aware routing requires --train-cache")
        base_count = class_counts[base_prediction]
        alt_count = class_counts[alt_prediction]
        if mode == "alt_c2":
            return alt_count.eq(2)
        if mode == "base_c2":
            return base_count.eq(2)
        if mode == "either_c2":
            return alt_count.eq(2) | base_count.eq(2)
        if mode == "alt_le5":
            return alt_count.le(5)
        if mode == "base_le5":
            return base_count.le(5)
        if mode == "either_le5":
            return alt_count.le(5) | base_count.le(5)
        raise ValueError(f"Unknown frequency mode: {mode}")

    base_fractions = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
    alt_fractions = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    require_values = (0, 5, 20)
    frequency_modes = ("all",) if class_counts is None else (
        "all",
        "alt_c2",
        "base_c2",
        "either_c2",
        "alt_le5",
        "base_le5",
        "either_le5",
    )

    def evaluate_config(indices: torch.Tensor, config: tuple[float, float, int, str]) -> tuple[dict, float, float]:
        base_fraction, alt_fraction, require_topk, frequency_mode = config
        base_threshold = float(torch.quantile(base_margin[indices], base_fraction).item())
        alt_threshold = float(torch.quantile(alt_margin[indices], 1.0 - alt_fraction).item())
        prediction = route(
            base_prediction,
            alt_prediction,
            base_margin,
            alt_margin,
            base_topk,
            make_frequency_mask(frequency_mode),
            base_threshold=base_threshold,
            alt_threshold=alt_threshold,
            require_base_topk=require_topk,
        )
        prediction = torch.where(agreement_mask, prediction, base_prediction)
        return metrics(prediction[indices], target[indices], base_prediction[indices]), base_threshold, alt_threshold

    all_indices = torch.arange(len(target))
    configs = [
        (base_fraction, alt_fraction, require_topk, frequency_mode)
        for base_fraction in base_fractions
        for alt_fraction in alt_fractions
        for require_topk in require_values
        for frequency_mode in frequency_modes
    ]
    grid = []
    for config in configs:
        row, base_threshold, alt_threshold = evaluate_config(all_indices, config)
        grid.append(
            {
                "base_low_fraction": config[0],
                "alt_high_fraction": config[1],
                "require_alt_in_base_topk": config[2],
                "frequency_mode": config[3],
                "base_threshold": base_threshold,
                "alt_threshold": alt_threshold,
                **row,
            }
        )

    oof_prediction = base_prediction.clone()
    fold_rows = []
    for fold in range(args.folds):
        train_indices = torch.where(folds.ne(fold))[0]
        heldout_indices = torch.where(folds.eq(fold))[0]
        candidates = []
        for config in configs:
            train_row, base_threshold, alt_threshold = evaluate_config(train_indices, config)
            candidates.append((train_row, config, base_threshold, alt_threshold))
        train_row, config, base_threshold, alt_threshold = max(
            candidates,
            key=lambda value: (value[0]["top1"], -value[0]["changed"], -value[1][0], -value[1][1]),
        )
        prediction = route(
            base_prediction,
            alt_prediction,
            base_margin,
            alt_margin,
            base_topk,
            make_frequency_mask(config[3]),
            base_threshold=base_threshold,
            alt_threshold=alt_threshold,
            require_base_topk=config[2],
        )
        prediction = torch.where(agreement_mask, prediction, base_prediction)
        oof_prediction[heldout_indices] = prediction[heldout_indices]
        fold_rows.append(
            {
                "fold": fold,
                "rows": len(heldout_indices),
                "selected": {
                    "base_low_fraction": config[0],
                    "alt_high_fraction": config[1],
                    "require_alt_in_base_topk": config[2],
                    "frequency_mode": config[3],
                    "base_threshold": base_threshold,
                    "alt_threshold": alt_threshold,
                },
                "train": train_row,
                "heldout": metrics(
                    prediction[heldout_indices], target[heldout_indices], base_prediction[heldout_indices]
                ),
            }
        )

    fixed_config = (0.3, 0.3, 20, "all")
    fixed_row, fixed_base_threshold, fixed_alt_threshold = evaluate_config(all_indices, fixed_config)
    summary = {
        "rows": len(target),
        "base": metrics(base_prediction, target, base_prediction),
        "alternate": metrics(alt_prediction, target, base_prediction),
        "complementarity": {
            "base_wrong_alternate_correct": int((base_prediction.ne(target) & alt_prediction.eq(target)).sum().item()),
            "base_correct_alternate_wrong": int((base_prediction.eq(target) & alt_prediction.ne(target)).sum().item()),
            "either_correct_top1": float(
                (base_prediction.eq(target) | alt_prediction.eq(target)).float().mean().item()
            ),
            "alternate_agreement_rows": int(agreement_mask.sum().item()),
            "alternate_agreement_rate": float(agreement_mask.float().mean().item()),
        },
        "fixed": {
            "base_low_fraction": fixed_config[0],
            "alt_high_fraction": fixed_config[1],
            "require_alt_in_base_topk": fixed_config[2],
            "frequency_mode": fixed_config[3],
            "base_threshold": fixed_base_threshold,
            "alt_threshold": fixed_alt_threshold,
            **fixed_row,
        },
        "gate_best": max(grid, key=lambda row: (row["top1"], -row["changed"])),
        "genus_grouped_oof": {**metrics(oof_prediction, target, base_prediction), "folds": fold_rows},
        "grid": grid,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

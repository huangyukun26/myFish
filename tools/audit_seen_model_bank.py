from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "logits" not in payload:
        raise RuntimeError(f"{path} is not a logits payload")
    return payload


def validate_alignment(reference: dict[str, Any], candidate: dict[str, Any], path: Path) -> None:
    for key in ("image_ids", "class_ids", "classes"):
        if list(candidate[key]) != list(reference[key]):
            raise RuntimeError(f"{path}: {key} does not match reference order")
    if candidate["logits"].shape != reference["logits"].shape:
        raise RuntimeError(
            f"{path}: logits shape {tuple(candidate['logits'].shape)} "
            f"does not match {tuple(reference['logits'].shape)}"
        )


def top1(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)


def row_normalize(logits: torch.Tensor) -> torch.Tensor:
    centered = logits.float() - logits.float().mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True).clamp_min(1e-6)
    return centered / scale


def score_predictions(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    correct = prediction == target
    return {
        "correct": int(correct.sum().item()),
        "accuracy": float(correct.float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=logits.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ensemble-weights", default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    args = parser.parse_args()

    base = load_payload(args.base)
    base_logits = base["logits"].float()
    target = base["class_ids"].long()
    base_prediction = top1(base_logits)
    base_correct = base_prediction == target
    results: dict[str, Any] = {
        "base": {
            "path": str(args.base),
            "rows": int(target.numel()),
            "classes": int(base_logits.shape[1]),
            **score_predictions(base_prediction, target),
        },
        "candidates": {},
        "pairwise_ensemble": {},
    }

    for item in args.candidate:
        name, separator, value = item.partition("=")
        if not separator or not name.strip() or not value.strip():
            raise ValueError(f"Expected name=path, got {item!r}")
        path = Path(value.strip())
        payload = load_payload(path)
        validate_alignment(base, payload, path)
        logits = payload["logits"].float()
        prediction = top1(logits)
        correct = prediction == target
        wins = correct & ~base_correct
        losses = ~correct & base_correct
        oracle = (~base_correct) & correct
        results["candidates"][name.strip()] = {
            "path": str(path),
            **score_predictions(prediction, target),
            "wins_vs_base": int(wins.sum().item()),
            "losses_vs_base": int(losses.sum().item()),
            "net_vs_base": int(wins.sum().item() - losses.sum().item()),
            "oracle_complement_vs_base": int(oracle.sum().item()),
            "base_wrong_rows": int((~base_correct).sum().item()),
        }

        normalized_base = row_normalize(base_logits)
        normalized_candidate = row_normalize(logits)
        ensemble_rows: list[dict[str, Any]] = []
        for raw_weight in args.ensemble_weights.split(","):
            weight = float(raw_weight.strip())
            fused = (1.0 - weight) * normalized_base + weight * normalized_candidate
            fused_prediction = top1(fused)
            fused_correct = fused_prediction == target
            ensemble_rows.append(
                {
                    "candidate_weight": weight,
                    **score_predictions(fused_prediction, target),
                    "wins_vs_base": int((fused_correct & ~base_correct).sum().item()),
                    "losses_vs_base": int((~fused_correct & base_correct).sum().item()),
                    "net_vs_base": int((fused_correct & ~base_correct).sum().item() - (~fused_correct & base_correct).sum().item()),
                }
            )
        results["pairwise_ensemble"][name.strip()] = ensemble_rows

    candidate_names = list(results["candidates"])
    if candidate_names:
        correct_masks = [
            top1(load_payload(Path(next(item.split("=", 1)[1] for item in args.candidate if item.split("=", 1)[0] == name)))["logits"].float()) == target
            for name in candidate_names
        ]
        stacked = torch.stack(correct_masks, dim=0)
        results["union_oracle"] = {
            "candidate_count": len(candidate_names),
            "any_candidate_correct": int(stacked.any(dim=0).sum().item()),
            "base_wrong_any_candidate_correct": int((stacked.any(dim=0) & ~base_correct).sum().item()),
            "base_wrong_rows": int((~base_correct).sum().item()),
            "candidate_names": candidate_names,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

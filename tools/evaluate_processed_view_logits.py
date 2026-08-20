from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def split_is_sealed(image_id: str, seed: int, sealed_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < sealed_fraction


def align_logits(
    payload: dict[str, Any],
    target_ids: list[str],
    target_classes: list[str],
) -> torch.Tensor:
    source_ids = list(payload["image_ids"])
    source_classes = list(payload["classes"])
    row_index = {image_id: index for index, image_id in enumerate(source_ids)}
    missing = [image_id for image_id in target_ids if image_id not in row_index]
    if missing:
        raise RuntimeError(f"{len(missing)} image IDs are missing; first={missing[:5]}")
    logits = payload["logits"].float()[
        torch.tensor([row_index[image_id] for image_id in target_ids])
    ]
    if source_classes == target_classes:
        return logits
    class_index = {label: index for index, label in enumerate(source_classes)}
    missing_classes = [label for label in target_classes if label not in class_index]
    if missing_classes:
        raise RuntimeError(
            f"{len(missing_classes)} target classes are absent; first={missing_classes[:5]}"
        )
    return logits[:, torch.tensor([class_index[label] for label in target_classes])]


def align_targets(
    payload: dict[str, Any],
    target_ids: list[str],
    target_classes: list[str],
) -> torch.Tensor:
    source_ids = list(payload["image_ids"])
    row_index = {image_id: index for index, image_id in enumerate(source_ids)}
    if payload.get("class_ids") is not None and list(payload["classes"]) == target_classes:
        values = payload["class_ids"].long()
        return values[torch.tensor([row_index[image_id] for image_id in target_ids])]
    labels = list(payload.get("labels") or [])
    if not labels:
        raise RuntimeError("Reference payload has neither compatible class_ids nor labels")
    class_to_idx = {label: index for index, label in enumerate(target_classes)}
    return torch.tensor(
        [class_to_idx[labels[row_index[image_id]]] for image_id in target_ids],
        dtype=torch.long,
    )


def metrics(
    base_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    target: torch.Tensor,
    select: torch.Tensor,
) -> dict[str, Any]:
    base_correct = base_pred.eq(target)
    candidate_correct = candidate_pred.eq(target)
    base_count = int(base_correct[select].sum())
    candidate_count = int(candidate_correct[select].sum())
    wins = int((~base_correct & candidate_correct & select).sum())
    losses = int((base_correct & ~candidate_correct & select).sum())
    rows = int(select.sum())
    return {
        "rows": rows,
        "base_correct": base_count,
        "candidate_correct": candidate_count,
        "base_accuracy": base_count / rows if rows else None,
        "candidate_accuracy": candidate_count / rows if rows else None,
        "net": candidate_count - base_count,
        "wins": wins,
        "losses": losses,
        "changed": int((base_pred.ne(candidate_pred) & select).sum()),
        "oracle_complement": wins,
        "oracle_correct": base_count + wins,
        "oracle_accuracy": (base_count + wins) / rows if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--sealed-fraction", type=float, default=0.40)
    parser.add_argument(
        "--alphas",
        default="0,0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1",
    )
    parser.add_argument("--full-val-rows", type=int, default=10790)
    args = parser.parse_args()

    candidate = load(args.candidate_logits)
    base = load(args.base_logits)
    image_ids = list(candidate["image_ids"])
    classes = list(base["classes"])
    candidate_logits = align_logits(candidate, image_ids, classes)
    base_logits = align_logits(base, image_ids, classes)
    target = align_targets(base, image_ids, classes)
    base_pred = base_logits.argmax(dim=1)
    candidate_pred = candidate_logits.argmax(dim=1)

    sealed = torch.tensor(
        [split_is_sealed(image_id, args.seed, args.sealed_fraction) for image_id in image_ids],
        dtype=torch.bool,
    )
    dev = ~sealed
    all_rows = torch.ones(len(image_ids), dtype=torch.bool)

    raw = {
        "all": metrics(base_pred, candidate_pred, target, all_rows),
        "dev": metrics(base_pred, candidate_pred, target, dev),
        "sealed": metrics(base_pred, candidate_pred, target, sealed),
    }

    base_normalized = F.normalize(
        base_logits - base_logits.mean(dim=1, keepdim=True),
        dim=1,
    )
    candidate_normalized = F.normalize(
        candidate_logits - candidate_logits.mean(dim=1, keepdim=True),
        dim=1,
    )
    alpha_values = [float(value) for value in args.alphas.split(",") if value.strip()]
    trials: list[dict[str, Any]] = []
    for alpha in alpha_values:
        blended = (1.0 - alpha) * base_normalized + alpha * candidate_normalized
        pred = blended.argmax(dim=1)
        trials.append(
            {
                "alpha_candidate": alpha,
                "all": metrics(base_pred, pred, target, all_rows),
                "dev": metrics(base_pred, pred, target, dev),
                "sealed": metrics(base_pred, pred, target, sealed),
            }
        )
    selected = max(
        trials,
        key=lambda trial: (
            trial["dev"]["net"],
            -trial["dev"]["losses"],
            -trial["alpha_candidate"],
        ),
    )

    scale = args.full_val_rows / len(image_ids)
    result = {
        "candidate_logits": str(args.candidate_logits),
        "base_logits": str(args.base_logits),
        "rows": len(image_ids),
        "classes": len(classes),
        "split": {
            "seed": args.seed,
            "sealed_fraction": args.sealed_fraction,
            "dev_rows": int(dev.sum()),
            "sealed_rows": int(sealed.sum()),
        },
        "raw": raw,
        "blend": {
            "normalization": "per_row_centered_l2",
            "selection": "maximum dev net; ties prefer fewer dev losses then smaller alpha",
            "selected": selected,
            "trials": trials,
        },
        "diagnostic_extrapolation_to_full_val": {
            "scale": scale,
            "raw_net": raw["all"]["net"] * scale,
            "oracle_complement": raw["all"]["oracle_complement"] * scale,
            "selected_alpha_net": selected["all"]["net"] * scale,
            "selected_alpha_sealed_net": selected["sealed"]["net"]
            * args.full_val_rows
            / max(int(sealed.sum()), 1),
            "warning": "Pilot-only linear extrapolation; not a substitute for full validation.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(image_ids),
                "raw_net": raw["all"]["net"],
                "oracle_complement": raw["all"]["oracle_complement"],
                "selected_alpha": selected["alpha_candidate"],
                "selected_alpha_net": selected["all"]["net"],
                "selected_alpha_sealed_net": selected["sealed"]["net"],
                "out": str(args.out),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

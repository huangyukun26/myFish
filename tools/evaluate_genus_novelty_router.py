from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def genus(name: str) -> str:
    parts = name.split()
    return parts[0] if parts else name


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload) if not isinstance(payload, dict) else list(payload.keys())


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"Expected name=path, got {value}")
    return name.strip(), Path(path.strip())


def build_known_genera(
    known_classes_path: Path,
    holdout_classes_path: Path | None,
    exclude_holdout_genera: bool,
) -> set[str]:
    classes = load_classes(known_classes_path)
    holdout = set(load_classes(holdout_classes_path)) if holdout_classes_path else set()
    heldout_genera = {genus(name) for name in holdout} if exclude_holdout_genera else set()
    train_classes = [
        name
        for name in classes
        if name not in holdout and genus(name) not in heldout_genera
    ]
    return {genus(name) for name in train_classes}


def metrics(pred: torch.Tensor, base: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    base_correct = base == truth
    correct = pred == truth
    changed = pred != base
    wins = (~base_correct & correct).sum().item()
    losses = (base_correct & ~correct).sum().item()
    return {
        "top1": correct.float().mean().item(),
        "base_top1": base_correct.float().mean().item(),
        "changed": changed.sum().item(),
        "wins": wins,
        "losses": losses,
        "net": wins - losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="name=predictions.pt")
    parser.add_argument("--known-classes", type=Path, required=True)
    parser.add_argument("--holdout-classes", type=Path, default=None)
    parser.add_argument("--exclude-holdout-genera", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    known_genera = build_known_genera(
        args.known_classes,
        args.holdout_classes,
        args.exclude_holdout_genera,
    )
    rows: list[dict[str, Any]] = []
    for split_name, path in map(parse_named_path, args.predictions):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        candidates = list(payload["candidates"])
        labels = list(payload["labels"])
        candidate_to_idx = {name: idx for idx, name in enumerate(candidates)}
        missing = [label for label in labels if label not in candidate_to_idx]
        if missing:
            raise RuntimeError(f"{split_name}: {len(missing)} labels missing from candidates")
        truth = torch.tensor([candidate_to_idx[label] for label in labels], dtype=torch.long)
        base = payload["base_pred_indices"].long().cpu()
        rerank = payload["best_pred_indices"].long().cpu()
        candidate_is_novel = torch.tensor(
            [genus(name) not in known_genera for name in candidates],
            dtype=torch.bool,
        )
        base_novel = candidate_is_novel[base]
        rerank_novel = candidate_is_novel[rerank]
        base_genus = [genus(candidates[int(idx)]) for idx in base]
        rerank_genus = [genus(candidates[int(idx)]) for idx in rerank]
        same_genus = torch.tensor(
            [left == right for left, right in zip(base_genus, rerank_genus)],
            dtype=torch.bool,
        )
        gates = {
            "none": torch.ones_like(base_novel),
            "base_novel": base_novel,
            "rerank_novel": rerank_novel,
            "either_novel": base_novel | rerank_novel,
            "both_novel": base_novel & rerank_novel,
            "same_genus_novel": base_novel & rerank_novel & same_genus,
            "move_to_novel_genus": (~base_novel) & rerank_novel & (~same_genus),
        }
        true_novel = torch.tensor(
            [genus(label) not in known_genera for label in labels],
            dtype=torch.bool,
        )
        for gate_name, eligible in gates.items():
            pred = base.clone()
            pred[eligible] = rerank[eligible]
            rows.append(
                {
                    "split": split_name,
                    "source": str(path),
                    "gate": gate_name,
                    "known_genera": len(known_genera),
                    "true_novel_rows": int(true_novel.sum().item()),
                    "eligible_rows": int(eligible.sum().item()),
                    **metrics(pred, base, truth),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "rows": len(rows), "known_genera": len(known_genera)}, indent=2))


if __name__ == "__main__":
    main()

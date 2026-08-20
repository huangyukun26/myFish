from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from train_external_fish_bridge import IdentityAdapter


def load_checkpoint(path: Path) -> dict[str, Any]:
    posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = posix_path


def load_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [name for name, _idx in sorted(payload.items(), key=lambda item: int(item[1]))]
    return list(payload)


def load_candidate(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    return load_classes(path)


def load_text(path: Path) -> tuple[list[str], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload["classes"]), F.normalize(payload["features"].float(), dim=1)


def load_model(path: Path, dim: int) -> IdentityAdapter:
    checkpoint = load_checkpoint(path)
    model = IdentityAdapter(dim)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def score(model: IdentityAdapter, image_features: torch.Tensor, text_features: torch.Tensor, batch_size: int) -> torch.Tensor:
    rows = []
    with torch.inference_mode():
        for start in range(0, image_features.shape[0], batch_size):
            x = F.normalize(image_features[start : start + batch_size].float(), dim=1)
            rows.append(model(x) @ text_features.T)
    return torch.cat(rows, dim=0)


def metrics(pred: torch.Tensor, labels: list[str], classes: list[str], base_pred: torch.Tensor | None = None) -> dict[str, Any]:
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    target = torch.tensor([class_to_idx.get(label, -1) for label in labels], dtype=torch.long)
    known = target >= 0
    pred = pred[known]
    target = target[known]
    correct = pred == target
    result: dict[str, Any] = {
        "known": int(known.sum().item()),
        "correct": int(correct.sum().item()),
        "top1": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "top5": 0.0,
        "top20": 0.0,
    }
    if correct.numel():
        # This function receives top-1 indices; top-k is reported by the
        # caller only when it has the full score matrix.
        result["top5"] = None
    if base_pred is not None:
        base_pred = base_pred[known]
        base_correct = base_pred == target
        result.update(
            {
                "base_correct": int(base_correct.sum().item()),
                "base_top1": float(base_correct.float().mean().item()),
                "changed": int((pred != base_pred).sum().item()),
                "wins": int((correct & ~base_correct).sum().item()),
                "losses": int((~correct & base_correct).sum().item()),
                "net": int((correct & ~base_correct).sum().item() - (~correct & base_correct).sum().item()),
            }
        )
    return result


def evaluate_payload(
    name: str,
    feature_path: Path,
    label_classes: list[str],
    candidate_path: Path | None,
    text_classes: list[str],
    text_features: torch.Tensor,
    base_model: IdentityAdapter,
    bridge_model: IdentityAdapter,
    out_dir: Path,
    batch_size: int,
    strong_path: Path | None,
) -> dict[str, Any]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    labels = list(payload.get("labels", [""] * len(payload["image_ids"])))
    candidates = load_candidate(candidate_path, label_classes) if candidate_path else label_classes
    text_index = {name: idx for idx, name in enumerate(text_classes)}
    missing = [name for name in candidates if name not in text_index]
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} candidates missing from text features; first={missing[:5]}")
    candidate_text = text_features[torch.tensor([text_index[value] for value in candidates], dtype=torch.long)]
    base_scores = score(base_model, payload["features"], candidate_text, batch_size)
    bridge_scores = score(bridge_model, payload["features"], candidate_text, batch_size)
    base_pred = base_scores.argmax(dim=1)
    bridge_pred = bridge_scores.argmax(dim=1)
    row: dict[str, Any] = {
        "name": name,
        "features": str(feature_path),
        "candidate_classes": len(candidates),
        "base": metrics(base_pred, labels, candidates),
        "bridge": metrics(bridge_pred, labels, candidates, base_pred),
    }
    if strong_path is not None:
        strong = torch.load(strong_path, map_location="cpu", weights_only=False)
        strong_pred = strong["logits"].argmax(dim=1)
        if list(strong["image_ids"]) != list(payload["image_ids"]):
            raise RuntimeError(f"{name}: strong reference image order differs")
        strong_classes = list(strong["classes"])
        strong_names = [strong_classes[int(idx)] for idx in strong_pred.tolist()]
        strong_idx = torch.tensor([candidates.index(value) if value in candidates else -1 for value in strong_names], dtype=torch.long)
        known = strong_idx >= 0
        bridge_known = bridge_pred[known]
        target = torch.tensor([candidates.index(label) if label in candidates else -1 for label in labels], dtype=torch.long)[known]
        strong_candidate = strong_idx[known]
        bridge_correct = bridge_known == target
        strong_correct = strong_candidate == target
        row["bridge_vs_strong"] = {
            "rows_with_strong_label_in_candidates": int(known.sum().item()),
            "strong_correct": int(strong_correct.sum().item()),
            "bridge_correct": int(bridge_correct.sum().item()),
            "wins": int((bridge_correct & ~strong_correct).sum().item()),
            "losses": int((~bridge_correct & strong_correct).sum().item()),
            "net": int((bridge_correct & ~strong_correct).sum().item() - (~bridge_correct & strong_correct).sum().item()),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_ids": payload["image_ids"],
            "labels": labels,
            "candidates": candidates,
            "base_pred": base_pred,
            "bridge_pred": bridge_pred,
        },
        out_dir / f"{name}.pt",
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--bridge-adapter", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--seen-features", type=Path, default=None)
    parser.add_argument("--seen-classes", type=Path, default=None)
    parser.add_argument("--strong-seen-logits", type=Path, default=None)
    parser.add_argument("--pseudo", action="append", default=[], help="name=features.pt[:candidate.json]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    text_classes, text_features = load_text(args.text_features)
    dim = int(text_features.shape[1])
    base_model = load_model(args.base_adapter, dim)
    bridge_model = load_model(args.bridge_adapter, dim)
    rows = []
    if args.seen_features is not None:
        if args.seen_classes is None:
            raise ValueError("--seen-classes is required with --seen-features")
        rows.append(
            evaluate_payload(
                "seen_val",
                args.seen_features,
                load_classes(args.seen_classes),
                None,
                text_classes,
                text_features,
                base_model,
                bridge_model,
                args.out.parent,
                args.batch_size,
                args.strong_seen_logits,
            )
        )
    for item in args.pseudo:
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Expected name=features.pt[:candidate.json], got {item}")
        parts = value.split(":", 1)
        feature_path = Path(parts[0])
        candidate_path = Path(parts[1]) if len(parts) == 2 and parts[1] else None
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        labels = sorted(set(str(label) for label in payload.get("labels", []) if label))
        rows.append(
            evaluate_payload(
                name,
                feature_path,
                labels,
                candidate_path,
                text_classes,
                text_features,
                base_model,
                bridge_model,
                args.out.parent,
                args.batch_size,
                None,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

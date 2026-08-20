from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch


def load_classes(path: Optional[Path], text_classes: List[str]) -> List[str]:
    if path is None:
        return text_classes
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.keys())
    return list(data)


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_candidate_features(path: Path, candidates: List[str]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    class_to_idx = {name: idx for idx, name in enumerate(payload["classes"])}
    missing = [name for name in candidates if name not in class_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} candidates missing from {path}; first={missing[:10]}")
    idx = torch.tensor([class_to_idx[name] for name in candidates], dtype=torch.long)
    return normalize_features(payload["features"].float()[idx])


def genus(name: str) -> str:
    return name.split()[0] if name.split() else name


def top1(logits: torch.Tensor, candidates: List[str]) -> List[str]:
    indices = logits.argmax(dim=1).tolist()
    return [candidates[int(idx)] for idx in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--base-text-features", type=Path, required=True)
    parser.add_argument("--extra-text-features", type=Path, required=True)
    parser.add_argument("--candidate-classes", type=Path, required=True)
    parser.add_argument("--weight-a", type=float, default=0.0, help="Extra/taxon weight for prediction A.")
    parser.add_argument("--weight-b", type=float, default=0.5, help="Extra/taxon weight for prediction B.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    image_payload = torch.load(args.image_features, map_location="cpu", weights_only=False)
    image_features = normalize_features(image_payload["features"].float())
    labels = list(image_payload["labels"])
    base_payload = torch.load(args.base_text_features, map_location="cpu", weights_only=False)
    candidates = load_classes(args.candidate_classes, list(base_payload["classes"]))
    base_features = load_candidate_features(args.base_text_features, candidates)
    extra_features = load_candidate_features(args.extra_text_features, candidates)
    base_logits = image_features @ base_features.T
    extra_logits = image_features @ extra_features.T
    logits_a = base_logits * (1.0 - args.weight_a) + extra_logits * args.weight_a
    logits_b = base_logits * (1.0 - args.weight_b) + extra_logits * args.weight_b
    pred_a = top1(logits_a, candidates)
    pred_b = top1(logits_b, candidates)

    changed = []
    a_correct = b_correct = both_correct = both_wrong = b_wins = b_loses = 0
    same_genus_changed = 0
    for image_id, label, a, b in zip(image_payload["image_ids"], labels, pred_a, pred_b):
        ac = a == label
        bc = b == label
        a_correct += int(ac)
        b_correct += int(bc)
        both_correct += int(ac and bc)
        both_wrong += int((not ac) and (not bc))
        b_wins += int((not ac) and bc)
        b_loses += int(ac and (not bc))
        if a != b:
            same_genus = genus(a) == genus(b)
            same_genus_changed += int(same_genus)
            changed.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "pred_a": a,
                    "pred_b": b,
                    "a_correct": ac,
                    "b_correct": bc,
                    "same_genus": same_genus,
                }
            )

    summary = {
        "image_features": str(args.image_features),
        "candidate_classes": str(args.candidate_classes),
        "weight_a": args.weight_a,
        "weight_b": args.weight_b,
        "rows": len(labels),
        "changed": len(changed),
        "changed_ratio": len(changed) / len(labels),
        "same_genus_changed": same_genus_changed,
        "same_genus_changed_ratio": (same_genus_changed / len(changed)) if changed else 0.0,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "a_top1": a_correct / len(labels),
        "b_top1": b_correct / len(labels),
        "net_correct_delta": b_correct - a_correct,
        "b_wins": b_wins,
        "b_loses": b_loses,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "changed_examples": changed[:200]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

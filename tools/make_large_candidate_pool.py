from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


def load_classes(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        try:
            return [name for name, _idx in sorted(data.items(), key=lambda item: int(item[1]))]
        except Exception:
            return list(data.keys())
    return list(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-classes", type=Path, required=True)
    parser.add_argument("--source-classes", type=Path, default=Path("work/full_manifests/all_classes.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=11598)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    true_classes = load_classes(args.true_classes)
    source_classes = load_classes(args.source_classes)
    true_set = set(true_classes)
    missing_true = [name for name in true_classes if name not in set(source_classes)]
    if missing_true:
        raise RuntimeError(f"True classes missing from source; first={missing_true[:10]}")
    if args.candidate_count < len(true_classes):
        raise ValueError("--candidate-count must be >= number of true classes")
    if args.candidate_count > len(source_classes):
        raise ValueError("--candidate-count must be <= number of source classes")

    distractors = [name for name in source_classes if name not in true_set]
    rng = random.Random(args.seed)
    sampled = rng.sample(distractors, args.candidate_count - len(true_classes))
    candidate_set = set(true_classes) | set(sampled)
    ordered = [name for name in source_classes if name in candidate_set]
    candidate_to_idx = {name: idx for idx, name in enumerate(ordered)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidate_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "true_classes": str(args.true_classes),
        "source_classes": str(args.source_classes),
        "out": str(args.out),
        "candidate_count": len(ordered),
        "true_class_count": len(true_classes),
        "distractor_count": len(sampled),
        "seed": args.seed,
    }
    (args.out.parent / f"{args.out.stem}.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

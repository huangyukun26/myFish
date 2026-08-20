from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import statistics
from pathlib import Path
from typing import Any, Dict


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)


def percentile(values, q: float):
    values = sorted(values)
    if not values:
        return None
    idx = max(0, min(len(values) - 1, int(q * len(values)) - 1))
    return values[idx]


def inspect(dataset_root: Path) -> Dict[str, Any]:
    labels = json.loads((dataset_root / "label_train.json").read_text(encoding="utf-8"))
    descriptions = json.loads((dataset_root / "descriptions.json").read_text(encoding="utf-8"))
    all_classes = load_pickle(dataset_root / "all_classes.pkl")
    train_split = load_pickle(dataset_root / "splits" / "train.pkl")
    test_split = load_pickle(dataset_root / "splits" / "test.pkl")
    unseen_split = load_pickle(dataset_root / "splits" / "unseen.pkl")

    counts = collections.Counter(labels.values())
    image_counts = list(counts.values())
    all_class_set = set(all_classes)
    train_class_set = set(counts)
    unseen_candidate_classes = all_class_set - train_class_set
    seen_genera = {name.split()[0] for name in train_class_set}
    unseen_genera = {name.split()[0] for name in unseen_candidate_classes}
    desc_word_lens = [len(re.findall(r"\w+", text)) for text in descriptions.values()]

    summary = {
        "train_images": len(labels),
        "train_split_images": len(train_split),
        "test_split_images": len(test_split),
        "unseen_split_images": len(unseen_split),
        "train_split_matches_labels": set(train_split) == set(labels),
        "split_intersections": {
            "train_test": len(set(train_split) & set(test_split)),
            "train_unseen": len(set(train_split) & set(unseen_split)),
            "test_unseen": len(set(test_split) & set(unseen_split)),
        },
        "all_classes": len(all_classes),
        "train_classes": len(train_class_set),
        "unseen_candidate_classes": len(unseen_candidate_classes),
        "description_coverage_all_classes": len(all_class_set & set(descriptions)),
        "description_extra_classes": len(set(descriptions) - all_class_set),
        "class_image_count": {
            "min": min(image_counts),
            "median": statistics.median(image_counts),
            "mean": round(statistics.mean(image_counts), 3),
            "p90": percentile(image_counts, 0.90),
            "p95": percentile(image_counts, 0.95),
            "p99": percentile(image_counts, 0.99),
            "max": max(image_counts),
        },
        "long_tail": {f"classes_le_{n}": sum(v <= n for v in image_counts) for n in [2, 3, 5, 10, 20, 50]},
        "seen_genera": len(seen_genera),
        "unseen_candidate_genera": len(unseen_genera),
        "unseen_candidate_genera_seen_overlap": len(unseen_genera & seen_genera),
        "unseen_candidate_classes_with_seen_genus": sum(name.split()[0] in seen_genera for name in unseen_candidate_classes),
        "description_word_count": {
            "median": statistics.median(desc_word_lens),
            "mean": round(statistics.mean(desc_word_lens), 3),
            "p90": percentile(desc_word_lens, 0.90),
            "max": max(desc_word_lens),
        },
        "top_train_classes": counts.most_common(20),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = inspect(args.dataset_root)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


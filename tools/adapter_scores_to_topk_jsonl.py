from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.score_file, map_location="cpu", weights_only=False)
    image_ids = list(payload["image_ids"])
    predictions = list(payload["predictions"])
    scores = list(payload["base_scores"])
    labels = list(payload.get("labels", [""] * len(image_ids)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for image_id, label, preds, row_scores in zip(image_ids, labels, predictions, scores):
            true_rank = ""
            if label:
                try:
                    true_rank = list(preds).index(label) + 1
                except ValueError:
                    true_rank = len(preds) + 1
            fp.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "label": label,
                        "predictions": list(preds),
                        "scores": [float(value) for value in row_scores],
                        "true_rank": true_rank,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {"score_file": str(args.score_file), "out": str(args.out), "rows": len(image_ids)}
    (args.out.parent / f"{args.out.stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit high-similarity train-to-val/test retrieval on paired BioCLIP views.

This is a diagnostic, not a submission builder.  It measures how precise a
nearest-neighbour label is at increasingly strict cosine thresholds and, on
test_seen, reports how often that label disagrees with the supplied base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def paired_search(
    query_hflip: torch.Tensor,
    query_letterbox: torch.Tensor,
    gallery_hflip: torch.Tensor,
    gallery_letterbox: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gallery_hflip = F.normalize(gallery_hflip.float(), dim=1).to(device)
    gallery_letterbox = F.normalize(gallery_letterbox.float(), dim=1).to(device)
    query_hflip = F.normalize(query_hflip.float(), dim=1)
    query_letterbox = F.normalize(query_letterbox.float(), dim=1)

    values: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, len(query_hflip), batch_size):
        qh = query_hflip[start : start + batch_size].to(device)
        ql = query_letterbox[start : start + batch_size].to(device)
        similarities = 0.5 * (qh @ gallery_hflip.T + ql @ gallery_letterbox.T)
        value, index = similarities.max(dim=1)
        values.append(value.cpu())
        indices.append(index.cpu())
    return torch.cat(values), torch.cat(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hflip-root", type=Path, required=True)
    parser.add_argument("--letterbox-root", type=Path, required=True)
    parser.add_argument("--base-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    hflip = {
        split: load_cache(args.hflip_root / f"{split}_hflip.pt")
        for split in ("train", "val", "test_seen")
    }
    letterbox = {
        split: load_cache(args.letterbox_root / f"{split}_letterbox.pt")
        for split in ("train", "val", "test_seen")
    }
    for split in hflip:
        if hflip[split]["image_ids"] != letterbox[split]["image_ids"]:
            raise ValueError(f"Image order mismatch for {split}")

    with args.base_prediction.open(encoding="utf-8") as handle:
        base = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = (0.99999, 0.9999, 0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.90, 0.85)
    gallery_labels = hflip["train"]["labels"]
    gallery_ids = hflip["train"]["image_ids"]
    report: dict[str, object] = {"device": str(device), "thresholds": {}}

    for split in ("val", "test_seen"):
        values, indices = paired_search(
            hflip[split]["features"],
            letterbox[split]["features"],
            hflip["train"]["features"],
            letterbox["train"]["features"],
            device,
            args.batch_size,
        )
        predictions = [gallery_labels[index] for index in indices.tolist()]
        image_ids = hflip[split]["image_ids"]
        split_report: dict[str, object] = {
            "rows": len(values),
            "similarity_quantiles": {
                str(q): float(torch.quantile(values, torch.tensor(q)))
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0)
            },
            "thresholds": {},
        }
        for threshold in thresholds:
            selected = torch.where(values >= threshold)[0].tolist()
            item: dict[str, object] = {"rows": len(selected)}
            if split == "val":
                true_labels = hflip[split]["labels"]
                correct = sum(predictions[row] == true_labels[row] for row in selected)
                item.update(
                    correct=correct,
                    accuracy=(correct / len(selected) if selected else None),
                )
            else:
                disagreements = [row for row in selected if predictions[row] != base[image_ids[row]]]
                item.update(disagree_base=len(disagreements))
            split_report["thresholds"][str(threshold)] = item

        if split == "test_seen":
            disagreements = [
                {
                    "image_id": image_ids[row],
                    "similarity": float(values[row]),
                    "base_label": base[image_ids[row]],
                    "nn_label": predictions[row],
                    "gallery_image_id": gallery_ids[int(indices[row])],
                }
                for row in range(len(values))
                if predictions[row] != base[image_ids[row]]
            ]
            disagreements.sort(key=lambda item: item["similarity"], reverse=True)
            split_report["top_base_disagreements"] = disagreements[:500]
        report[split] = split_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

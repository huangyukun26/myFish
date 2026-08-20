from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("avg3", "concat3"), default="avg3")
    args = parser.parse_args()
    x = torch.load(args.input, map_location="cpu", weights_only=False)
    labels = list(x["labels"])
    genera = sorted({label.split()[0] for label in labels})
    pos = {name: i for i, name in enumerate(genera)}
    out = dict(x)
    features = x["features"].float()
    if features.shape[1] == 3072 and args.mode == "avg3":
        features = F.normalize(features.reshape(len(features), 3, 1024), dim=2).mean(1)
    out["features"] = F.normalize(features, dim=1)
    out["species_labels"] = labels
    out["labels"] = [label.split()[0] for label in labels]
    out["class_ids"] = torch.tensor([pos[label] for label in out["labels"]], dtype=torch.long)
    out["classes"] = genera
    args.out.parent.mkdir(parents=True, exist_ok=True); torch.save(out, args.out)
    print({"rows": len(labels), "genera": len(genera), "dim": out["features"].shape[1], "out": str(args.out)})


if __name__ == "__main__": main()

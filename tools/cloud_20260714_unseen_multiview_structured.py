from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("runs/cloud_20260714/unseen_multiview_structured")
STRUCT = Path("runs/structural_backbones_20260713")
KINDS = {
    "species": {
        "base": Path("work/cloud_20260713/pseudo_species_hflip_letterbox_avg.pt"),
        "five": Path("runs/cloud_20260714/bioclip_fivecrop_unseen_gate/species42.pt"),
        "text": STRUCT / "structured_taxonomy_combo/species_taxon95_structured05.pt",
        "adapter": STRUCT / "unseen_adapter/species_excluded_taxon_b075.pt",
        "pseudo": Path("work/pseudo_unseen/species_1000_seed42"),
    },
    "genus": {
        "base": Path("work/cloud_20260713/pseudo_genus_hflip_letterbox_avg.pt"),
        "five": Path("runs/cloud_20260714/bioclip_fivecrop_unseen_gate/genus42.pt"),
        "text": STRUCT / "structured_taxonomy_combo/genus_taxon95_structured05.pt",
        "adapter": STRUCT / "unseen_adapter/genus_excluded_taxon_b075.pt",
        "pseudo": Path("work/pseudo_unseen/genus_1000_seed42"),
    },
}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    weights = [0.0, 0.05, 0.1, 0.2, 0.3]
    for kind, cfg in KINDS.items():
        base = torch.load(cfg["base"], map_location="cpu", weights_only=False)
        five = torch.load(cfg["five"], map_location="cpu", weights_only=False)
        assert base["image_ids"] == five["image_ids"]
        for weight in weights:
            payload = dict(base)
            payload["features"] = F.normalize((1 - weight) * F.normalize(base["features"].float(), dim=1)
                                                + weight * F.normalize(five["features"].float(), dim=1), dim=1)
            cache = ROOT / f"{kind}_w{int(weight * 100):02d}.pt"; torch.save(payload, cache)
            for seed in (42, 43, 44):
                out = ROOT / f"{kind}{seed}_w{int(weight * 100):02d}"
                command = [
                    "python", "tools/evaluate_taxonomy_local_rerank.py",
                    "--image-features", str(cache), "--base-text-features", str(cfg["text"]),
                    "--adapter-text-features", str(cfg["adapter"]),
                    "--candidate-classes", str(cfg["pseudo"] / f"candidate_classes_11598_seed{seed}.json"),
                    "--known-classes", "work/full_manifests/seen_class_to_idx.json",
                    "--holdout-classes", str(cfg["pseudo"] / "classes.json"),
                    "--novelty-gate-grid", "move_to_novel_genus", "--topk-grid", "50",
                    "--branch-source-grid", "adapter_logits", "--species-weight-grid", "0.5,1.0",
                    "--mode-grid", "pooled_genus", "--support-weight-grid", "0.1,0.25",
                    "--support-temperature-grid", "1.0", "--out-dir", str(out),
                ]
                subprocess.run(command, check=True)

    configs: dict[tuple, list[dict]] = {}
    for kind in KINDS:
        for weight in weights:
            for seed in (42, 43, 44):
                path = ROOT / f"{kind}{seed}_w{int(weight * 100):02d}/sweep.csv"
                for row in csv.DictReader(path.open(encoding="utf-8")):
                    key = (weight, row["config_id"])
                    record = {"split": f"{kind}{seed}", "net": int(row["net"]),
                              "wins": int(row["wins"]), "losses": int(row["losses"]),
                              "changed": int(row["changed"]), "top1": float(row["top1"])}
                    configs.setdefault(key, []).append(record)
    rows = []
    for (weight, config_id), splits in configs.items():
        nets = [x["net"] for x in splits]
        rows.append({"weight": weight, "config_id": config_id, "total_net": sum(nets),
                     "worst_net": min(nets), "nonnegative": sum(x >= 0 for x in nets),
                     "total_changed": sum(x["changed"] for x in splits),
                     "total_wins": sum(x["wins"] for x in splits),
                     "total_losses": sum(x["losses"] for x in splits), "splits": splits})
    rows.sort(key=lambda x: (x["worst_net"], x["total_net"], -x["total_changed"]), reverse=True)
    (ROOT / "robust_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows[:20], indent=2))


if __name__ == "__main__": main()

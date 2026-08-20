from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_embedding_mlp_classifier import MLPClassifier, normalize
from transductive_active_sinkhorn import load_text_features


MODE = os.environ.get("GENUS_ENSEMBLE", "single")
INPUT_MODE = os.environ.get("GENUS_INPUT", "avg3")
OUT = Path("runs/cloud_20260714/unseen_genus_gate" + ("_ensemble" if MODE == "three" else "") + ("_concat3" if INPUT_MODE == "concat3" else ""))


def infer(model, x, device):
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 512): parts.append(model(normalize(x[start:start + 512]).to(device)).cpu())
    return torch.cat(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    paths = [Path("runs/cloud_20260714/genus_gate/mlp_concat3/best_model.pt") if INPUT_MODE == "concat3" else Path("runs/cloud_20260714/genus_gate/mlp/best_model.pt")]
    if MODE == "three":
        paths += [Path(f"runs/cloud_20260714/genus_gate/mlp_seed{s}/best_model.pt") for s in (2051, 2052)]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    ck = checkpoints[0]; known = {g: i for i, g in enumerate(ck["classes"])}
    base_image = torch.load("work/cloud_20260713/pseudo_species_hflip_letterbox_avg.pt", map_location="cpu", weights_only=False)
    five = torch.load("runs/cloud_20260714/bioclip_fivecrop_unseen_gate/species42.pt", map_location="cpu", weights_only=False)
    assert base_image["image_ids"] == five["image_ids"]
    base_features = F.normalize(base_image["features"].float(), dim=1); five_features = F.normalize(five["features"].float(), dim=1)
    semantic_query = F.normalize((2 * base_features + five_features) / 3, dim=1)
    query = torch.cat([base_features, base_features, five_features], dim=1) if INPUT_MODE == "concat3" else semantic_query
    logits = []
    for checkpoint in checkpoints:
        assert checkpoint["classes"] == ck["classes"]
        arch = checkpoint["arch"]; model = MLPClassifier(arch["in_dim"], arch["hidden_dim"], len(ck["classes"]), arch["dropout"])
        model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
        value = infer(model, query, device); value = (value - value.mean(1, keepdim=True)) / value.std(1, keepdim=True).clamp_min(1e-6)
        logits.append(value)
    genus_logits = torch.stack(logits).mean(0); gv, gi = genus_logits.topk(2, 1); margin = (gv[:, 0] - gv[:, 1]).numpy()
    predicted_genus = [ck["classes"][int(i)] for i in gi[:, 0]]
    text_path = Path("runs/structural_backbones_20260713/structured_taxonomy_combo/species_taxon95_structured05.pt")
    split_data = []
    for seed in (42, 43, 44):
        p = torch.load(f"runs/structural_backbones_20260713/structured_taxonomy_combo/species{seed}/predictions.pt", map_location="cpu", weights_only=False)
        candidates = p["candidates"]; text = load_text_features(text_path, candidates)
        scores = semantic_query.to(device) @ text.to(device).T
        values, top = scores.topk(100, 1); del scores
        truth_pos = {name: i for i, name in enumerate(candidates)}
        truth = torch.tensor([truth_pos[x] for x in p["labels"]]); base = p["base_pred_indices"].long()
        split_data.append({"seed": seed, "candidates": candidates, "top": top.cpu(), "truth": truth, "base": base})

    rows = []
    for topk in (10, 20, 50, 100):
        for quantile in (0.0, .25, .5, .75, .9, .95):
            threshold = float(np.quantile(margin, quantile)); nets = []; details = []
            for data in split_data:
                candidate_genera = [x.split()[0] for x in data["candidates"]]
                alt = data["base"].clone(); eligible = torch.zeros(len(alt), dtype=torch.bool)
                for row in range(len(alt)):
                    if margin[row] < threshold: continue
                    genus = predicted_genus[row]
                    base_genus = candidate_genera[int(data["base"][row])]
                    if genus == base_genus or base_genus not in known: continue
                    for idx in data["top"][row, :topk]:
                        if candidate_genera[int(idx)] == genus:
                            alt[row] = idx; eligible[row] = True; break
                delta = alt.eq(data["truth"]).long() - data["base"].eq(data["truth"]).long()
                net = int(delta[eligible].sum()); nets.append(net)
                details.append({"split": f"species{data['seed']}", "changed": int((eligible & alt.ne(data["base"])).sum()),
                                "wins": int((delta[eligible] == 1).sum()), "losses": int((delta[eligible] == -1).sum()), "net": net})
                if topk == 100 and quantile in (0.0, 0.25, 0.5):
                    torch.save({"base": data["base"], "alternate": alt, "truth": data["truth"],
                                "eligible": eligible, "candidates": data["candidates"]},
                               OUT / f"species{data['seed']}_q{int(quantile * 100)}.pt")
            rows.append({"topk": topk, "margin_quantile": quantile, "threshold": threshold,
                         "nets": nets + [0, 0, 0], "worst_net": min(nets + [0]), "total_net": sum(nets), "details": details})
    rows.sort(key=lambda x: (x["worst_net"], x["total_net"], -sum(d["changed"] for d in x["details"])), reverse=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows[:20], indent=2))


if __name__ == "__main__": main()

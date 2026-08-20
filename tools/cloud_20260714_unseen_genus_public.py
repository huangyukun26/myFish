from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from train_embedding_mlp_classifier import MLPClassifier, normalize
from transductive_active_sinkhorn import load_text_features


MODE = os.environ.get("GENUS_ENSEMBLE", "single")
OUT = Path("runs/cloud_20260714/unseen_genus_gate_public" + ("_ensemble" if MODE == "three" else ""))
BASE_JSON = Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    paths = [Path("runs/cloud_20260714/genus_gate/mlp/best_model.pt")]
    if MODE == "three": paths += [Path(f"runs/cloud_20260714/genus_gate/mlp_seed{s}/best_model.pt") for s in (2051, 2052)]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    ck = checkpoints[0]; known = set(ck["classes"])
    base_image = torch.load("runs/cloud_20260714/unseen_structured_public/test_unseen_hflip_letterbox_avg.pt", map_location="cpu", weights_only=False)
    five = torch.load("runs/cloud_20260714/bioclip_fivecrop_unseen_public/test_unseen.pt", map_location="cpu", weights_only=False)
    assert base_image["image_ids"] == five["image_ids"]
    query = F.normalize((2 * F.normalize(base_image["features"].float(), dim=1) + F.normalize(five["features"].float(), dim=1)) / 3, dim=1)
    genus_models = []
    for checkpoint in checkpoints:
        arch = checkpoint["arch"]; model = MLPClassifier(arch["in_dim"], arch["hidden_dim"], len(ck["classes"]), arch["dropout"])
        model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval(); parts = []
        with torch.inference_mode():
            for start in range(0, len(query), 512): parts.append(model(normalize(query[start:start + 512]).to(device)).cpu())
        value = torch.cat(parts); value = (value - value.mean(1, keepdim=True)) / value.std(1, keepdim=True).clamp_min(1e-6)
        genus_models.append(value)
    genus_logits = torch.stack(genus_models).mean(0); gv, gi = genus_logits.topk(2, 1); margin = gv[:, 0] - gv[:, 1]
    predicted_genus = [ck["classes"][int(i)] for i in gi[:, 0]]

    tool = torch.load("runs/cloud_20260714/unseen_structured_public/robust_v2/predictions.pt", map_location="cpu", weights_only=False)
    candidates = tool["candidates"]; assert tool["image_ids"] == base_image["image_ids"]
    text = load_text_features(Path("runs/structural_backbones_20260713/structured_taxonomy_combo/species_taxon95_structured05.pt"), candidates).to(device)
    top_parts = []
    with torch.inference_mode():
        for start in range(0, len(query), 256): top_parts.append((query[start:start + 256].to(device) @ text.T).topk(100, 1).indices.cpu())
    top = torch.cat(top_parts); candidate_genera = [x.split()[0] for x in candidates]
    current_json = json.loads(BASE_JSON.read_text(encoding="utf-8")); pos = {x: i for i, x in enumerate(candidates)}
    current = torch.tensor([pos[current_json[i]] for i in tool["image_ids"]]); toolbase = tool["base_pred_indices"].long()
    alt = current.clone(); gate = torch.zeros(len(current), dtype=torch.bool)
    for row in range(len(current)):
        pg = predicted_genus[row]; cg = candidate_genera[int(current[row])]
        if cg not in known or pg == cg: continue
        for idx in top[row]:
            if candidate_genera[int(idx)] == pg:
                alt[row] = idx; gate[row] = True; break
    summary = {"rows": len(current), "raw_gate": int(gate.sum()), "top_changed": int((gate & alt.ne(current)).sum())}
    masks = {}
    thresholds = (("q0", 0.0), ("q25", 0.8573753237724304), ("q35", float(torch.quantile(margin, 0.35))), ("q40", float(torch.quantile(margin, 0.40))), ("q50", 1.6326310634613037)) if MODE == "three" else (("q0", 0.0), ("q25", 2.1605011224746704), ("q35", float(torch.quantile(margin, 0.35))), ("q40", float(torch.quantile(margin, 0.40))), ("q50", 4.225916862487793))
    for tag, threshold in thresholds:
        threshold_gate = gate & margin.ge(threshold)
        masks[f"{tag}_strict_current_equals_toolbase"] = threshold_gate & current.eq(toolbase)
        masks[f"{tag}_current_known_genus"] = threshold_gate
    for name, mask in masks.items():
        pred = current.clone(); pred[mask] = alt[mask]; out = dict(current_json)
        for image_id, idx in zip(tool["image_ids"], pred): out[image_id] = candidates[int(idx)]
        folder = OUT / name; folder.mkdir(parents=True, exist_ok=True); path = folder / "prediction.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(folder / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(path, "prediction.json")
        summary[name] = {"eligible": int(mask.sum()), "changed": sum(out[k] != current_json[k] for k in current_json)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

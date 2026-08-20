from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cloud_20260714_full_fivecrop_ensemble as ensemble_module
from cloud_20260714_full_fivecrop_ensemble import infer, make_package, regressor
from cloud_20260714_seen_dino_router import aligned_text, fused_topk, prototypes, zscore
from cloud_20260714_seen_tri_router import ROOT, feats


NEW = Path("runs/cloud_20260714/concat_fivecrop_gate")
OUT = Path("runs/cloud_20260714/seen_fullfit_ensemble")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); ensemble_module.OUT = OUT; device = torch.device("cuda")
    fixed = torch.load(ROOT / "concat_balanced_gate/fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    val_alt = torch.load("runs/cloud_20260714/seen_full_fivecrop_ensemble_five/val_fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    da = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    db = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    x, base, alt, _, _ = feats(fixed, val_alt, da, db, fixed["full_class_counts"].long())
    truth = fixed["class_ids"].long(); y = (alt.eq(truth).long() - base.eq(truth).long()).numpy()

    test = torch.load(NEW / "test_seen.pt", map_location="cpu", weights_only=False); logits = []; ck = None
    for seed in (2033, 2034, 2035):
        out, ck = infer(NEW / f"fullfit_seed{seed}/best_model.pt", test["features"], device); logits.append(zscore(out))
    ensemble = torch.stack(logits).mean(0)
    train = torch.load(NEW / "full_train_64259.pt", map_location="cpu", weights_only=False)
    proto = prototypes(train["features"][:, :1024], train["class_ids"].long(), len(ck["classes"]))
    text = aligned_text(Path("work/clip_text_features/seen_bioclip25_taxon.pt"), ck["classes"])
    values, indices = fused_topk(ensemble, test["features"][:, :1024], proto, text, device)
    public_alt = {"topk_values": values, "topk_indices": indices, "classes": ck["classes"], "image_ids": test["image_ids"]}
    torch.save(public_alt, OUT / "public_fusion.pt")

    public_base = torch.load(OUT.parent / "seen_dino_router/public_fusion.pt", map_location="cpu", weights_only=False)
    pda = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    pdb = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    px, pbase, palt, _, _ = feats(public_base, public_alt, pda, pdb, fixed["full_class_counts"].long())
    score = regressor().fit(x, y).predict(px)
    cur = json.loads(Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json").read_text(encoding="utf-8"))
    pos = {name: i for i, name in enumerate(ck["classes"])}; current = torch.tensor([pos[cur[i]] for i in test["image_ids"]]); agree = current.eq(pbase)
    old = torch.load("runs/cloud_20260714/seen_full_fivecrop_ensemble_five/public_fusion.pt", map_location="cpu", weights_only=False)
    public_rows = {}
    for pct in (20, 15, 12, 10, 5, 3, 2, 1):
        threshold = float(np.quantile(score, 1 - pct / 100)); mask = torch.from_numpy(score >= threshold) & agree
        public_rows[str(pct)] = {"selected": int(mask.sum()), "changed": make_package(cur, test["image_ids"], ck["classes"], current, palt, mask, f"top{pct}pct")}
    summary = {"fullfit_vs_holdout_top1_agreement": float(indices[:, 0].eq(old["topk_indices"][:, 0]).float().mean()),
               "fullfit_vs_base_agreement": float(indices[:, 0].eq(pbase).float().mean()), "public": public_rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

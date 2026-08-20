from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from cloud_20260714_build_v6_candidates import oof_score, regressor
from cloud_20260714_seen_tri_router import ROOT, feats, fold


def main() -> None:
    fixed = torch.load(ROOT / "concat_balanced_gate/fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    truth = fixed["class_ids"].long(); base = fixed["logits"].argmax(1)
    groups = np.array([fold(x) for x in fixed["labels"]])
    da = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    db = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)

    experts = {}
    dino = torch.load("runs/cloud_20260714/seen_meta_router_tri_best/oof_router_scores.pt", map_location="cpu", weights_only=False)
    experts["dino"] = (dino["base"], dino["alternate"],
                       torch.from_numpy((dino["oof_score"].numpy() >= max(dino["thresholds"])) & dino["guard"].numpy().astype(bool)))
    tri = torch.load("runs/cloud_20260714/seen_tri_router/oof_router_scores.pt", map_location="cpu", weights_only=False)
    experts["tri"] = (tri["base"], tri["alternate"],
                      torch.from_numpy((tri["oof_score"].numpy() >= max(tri["thresholds"])) & tri["guard"].numpy().astype(bool)))

    scores = {}
    full_path = os.environ.get("FULL_ALT", "runs/cloud_20260714/seen_full_fivecrop_router/val_fixed_fusion_logits.pt")
    for name, path in [("full", full_path),
                       ("crop", "runs/cloud_20260714/bioclip_fivecrop_priority/val_fused_topk.pt")]:
        alt = torch.load(path, map_location="cpu", weights_only=False)
        x, b, a, _, _ = feats(fixed, alt, da, db, fixed["full_class_counts"].long())
        y = (a.eq(truth).long() - b.eq(truth).long()).numpy()
        scores[name] = (b, a, oof_score(x, y, groups))

    rows = []
    pcts = [0, 20, 15, 12, 10, 5, 3, 2, 1]
    for full_pct, crop_pct in itertools.product(pcts, pcts):
        selected = dict(experts)
        for name, pct in [("full", full_pct), ("crop", crop_pct)]:
            if pct:
                b, a, score = scores[name]
                selected[name] = (b, a, torch.from_numpy(score >= np.quantile(score, 1 - pct / 100)))
        for order in itertools.permutations(selected):
            current = base.clone(); applied = []; overlaps = []
            for name in order:
                b, a, mask = selected[name]; eligible = mask & current.eq(b)
                overlaps.append(int((mask & ~current.eq(b)).sum())); applied.append(int(eligible.sum()))
                current[eligible] = a[eligible]
            delta = current.eq(truth).long() - base.eq(truth).long()
            fold_nets = [int(delta[torch.from_numpy(groups == f)].sum()) for f in range(4)]
            rows.append({"full_pct": full_pct, "crop_pct": crop_pct, "order": order,
                         "changed": int(current.ne(base).sum()), "net": int(delta.sum()),
                         "worst_fold": min(fold_nets), "fold_nets": fold_nets,
                         "applied": applied, "overlaps": overlaps})
    best_by_net = sorted(rows, key=lambda x: (x["net"], x["worst_fold"], -x["changed"]), reverse=True)[:100]
    rows.sort(key=lambda x: (x["worst_fold"], x["net"], -x["changed"]), reverse=True)
    suffix = "_ensemble" if "ensemble" in full_path else ""
    out = Path("runs/cloud_20260714/combo_sweep" + suffix); out.mkdir(parents=True, exist_ok=True)
    result = {"best_balanced": rows[:100], "best_net": best_by_net}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_balanced": rows[:5], "best_net": best_by_net[:10]}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from cloud_20260714_seen_tri_router import ROOT, feats, fold


BASE_PATH = Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json")
OUT = Path("runs/cloud_20260714/final_candidates_v6")


def regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, max_depth=2,
                                         min_samples_leaf=25, l2_regularization=4., random_state=2027)


def oof_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    score = np.zeros(len(y), dtype=np.float32)
    for f in range(4):
        tr, te = groups != f, groups == f
        score[te] = regressor().fit(x[tr], y[tr]).predict(x[te])
    return score


def apply(current: torch.Tensor, base: torch.Tensor, alternate: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    eligible = mask & current.eq(base)
    overlap = int((mask & ~current.eq(base)).sum())
    current[eligible] = alternate[eligible]
    return int(eligible.sum()), overlap


def oof_audit() -> dict:
    fixed = torch.load(ROOT / "concat_balanced_gate/fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    truth = fixed["class_ids"].long(); base = fixed["logits"].argmax(1); current = base.clone()
    groups = np.array([fold(x) for x in fixed["labels"]])
    stages = []

    dino = torch.load("runs/cloud_20260714/seen_meta_router_tri_best/oof_router_scores.pt", map_location="cpu", weights_only=False)
    dmask = dino["oof_score"].numpy() >= max(dino["thresholds"])
    dmask &= dino["guard"].numpy().astype(bool)
    n, overlap = apply(current, dino["base"], dino["alternate"], torch.from_numpy(dmask))
    stages.append({"name": "dino", "applied": n, "overlap": overlap})

    da = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    db = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    full = torch.load("runs/cloud_20260714/seen_full_fivecrop_router/val_fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    x, b, a, _, _ = feats(fixed, full, da, db, fixed["full_class_counts"].long())
    y = (a.eq(truth).long() - b.eq(truth).long()).numpy(); score = oof_score(x, y, groups)
    threshold = float(np.quantile(score, .95)); mask = torch.from_numpy(score >= threshold)
    n, overlap = apply(current, b, a, mask); stages.append({"name": "full_fivecrop_top5", "applied": n, "overlap": overlap})

    tri = torch.load("runs/cloud_20260714/seen_tri_router/oof_router_scores.pt", map_location="cpu", weights_only=False)
    tmask = tri["oof_score"].numpy() >= max(tri["thresholds"])
    tmask &= tri["guard"].numpy().astype(bool)
    n, overlap = apply(current, tri["base"], tri["alternate"], torch.from_numpy(tmask))
    stages.append({"name": "tri", "applied": n, "overlap": overlap})

    crop = torch.load("runs/cloud_20260714/bioclip_fivecrop_priority/val_fused_topk.pt", map_location="cpu", weights_only=False)
    x, b, a, _, _ = feats(fixed, crop, da, db, fixed["full_class_counts"].long())
    y = (a.eq(truth).long() - b.eq(truth).long()).numpy(); score = oof_score(x, y, groups)
    threshold = float(np.quantile(score, .90)); mask = torch.from_numpy(score >= threshold)
    n, overlap = apply(current, b, a, mask); stages.append({"name": "priority_fivecrop_top10", "applied": n, "overlap": overlap})

    delta = current.eq(truth).long() - base.eq(truth).long()
    genera = np.array([x.split()[0] for x in fixed["labels"]]); unique = np.unique(genera)
    genus_delta = np.array([int(delta[torch.from_numpy(genera == g)].sum()) for g in unique])
    rng = np.random.default_rng(2027)
    bootstrap = np.empty(20000, dtype=np.int32)
    for start in range(0, len(bootstrap), 500):
        draws = rng.integers(0, len(unique), size=(min(500, len(bootstrap) - start), len(unique)))
        bootstrap[start:start + len(draws)] = genus_delta[draws].sum(1)
    return {
        "changed": int(current.ne(base).sum()), "net": int(delta.sum()), "stages": stages,
        "fold_nets": [int(delta[torch.from_numpy(groups == f)].sum()) for f in range(4)],
        "positive_genera": int((genus_delta > 0).sum()), "negative_genera": int((genus_delta < 0).sum()),
        "bootstrap_mean": float(bootstrap.mean()), "ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "prob_positive": float((bootstrap > 0).mean()),
    }


def compose(name: str, stage_paths: list[tuple[str, Path]], seen_ids: set[str]) -> dict:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8")); out = dict(base)
    rows = []
    for stage, path in stage_paths:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        changed = {k for k in base if candidate[k] != base[k]}
        overlap = sum(out[k] != base[k] for k in changed)
        for k in changed:
            out[k] = candidate[k]
        rows.append({"stage": stage, "candidate_changes": len(changed), "overlap": overlap})
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    prediction = folder / "prediction.json"
    prediction.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(folder / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction, "prediction.json")
    seen = sum(out[k] != base[k] for k in seen_ids); unseen = sum(out[k] != base[k] for k in base.keys() - seen_ids)
    digest = hashlib.sha256((folder / "submission.zip").read_bytes()).hexdigest()
    return {"stages": rows, "seen_changed": seen, "unseen_changed": unseen,
            "total_changed": seen + unseen, "rows": len(out), "sha256": digest}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    public = torch.load("runs/cloud_20260714/seen_full_fivecrop_router/public_scores.pt", map_location="cpu", weights_only=False)
    seen_ids = set(public["image_ids"])
    common = [
        ("dino", Path("runs/cloud_20260714/seen_meta_router_tri_best/strict_high/prediction.json")),
    ]
    tail = [
        ("tri", Path("runs/cloud_20260714/seen_tri_router/strict_high/prediction.json")),
        ("priority_fivecrop", Path("runs/cloud_20260714/seen_fivecrop_router/top10pct/prediction.json")),
        ("unseen_fivecrop", Path("runs/cloud_20260714/unseen_fivecrop_ar125_150_w010/prediction.json")),
        ("unseen_structured", Path("runs/cloud_20260714/unseen_structured_public/packages_robust_v2/strict_current_equals_toolbase/prediction.json")),
    ]
    candidates = {}
    candidates["v6_recommended"] = compose("v6_recommended", common + [
        ("full_fivecrop", Path("runs/cloud_20260714/seen_full_fivecrop_router/top5pct/prediction.json")),
    ] + tail, seen_ids)
    broad_tail = list(tail); broad_tail[1] = ("priority_fivecrop", Path("runs/cloud_20260714/seen_fivecrop_router/top15pct/prediction.json"))
    candidates["v6_broader"] = compose("v6_broader", common + [
        ("full_fivecrop", Path("runs/cloud_20260714/seen_full_fivecrop_router/top10pct/prediction.json")),
    ] + broad_tail, seen_ids)
    result = {"oof_audit": oof_audit(), "candidates": candidates}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

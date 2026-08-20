from __future__ import annotations

import hashlib
import json
import os
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from cloud_20260714_seen_dino_router import aligned_text, fused_topk, prototypes, zscore
from cloud_20260714_seen_tri_router import ROOT, feats
from train_embedding_mlp_classifier import MLPClassifier, normalize

NEW = Path("runs/cloud_20260714/concat_fivecrop_gate")
MODELS = (["mlp_h4096_balsoft"] + [f"mlp_h4096_balsoft_seed{s}" for s in range(2028, 2033)]
          + ["ablate_ls005", "ablate_ls002", "ablate_ls010", "ablate_ls015", "ablate_ls020"])
MODE = os.environ.get("FISH_ENSEMBLE", "triple")
SELECTED = ([1, 2] if MODE == "pair2829" else
            (list(range(5)) if MODE == "five" else
             (list(range(5)) + [6] if MODE == "five_ls" else
              (list(range(5)) + [9] if MODE == "five_ls015" else
               (list(range(5)) + list(range(6, 11)) if MODE == "five_ls_all" else [0, 1, 2])))))
suffix = ("_pair2829" if MODE == "pair2829" else
          ("_five" if MODE == "five" else
           ("_five_ls" if MODE == "five_ls" else
            ("_five_ls015" if MODE == "five_ls015" else ("_five_ls_all" if MODE == "five_ls_all" else "")))))
OUT = Path("runs/cloud_20260714/seen_full_fivecrop_ensemble" + suffix)


def fold(label: str) -> int:
    return int.from_bytes(hashlib.sha1(label.split()[0].encode()).digest()[:4], "little") % 4


def regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, max_depth=2,
                                         min_samples_leaf=25, l2_regularization=4., random_state=2027)


def infer(path: Path, features: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False); arch = ck["arch"]
    net = MLPClassifier(arch["in_dim"], arch["hidden_dim"], len(ck["classes"]), arch["dropout"])
    net.load_state_dict(ck["state_dict"]); net.to(device).eval(); parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            parts.append(net(normalize(features[start:start + 256]).to(device)).cpu())
    return torch.cat(parts), ck


def make_package(cur, ids, classes, current, alternate, mask, name):
    pred = current.clone(); pred[mask] = alternate[mask]; out = dict(cur)
    for image_id, idx in zip(ids, pred): out[image_id] = classes[int(idx)]
    q = OUT / name; q.mkdir(parents=True, exist_ok=True); p = q / "prediction.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(q / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf: zf.write(p, "prediction.json")
    return sum(out[k] != cur[k] for k in cur)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    fixed = torch.load(ROOT / "concat_balanced_gate/fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    old_raw = torch.load(ROOT / "concat_balanced_gate/mlp_h4096_balsoft/val_logits.pt", map_location="cpu", weights_only=False)
    payloads = [torch.load(NEW / m / "val_logits.pt", map_location="cpu", weights_only=False) for m in MODELS]
    truth = fixed["class_ids"].long(); ancillary = fixed["logits"].float() - zscore(old_raw["logits"])
    subset_results = {}
    best = None
    for size in (1, 2, 3):
        for subset in combinations(range(3), size):
            ensemble = torch.stack([zscore(payloads[i]["logits"]) for i in subset]).mean(0)
            fused = ensemble + ancillary
            row = {"raw_top1": float(ensemble.argmax(1).eq(truth).float().mean()),
                   "fixed_top1": float(fused.argmax(1).eq(truth).float().mean()),
                   "fixed_delta": int(fused.argmax(1).eq(truth).sum() - fixed["logits"].argmax(1).eq(truth).sum())}
            key = "+".join(str(2027 + i) for i in subset); subset_results[key] = row
            if best is None or row["fixed_top1"] > best[0]: best = (row["fixed_top1"], subset, ensemble, fused)
    # Triple average is the prespecified robust ensemble; report the validation-best subset separately.
    ensemble = torch.stack([zscore(payloads[i]["logits"]) for i in SELECTED]).mean(0); fused = ensemble + ancillary
    val_alt = dict(fixed); val_alt["logits"] = fused; torch.save(val_alt, OUT / "val_fixed_fusion_logits.pt")
    da = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    db = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    x, base, alt, _, _ = feats(fixed, val_alt, da, db, fixed["full_class_counts"].long())
    y = (alt.eq(truth).long() - base.eq(truth).long()).numpy(); groups = np.array([fold(s) for s in fixed["labels"]])
    oof = np.zeros(len(y), dtype=np.float32)
    for f in range(4):
        tr, te = groups != f, groups == f; oof[te] = regressor().fit(x[tr], y[tr]).predict(x[te])
    oof_rows = {}
    for pct in (20, 15, 12, 10, 5, 3, 2, 1):
        th = float(np.quantile(oof, 1 - pct / 100)); take = oof >= th
        oof_rows[str(pct)] = {"net": int(y[take].sum()), "changed": int(take.sum()),
                              "fold_nets": [int(y[take & (groups == f)].sum()) for f in range(4)]}

    test = torch.load(NEW / "test_seen.pt", map_location="cpu", weights_only=False); test_logits = []
    checkpoint = None
    for name in [MODELS[i] for i in SELECTED]:
        logits, checkpoint = infer(NEW / name / "best_model.pt", test["features"], device); test_logits.append(zscore(logits))
    test_ensemble = torch.stack(test_logits).mean(0)
    train = torch.load(NEW / "random2027/train.pt", map_location="cpu", weights_only=False)
    proto = prototypes(train["features"][:, :1024], train["class_ids"].long(), len(checkpoint["classes"]))
    text = aligned_text(Path("work/clip_text_features/seen_bioclip25_taxon.pt"), checkpoint["classes"])
    values, indices = fused_topk(test_ensemble, test["features"][:, :1024], proto, text, device)
    public_alt = {"topk_values": values, "topk_indices": indices, "classes": checkpoint["classes"], "image_ids": test["image_ids"]}
    torch.save(public_alt, OUT / "public_fusion.pt")
    public_base = torch.load(OUT.parent / "seen_dino_router/public_fusion.pt", map_location="cpu", weights_only=False)
    pda = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    pdb = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    px, pbase, palt, _, _ = feats(public_base, public_alt, pda, pdb, fixed["full_class_counts"].long())
    score = regressor().fit(x, y).predict(px)
    cur = json.loads(Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json").read_text(encoding="utf-8"))
    pos = {name: i for i, name in enumerate(checkpoint["classes"])}; current = torch.tensor([pos[cur[i]] for i in test["image_ids"]])
    agree = current.eq(pbase); public_rows = {}
    for pct in (15, 12, 10, 5, 3, 2, 1):
        th = float(np.quantile(score, 1 - pct / 100)); mask = torch.from_numpy(score >= th) & agree
        public_rows[str(pct)] = {"selected": int(mask.sum()), "changed": make_package(cur, test["image_ids"], checkpoint["classes"], current, palt, mask, f"top{pct}pct")}
    torch.save({"score": torch.from_numpy(score), "base": pbase, "alternate": palt, "current": current,
                "image_ids": test["image_ids"]}, OUT / "public_scores.pt")
    summary = {"mode": MODE, "selected_seeds": [2027 + i for i in SELECTED],
               "subsets": subset_results, "validation_best_subset": [2027 + i for i in best[1]],
               "selected_fixed_top1": float(fused.argmax(1).eq(truth).float().mean()),
               "selected_fixed_delta": int(fused.argmax(1).eq(truth).sum() - fixed["logits"].argmax(1).eq(truth).sum()),
               "oof_quantiles": oof_rows, "public": public_rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

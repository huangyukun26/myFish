from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from cloud_20260714_seen_dino_router import aligned_text, fused_topk, prototypes, zscore
from cloud_20260714_seen_tri_router import ROOT, feats
from train_embedding_mlp_classifier import MLPClassifier, normalize


NEW = Path("runs/cloud_20260714/concat_fivecrop_gate")
OUT = Path("runs/cloud_20260714/seen_full_fivecrop_router")


def fold(label: str) -> int:
    genus = label.split()[0]
    return int.from_bytes(hashlib.sha1(genus.encode()).digest()[:4], "little") % 4


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.04, max_depth=2,
        min_samples_leaf=25, l2_regularization=4.0, random_state=2027,
    )


def infer(checkpoint: dict, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    arch = checkpoint["arch"]
    net = MLPClassifier(arch["in_dim"], arch["hidden_dim"], len(checkpoint["classes"]), arch["dropout"])
    net.load_state_dict(checkpoint["state_dict"])
    net.to(device).eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            parts.append(net(normalize(features[start:start + 256]).to(device)).cpu())
    return torch.cat(parts)


def package(cur: dict[str, str], ids: list[str], classes: list[str], current: torch.Tensor,
            alt: torch.Tensor, mask: torch.Tensor, name: str) -> int:
    pred = current.clone(); pred[mask] = alt[mask]
    out = dict(cur)
    for image_id, class_id in zip(ids, pred):
        out[image_id] = classes[int(class_id)]
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    path = folder / "prediction.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(folder / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, "prediction.json")
    return sum(out[k] != cur[k] for k in cur)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    old_dir = ROOT / "concat_balanced_gate"
    old_fixed = torch.load(old_dir / "fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    old_raw = torch.load(old_dir / "mlp_h4096_balsoft/val_logits.pt", map_location="cpu", weights_only=False)
    new_raw = torch.load(NEW / "mlp_h4096_balsoft/val_logits.pt", map_location="cpu", weights_only=False)
    assert old_fixed["image_ids"] == old_raw["image_ids"] == new_raw["image_ids"]
    truth = old_fixed["class_ids"].long()

    # Keep the exact prototype/text contribution of the established fusion and
    # replace only its supervised MLP term with the new multiview MLP term.
    new_logits = zscore(new_raw["logits"]) + old_fixed["logits"].float() - zscore(old_raw["logits"])
    new_fixed = dict(old_fixed); new_fixed["logits"] = new_logits
    torch.save(new_fixed, OUT / "val_fixed_fusion_logits.pt")

    da = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    db = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    x, base, alt, _, _ = feats(old_fixed, new_fixed, da, db, old_fixed["full_class_counts"].long())
    y = (alt.eq(truth).long() - base.eq(truth).long()).numpy()
    groups = np.array([fold(s) for s in old_fixed["labels"]])
    oof = np.zeros(len(y), dtype=np.float32)
    rows = []
    for f in range(4):
        tr, te = groups != f, groups == f
        reg = model().fit(x[tr], y[tr]); score = reg.predict(x[te]); oof[te] = score
        rows.append({"fold": f, "raw_delta": int(y[te].sum())})
    quantiles = {}
    for pct in (20, 15, 12, 10, 5, 3, 2, 1):
        threshold = float(np.quantile(oof, 1 - pct / 100))
        take = oof >= threshold
        quantiles[str(pct)] = {
            "threshold": threshold, "changed": int(take.sum()), "net": int(y[take].sum()),
            "fold_nets": [int(y[take & (groups == f)].sum()) for f in range(4)],
        }

    checkpoint = torch.load(NEW / "mlp_h4096_balsoft/best_model.pt", map_location="cpu", weights_only=False)
    train = torch.load(NEW / "random2027/train.pt", map_location="cpu", weights_only=False)
    test = torch.load(NEW / "test_seen.pt", map_location="cpu", weights_only=False)
    test_logits = infer(checkpoint, test["features"], device)
    proto = prototypes(train["features"][:, :1024], train["class_ids"].long(), len(checkpoint["classes"]))
    text = aligned_text(Path("work/clip_text_features/seen_bioclip25_taxon.pt"), checkpoint["classes"])
    values, indices = fused_topk(test_logits, test["features"][:, :1024], proto, text, device)
    public_alt = {"topk_values": values, "topk_indices": indices, "classes": checkpoint["classes"], "image_ids": test["image_ids"]}
    torch.save(public_alt, OUT / "public_fusion.pt")

    public_base = torch.load(OUT.parent / "seen_dino_router/public_fusion.pt", map_location="cpu", weights_only=False)
    pda = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    pdb = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    px, pbase, palt, _, _ = feats(public_base, public_alt, pda, pdb, old_fixed["full_class_counts"].long())
    reg = model().fit(x, y); public_score = reg.predict(px)
    base_path = Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json")
    cur = json.loads(base_path.read_text(encoding="utf-8"))
    pos = {name: i for i, name in enumerate(checkpoint["classes"])}
    current = torch.tensor([pos[cur[i]] for i in test["image_ids"]])
    agree = current.eq(pbase)
    public = {}
    for pct in (10, 5, 3, 2, 1):
        threshold = float(np.quantile(public_score, 1 - pct / 100))
        mask = torch.from_numpy(public_score >= threshold) & agree
        public[str(pct)] = {"selected": int(mask.sum()), "changed": package(cur, test["image_ids"], checkpoint["classes"], current, palt, mask, f"top{pct}pct")}
    torch.save({"score": torch.from_numpy(public_score), "base": pbase, "alternate": palt,
                "current": current, "image_ids": test["image_ids"]}, OUT / "public_scores.pt")
    summary = {
        "raw_top1": float(new_raw["logits"].argmax(1).eq(truth).float().mean()),
        "fixed_top1": float(new_logits.argmax(1).eq(truth).float().mean()),
        "old_fixed_top1": float(old_fixed["logits"].argmax(1).eq(truth).float().mean()),
        "direct_fixed_delta": int(new_logits.argmax(1).eq(truth).sum() - old_fixed["logits"].argmax(1).eq(truth).sum()),
        "folds": rows, "oof_quantiles": quantiles, "public": public,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

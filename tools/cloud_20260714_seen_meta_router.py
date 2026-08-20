from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path("runs/structural_backbones_20260713")
OUT = Path("runs/cloud_20260714/seen_meta_router")


def fold(label: str) -> int:
    g = label.split(maxsplit=1)[0]
    return int.from_bytes(hashlib.sha1(g.encode()).digest()[:4], "little") % 4


def load_quality(path: Path, image_ids: list[str]) -> torch.Tensor:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["image_id"]: row for row in rows}
    output = []
    for image_id in image_ids:
        r = by_id[image_id]
        output.append([
            float(r["aspect_ratio"]), abs(np.log(max(float(r["aspect_ratio"]), 1e-6))),
            float(r["crop_area_fraction"]), float(r["component_patches"]) / max(float(r["valid_patches"]), 1),
            float(r["selected_patches"]) / max(float(r["valid_patches"]), 1), float(r["fallback"]),
            np.log1p(float(r["original_width"]) * float(r["original_height"])),
        ])
    return torch.tensor(output, dtype=torch.float32)


def features(base_source, a: dict, b: dict, counts: torch.Tensor,
             quality: torch.Tensor | None = None, tri: dict | None = None) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    if isinstance(base_source, dict):
        bv, bi = base_source["topk_values"].float(), base_source["topk_indices"].long()
    else:
        bv, bi = base_source.float().topk(20, dim=1)
    av, ai = a["topk_values"].float(), a["topk_indices"].long()
    bvz = (bv - bv.mean(1, keepdim=True)) / bv.std(1, keepdim=True).clamp_min(1e-6)
    avz = (av - av.mean(1, keepdim=True)) / av.std(1, keepdim=True).clamp_min(1e-6)
    base, alt = bi[:, 0], ai[:, 0]
    alt_rank = bi.eq(alt[:, None]).float().argmax(1)
    in_top = bi.eq(alt[:, None]).any(1)
    alt_rank = torch.where(in_top, alt_rank, torch.full_like(alt_rank, 20))
    same_genus = torch.tensor([
        a["classes"][int(x)].split()[0] == a["classes"][int(y)].split()[0]
        for x, y in zip(base, alt)
    ])
    agree = alt.eq(b["topk_indices"].long()[:, 0])
    amin = torch.minimum(av[:, 0] - av[:, 1], b["topk_values"][:, 0] - b["topk_values"][:, 1])
    x = torch.stack([
        bv[:, 0] - bv[:, 1], bv[:, 0] - bv[:, 4], bvz[:, 0], bvz[:, 0] - bvz[:, 1],
        amin, av[:, 0] - av[:, 4], avz[:, 0], avz[:, 0] - avz[:, 1],
        alt_rank.float() / 20, in_top.float(), agree.float(), same_genus.float(),
        torch.log1p(counts[base].float()), torch.log1p(counts[alt].float()),
        (counts[alt] <= 2).float(), (counts[alt] <= 5).float(),
    ], dim=1)
    if quality is not None:
        x = torch.cat([x, quality], dim=1)
    if tri is not None:
        if "logits" in tri:
            tv, ti = tri["logits"].float().topk(20, dim=1)
        else:
            tv, ti = tri["topk_values"].float(), tri["topk_indices"].long()
        tvz = (tv - tv.mean(1, keepdim=True)) / tv.std(1, keepdim=True).clamp_min(1e-6)
        tpred = ti[:, 0]
        tri_extra = torch.stack([
            tvz[:, 0], tvz[:, 0] - tvz[:, 1],
            tpred.eq(base).float(), tpred.eq(alt).float(),
            ti[:, :5].eq(alt[:, None]).any(1).float(), ti.eq(alt[:, None]).any(1).float(),
            bi.eq(tpred[:, None]).any(1).float(),
        ], dim=1)
        x = torch.cat([x, tri_extra], dim=1)
    return x.numpy(), base, alt


def load_jsonl_topk(path: Path, classes: list[str]) -> dict:
    pos = {name: i for i, name in enumerate(classes)}
    indices, values, image_ids = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); image_ids.append(row["image_id"])
        indices.append([pos[name] for name in row["predictions"]]); values.append(row["scores"])
    return {"topk_indices": torch.tensor(indices), "topk_values": torch.tensor(values), "image_ids": image_ids}


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=0.04, max_depth=3,
                                         min_samples_leaf=70, l2_regularization=20.0,
                                         random_state=2027)


def best_threshold(score: np.ndarray, y: np.ndarray) -> float:
    choices = np.unique(np.quantile(score, np.linspace(0.5, 0.995, 120)))
    return float(max(choices, key=lambda t: (y[score >= t].sum(), -(score >= t).sum())))


def guard_mask(base_source, a: dict, b: dict, counts: torch.Tensor, alt: torch.Tensor) -> torch.Tensor:
    if isinstance(base_source, dict):
        bi = base_source["topk_indices"].long()
    else:
        bi = base_source.float().topk(20, dim=1).indices
    return (
        alt.eq(b["topk_indices"].long()[:, 0])
        & bi.eq(alt[:, None]).any(1)
        & counts[alt].le(5)
    )


def write_package(current_json: dict[str, str], image_ids: list[str], classes: list[str],
                  current: torch.Tensor, alt: torch.Tensor, mask: torch.Tensor, name: str) -> int:
    out_dir = OUT / name; out_dir.mkdir(parents=True, exist_ok=True)
    pred = current.clone(); pred[mask] = alt[mask]
    merged = dict(current_json)
    for image_id, idx in zip(image_ids, pred): merged[image_id] = classes[int(idx)]
    out_json = out_dir / "prediction.json"
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    return int(mask.sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    val = torch.load(ROOT / "concat_balanced_gate/fixed_fusion_logits.pt", map_location="cpu", weights_only=False)
    a = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    b = torch.load(ROOT / "dino_metric_full_holdout/prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    assert val["image_ids"] == a["image_ids"] == b["image_ids"]
    counts = val["full_class_counts"].long()
    tri_val = torch.load(ROOT / "triview_concat_gate/paired_random2027/fixed_fusion_taxon_logits.pt", map_location="cpu", weights_only=False)
    x, base, alt = features(val["logits"], a, b, counts, tri=tri_val)
    truth = val["class_ids"].long()
    y = (alt.eq(truth).long() - base.eq(truth).long()).numpy()
    groups = np.array([fold(label) for label in val["labels"]])
    oof = np.zeros(len(y)); thresholds, rows = [], []
    for f in range(4):
        tr, te = groups != f, groups == f
        m = model().fit(x[tr], y[tr]); train_score = m.predict(x[tr]); threshold = best_threshold(train_score, y[tr])
        score = m.predict(x[te]); oof[te] = score; thresholds.append(threshold)
        take = score >= threshold
        rows.append({"fold": f, "rows": int(te.sum()), "threshold": threshold,
                     "changed": int(take.sum()), "net": int(y[te][take].sum()),
                     "wins": int((y[te][take] == 1).sum()), "losses": int((y[te][take] == -1).sum())})
    take = oof >= np.array([thresholds[f] for f in groups])
    oof_summary = {"changed": int(take.sum()), "net": int(y[take].sum()),
                   "wins": int((y[take] == 1).sum()), "losses": int((y[take] == -1).sum()), "folds": rows}
    val_guard = guard_mask(val["logits"], a, b, counts, alt).numpy()
    strict_take = take & val_guard
    strict_oof = {"changed": int(strict_take.sum()), "net": int(y[strict_take].sum()),
                  "wins": int((y[strict_take] == 1).sum()), "losses": int((y[strict_take] == -1).sum())}
    torch.save({"oof_score": torch.from_numpy(oof), "thresholds": thresholds,
                "groups": torch.from_numpy(groups), "base": base, "alternate": alt,
                "truth": truth, "guard": torch.from_numpy(val_guard)}, OUT / "oof_router_scores.pt")
    final = model().fit(x, y)
    threshold = float(np.median(thresholds))

    # Public fusion top-k/logits were saved by the preceding deterministic router job.
    public = torch.load(OUT.parent / "seen_dino_router/public_fusion.pt", map_location="cpu", weights_only=False)
    pa = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    pb = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    tri_public = load_jsonl_topk(Path("runs/cloud_20260714/triview_public_direct/topk.jsonl"), list(pa["classes"]))
    assert tri_public["image_ids"] == pa["image_ids"]
    px, pbase, palt = features(public, pa, pb, counts, tri=tri_public)
    score = final.predict(px)
    current_json = json.loads(Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json").read_text(encoding="utf-8"))
    class_to_idx = {name: i for i, name in enumerate(pa["classes"])}
    current = torch.tensor([class_to_idx[current_json[i]] for i in pa["image_ids"]])
    agree_base = current.eq(pbase)
    public_guard = guard_mask(public, pa, pb, counts, palt)
    masks = {
        "medium": torch.from_numpy(score >= threshold) & agree_base,
        "strict": torch.from_numpy(score >= threshold) & agree_base & public_guard,
        "strict_high": torch.from_numpy(score >= max(thresholds)) & agree_base & public_guard,
    }
    outputs = {name: {"changed": write_package(current_json, list(pa["image_ids"]), pa["classes"],
                                                 current, palt, mask, name)}
               for name, mask in masks.items()}
    torch.save({"image_ids": pa["image_ids"], "score": torch.from_numpy(score), "current": current,
                "base": pbase, "alternate": palt, "guard": public_guard,
                "thresholds": thresholds}, OUT / "public_router_scores.pt")
    summary = {"oof": oof_summary, "strict_oof": strict_oof,
               "public_threshold": threshold, "outputs": outputs}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

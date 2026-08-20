"""Group-aware taxon-vs-fish selector (P4 light control).

The selector never sees candidate identity or the true label as an input.  It
only receives the two experts' confidence/overlap signals and is evaluated on
the same held-out species/genus folds as the candidate reranker.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def query_groups(qid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    unique, starts, counts = np.unique(qid, return_index=True, return_counts=True)
    return [(int(q), np.arange(s, s + c, dtype=np.int64)) for q, s, c in zip(unique.tolist(), starts.tolist(), counts.tolist())]


def build_query_table(table: dict[str, Any], tax_top: np.ndarray, fish_top: np.ndarray) -> dict[str, Any]:
    qid = table["query_id"].numpy().astype(np.int64)
    fold_species = table["species_fold"].numpy().astype(np.int64)
    fold_genus = table["genus_fold"].numpy().astype(np.int64)
    true = table["true_id"].numpy().astype(np.int64)
    cand = table["candidate_id"].numpy().astype(np.int64)
    feat = table["features"].float().numpy()
    classes = [str(x) for x in table["candidate_classes"]]
    genus_names = [x.split()[0] for x in classes]
    genus_to_id = {x: i for i, x in enumerate(sorted(set(genus_names)))}
    cand_genus = np.asarray([genus_to_id[x] for x in genus_names], dtype=np.int64)
    names = [
        "taxon_top1_score", "fish_top1_score", "taxon_margin", "fish_margin", "taxon_entropy", "fish_entropy",
        "top1_same_species", "top1_same_genus", "overlap5", "overlap10", "overlap20", "overlap50",
        "fish_top1_in_tax_rank", "tax_top1_in_fish_rank", "top1_score_diff",
    ]
    rows: list[list[float]] = []
    qids: list[int] = []
    species: list[int] = []
    genera: list[int] = []
    true_out: list[int] = []
    tax_correct: list[bool] = []
    fish_correct: list[bool] = []
    for q, idx in query_groups(qid):
        tax1 = int(tax_top[q, 0])
        fish1 = int(fish_top[q, 0])
        tr = int(true[idx[0]])
        tax_rows = idx[feat[idx, 2] == 1]
        fish_rows = idx[feat[idx, 5] == 1]
        tax_score = float(feat[tax_rows[0], 0]) if len(tax_rows) else 0.0
        fish_score = float(feat[fish_rows[0], 3]) if len(fish_rows) else 0.0
        tax_margin = float(feat[idx[0], 7])
        fish_margin = float(feat[idx[0], 8])
        tax_entropy = float(feat[idx[0], 9])
        fish_entropy = float(feat[idx[0], 10])
        same_species = float(tax1 == fish1)
        same_genus = float(cand_genus[tax1] == cand_genus[fish1])
        overlaps = [float(len(set(tax_top[q, :k].tolist()) & set(fish_top[q, :k].tolist())) / k) for k in (5, 10, 20, 50)]
        fish_in_tax = int(np.where(tax_top[q] == fish1)[0][0] + 1) if np.any(tax_top[q] == fish1) else 101
        tax_in_fish = int(np.where(fish_top[q] == tax1)[0][0] + 1) if np.any(fish_top[q] == tax1) else 101
        rows.append([tax_score, fish_score, tax_margin, fish_margin, tax_entropy, fish_entropy, same_species, same_genus, *overlaps, float(fish_in_tax), float(tax_in_fish), tax_score - fish_score])
        qids.append(q)
        species.append(int(fold_species[idx[0]]))
        genera.append(int(fold_genus[idx[0]]))
        true_out.append(tr)
        tax_correct.append(tax1 == tr)
        fish_correct.append(fish1 == tr)
    return {
        "features": torch.tensor(rows, dtype=torch.float32), "query_id": torch.tensor(qids, dtype=torch.int32),
        "species_fold": torch.tensor(species, dtype=torch.int16), "genus_fold": torch.tensor(genera, dtype=torch.int16),
        "true_id": torch.tensor(true_out, dtype=torch.int32), "tax_correct": torch.tensor(tax_correct), "fish_correct": torch.tensor(fish_correct),
        "feature_names": names,
    }


def fit(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=200, random_state=seed)).fit(x, y)


def select_threshold(model: Any, x: np.ndarray, tax: np.ndarray, fish: np.ndarray, true: np.ndarray, threshold_grid: tuple[float, ...] = (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)) -> float:
    prob = model.predict_proba(x)[:, 1]
    best = (-10**9, 0.5)
    for t in threshold_grid:
        pred = np.where(prob >= t, fish, tax)
        base = tax == true
        wins = np.logical_and(pred == true, ~base).sum()
        losses = np.logical_and(pred != true, base).sum()
        net = int(wins - losses)
        if net > best[0] or (net == best[0] and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (net, t)
    return float(best[1])


def metric(protocol: str, fold: int, mode: str, threshold: float, prob: np.ndarray, tax: np.ndarray, fish: np.ndarray, true: np.ndarray, seed: int) -> dict[str, Any]:
    pred = np.where(prob >= threshold, fish, tax) if mode == "selector" else tax
    base = tax == true
    changed = pred != tax
    wins = np.logical_and(changed, np.logical_and(pred == true, ~base)).sum()
    losses = np.logical_and(changed, np.logical_and(pred != true, base)).sum()
    delta = (pred == true).astype(np.float32) - base.astype(np.float32)
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(delta, size=len(delta), replace=True).sum() for _ in range(300)], dtype=np.float32)
    return {
        "protocol": protocol, "fold": fold, "mode": mode, "threshold": threshold, "queries": int(len(true)),
        "base_accuracy": float(base.mean()), "selector_accuracy": float((pred == true).mean()), "changed": int(changed.sum()),
        "wins": int(wins), "losses": int(losses), "net": int(wins - losses), "change_rate": float(changed.mean()),
        "bootstrap_lower_net": float(np.quantile(boot, 0.025)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("runs/research_next_20260820/full_universe/candidate_table.pt"))
    parser.add_argument("--tax-top", type=Path, default=Path("runs/research_next_20260820/full_universe/top100_taxon.pt"))
    parser.add_argument("--fish-top", type=Path, default=Path("runs/research_next_20260820/full_universe/top100_fish.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/full_universe/selector"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = load(args.table)
    tax_top = load(args.tax_top)["top_ids"].numpy().astype(np.int64)
    fish_top = load(args.fish_top)["top_ids"].numpy().astype(np.int64)
    qtable = build_query_table(table, tax_top, fish_top)
    torch.save(qtable, args.out_dir / "selector_query_features.pt")
    x = qtable["features"].numpy()
    true = qtable["true_id"].numpy().astype(np.int64)
    tax = tax_top[:, 0]
    fish = fish_top[:, 0]
    results: list[dict[str, Any]] = []
    for protocol, fold_tensor in (("species", qtable["species_fold"].numpy()), ("genus", qtable["genus_fold"].numpy())):
        for outer in range(5):
            train = fold_tensor != outer
            inner_fold = int(np.where(train)[0][0])  # replaced below by smallest fold value in training
            inner_value = min(set(fold_tensor.tolist()) - {outer})
            inner = fold_tensor == inner_value
            train_inner = train & ~inner
            train_mask = train
            # Train on three folds to choose a threshold; then refit on four.
            target_inner = np.logical_xor(tax[inner] == true[inner], fish[inner] == true[inner])
            if target_inner.sum() == 0:
                threshold = 0.5
            else:
                use = inner & np.logical_xor(tax == true, fish == true)
                y = (fish[use] == true[use]).astype(np.int64)
                inner_model = fit(x[train_inner & np.logical_xor(tax == true, fish == true)], (fish[train_inner & np.logical_xor(tax == true, fish == true)] == true[train_inner & np.logical_xor(tax == true, fish == true)]).astype(np.int64), 5000 + outer)
                threshold = select_threshold(inner_model, x[inner & np.logical_xor(tax == true, fish == true)], tax[inner & np.logical_xor(tax == true, fish == true)], fish[inner & np.logical_xor(tax == true, fish == true)], true[inner & np.logical_xor(tax == true, fish == true)])
            use_train = train & np.logical_xor(tax == true, fish == true)
            model = fit(x[use_train], (fish[use_train] == true[use_train]).astype(np.int64), 6000 + outer)
            test = fold_tensor == outer
            prob = model.predict_proba(x[test])[:, 1]
            results.append(metric(protocol, outer, "selector", threshold, prob, tax[test], fish[test], true[test], 7000 + outer))
            results.append(metric(protocol, outer, "taxon_baseline", 0.5, np.zeros(int(test.sum()), dtype=np.float32), tax[test], fish[test], true[test], 8000 + outer))
            print(protocol, outer, "threshold", threshold)
    write_csv(args.out_dir / "selector_metrics.csv", results)
    summary = {"features": qtable["feature_names"], "results": results, "note": "only confidence/overlap/cross-rank features; no candidate identity"}
    (args.out_dir / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

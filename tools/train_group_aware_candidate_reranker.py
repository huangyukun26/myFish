"""Train/evaluate the small group-aware candidate reranker.

Outer folds are the species- or genus-held-out folds from the manifest.  An
inner fold chooses the selective margin threshold; the outer fold is evaluated
once.  The direct reranker never sees candidate identity, frequency, or split
membership as an input feature.
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
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def groups(qid: np.ndarray, fold: np.ndarray) -> list[tuple[int, np.ndarray, int]]:
    unique, starts, counts = np.unique(qid, return_index=True, return_counts=True)
    out: list[tuple[int, np.ndarray, int]] = []
    for q, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist()):
        idx = np.arange(start, start + count, dtype=np.int64)
        out.append((int(q), idx, int(fold[start])))
    return out


def sample_rows(table: dict[str, Any], group_list: list[tuple[int, np.ndarray, int]], train_folds: set[int], candidate_genus: np.ndarray, max_neg: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cand = table["candidate_id"].numpy().astype(np.int64)
    true = table["true_id"].numpy().astype(np.int64)
    selected: list[int] = []
    for _, idx, f in group_list:
        if f not in train_folds:
            continue
        pos = idx[cand[idx] == true[idx]]
        neg = idx[cand[idx] != true[idx]]
        if len(pos):
            selected.extend(pos.tolist())
        if len(neg) > max_neg:
            same = neg[candidate_genus[cand[neg]] == candidate_genus[true[idx[0]]]]
            other = neg[candidate_genus[cand[neg]] != candidate_genus[true[idx[0]]]]
            rng.shuffle(same)
            rng.shuffle(other)
            take_same = min(len(same), max(1, max_neg // 2))
            chosen = np.concatenate([same[:take_same], other[: max_neg - take_same]])
            if len(chosen) < max_neg:
                remaining = np.setdiff1d(neg, chosen, assume_unique=False)
                rng.shuffle(remaining)
                chosen = np.concatenate([chosen, remaining[: max_neg - len(chosen)]])
            selected.extend(chosen[:max_neg].tolist())
        else:
            selected.extend(neg.tolist())
    return np.asarray(selected, dtype=np.int64)


def fit_model(model_name: str, x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    if model_name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=120, solver="lbfgs", random_state=seed)).fit(x, y)
    if model_name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-4, batch_size=512, learning_rate_init=2e-3, max_iter=35, early_stopping=True, validation_fraction=0.1, random_state=seed, verbose=False)).fit(x, y)
    raise ValueError(model_name)


def predict_group_scores(model: Any, table: dict[str, Any], group_list: list[tuple[int, np.ndarray, int]], wanted_folds: set[int], feature_count: int) -> dict[int, dict[str, Any]]:
    x = table["features"].float().numpy()
    both = table["features"][:, 6].numpy()
    cand = table["candidate_id"].numpy().astype(np.int64)
    true = table["true_id"].numpy().astype(np.int64)
    rank_tax = table["features"][:, 2].numpy()
    out: dict[int, dict[str, Any]] = {}
    for q, idx, f in group_list:
        if f not in wanted_folds:
            continue
        score = model.predict_proba(x[idx, :feature_count])[:, 1]
        order = np.argsort(-score)
        chosen = int(cand[idx[order[0]]])
        baseline_idx = idx[np.argmin(rank_tax[idx])]
        baseline = int(cand[baseline_idx])
        top2 = score[order[:2]]
        margin = float(top2[0] - top2[1]) if len(top2) > 1 else float(top2[0])
        out[q] = {
            "idx": idx,
            "candidate": chosen,
            "baseline": baseline,
            "true": int(true[idx[0]]),
            "margin": margin,
            "both": float(both[idx[order[0]]]) >= 0.5,
            "direct_correct": chosen == int(true[idx[0]]),
            "base_correct": baseline == int(true[idx[0]]),
            "union_recall": bool(np.any(cand[idx] == int(true[idx[0]]))),
        }
    return out


def select_threshold(preds: dict[int, dict[str, Any]]) -> float:
    best = (0, 0.0)
    for threshold in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        wins = losses = changed = 0
        for p in preds.values():
            use = p["margin"] >= threshold and p["both"]
            final = p["candidate"] if use else p["baseline"]
            changed += int(final != p["baseline"])
            wins += int(final == p["true"] and p["baseline"] != p["true"])
            losses += int(final != p["true"] and p["baseline"] == p["true"])
        score = wins - losses
        if score > best[0] or (score == best[0] and threshold < best[1]):
            best = (score, threshold)
    return best[1]


def summarize(preds: dict[int, dict[str, Any]], threshold: float, mode: str, protocol: str, fold: int, model: str) -> dict[str, Any]:
    base_correct = direct_correct = final_correct = 0
    changed = wins = losses = 0
    union_hits = 0
    delta: list[float] = []
    for p in preds.values():
        base_correct += int(p["base_correct"])
        direct_correct += int(p["direct_correct"])
        use = mode == "direct" or (p["margin"] >= threshold and p["both"])
        final = p["candidate"] if use else p["baseline"]
        fc = final == p["true"]
        final_correct += int(fc)
        ch = final != p["baseline"]
        changed += int(ch)
        wins += int(ch and fc and not p["base_correct"])
        losses += int(ch and not fc and p["base_correct"])
        union_hits += int(p["union_recall"])
        delta.append(float(fc) - float(p["base_correct"]))
    arr = np.asarray(delta, dtype=np.float32)
    rng = np.random.default_rng(20260820 + fold)
    if len(arr):
        boot = np.asarray([rng.choice(arr, size=len(arr), replace=True).sum() for _ in range(300)], dtype=np.float32)
        lower = float(np.quantile(boot, 0.025))
    else:
        lower = 0.0
    n = len(preds)
    return {
        "protocol": protocol, "fold": fold, "model": model, "mode": mode, "threshold": float(threshold), "queries": n,
        "base_accuracy": float(base_correct / max(1, n)), "direct_accuracy": float(direct_correct / max(1, n)), "final_accuracy": float(final_correct / max(1, n)),
        "union_recall50": float(union_hits / max(1, n)), "changed": changed, "wins": wins, "losses": losses, "net": wins - losses,
        "change_rate": float(changed / max(1, n)), "bootstrap_lower_net": lower,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("runs/research_next_20260820/full_universe/candidate_table.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/full_universe/reranker"))
    parser.add_argument("--protocols", nargs="+", default=["species", "genus"])
    parser.add_argument("--models", nargs="+", default=["logistic", "mlp"])
    parser.add_argument("--max-negatives", type=int, default=6)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = load(args.table)
    feature_count = len(table["feature_names"])
    features = table["features"].float()
    qid = table["query_id"].numpy().astype(np.int64)
    species = table["species_fold"].numpy().astype(np.int64)
    genus = table["genus_fold"].numpy().astype(np.int64)
    classes = [str(x) for x in table["candidate_classes"]]
    genus_names = [x.split()[0] for x in classes]
    genus_to_id = {x: i for i, x in enumerate(sorted(set(genus_names)))}
    candidate_genus = np.asarray([genus_to_id[x] for x in genus_names], dtype=np.int64)
    group_cache = {
        "species": groups(qid, species),
        "genus": groups(qid, genus),
    }
    results: list[dict[str, Any]] = []
    for protocol in args.protocols:
        fold_values = species if protocol == "species" else genus
        group_list = group_cache[protocol]
        for model_name in args.models:
            for outer in range(5):
                train_folds = set(range(5)) - {outer}
                # Inner selection fold is held out from the model used for
                # threshold selection, keeping the outer fold sealed.
                inner = min(train_folds)
                inner_train = train_folds - {inner}
                inner_idx = sample_rows(table, group_list, inner_train, candidate_genus, args.max_negatives, 1000 + outer)
                x_inner = features[inner_idx].numpy()
                y_inner = (table["candidate_id"][inner_idx] == table["true_id"][inner_idx]).numpy().astype(np.int64)
                inner_model = fit_model(model_name, x_inner, y_inner, 1000 + outer)
                inner_preds = predict_group_scores(inner_model, table, group_list, {inner}, feature_count)
                threshold = select_threshold(inner_preds)
                # Refit on all non-outer groups for the sealed outer fold.
                train_idx = sample_rows(table, group_list, train_folds, candidate_genus, args.max_negatives, 2000 + outer)
                x_train = features[train_idx].numpy()
                y_train = (table["candidate_id"][train_idx] == table["true_id"][train_idx]).numpy().astype(np.int64)
                model = fit_model(model_name, x_train, y_train, 3000 + outer)
                preds = predict_group_scores(model, table, group_list, {outer}, feature_count)
                results.append(summarize(preds, threshold, "direct", protocol, outer, model_name))
                results.append(summarize(preds, threshold, "selective", protocol, outer, model_name))
                print(protocol, model_name, "fold", outer, "threshold", threshold, "queries", len(preds))
    write_csv(args.out_dir / "reranker_metrics.csv", results)
    summary = {"results": results, "models": args.models, "protocols": args.protocols, "feature_names": table["feature_names"], "max_negatives": args.max_negatives}
    (args.out_dir / "reranker_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

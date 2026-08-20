from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def row_zscore(values):
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def paired_stats(base_prediction, prediction, labels, mask):
    base_correct = base_prediction.eq(labels)
    prediction_correct = prediction.eq(labels)
    changed = mask & prediction.ne(base_prediction)
    wins = changed & ~base_correct & prediction_correct
    losses = changed & base_correct & ~prediction_correct
    return {
        "rows": int(mask.sum()),
        "base_correct": int((mask & base_correct).sum()),
        "candidate_correct": int((mask & prediction_correct).sum()),
        "net": int(wins.sum() - losses.sum()),
        "changed": int(changed.sum()),
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "efficiency": float((wins.sum() - losses.sum()) / max(1, changed.sum())),
    }


class LinearFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.correction = nn.Linear(channels, 1, bias=False)
        nn.init.zeros_(self.correction.weight)

    def forward(self, features):
        return features[..., 0] + self.correction(features).squeeze(-1)


class GatedFusion(nn.Module):
    def __init__(self, channels, hidden=32, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels * 3 + 1),
            nn.Linear(channels * 3 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        # Candidate-local values plus row-level source confidence summaries.
        sorted_values = features.sort(dim=1, descending=True).values
        source_max = sorted_values[:, 0]
        source_margin = sorted_values[:, 0] - sorted_values[:, 1]
        batch, candidates, _channels = features.shape
        rank = torch.linspace(1, 0, candidates, device=features.device)[None, :, None]
        context = torch.cat([source_max, source_margin], dim=1)[:, None].expand(
            -1, candidates, -1
        )
        rank = rank.expand(batch, -1, -1)
        correction = self.net(torch.cat([features, context, rank], dim=2)).squeeze(-1)
        return features[..., 0] + correction


def load_feature_stack(paths):
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference = payloads[0]
    channels = [row_zscore(reference["top_values"].float())]
    names = ["base"]
    for source_index, payload in enumerate(payloads):
        if list(payload["image_ids"]) != list(reference["image_ids"]):
            raise RuntimeError(f"score payload {paths[source_index]} image order differs")
        if not torch.equal(payload["top_indices"].long(), reference["top_indices"].long()):
            raise RuntimeError(f"score payload {paths[source_index]} top-k differs")
        score_dict = payload.get("scores", payload.get("score_families"))
        for name, values in score_dict.items():
            channels.append(row_zscore(values.float()))
            names.append(f"source{source_index}:{name}")
    return reference, torch.stack(channels, dim=2), names


def make_crossfit_folds(image_ids, dev, folds):
    fold_ids = torch.full((len(image_ids),), -1, dtype=torch.long)
    for row, image_id in enumerate(image_ids):
        if dev[row]:
            fold_ids[row] = stable_hash("fusion-fold:" + image_id) % folds
    return fold_ids


def build_model(architecture, channels, seed):
    torch.manual_seed(seed)
    if architecture == "linear":
        return LinearFusion(channels)
    if architecture == "gated":
        return GatedFusion(channels)
    raise ValueError(architecture)


def fit_model(
    features,
    target_slots,
    train_mask,
    architecture,
    error_weight,
    weight_decay,
    epochs,
    learning_rate,
    seed,
    device,
):
    model = build_model(architecture, features.shape[2], seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x = features[train_mask].to(device)
    y = target_slots[train_mask].to(device)
    weights = torch.where(
        y.eq(0), torch.ones_like(y, dtype=torch.float32), torch.full_like(y, error_weight, dtype=torch.float32)
    )
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = (F.cross_entropy(logits, y, reduction="none") * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def predict_scores(model, features, mask, device):
    output = torch.zeros(features.shape[:2], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        output[mask] = model(features[mask].to(device)).cpu()
    return output


def evaluate_prediction(top_indices, scores, labels, mask):
    prediction = top_indices.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
    return prediction, paired_stats(top_indices[:, 0], prediction, labels, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument(
        "--architectures", nargs="+", default=["linear", "gated"]
    )
    parser.add_argument(
        "--error-weights", type=float, nargs="+", default=[1.0, 2.0, 4.0]
    )
    parser.add_argument(
        "--weight-decays", type=float, nargs="+", default=[0.001, 0.01]
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference, features, channel_names = load_feature_stack(args.scores)
    labels = reference["labels"].long()
    top_indices = reference["top_indices"].long()
    dev = reference["dev"].bool()
    sealed = reference["sealed"].bool()
    in_topk = top_indices.eq(labels[:, None]).any(dim=1)
    target_slots = top_indices.eq(labels[:, None]).float().argmax(dim=1)
    fold_ids = make_crossfit_folds(reference["image_ids"], dev, args.folds)
    base_scores = features[..., 0]
    base_prediction = top_indices[:, 0]

    trials = []
    stored_oof = {}
    configs = [
        (architecture, error_weight, weight_decay)
        for architecture in args.architectures
        for error_weight in args.error_weights
        for weight_decay in args.weight_decays
    ]
    for config_index, (architecture, error_weight, weight_decay) in enumerate(configs):
        oof_correction = torch.zeros_like(base_scores)
        for fold in range(args.folds):
            train_mask = dev & in_topk & fold_ids.ne(fold)
            holdout_mask = dev & fold_ids.eq(fold)
            model = fit_model(
                features,
                target_slots,
                train_mask,
                architecture,
                error_weight,
                weight_decay,
                args.epochs,
                args.learning_rate,
                args.seed + config_index * 101 + fold,
                device,
            )
            fold_scores = predict_scores(model, features, holdout_mask, device)
            oof_correction[holdout_mask] = fold_scores[holdout_mask] - base_scores[holdout_mask]
        for beta in args.betas:
            scores = base_scores + beta * oof_correction
            _prediction, stats = evaluate_prediction(top_indices, scores, labels, dev)
            trials.append(
                {
                    "architecture": architecture,
                    "error_weight": error_weight,
                    "weight_decay": weight_decay,
                    "beta": beta,
                    "dev_oof": stats,
                }
            )
        stored_oof[(architecture, error_weight, weight_decay)] = oof_correction
        print(
            json.dumps(
                {
                    "stage": "crossfit",
                    "config": config_index + 1,
                    "configs": len(configs),
                    "architecture": architecture,
                    "error_weight": error_weight,
                    "weight_decay": weight_decay,
                    "best_net": max(row["dev_oof"]["net"] for row in trials if row["architecture"] == architecture and row["error_weight"] == error_weight and row["weight_decay"] == weight_decay),
                }
            ),
            flush=True,
        )

    selected = max(
        trials,
        key=lambda row: (
            row["dev_oof"]["net"],
            -row["dev_oof"]["changed"],
            -row["beta"],
        ),
    )
    config_key = (
        selected["architecture"],
        selected["error_weight"],
        selected["weight_decay"],
    )
    final_train_mask = dev & in_topk
    final_model = fit_model(
        features,
        target_slots,
        final_train_mask,
        selected["architecture"],
        selected["error_weight"],
        selected["weight_decay"],
        args.epochs,
        args.learning_rate,
        args.seed + 99991,
        device,
    )
    sealed_model_scores = predict_scores(final_model, features, sealed, device)
    final_scores = base_scores.clone()
    final_scores[dev] += selected["beta"] * stored_oof[config_key][dev]
    final_scores[sealed] += selected["beta"] * (
        sealed_model_scores[sealed] - base_scores[sealed]
    )
    final_prediction = top_indices.gather(
        1, final_scores.argmax(dim=1, keepdim=True)
    ).squeeze(1)
    masks = {
        "all_crossfit_plus_sealed": torch.ones(len(labels), dtype=torch.bool),
        "dev_oof": dev,
        "sealed_once": sealed,
    }
    final_stats = {
        name: paired_stats(base_prediction, final_prediction, labels, mask)
        for name, mask in masks.items()
    }

    method_predictions = []
    for channel in range(1, features.shape[2]):
        prediction = top_indices.gather(
            1, features[..., channel].argmax(dim=1, keepdim=True)
        ).squeeze(1)
        method_predictions.append(prediction)
    union = torch.zeros(len(labels), dtype=torch.bool)
    for prediction in method_predictions:
        union |= ~base_prediction.eq(labels) & prediction.eq(labels)
    union_oracle = {
        name: int((mask & union).sum()) for name, mask in masks.items()
    }
    result = {
        "protocol": {
            "score_payloads": [str(path.resolve()) for path in args.scores],
            "channels": channel_names,
            "candidate_scope": "frozen strong-base top-5",
            "training": "dev-only class-agnostic score fusion; 5-fold OOF for locked dev",
            "selection": "architecture/hyperparameters/beta selected on dev OOF only; sealed read once after selection",
            "global_assignment": False,
            "test_seen_touched": False,
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "base": {
            "rows": len(labels),
            "correct": int(base_prediction.eq(labels).sum()),
            "dev_correct": int((dev & base_prediction.eq(labels)).sum()),
            "sealed_correct": int((sealed & base_prediction.eq(labels)).sum()),
        },
        "feature_union_oracle": union_oracle,
        "selected": selected,
        "final": final_stats,
        "trials": trials,
    }
    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "selected": selected,
            "channels": channel_names,
            "final_scores": final_scores.half(),
            "prediction": final_prediction,
            "dev": dev,
            "sealed": sealed,
        },
        args.out_dir / "fusion.pt",
    )
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selected_stats = result["final"]
    report = f"""# Cross-fitted Comparator Fusion Scout

## Outcome

- Selected on dev OOF: `{selected['architecture']}`, error weight `{selected['error_weight']}`, weight decay `{selected['weight_decay']}`, beta `{selected['beta']}`.
- Locked dev OOF: net {selected_stats['dev_oof']['net']:+d} ({selected_stats['dev_oof']['wins']} wins / {selected_stats['dev_oof']['losses']} losses; {selected_stats['dev_oof']['changed']} changes).
- Sealed once: net {selected_stats['sealed_once']['net']:+d} ({selected_stats['sealed_once']['wins']} wins / {selected_stats['sealed_once']['losses']} losses; {selected_stats['sealed_once']['changed']} changes).
- Combined: net {selected_stats['all_crossfit_plus_sealed']['net']:+d}.
- Feature-union oracle: {union_oracle['all_crossfit_plus_sealed']} all / {union_oracle['dev_oof']} dev / {union_oracle['sealed_once']} sealed.

## Protocol

- Inputs are the frozen strong-base top-5 logits, six static 512px exemplar scores, and learned comparator scores at top-r 3 and 16.
- Each dev prediction is out-of-fold: its fusion model was trained on the other four folds.
- The final model was trained on all dev only after model selection; sealed was then read once.
- Fusion is class-agnostic and per-query. It does not use global class-count constraints or test-batch distribution.
- No test_seen predictions or submission were produced.
"""
    (args.out_dir / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"stage": "done", **result["final"]}), flush=True)


if __name__ == "__main__":
    main()

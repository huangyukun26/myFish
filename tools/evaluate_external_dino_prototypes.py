from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=1)


def row_z(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    out = torch.zeros_like(values, dtype=torch.float32)
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    for row in range(values.shape[0]):
        valid = mask[row]
        if int(valid.sum()) < 2:
            continue
        current = values[row, valid].float()
        out[row, valid] = (current - current.mean()) / current.std().clamp_min(1e-6)
    return out


def build_prototypes(
    payload: dict, class_count: int, classes: list[str], use_labels: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    features = normalize(payload["features"])
    if use_labels:
        class_to_idx = {name: index for index, name in enumerate(classes)}
        class_ids = torch.tensor(
            [class_to_idx.get(label, -1) for label in payload["labels"]], dtype=torch.long
        )
    else:
        class_ids = payload["class_ids"].long()
    sums = torch.zeros(class_count, features.shape[1])
    counts = torch.zeros(class_count, dtype=torch.long)
    valid = (class_ids >= 0) & (class_ids < class_count)
    sums.index_add_(0, class_ids[valid], features[valid])
    counts.index_add_(0, class_ids[valid], torch.ones(int(valid.sum()), dtype=torch.long))
    return normalize(sums / counts.clamp_min(1)[:, None]), counts


def metrics(pred: torch.Tensor, target: torch.Tensor, base: torch.Tensor) -> dict:
    correct = pred.eq(target)
    base_correct = base.eq(target)
    return {
        "rows": int(target.numel()),
        "accuracy": float(correct.float().mean()),
        "changed": int(pred.ne(base).sum()),
        "wins": int((correct & ~base_correct).sum()),
        "losses": int((~correct & base_correct).sum()),
        "net": int(correct.sum() - base_correct.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--external-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, default=None)
    args = parser.parse_args()

    base = load(args.base_logits)
    query = load(args.query_features)
    train = load(args.train_features)
    external = load(args.external_features)
    classes = list(base["classes"])
    if list(query["classes"]) != classes or list(train["classes"]) != classes:
        raise RuntimeError("base/query/train class order mismatch")
    class_count = len(classes)
    target = base["class_ids"].long()
    base_logits = base["logits"].float()
    base_pred = base_logits.argmax(1)
    query_features = normalize(query["features"])
    train_proto, train_counts = build_prototypes(train, class_count, classes)
    # External download manifests carry the iNat/fishnet source id, while the
    # competition caches use the sorted 5795-class index. Labels are the only
    # stable cross-cache key, so remap external rows by label here.
    ext_proto, ext_counts = build_prototypes(external, class_count, classes, use_labels=True)
    train_scores = query_features @ train_proto.T
    ext_scores = query_features @ ext_proto.T
    ext_mask = ext_counts.gt(0)[None, :].expand_as(ext_scores)

    output: dict[str, object] = {
        "rows": int(target.numel()),
        "classes": class_count,
        "external_rows": len(external["image_ids"]),
        "external_classes": int(ext_counts.gt(0).sum()),
        "base_accuracy": float(base_pred.eq(target).float().mean()),
        "train_proto_accuracy": float(train_scores.argmax(1).eq(target).float().mean()),
        "external_proto_accuracy_on_supported": None,
        "grid": [],
    }
    ext_direct = ext_scores.clone()
    ext_direct[:, ~ext_counts.gt(0)] = -1e9
    supported_rows = ext_counts[target].gt(0)
    if bool(supported_rows.any()):
        output["external_proto_accuracy_on_supported"] = float(
            ext_direct[supported_rows].argmax(1).eq(target[supported_rows]).float().mean()
        )

    # Standardize only over candidates that have external support. This keeps a sparse
    # gallery from turning missing classes into artificial high or low scores.
    for k in (5, 10, 20, 50):
        top = base_logits.topk(k, dim=1).indices
        candidate_mask = ext_mask.gather(1, top)
        base_top = base_logits.gather(1, top)
        base_top_z = row_z(base_top)
        ext_top = ext_scores.gather(1, top)
        ext_top_z = row_z(ext_top, candidate_mask)
        for weight in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0):
            reranked_scores = base_top_z + weight * ext_top_z
            best_local = reranked_scores.argmax(1)
            has_support = candidate_mask.any(1)
            prediction = base_pred.clone()
            prediction[has_support] = top[has_support].gather(1, best_local[has_support, None]).squeeze(1)
            output["grid"].append(
                {"k": k, "weight": weight, "support_rows": int(has_support.sum()), **metrics(prediction, target, base_pred)}
            )
        # Train visual prototype is dense and is evaluated as a second-stage signal.
        train_top = train_scores.gather(1, top)
        train_top_z = row_z(train_top)
        for weight in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            reranked = (base_top_z + weight * train_top_z).argmax(1)
            prediction = top.gather(1, reranked[:, None]).squeeze(1)
            output["grid"].append(
                {"k": k, "weight": weight, "signal": "train_proto", **metrics(prediction, target, base_pred)}
            )

    # The external gallery is valuable only where its score is reliable. Sweep a
    # conservative base-margin gate and preserve base predictions outside the gate.
    top2 = base_logits.topk(2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    top20 = base_logits.topk(20, dim=1).indices
    top20_mask = ext_mask.gather(1, top20)
    top20_base_z = row_z(base_logits.gather(1, top20))
    top20_ext_z = row_z(ext_scores.gather(1, top20), top20_mask)
    candidate = top20_ext_z.argmax(1)
    proposed = top20.gather(1, candidate[:, None]).squeeze(1)
    proposal_valid = top20_mask.any(1)
    for threshold in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        for weight in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            use = proposal_valid & margins.lt(threshold)
            prediction = base_pred.clone()
            prediction[use] = proposed[use]
            output["grid"].append(
                {"signal": "external_gate", "margin_lt": threshold, "weight": weight, "used_rows": int(use.sum()), **metrics(prediction, target, base_pred)}
            )

    output["best_positive"] = sorted(
        [row for row in output["grid"] if row["net"] > 0],
        key=lambda row: (-row["net"], row["losses"], row.get("changed", 0)),
    )[:20]
    output["best_by_low_loss"] = sorted(
        output["grid"], key=lambda row: (row["losses"], -row["net"], -row.get("wins", 0))
    )[:20]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.save_predictions:
        best = output["best_positive"][0] if output["best_positive"] else None
        if best and best.get("signal") == "external_gate":
            use = proposal_valid & margins.lt(float(best["margin_lt"]))
            final = base_pred.clone()
            final[use] = proposed[use]
        else:
            final = base_pred.clone()
        torch.save({"predictions": final, "target": target, "base_predictions": base_pred, "classes": classes}, args.save_predictions)
    print(json.dumps({k: output[k] for k in ("base_accuracy", "external_classes", "best_positive", "best_by_low_loss")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

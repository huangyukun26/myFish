from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


CURRENT_BASE_CORRECT = {"all": 9823, "dev": 6662, "sealed": 3161}


def stable_hash(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_sealed(image_ids, class_ids):
    groups = defaultdict(list)
    for index, (image_id, class_id) in enumerate(zip(image_ids, class_ids.tolist())):
        groups[int(class_id)].append((stable_hash(image_id), index))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for position, (_digest, index) in enumerate(rows):
            if position % 5 in {0, 1, 2}:
                dev[index] = True
    return dev, ~dev


def paired_stats(base_prediction, prediction, labels, mask):
    base_correct = base_prediction.eq(labels)
    candidate_correct = prediction.eq(labels)
    changed = mask & prediction.ne(base_prediction)
    wins = changed & ~base_correct & candidate_correct
    losses = changed & base_correct & ~candidate_correct
    return {
        "rows": int(mask.sum().item()),
        "base_correct": int((mask & base_correct).sum().item()),
        "candidate_correct": int((mask & candidate_correct).sum().item()),
        "net": int(wins.sum().item() - losses.sum().item()),
        "changed": int(changed.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "efficiency": float(
            (wins.sum().item() - losses.sum().item()) / max(1, changed.sum().item())
        ),
    }


def zscore_rows(values):
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True
    ).clamp_min(1e-6)


def evaluate_family(name, local_scores, top_values, top_indices, labels, masks, alphas):
    base_prediction = top_indices[:, 0]
    local_choice = local_scores.argmax(dim=1)
    local_prediction = top_indices.gather(1, local_choice[:, None]).squeeze(1)
    base_correct = base_prediction.eq(labels)
    local_correct = local_prediction.eq(labels)
    oracle = (~base_correct) & local_correct
    base_z = zscore_rows(top_values.float())
    local_z = zscore_rows(local_scores.float())
    trials = []
    for alpha in alphas:
        choice = (base_z + alpha * local_z).argmax(dim=1)
        prediction = top_indices.gather(1, choice[:, None]).squeeze(1)
        row = {"alpha": alpha}
        for split_name, mask in masks.items():
            row[split_name] = paired_stats(
                base_prediction,
                prediction,
                labels,
                mask,
            )
            row[split_name]["raw_vs_current_count"] = (
                row[split_name]["candidate_correct"] - CURRENT_BASE_CORRECT[split_name]
            )
        trials.append(row)
    best = max(
        trials,
        key=lambda row: (
            row["dev"]["net"],
            -row["dev"]["changed"],
            -row["alpha"],
        ),
    )
    return {
        "name": name,
        "local_standalone": {
            split_name: {
                "correct": int((local_correct & mask).sum().item()),
                "oracle_complement": int((oracle & mask).sum().item()),
                "raw_vs_current_count": int((local_correct & mask).sum().item())
                - CURRENT_BASE_CORRECT[split_name],
                **paired_stats(
                    base_prediction,
                    local_prediction,
                    labels,
                    mask,
                ),
            }
            for split_name, mask in masks.items()
        },
        "best_by_dev": best,
        "trials": trials,
    }


def top_r_mean(values, take):
    return values.topk(min(take, len(values))).values.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parts", type=Path, required=True)
    parser.add_argument("--val-parts", type=Path, required=True)
    parser.add_argument("--train-full-cls", type=Path, required=True)
    parser.add_argument("--val-full-cls", type=Path, required=True)
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--preselect", type=int, default=16)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.35, 0.50],
    )
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = torch.load(args.train_parts, map_location="cpu", weights_only=False)
    val = torch.load(args.val_parts, map_location="cpu", weights_only=False)
    train_full = torch.load(args.train_full_cls, map_location="cpu", weights_only=False)
    val_full = torch.load(args.val_full_cls, map_location="cpu", weights_only=False)
    base = torch.load(args.base_logits, map_location="cpu", weights_only=False)
    if list(train["image_ids"]) != list(train_full["image_ids"]):
        raise RuntimeError("Train part cache and full CLS cache are not aligned")
    if list(val["image_ids"]) != list(val_full["image_ids"]):
        raise RuntimeError("Val part cache and full CLS cache are not aligned")
    if list(val["image_ids"]) != list(base["image_ids"]):
        raise RuntimeError("Val part cache and base logits are not aligned")
    if not torch.equal(val["class_ids"].long(), base["class_ids"].long()):
        raise RuntimeError("Val labels and base labels do not match")

    labels = val["class_ids"].long()
    logits = base["logits"].float()
    top_values, top_indices = logits.topk(args.topk, dim=1)
    gallery_by_class = defaultdict(list)
    for index, class_id in enumerate(train["class_ids"].tolist()):
        gallery_by_class[int(class_id)].append(index)

    train_full_cls = train_full["features"].to(device)
    val_full_cls = val_full["features"].to(device)
    train_roi_cls = train["cls"].to(device)
    val_roi_cls = val["cls"].to(device)
    train_roi_mean = train["valid_mean"].to(device)
    val_roi_mean = val["valid_mean"].to(device)
    train_parts = train["parts"].to(device)
    val_parts = val["parts"].to(device)

    score_names = [
        "full_cls_control",
        "roi_global",
        "flip_aligned_parts",
        "unordered_part_chamfer",
        "local_flip_chamfer",
        "roi_global_0p3_local_0p7",
    ]
    scores = {
        name: torch.empty(
            (len(labels), args.topk),
            dtype=torch.float32,
            device=device,
        )
        for name in score_names
    }
    flip_order = torch.tensor([2, 1, 0, 5, 4, 3], dtype=torch.long, device=device)

    with torch.inference_mode():
        for query_index in range(len(labels)):
            candidate_classes = top_indices[query_index].tolist()
            query_full = val_full_cls[query_index].float()
            query_roi_cls = val_roi_cls[query_index].float()
            query_roi_mean = val_roi_mean[query_index].float()
            query_parts = val_parts[query_index].float()
            for owner, class_id in enumerate(candidate_classes):
                gallery = gallery_by_class.get(int(class_id), [])
                if not gallery:
                    for name in score_names:
                        scores[name][query_index, owner] = -1.0
                    continue
                gallery_tensor = torch.tensor(gallery, dtype=torch.long, device=device)
                preselect_similarity = train_full_cls.index_select(
                    0, gallery_tensor
                ).float().matmul(query_full)
                if len(gallery) > args.preselect:
                    selected_positions = preselect_similarity.topk(args.preselect).indices
                    gallery_tensor = gallery_tensor.index_select(0, selected_positions)
                    preselect_similarity = preselect_similarity.index_select(
                        0, selected_positions
                    )
                exemplar_roi_cls = train_roi_cls.index_select(0, gallery_tensor).float()
                exemplar_roi_mean = train_roi_mean.index_select(0, gallery_tensor).float()
                exemplar_parts = train_parts.index_select(0, gallery_tensor).float()
                roi_global = 0.5 * exemplar_roi_cls.matmul(query_roi_cls)
                roi_global += 0.5 * exemplar_roi_mean.matmul(query_roi_mean)
                aligned = (exemplar_parts * query_parts[None]).sum(dim=2).mean(dim=1)
                flipped = (
                    exemplar_parts.index_select(1, flip_order)
                    * query_parts[None]
                ).sum(dim=2).mean(dim=1)
                flip_aligned = torch.maximum(aligned, flipped)
                similarity = torch.einsum(
                    "nkd,qd->nkq",
                    exemplar_parts,
                    query_parts,
                )
                chamfer = 0.5 * (
                    similarity.max(dim=2).values.mean(dim=1)
                    + similarity.max(dim=1).values.mean(dim=1)
                )
                local = 0.5 * flip_aligned + 0.5 * chamfer
                combined = 0.3 * roi_global + 0.7 * local
                scores["full_cls_control"][query_index, owner] = top_r_mean(
                    preselect_similarity, args.top_r
                )
                scores["roi_global"][query_index, owner] = top_r_mean(
                    roi_global, args.top_r
                )
                scores["flip_aligned_parts"][query_index, owner] = top_r_mean(
                    flip_aligned, args.top_r
                )
                scores["unordered_part_chamfer"][query_index, owner] = top_r_mean(
                    chamfer, args.top_r
                )
                scores["local_flip_chamfer"][query_index, owner] = top_r_mean(
                    local, args.top_r
                )
                scores["roi_global_0p3_local_0p7"][query_index, owner] = top_r_mean(
                    combined, args.top_r
                )
            if args.log_every and (
                (query_index + 1) % args.log_every == 0
                or query_index + 1 == len(labels)
            ):
                print(
                    json.dumps(
                        {
                            "queries": query_index + 1,
                            "total": len(labels),
                            "device": str(device),
                        }
                    ),
                    flush=True,
                )

    scores = {name: value.cpu() for name, value in scores.items()}
    dev, sealed = split_dev_sealed(val["image_ids"], labels)
    masks = {
        "all": torch.ones(len(labels), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    base_prediction = top_indices[:, 0]
    base_correct = base_prediction.eq(labels)
    base_top5 = top_indices.eq(labels[:, None]).any(dim=1)
    classes = base.get("classes", [])
    genera = [name.split(maxsplit=1)[0] if name else "" for name in classes]
    same_genus_wrong = torch.tensor(
        [
            bool(genera[int(pred)] == genera[int(target)])
            for pred, target in zip(base_prediction.tolist(), labels.tolist())
        ],
        dtype=torch.bool,
    ) & ~base_correct
    result = {
        "protocol": {
            "train_parts": str(args.train_parts),
            "val_parts": str(args.val_parts),
            "train_full_cls": str(args.train_full_cls),
            "val_full_cls": str(args.val_full_cls),
            "base_logits": str(args.base_logits),
            "topk": args.topk,
            "preselect": args.preselect,
            "top_r": args.top_r,
            "alphas": args.alphas,
            "candidate_scope": "base top-k only",
            "selection": "alpha selected on deterministic per-class dev; sealed read once",
        },
        "base": {
            "rows": len(labels),
            "correct": int(base_correct.sum().item()),
            "errors": int((~base_correct).sum().item()),
            "top5_correct": int(base_top5.sum().item()),
            "true_in_top5_errors": int((~base_correct & base_top5).sum().item()),
            "same_genus_errors": int(same_genus_wrong.sum().item()),
            "dev_rows": int(dev.sum().item()),
            "sealed_rows": int(sealed.sum().item()),
            "dev_correct": int((dev & base_correct).sum().item()),
            "sealed_correct": int((sealed & base_correct).sum().item()),
        },
        "families": [
            evaluate_family(
                name,
                scores[name],
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            )
            for name in score_names
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    torch.save(
        {
            "image_ids": val["image_ids"],
            "labels": labels,
            "top_indices": top_indices,
            "top_values": top_values.half(),
            "scores": {name: value.half() for name, value in scores.items()},
            "dev": dev,
            "sealed": sealed,
        },
        args.out.with_suffix(".scores.pt"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

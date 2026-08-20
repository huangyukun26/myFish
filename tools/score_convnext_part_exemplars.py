from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


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
    base_z = zscore_rows(top_values.float())
    local_z = zscore_rows(local_scores.float())
    local_choice = local_scores.argmax(dim=1)
    local_prediction = top_indices.gather(1, local_choice[:, None]).squeeze(1)
    base_correct = base_prediction.eq(labels)
    local_correct = local_prediction.eq(labels)
    oracle = (~base_correct) & local_correct
    trials = []
    for alpha in alphas:
        choice = (base_z + alpha * local_z).argmax(dim=1)
        prediction = top_indices.gather(1, choice[:, None]).squeeze(1)
        row = {"alpha": alpha}
        row.update(
            {
                split_name: paired_stats(
                    base_prediction,
                    prediction,
                    labels,
                    split_mask,
                )
                for split_name, split_mask in masks.items()
            }
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--base-logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
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
    train = torch.load(str(args.train_cache), map_location="cpu")
    val = torch.load(str(args.val_cache), map_location="cpu")
    base = torch.load(str(args.base_logits), map_location="cpu")
    if list(val["image_ids"]) != list(base["image_ids"]):
        raise RuntimeError("Validation cache and base logits image_id order do not match")
    if not torch.equal(val["class_ids"].long(), base["class_ids"].long()):
        raise RuntimeError("Validation cache and base logits labels do not match")

    labels = val["class_ids"].long()
    logits = base["logits"].float()
    top_values, top_indices = logits.topk(args.topk, dim=1)
    train_classes = train["class_ids"].long()
    gallery_by_class = defaultdict(list)
    for index, class_id in enumerate(train_classes.tolist()):
        gallery_by_class[int(class_id)].append(index)

    train_global = train["global"].to(device)
    train_mid_parts = train["mid_parts"].to(device)
    train_parts = train["parts"].to(device)
    val_global = val["global"].to(device)
    val_mid_parts = val["mid_parts"].to(device)
    val_parts = val["parts"].to(device)
    global_scores = torch.empty(
        (len(labels), args.topk), dtype=torch.float32, device=device
    )
    part_scores = torch.empty_like(global_scores)
    mid_part_scores = torch.empty_like(global_scores)
    two_level_part_scores = torch.empty_like(global_scores)
    combined_scores = torch.empty_like(global_scores)

    with torch.no_grad():
        for query_index in range(len(labels)):
            candidate_classes = top_indices[query_index].tolist()
            gallery_indices = []
            owners = []
            for owner, class_id in enumerate(candidate_classes):
                class_gallery = gallery_by_class.get(int(class_id), [])
                gallery_indices.extend(class_gallery)
                owners.extend([owner] * len(class_gallery))
            if not gallery_indices:
                raise RuntimeError("No gallery exemplars for query %d" % query_index)
            gallery_index_tensor = torch.tensor(
                gallery_indices, dtype=torch.long, device=device
            )
            owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
            exemplar_global = train_global.index_select(0, gallery_index_tensor)
            exemplar_mid_parts = train_mid_parts.index_select(0, gallery_index_tensor)
            exemplar_parts = train_parts.index_select(0, gallery_index_tensor)
            query_global = val_global[query_index].float()
            query_mid_parts = val_mid_parts[query_index].float()
            query_parts = val_parts[query_index].float()
            global_per_exemplar = exemplar_global.float().matmul(query_global)
            mid_similarity = torch.einsum(
                "nkd,qd->nkq",
                exemplar_mid_parts.float(),
                query_mid_parts,
            )
            mid_part_per_exemplar = 0.5 * (
                mid_similarity.max(dim=2).values.mean(dim=1)
                + mid_similarity.max(dim=1).values.mean(dim=1)
            )
            similarity = torch.einsum(
                "nkd,qd->nkq",
                exemplar_parts.float(),
                query_parts,
            )
            part_per_exemplar = 0.5 * (
                similarity.max(dim=2).values.mean(dim=1)
                + similarity.max(dim=1).values.mean(dim=1)
            )
            two_level_part_per_exemplar = (
                0.5 * mid_part_per_exemplar + 0.5 * part_per_exemplar
            )
            combined_per_exemplar = (
                0.3 * global_per_exemplar + 0.7 * two_level_part_per_exemplar
            )
            for owner in range(args.topk):
                selected = owner_tensor.eq(owner)
                count = int(selected.sum().item())
                if count == 0:
                    global_scores[query_index, owner] = -1.0
                    mid_part_scores[query_index, owner] = -1.0
                    part_scores[query_index, owner] = -1.0
                    two_level_part_scores[query_index, owner] = -1.0
                    combined_scores[query_index, owner] = -1.0
                    continue
                take = min(args.top_r, count)
                global_scores[query_index, owner] = (
                    global_per_exemplar[selected].topk(take).values.mean()
                )
                mid_part_scores[query_index, owner] = (
                    mid_part_per_exemplar[selected].topk(take).values.mean()
                )
                part_scores[query_index, owner] = (
                    part_per_exemplar[selected].topk(take).values.mean()
                )
                two_level_part_scores[query_index, owner] = (
                    two_level_part_per_exemplar[selected].topk(take).values.mean()
                )
                combined_scores[query_index, owner] = (
                    combined_per_exemplar[selected].topk(take).values.mean()
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

    global_scores = global_scores.cpu()
    mid_part_scores = mid_part_scores.cpu()
    part_scores = part_scores.cpu()
    two_level_part_scores = two_level_part_scores.cpu()
    combined_scores = combined_scores.cpu()
    dev, sealed = split_dev_sealed(val["image_ids"], labels)
    masks = {
        "all": torch.ones(len(labels), dtype=torch.bool),
        "dev": dev,
        "sealed": sealed,
    }
    base_prediction = top_indices[:, 0]
    base_top5 = top_indices.eq(labels[:, None]).any(dim=1)
    classes = base.get("classes", [])
    genera = [name.split(maxsplit=1)[0] if name else "" for name in classes]
    wrong = base_prediction.ne(labels)
    same_genus_wrong = torch.tensor(
        [
            bool(genera[int(pred)] == genera[int(target)])
            for pred, target in zip(base_prediction.tolist(), labels.tolist())
        ],
        dtype=torch.bool,
    ) & wrong
    result = {
        "protocol": {
            "train_cache": str(args.train_cache),
            "val_cache": str(args.val_cache),
            "base_logits": str(args.base_logits),
            "topk": args.topk,
            "top_r": args.top_r,
            "alphas": args.alphas,
            "selection": "alpha selected on deterministic per-class dev; sealed read once",
            "candidate_scope": "base top-k only",
        },
        "base": {
            "rows": len(labels),
            "correct": int(base_prediction.eq(labels).sum().item()),
            "errors": int(wrong.sum().item()),
            "top5_correct": int(base_top5.sum().item()),
            "true_in_top5_errors": int((wrong & base_top5).sum().item()),
            "same_genus_errors": int(same_genus_wrong.sum().item()),
            "dev_rows": int(dev.sum().item()),
            "sealed_rows": int(sealed.sum().item()),
        },
        "families": [
            evaluate_family(
                "global_exemplar",
                global_scores,
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            ),
            evaluate_family(
                "mid_unordered_part_chamfer",
                mid_part_scores,
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            ),
            evaluate_family(
                "unordered_part_chamfer",
                part_scores,
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            ),
            evaluate_family(
                "two_level_unordered_part_chamfer",
                two_level_part_scores,
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            ),
            evaluate_family(
                "global_0p3_two_level_part_0p7",
                combined_scores,
                top_values,
                top_indices,
                labels,
                masks,
                args.alphas,
            ),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    scores_path = args.out.with_suffix(".scores.pt")
    torch.save(
        {
            "image_ids": val["image_ids"],
            "labels": labels,
            "top_indices": top_indices,
            "top_values": top_values.half(),
            "global_scores": global_scores.half(),
            "mid_part_scores": mid_part_scores.half(),
            "part_scores": part_scores.half(),
            "two_level_part_scores": two_level_part_scores.half(),
            "combined_scores": combined_scores.half(),
            "dev": dev,
            "sealed": sealed,
        },
        str(scores_path),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Build taxon/fish top-50-union features for the group-aware reranker.

The table contains only query/candidate scores and geometry.  Candidate class
identity is represented by scores/ranks/genus agreement, never by a learned
class embedding or class frequency.  It is saved as a compact tensor package
so the 6M-ish rows do not expand into a multi-gigabyte CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=1)


def aspect_ratios(image_ids: list[str], image_root: Path) -> torch.Tensor:
    from PIL import Image

    values: list[float] = []
    for image_id in image_ids:
        try:
            with Image.open(image_root / Path(image_id).name) as im:
                w, h = im.size
            values.append(float(max(w, h) / max(1, min(w, h))))
        except Exception:
            values.append(1.0)
    return torch.tensor(values, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("runs/research_next_20260820/full_universe"))
    parser.add_argument("--text-dir", type=Path, default=Path("work/clip_text_features"))
    parser.add_argument("--train-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_complement_train.pt"))
    parser.add_argument("--val-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_val.pt"))
    parser.add_argument("--image-root", type=Path, default=Path("dataset/images"))
    parser.add_argument("--top-dir", type=Path, default=Path("runs/research_next_20260820/full_universe"))
    parser.add_argument("--out", type=Path, default=Path("runs/research_next_20260820/full_universe/candidate_table.pt"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    manifest = list(csv.DictReader((args.manifest_dir / "query_manifest.csv").open("r", encoding="utf-8", newline="")))
    true_ids = torch.tensor([int(r["class_id"]) for r in manifest], dtype=torch.long)
    true_genus = [r["genus"] for r in manifest]
    species_fold = torch.tensor([int(r["species_fold"]) for r in manifest], dtype=torch.int16)
    genus_fold = torch.tensor([int(r["genus_fold"]) for r in manifest], dtype=torch.int16)
    image_ids = [r["image_id"] for r in manifest]
    aspect = aspect_ratios(image_ids, args.image_root)

    train = load(args.train_feature)
    val = load(args.val_feature)
    query = normalize(torch.cat([train["features"].float(), val["features"].float()], dim=0))
    tax_text = load(args.text_dir / "all_bioclip25_taxon.pt")
    fish_text = load(args.text_dir / "all_bioclip25_fish.pt")
    classes = [str(x) for x in tax_text["classes"]]
    if [str(x) for x in fish_text["classes"]] != classes:
        raise RuntimeError("taxon/fish class order mismatch")
    genus_to_id: dict[str, int] = {}
    candidate_genus: list[int] = []
    for name in classes:
        genus = name.split()[0] if name else ""
        if genus not in genus_to_id:
            genus_to_id[genus] = len(genus_to_id)
        candidate_genus.append(genus_to_id[genus])
    candidate_genus_t = torch.tensor(candidate_genus, dtype=torch.long)
    true_genus_id = torch.tensor([genus_to_id[x] for x in true_genus], dtype=torch.long)

    tax_top = load(args.top_dir / "top100_taxon.pt")["top_ids"].long()
    fish_top = load(args.top_dir / "top100_fish.pt")["top_ids"].long()
    if tax_top.shape[0] != query.shape[0] or fish_top.shape != tax_top.shape:
        raise RuntimeError("top cache/query shape mismatch")
    tax_text_n = normalize(tax_text["features"])
    fish_text_n = normalize(fish_text["features"])

    # Feature names are intentionally explicit and class-independent.
    feature_names = [
        "taxon_score", "taxon_z", "taxon_rank", "fish_score", "fish_z", "fish_rank",
        "both_support", "taxon_top1_margin", "fish_top1_margin", "taxon_entropy_top50", "fish_entropy_top50",
        "taxon_genus_agree", "fish_genus_agree", "candidate_genus_same_taxon_top1", "candidate_genus_same_fish_top1",
        "taxon_gap", "fish_gap", "aspect_ratio",
    ]
    feature_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    true_parts: list[torch.Tensor] = []
    species_parts: list[torch.Tensor] = []
    genus_parts: list[torch.Tensor] = []
    valid_rows = 0

    with torch.inference_mode():
        for start in range(0, query.shape[0], args.batch_size):
            end = min(query.shape[0], start + args.batch_size)
            q = query[start:end].to(device)
            tt = tax_top[start:end, :50].to(device)
            ft = fish_top[start:end, :50].to(device)
            # Union slots are fixed at 100; duplicate IDs are masked.
            union = torch.cat([tt, ft], dim=1)
            keep = torch.ones(union.shape, dtype=torch.bool, device=device)
            for j in range(1, union.shape[1]):
                keep[:, j] = ~(union[:, j : j + 1] == union[:, :j]).any(dim=1)
            union = union.masked_fill(~keep, 0)
            valid = keep
            union_cpu = union.cpu()
            tc = tax_text_n[union_cpu.view(-1)].view(union.shape[0], union.shape[1], -1).to(device)
            fc = fish_text_n[union_cpu.view(-1)].view(union.shape[0], union.shape[1], -1).to(device)
            tax_s = (tc * q[:, None, :]).sum(dim=2)
            fish_s = (fc * q[:, None, :]).sum(dim=2)
            # A candidate absent from an expert's top-50 receives a sentinel;
            # union slots are always drawn from at least one source.
            eq_tax = union[:, :, None].eq(tt[:, None, :])
            eq_fish = union[:, :, None].eq(ft[:, None, :])
            tax_rank = torch.where(eq_tax.any(dim=2), eq_tax.float().argmax(dim=2) + 1, torch.full(union.shape, 101.0, device=device))
            fish_rank = torch.where(eq_fish.any(dim=2), eq_fish.float().argmax(dim=2) + 1, torch.full(union.shape, 101.0, device=device))
            tax_valid = tax_rank <= 50
            fish_valid = fish_rank <= 50
            tax_score = torch.where(tax_valid, tax_s, torch.full_like(tax_s, -5.0))
            fish_score = torch.where(fish_valid, fish_s, torch.full_like(fish_s, -5.0))
            tax_vals = (q[:, None, :] * tax_text_n[tt.cpu()].to(device)).sum(dim=2)
            fish_vals = (q[:, None, :] * fish_text_n[ft.cpu()].to(device)).sum(dim=2)
            tax_top1 = tax_vals[:, 0]
            fish_top1 = fish_vals[:, 0]
            tax_margin = tax_vals[:, 0] - tax_vals[:, 1]
            fish_margin = fish_vals[:, 0] - fish_vals[:, 1]
            tax_prob = torch.softmax(tax_vals.float(), dim=1)
            fish_prob = torch.softmax(fish_vals.float(), dim=1)
            tax_entropy = -(tax_prob * (tax_prob.clamp_min(1e-8).log())).sum(dim=1)
            fish_entropy = -(fish_prob * (fish_prob.clamp_min(1e-8).log())).sum(dim=1)
            cand_gen = candidate_genus_t[union.cpu()].to(device)
            top_tax_gen = candidate_genus_t[tt[:, 0].cpu()].to(device)
            top_fish_gen = candidate_genus_t[ft[:, 0].cpu()].to(device)
            def zscore(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
                count = valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                safe = values * valid_mask.float()
                mean = safe.sum(dim=1, keepdim=True) / count
                var = ((safe - mean) * valid_mask.float()).pow(2).sum(dim=1, keepdim=True) / count
                out = (values - mean) / var.sqrt().clamp_min(1e-5)
                return torch.where(valid_mask, out, torch.full_like(out, -5.0))
            tax_z = zscore(tax_score, tax_valid)
            fish_z = zscore(fish_score, fish_valid)
            feats = torch.stack([
                tax_score, tax_z, tax_rank.float(), fish_score, fish_z, fish_rank.float(),
                (tax_valid & fish_valid).float(), tax_margin[:, None].expand_as(tax_score), fish_margin[:, None].expand_as(fish_score),
                tax_entropy[:, None].expand_as(tax_score), fish_entropy[:, None].expand_as(fish_score),
                (cand_gen == top_tax_gen[:, None]).float(), (cand_gen == top_fish_gen[:, None]).float(),
                (cand_gen == top_tax_gen[:, None]).float(), (cand_gen == top_fish_gen[:, None]).float(),
                (tax_top1[:, None] - tax_score), (fish_top1[:, None] - fish_score), aspect[start:end].to(device)[:, None].expand_as(tax_score),
            ], dim=2)
            flat_valid = valid.view(-1)
            flat_valid_cpu = flat_valid.cpu()
            feature_parts.append(feats.view(-1, len(feature_names))[flat_valid].cpu().half())
            query_parts.append(torch.arange(start, end, dtype=torch.int32, device=device)[:, None].expand_as(union).reshape(-1)[flat_valid].cpu())
            candidate_parts.append(union.view(-1)[flat_valid].cpu().to(torch.int32))
            true_parts.append(true_ids[start:end, None].expand(union.shape[0], union.shape[1]).reshape(-1)[flat_valid_cpu].to(torch.int32))
            species_parts.append(species_fold[start:end, None].expand(union.shape[0], union.shape[1]).reshape(-1)[flat_valid_cpu])
            genus_parts.append(genus_fold[start:end, None].expand(union.shape[0], union.shape[1]).reshape(-1)[flat_valid_cpu])
            valid_rows += int(flat_valid.sum())
            if start == 0 or end == query.shape[0] or (start // args.batch_size) % 50 == 0:
                print(f"table {end}/{query.shape[0]} rows={valid_rows}")

    payload = {
        "features": torch.cat(feature_parts, dim=0),
        "query_id": torch.cat(query_parts, dim=0),
        "candidate_id": torch.cat(candidate_parts, dim=0),
        "true_id": torch.cat(true_parts, dim=0),
        "species_fold": torch.cat(species_parts, dim=0),
        "genus_fold": torch.cat(genus_parts, dim=0),
        "feature_names": feature_names,
        "candidate_classes": classes,
        "rows": valid_rows,
        "queries": int(query.shape[0]),
        "candidate_k": 50,
        "protocol": "taxon_fish_top50_union",
    }
    torch.save(payload, args.out)
    (args.out.with_suffix(".json")).write_text(json.dumps({k: v for k, v in payload.items() if isinstance(v, (str, int, float, list))}, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": valid_rows, "queries": int(query.shape[0]), "features": feature_names}, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate frozen BioCLIP experts against all 17,393 official classes.

The evaluator is deliberately chunked: a 6 GB GPU only holds one query batch
and a 17,393-way score matrix at a time.  Top-100 IDs are checkpointed per
expert, so an interrupted run can be resumed without recomputing completed
experts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


EXPERTS = {
    "taxon": "all_bioclip25_taxon.pt",
    "fish": "all_bioclip25_fish.pt",
    "fish_taxon_avg": "all_bioclip25_fish_taxon_avg.pt",
    "fish01_taxon09_avg": "all_bioclip25_fish01_taxon09_avg.pt",
    "visual_traits": "all_bioclip25_visual_traits.pt",
}
KS = (1, 5, 10, 20, 50, 100)


def load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=1)


def compute_top100(query: torch.Tensor, text: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    q = normalize(query)
    t = normalize(text)
    out: list[torch.Tensor] = []
    use_half = device.type == "cuda"
    if use_half:
        t = t.to(device=device, dtype=torch.float16)
    else:
        t = t.to(device=device)
    with torch.inference_mode():
        for start in range(0, q.shape[0], batch_size):
            qb = q[start : start + batch_size].to(device=device)
            if use_half:
                qb = qb.half()
            scores = qb @ t.T
            top = torch.topk(scores, k=100, dim=1, largest=True, sorted=True).indices
            out.append(top.cpu().to(torch.int32))
    return torch.cat(out, dim=0)


def metric_rows(top: torch.Tensor, true_ids: torch.Tensor, true_genus: torch.Tensor, candidate_genus: torch.Tensor, protocol: str, fold: str, expert: str) -> list[dict[str, Any]]:
    top = top.long()
    rows: list[dict[str, Any]] = []
    n = int(true_ids.numel())
    if n == 0:
        return rows
    genus_top = candidate_genus[top]
    for k in KS:
        species_hit = top[:, :k].eq(true_ids[:, None]).any(dim=1)
        genus_hit = genus_top[:, :k].eq(true_genus[:, None]).any(dim=1)
        rows.append({
            "protocol": protocol,
            "fold": fold,
            "expert": expert,
            "k": k,
            "rows": n,
            "species_recall": float(species_hit.float().mean()),
            "genus_recall": float(genus_hit.float().mean()),
        })
    return rows


def union_metric_rows(taxon: torch.Tensor, fish: torch.Tensor, all_tops: list[torch.Tensor], true_ids: torch.Tensor, protocol: str, fold: str, candidate_genus: torch.Tensor, true_genus: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(true_ids.numel())
    if n == 0:
        return rows
    for k in KS:
        pair = torch.cat([taxon[:, :k], fish[:, :k]], dim=1)
        pair_hit = pair.eq(true_ids[:, None]).any(dim=1)
        pair_genus = candidate_genus[pair].eq(true_genus[:, None]).any(dim=1)
        any_top = torch.cat([x[:, :k] for x in all_tops], dim=1)
        any_hit = any_top.eq(true_ids[:, None]).any(dim=1)
        rows.append({
            "protocol": protocol,
            "fold": fold,
            "expert": "union_taxon_fish",
            "k": k,
            "rows": n,
            "species_recall": float(pair_hit.float().mean()),
            "genus_recall": float(pair_genus.float().mean()),
            "all_expert_union_species_recall": float(any_hit.float().mean()),
        })
    # Oracle expert selection is top-1 only: whether any frozen expert had the
    # answer at rank one.  This estimates routing headroom, not a rule.
    top1 = torch.cat([x[:, :1] for x in all_tops], dim=1)
    oracle = top1.eq(true_ids[:, None]).any(dim=1)
    rows.append({
        "protocol": protocol,
        "fold": fold,
        "expert": "oracle_any_expert_top1",
        "k": 1,
        "rows": n,
        "species_recall": float(oracle.float().mean()),
        "genus_recall": float(candidate_genus[top1].eq(true_genus[:, None]).any(dim=1).float().mean()),
        "all_expert_union_species_recall": float(oracle.float().mean()),
    })
    return rows


def plot_bars(path: Path, series: dict[str, list[float]], labels: list[str], title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 850
    left, right, top, bottom = 120, 50, 90, 130
    im = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
    dr.text((left, 25), title, fill="black", font=title_font)
    pw, ph = width - left - right, height - top - bottom
    vmax = max([max(v) for v in series.values()] + [0.01])
    group = pw / max(1, len(labels))
    count = max(1, len(series))
    colors = [(55, 110, 190), (210, 95, 65), (70, 150, 95), (145, 90, 170), (220, 150, 45)]
    for si, (name, values) in enumerate(series.items()):
        for i, val in enumerate(values):
            bw = group / (count + 1) * 0.8
            x = left + i * group + (si + 0.5) * group / count - bw / 2
            y = height - bottom - int(float(val) / vmax * ph)
            dr.rectangle((x, y, x + bw, height - bottom), fill=colors[si % len(colors)])
    dr.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    for i, label in enumerate(labels):
        dr.text((left + i * group + group * 0.3, height - bottom + 10), str(label), fill="black", font=font)
    for si, name in enumerate(series):
        x = left + si * 230
        y = height - 55
        dr.rectangle((x, y, x + 18, y + 18), fill=colors[si % len(colors)])
        dr.text((x + 24, y + 1), name, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("runs/research_next_20260820/full_universe"))
    parser.add_argument("--text-dir", type=Path, default=Path("work/clip_text_features"))
    parser.add_argument("--train-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_complement_train.pt"))
    parser.add_argument("--val-feature", type=Path, default=Path("work/cloud_20260713/bioclip25_hflip_priority_val.pt"))
    parser.add_argument("--pseudo-dir", type=Path, default=Path("work/cloud_20260713"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/research_next_20260820/full_universe"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experts", nargs="+", default=list(EXPERTS))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    manifest_path = args.manifest_dir / "query_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"build the manifest first: {manifest_path}")

    manifest = list(csv.DictReader(manifest_path.open("r", encoding="utf-8", newline="")))
    true_ids = torch.tensor([int(r["class_id"]) for r in manifest], dtype=torch.long)
    genus_names = [r["genus"] for r in manifest]
    species_fold = torch.tensor([int(r["species_fold"]) for r in manifest], dtype=torch.long)
    genus_fold = torch.tensor([int(r["genus_fold"]) for r in manifest], dtype=torch.long)
    # The full class order is read once from the taxon cache and is asserted for
    # every expert to prevent silent all-vs-unseen index mixing.
    taxon_payload = load(args.text_dir / EXPERTS["taxon"])
    full_classes = [str(x) for x in taxon_payload["classes"]]
    name_to_genus = torch.tensor([], dtype=torch.long)
    genus_to_id: dict[str, int] = {}
    candidate_genus_ids: list[int] = []
    for name in full_classes:
        genus = name.split()[0] if name else ""
        if genus not in genus_to_id:
            genus_to_id[genus] = len(genus_to_id)
        candidate_genus_ids.append(genus_to_id[genus])
    candidate_genus = torch.tensor(candidate_genus_ids, dtype=torch.long)
    true_genus = torch.tensor([genus_to_id[x] for x in genus_names], dtype=torch.long)

    train_payload = load(args.train_feature)
    val_payload = load(args.val_feature)
    query_features = torch.cat([train_payload["features"].float(), val_payload["features"].float()], dim=0)
    if len(manifest) != query_features.shape[0]:
        raise RuntimeError(f"manifest/features mismatch: {len(manifest)} vs {query_features.shape[0]}")

    top_by_expert: dict[str, torch.Tensor] = {}
    for expert in args.experts:
        if expert not in EXPERTS:
            raise ValueError(f"unknown expert {expert}; choose from {list(EXPERTS)}")
        cache_path = args.text_dir / EXPERTS[expert]
        payload = load(cache_path)
        classes = [str(x) for x in payload["classes"]]
        if classes != full_classes:
            raise RuntimeError(f"class order mismatch in {cache_path}")
        top_path = args.out_dir / f"top100_{expert}.pt"
        if top_path.exists():
            top = load(top_path)
            top = top["top_ids"] if isinstance(top, dict) else top
            top_by_expert[expert] = top.to(torch.int32)
            print(f"reuse {expert}: {top_path}")
            continue
        print(f"computing {expert} on {device} ({query_features.shape[0]} queries)")
        top = compute_top100(query_features, payload["features"], device, args.batch_size)
        torch.save({"top_ids": top, "expert": expert, "candidate_classes": full_classes}, top_path)
        top_by_expert[expert] = top
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_rows_out: list[dict[str, Any]] = []
    for protocol, fold_tensor in (("species_heldout", species_fold), ("genus_heldout", genus_fold)):
        for fold in sorted(set(int(x) for x in fold_tensor.tolist())):
            mask = fold_tensor.eq(fold)
            ids = true_ids[mask]
            tg = true_genus[mask]
            selected = [top_by_expert[name][mask] for name in args.experts]
            for name, top in zip(args.experts, selected):
                metric_rows_out.extend(metric_rows(top, ids, tg, candidate_genus, protocol, str(fold), name))
            if "taxon" in top_by_expert and "fish" in top_by_expert:
                metric_rows_out.extend(union_metric_rows(top_by_expert["taxon"][mask], top_by_expert["fish"][mask], selected, ids, protocol, str(fold), candidate_genus, tg))
        for name in args.experts:
            metric_rows_out.extend(metric_rows(top_by_expert[name], true_ids, true_genus, candidate_genus, protocol, "all", name))
        if "taxon" in top_by_expert and "fish" in top_by_expert:
            metric_rows_out.extend(union_metric_rows(top_by_expert["taxon"], top_by_expert["fish"], [top_by_expert[n] for n in args.experts], true_ids, protocol, "all", candidate_genus, true_genus))

    # Re-evaluate the historical 1,000/1,001-query proxy against the full
    # 17,393-way pool.  It is explicitly labelled query-count proxy below,
    # never presented as a 1,000-class benchmark.
    small_specs = [("small_species_1000", "pseudo_species_hflip_letterbox_avg.pt"), ("small_genus_1001", "pseudo_genus_hflip_letterbox_avg.pt")]
    small_metrics: list[dict[str, Any]] = []
    for protocol, filename in small_specs:
        path = args.pseudo_dir / filename
        if not path.exists():
            continue
        payload = load(path)
        labels = [str(x) for x in payload["labels"]]
        ids = torch.tensor([full_classes.index(x) for x in labels], dtype=torch.long)
        gens = torch.tensor([genus_to_id[x.split()[0]] for x in labels], dtype=torch.long)
        tops: list[torch.Tensor] = []
        for expert in args.experts:
            ep = load(args.text_dir / EXPERTS[expert])
            top_path = args.out_dir / f"top100_{protocol}_{expert}.pt"
            if top_path.exists():
                top = load(top_path)["top_ids"]
            else:
                top = compute_top100(payload["features"].float(), ep["features"], device, args.batch_size)
                torch.save({"top_ids": top, "expert": expert, "candidate_classes": full_classes}, top_path)
            tops.append(top)
            small_metrics.extend(metric_rows(top, ids, gens, candidate_genus, protocol, "all", expert))
        if "taxon" in args.experts and "fish" in args.experts:
            ti, fi = tops[args.experts.index("taxon")], tops[args.experts.index("fish")]
            small_metrics.extend(union_metric_rows(ti, fi, tops, ids, protocol, "all", candidate_genus, gens))
    metric_rows_out.extend(small_metrics)
    write_csv(args.out_dir / "recall_metrics.csv", metric_rows_out)

    # Compact summary for routing decisions.
    summary: dict[str, Any] = {"device": str(device), "query_rows": len(manifest), "candidate_classes": len(full_classes), "experts": args.experts, "metrics_csv": str(args.out_dir / "recall_metrics.csv")}
    summary_rows = [r for r in metric_rows_out if r["fold"] == "all"]
    summary["all_fold_metrics"] = summary_rows
    (args.out_dir / "recall_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot the decision-relevant curves for taxon, fish and their union.
    plot_labels = [str(k) for k in KS]
    for protocol in ("species_heldout", "genus_heldout"):
        rows = [r for r in summary_rows if r["protocol"] == protocol and r["expert"] in ("taxon", "fish", "union_taxon_fish") and r["k"] in KS]
        series: dict[str, list[float]] = {}
        for expert in ("taxon", "fish", "union_taxon_fish"):
            series[expert] = [float(next((r["species_recall"] for r in rows if r["expert"] == expert and r["k"] == k), 0.0)) for k in KS]
        plot_bars(args.out_dir / f"{protocol}_recall.png", series, plot_labels, f"{protocol} exact species recall (17,393 candidates)")
    # Query-count proxy versus full-universe folds on the same axes.  The
    # candidate pool is 17,393 in both series; only query construction changes.
    series = {}
    for expert in ("taxon", "fish", "union_taxon_fish"):
        series[f"proxy_{expert}"] = [float(next((r["species_recall"] for r in summary_rows if r["protocol"] == "small_species_1000" and r["expert"] == expert and r["k"] == k), 0.0)) for k in KS]
        series[f"full_{expert}"] = [float(next((r["species_recall"] for r in summary_rows if r["protocol"] == "species_heldout" and r["expert"] == expert and r["k"] == k), 0.0)) for k in KS]
    plot_bars(args.out_dir / "proxy_1000_vs_full_universe_recall.png", series, plot_labels, "1,000-query proxy vs full held-out queries (17,393 candidates)")
    union_rows = [r for r in summary_rows if r["protocol"] in ("species_heldout", "genus_heldout") and r["expert"] == "union_taxon_fish"]
    union_gain: dict[str, list[float]] = {}
    for protocol in ("species_heldout", "genus_heldout"):
        vals: list[float] = []
        for k in KS:
            u = next((r["species_recall"] for r in union_rows if r["protocol"] == protocol and r["k"] == k), 0.0)
            b = max((r["species_recall"] for r in summary_rows if r["protocol"] == protocol and r["expert"] in ("taxon", "fish") and r["k"] == k), default=0.0)
            vals.append(float(u - b))
        union_gain[protocol] = vals
    plot_bars(args.out_dir / "candidate_union_gain.png", union_gain, plot_labels, "Taxon+fish candidate-union gain over best single expert")
    print(json.dumps({"device": str(device), "query_rows": len(manifest), "candidate_classes": len(full_classes), "metrics": len(metric_rows_out)}, indent=2))


if __name__ == "__main__":
    main()

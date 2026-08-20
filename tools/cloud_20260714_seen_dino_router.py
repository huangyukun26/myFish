from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from predict_embedding_classifier import MLPClassifier


ROOT = Path("runs/structural_backbones_20260713")
OUT = Path("runs/cloud_20260714/seen_dino_router")


def zscore(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch = checkpoint["arch"]
    model = MLPClassifier(
        int(arch["in_dim"]), int(arch["hidden_dim"]), len(checkpoint["classes"]),
        float(arch.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), list(checkpoint["classes"])


def infer(model, features: torch.Tensor, device: torch.device, batch: int = 512) -> torch.Tensor:
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            parts.append(model(F.normalize(features[start:start + batch].float(), dim=1).to(device)).cpu())
    return torch.cat(parts)


def prototypes(features: torch.Tensor, class_ids: torch.Tensor, nclasses: int) -> torch.Tensor:
    features = F.normalize(features.float(), dim=1)
    sums = torch.zeros(nclasses, features.shape[1])
    sums.index_add_(0, class_ids, features)
    counts = torch.bincount(class_ids, minlength=nclasses).clamp_min(1)
    return F.normalize(sums / counts[:, None], dim=1)


def aligned_text(path: Path, classes: list[str]) -> torch.Tensor:
    x = torch.load(path, map_location="cpu", weights_only=False)
    pos = {name: i for i, name in enumerate(x["classes"])}
    return F.normalize(x["features"][[pos[name] for name in classes]].float(), dim=1)


def fused_topk(
    mlp_logits: torch.Tensor,
    query: torch.Tensor,
    proto: torch.Tensor,
    text: torch.Tensor,
    device: torch.device,
    topk: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = [], []
    query = F.normalize(query.float(), dim=1)
    proto, text = proto.to(device), text.to(device)
    with torch.inference_mode():
        for start in range(0, len(query), 256):
            mlp = zscore(mlp_logits[start:start + 256]).to(device)
            visual = zscore(query[start:start + 256].to(device) @ proto.T)
            semantic = zscore(query[start:start + 256].to(device) @ text.T)
            score = mlp + 0.1 * visual + 0.75 * semantic
            v, i = score.topk(topk, dim=1)
            values.append(v.cpu()); indices.append(i.cpu())
    return torch.cat(values), torch.cat(indices)


def write_package(base: dict[str, str], image_ids: list[str], classes: list[str], pred: torch.Tensor, name: str):
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    override = {image_id: classes[int(idx)] for image_id, idx in zip(image_ids, pred)}
    changed = sum(base[image_id] != label for image_id, label in override.items())
    merged = dict(base); merged.update(override)
    ordered_keys = list(base)
    ordered = {key: merged[key] for key in ordered_keys}
    json_path = out_dir / "prediction.json"
    json_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction.json")
    with (out_dir / "seen_override.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp); writer.writerow(["image_id", "prediction"])
        writer.writerows((i, override[i]) for i in image_ids)
    return changed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    concat_dir = ROOT / "concat_balanced_gate"
    model, classes = load_model(concat_dir / "mlp_h4096_balsoft/best_model.pt", device)
    train = torch.load(concat_dir / "random2027/train.pt", map_location="cpu", weights_only=False)
    val = torch.load(concat_dir / "random2027/val.pt", map_location="cpu", weights_only=False)
    test = torch.load(concat_dir / "test_seen_hflip_letterbox_concat.pt", map_location="cpu", weights_only=False)
    proto = prototypes(train["features"][:, :1024], train["class_ids"].long(), len(classes))
    val_logits = infer(model, val["features"], device)
    test_logits = infer(model, test["features"], device)

    saved = torch.load(concat_dir / "fixed_fusion_structured_text_logits.pt", map_location="cpu", weights_only=False)
    text_options = {
        "taxon": Path("work/clip_text_features/seen_bioclip25_taxon.pt"),
        "structured_genus": ROOT / "structured_trait_mapper/genus/mapped_h0_seed2027.pt",
        "structured_species": ROOT / "structured_trait_mapper/species/mapped_h0_seed2027.pt",
    }
    selected = None
    validation = {}
    saved_top = saved["logits"].argmax(1)
    for name, path in text_options.items():
        text = aligned_text(path, classes)
        vv, vi = fused_topk(val_logits, val["features"][:, :1024], proto, text, device)
        acc = float(vi[:, 0].eq(val["class_ids"].long()).float().mean())
        agreement = float(vi[:, 0].eq(saved_top).float().mean())
        validation[name] = {"top1": acc, "agreement_saved": agreement}
        if selected is None or agreement > validation[selected]["agreement_saved"]:
            selected = name
    assert selected is not None
    text = aligned_text(text_options[selected], classes)
    base_values, base_topk = fused_topk(test_logits, test["features"][:, :1024], proto, text, device)
    torch.save(
        {"topk_values": base_values, "topk_indices": base_topk,
         "image_ids": list(test["image_ids"]), "classes": classes},
        OUT / "public_fusion.pt",
    )

    d1 = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    d2 = torch.load(ROOT / "dino_metric_full_prediction/test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    assert list(test["image_ids"]) == list(d1["image_ids"]) == list(d2["image_ids"])
    alt = d1["topk_indices"].long()[:, 0]
    alt_agree = alt.eq(d2["topk_indices"].long()[:, 0])
    alt_margin = torch.minimum(
        d1["topk_values"][:, 0] - d1["topk_values"][:, 1],
        d2["topk_values"][:, 0] - d2["topk_values"][:, 1],
    )
    base_margin = base_values[:, 0] - base_values[:, 1]
    full_train = torch.load(concat_dir / "train_hflip_letterbox_concat.pt", map_location="cpu", weights_only=False)
    counts = torch.bincount(full_train["class_ids"].long(), minlength=len(classes))
    base_json = json.loads(Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json").read_text(encoding="utf-8"))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    current = torch.tensor([class_to_idx[base_json[i]] for i in test["image_ids"]])
    common = current.eq(base_topk[:, 0]) & alt_agree & alt.ne(current) & counts[alt].le(5)

    # Thresholds are transferred from the four-fold, genus-grouped validation gate.
    common &= base_margin.le(0.7146902084350586) & alt_margin.ge(0.0385)
    outputs = {}
    for rank in (5, 20):
        mask = common & base_topk[:, :rank].eq(alt[:, None]).any(1)
        pred = current.clone(); pred[mask] = alt[mask]
        outputs[f"rank{rank}"] = {
            "eligible": int(mask.sum()),
            "changed": write_package(base_json, list(test["image_ids"]), classes, pred, f"rank{rank}"),
        }
    summary = {
        "selected_text": selected,
        "validation": validation,
        "rows": len(test["image_ids"]),
        "dino_seed_agreement": int(alt_agree.sum()),
        "current_equals_fused": int(current.eq(base_topk[:, 0]).sum()),
        "outputs": outputs,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "runs" / "submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox"
REF = ROOT / "runs" / "local_reference_7p2"
DIR = ROOT / "runs" / "cloud_20260714_v2" / "dirichlet_public"
DINO = ROOT / "runs" / "structural_backbones_20260713" / "dino_metric_full_prediction"
OUT = ROOT / "runs" / "cloud_20260714_v2" / "submission_v20_seen500_dirichlet_all3"


def load_prediction(path: Path) -> dict[str, str]:
    if path.is_dir():
        if (path / "prediction.json").exists():
            return json.loads((path / "prediction.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(path / "submission.zip") as zf:
            return json.loads(zf.read("prediction.json"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("prediction.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def load_dirichlet(path: Path, key: str = "t0.033333_s0.25") -> dict[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pred = payload["predictions"][key]["pred"].long()
    classes = list(payload["classes"])
    return {image_id: classes[int(idx)] for image_id, idx in zip(payload["image_ids"], pred)}


def main() -> None:
    base = load_prediction(BASE_DIR)
    fixed = load_prediction(REF / "seen_fixed_unseen_pair_prediction.json")
    letterbox = load_prediction(REF / "seen_concat_unseen_letterbox_submission.zip")
    seen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "test.pkl").read_bytes()))
    unseen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "unseen.pkl").read_bytes()))
    all_labels = set(pickle.loads((ROOT / "dataset" / "all_classes.pkl").read_bytes()))
    train_labels = json.loads((ROOT / "dataset" / "label_train.json").read_text(encoding="utf-8"))
    class_counts = collections.Counter(train_labels.values())
    known_genera = {label.split()[0] for label in train_labels.values()}
    protected_seen = {key for key in seen_ids if base[key] != fixed[key]}
    protected_unseen = {key for key in unseen_ids if base[key] != letterbox[key]}

    d1 = torch.load(DINO / "test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    d2 = torch.load(DINO / "test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    alt1 = d1["topk_indices"][:, 0].long()
    alt2 = d2["topk_indices"][:, 0].long()
    margin = torch.minimum(
        d1["topk_values"][:, 0] - d1["topk_values"][:, 1],
        d2["topk_values"][:, 0] - d2["topk_values"][:, 1],
    )
    classes = list(d1["classes"])
    seen_rows = []
    for image_id, idx1, idx2, score_margin in zip(d1["image_ids"], alt1, alt2, margin):
        if image_id not in seen_ids or image_id in protected_seen or int(idx1) != int(idx2):
            continue
        current = base[image_id]
        alternate = classes[int(idx1)]
        if alternate == current:
            continue
        if alternate.split()[0] != current.split()[0]:
            continue
        if class_counts[alternate] < class_counts[current]:
            continue
        seen_rows.append((float(score_margin), image_id, alternate))
    seen_rows.sort(reverse=True)
    seen_changes = {image_id: alternate for _margin, image_id, alternate in seen_rows[:500]}

    dirichlet_paths = sorted(DIR.glob("public_*_dirichlet.pt"))
    preds = [load_dirichlet(path) for path in dirichlet_paths]
    all3: dict[str, str] = {}
    all3_known_to_novel: dict[str, str] = {}
    vote2: dict[str, str] = {}
    for key in unseen_ids:
        if key in protected_unseen:
            continue
        votes = [p[key] for p in preds if key in p and p[key] != base[key]]
        if not votes:
            continue
        counter = collections.Counter(votes)
        label, count = counter.most_common(1)[0]
        if count >= 2:
            vote2[key] = label
        if count == len(preds):
            all3[key] = label
            if base[key].split()[0] in known_genera and label.split()[0] not in known_genera:
                all3_known_to_novel[key] = label

    all3_cap20: dict[str, str] = {}
    per_label = collections.Counter()
    for key, label in sorted(all3.items()):
        if per_label[label] >= 20:
            continue
        all3_cap20[key] = label
        per_label[label] += 1

    all3_cap10: dict[str, str] = {}
    per_label = collections.Counter()
    for key, label in sorted(all3.items()):
        if per_label[label] >= 10:
            continue
        all3_cap10[key] = label
        per_label[label] += 1

    variants = {
        "all3": all3,
        "all3_cap20": all3_cap20,
        "all3_cap10": all3_cap10,
        "all3_known_to_novel": all3_known_to_novel,
        "vote2": vote2,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    variant_audits = {}
    for name, unseen_changes in variants.items():
        output = dict(base)
        output.update(seen_changes)
        output.update(unseen_changes)
        assert list(output) == list(base)
        assert all(label in all_labels for label in output.values())
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "prediction.json"
        pred_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        zip_path = out_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_path, "prediction.json")
        changes = [key for key in base if output[key] != base[key]]
        split_seen = [key for key in changes if key in seen_ids]
        split_unseen = [key for key in changes if key in unseen_ids]
        unseen_counter = collections.Counter(output[key] for key in unseen_ids)
        variant_audits[name] = {
            "seen_changes": len(split_seen),
            "unseen_changes": len(split_unseen),
            "total_changes": len(changes),
            "protected_seen_overwrite": sum(key in protected_seen for key in split_seen),
            "protected_unseen_overwrite": sum(key in protected_unseen for key in split_unseen),
            "unseen_unique_labels": len(unseen_counter),
            "unseen_max_count": max(unseen_counter.values()),
            "sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        }
    audit = {
        "seen_rule": "DINO metric seeds agree; same genus; alternate train count >= current; exclude 7.2 seen protected; top500",
        "seen_candidates_available": len(seen_rows),
        "dirichlet_files": [str(path.relative_to(ROOT)) for path in dirichlet_paths],
        "variant_audits": variant_audits,
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

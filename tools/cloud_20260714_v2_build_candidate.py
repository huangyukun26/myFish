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
OUT = ROOT / "runs" / "cloud_20260714_v2" / "submission_v19_seen500_dirichlet_consensus"
DINO = ROOT / "runs" / "structural_backbones_20260713" / "dino_metric_full_prediction"
DIR = ROOT / "runs" / "cloud_20260714_v2" / "dirichlet_public"


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


def load_dirichlet_prediction(path: Path, key: str = "t0.033333_s0.25") -> dict[str, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pred = payload["predictions"][key]["pred"].long()
    classes = list(payload["classes"])
    ids = list(payload["image_ids"])
    return {image_id: classes[int(idx)] for image_id, idx in zip(ids, pred)}


def main() -> None:
    base = load_prediction(BASE_DIR)
    fixed = load_prediction(REF / "seen_fixed_unseen_pair_prediction.json")
    letterbox = load_prediction(REF / "seen_concat_unseen_letterbox_submission.zip")
    v17 = load_prediction(REF / "v17_prediction.json")
    seen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "test.pkl").read_bytes()))
    unseen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "unseen.pkl").read_bytes()))
    all_labels = set(pickle.loads((ROOT / "dataset" / "all_classes.pkl").read_bytes()))
    train_labels = json.loads((ROOT / "dataset" / "label_train.json").read_text(encoding="utf-8"))
    class_counts = collections.Counter(train_labels.values())
    protected_seen = {key for key in seen_ids if base[key] != fixed[key]}
    protected_unseen = {key for key in unseen_ids if base[key] != letterbox[key]}

    d1 = torch.load(DINO / "test_seen_metric_seed2027_topk.pt", map_location="cpu", weights_only=False)
    d2 = torch.load(DINO / "test_seen_metric_seed2028_topk.pt", map_location="cpu", weights_only=False)
    assert list(d1["image_ids"]) == list(d2["image_ids"])
    assert list(d1["classes"]) == list(d2["classes"])
    alt1 = d1["topk_indices"][:, 0].long()
    alt2 = d2["topk_indices"][:, 0].long()
    margin = torch.minimum(
        d1["topk_values"][:, 0] - d1["topk_values"][:, 1],
        d2["topk_values"][:, 0] - d2["topk_values"][:, 1],
    )
    classes = list(d1["classes"])
    seen_rows = []
    for image_id, idx1, idx2, score_margin in zip(d1["image_ids"], alt1, alt2, margin):
        if image_id not in seen_ids or image_id in protected_seen:
            continue
        if int(idx1) != int(idx2):
            continue
        current = base[image_id]
        alternate = classes[int(idx1)]
        if alternate == current:
            continue
        if alternate.split()[0] != current.split()[0]:
            continue
        if class_counts[alternate] < class_counts[current]:
            continue
        seen_rows.append(
            {
                "image_id": image_id,
                "current": current,
                "alternate": alternate,
                "margin": float(score_margin),
                "current_count": int(class_counts[current]),
                "alternate_count": int(class_counts[alternate]),
            }
        )
    seen_rows.sort(key=lambda row: row["margin"], reverse=True)
    selected_seen = seen_rows[:500]

    dirichlet_paths = sorted(DIR.glob("public_*_dirichlet.pt"))
    dirichlet_predictions = [load_dirichlet_prediction(path) for path in dirichlet_paths]
    unseen_changes: dict[str, str] = {}
    dirichlet_votes = []
    for key in sorted(unseen_ids):
        if key in protected_unseen:
            continue
        votes = [pred[key] for pred in dirichlet_predictions if key in pred and pred[key] != base[key]]
        if not votes:
            continue
        counts = collections.Counter(votes)
        label, vote_count = counts.most_common(1)[0]
        if vote_count >= 2:
            unseen_changes[key] = label
            dirichlet_votes.append({"image_id": key, "current": base[key], "alternate": label, "votes": vote_count})

    # If probability-space consensus is too sparse, use the prior low-coverage v17 unseen probe.
    unseen_source = "dirichlet_multiview_consensus"
    if len(unseen_changes) < 50:
        unseen_changes = {key: v17[key] for key in unseen_ids if v17.get(key) != base.get(key)}
        unseen_source = "fallback_v17_unseen_probe"

    output = dict(base)
    for row in selected_seen:
        output[row["image_id"]] = row["alternate"]
    output.update(unseen_changes)
    assert list(output) == list(base)
    assert all(label in all_labels for label in output.values())

    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUT / "prediction.json"
    pred_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = OUT / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, "prediction.json")

    changes = [key for key in base if output[key] != base[key]]
    seen_changes = [key for key in changes if key in seen_ids]
    unseen_changed = [key for key in changes if key in unseen_ids]
    unseen_counter = collections.Counter(output[key] for key in unseen_ids)
    audit = {
        "base": str(BASE_DIR.relative_to(ROOT)),
        "seen_rule": "DINO metric seeds agree; same genus; alternate train count >= current; exclude 7.2 seen protected; top500 by DINO margin",
        "seen_candidates_available": len(seen_rows),
        "seen_changes": len(seen_changes),
        "unseen_rule": unseen_source,
        "dirichlet_public_files": [str(path.relative_to(ROOT)) for path in dirichlet_paths],
        "dirichlet_consensus_raw_changes": len(dirichlet_votes),
        "unseen_changes": len(unseen_changed),
        "total_changes": len(changes),
        "protected_seen_overwrite": sum(key in protected_seen for key in seen_changes),
        "protected_unseen_overwrite": sum(key in protected_unseen for key in unseen_changed),
        "unseen_unique_labels": len(unseen_counter),
        "unseen_max_count": max(unseen_counter.values()),
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "seen_changes.json").open("w", encoding="utf-8") as fp:
        json.dump(selected_seen, fp, indent=2, ensure_ascii=False)
    with (OUT / "dirichlet_votes.json").open("w", encoding="utf-8") as fp:
        json.dump(dirichlet_votes, fp, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

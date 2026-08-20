from __future__ import annotations

import collections
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs" / "submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox"
FIXED = ROOT / "runs" / "submission_20260702_seen_fixed_unseen_pair_o70species_avg_letterbox"
V17 = ROOT / "runs" / "submission_20260714_v17_microaligned_dual_probe"
DINO = (
    ROOT
    / "work"
    / "cloud_20260713"
    / "artifacts"
    / "effective"
    / "runs"
    / "structural_backbones_20260713"
    / "dino_metric_full_prediction"
)
OUT = ROOT / "runs" / "submission_20260714_v18_local_seen_dino_nonrarer_top300_unseen_v17"


def load_prediction(folder: Path) -> dict[str, str]:
    path = folder / "prediction.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(folder / "submission.zip") as zf:
        return json.loads(zf.read("prediction.json"))


def main() -> None:
    base = load_prediction(BASE)
    fixed = load_prediction(FIXED)
    v17 = load_prediction(V17)
    seen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "test.pkl").read_bytes()))
    unseen_ids = set(pickle.loads((ROOT / "dataset" / "splits" / "unseen.pkl").read_bytes()))
    train_labels = json.loads((ROOT / "dataset" / "label_train.json").read_text(encoding="utf-8"))
    class_counts = collections.Counter(train_labels.values())

    protected_seen = {key for key in seen_ids if base[key] != fixed[key]}

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
    rows: list[dict[str, object]] = []
    for image_id, idx, idx2, score_margin in zip(d1["image_ids"], alt1, alt2, margin):
        if image_id not in seen_ids or idx.item() != idx2.item() or image_id in protected_seen:
            continue
        current = base[image_id]
        alternate = classes[int(idx)]
        if alternate == current:
            continue
        if alternate.split()[0] != current.split()[0]:
            continue
        if class_counts[alternate] < class_counts[current]:
            continue
        rows.append(
            {
                "image_id": image_id,
                "current": current,
                "alternate": alternate,
                "margin": float(score_margin),
                "current_count": int(class_counts[current]),
                "alternate_count": int(class_counts[alternate]),
            }
        )

    rows.sort(key=lambda row: float(row["margin"]), reverse=True)
    selected_seen = rows[:300]
    assert len(selected_seen) == 300

    unseen_changes = {
        key: v17[key]
        for key in unseen_ids
        if key in base and v17.get(key) is not None and v17[key] != base[key]
    }
    assert len(unseen_changes) == 106

    output = dict(base)
    for row in selected_seen:
        output[str(row["image_id"])] = str(row["alternate"])
    output.update(unseen_changes)
    assert list(output) == list(base)

    allowed = set(pickle.loads((ROOT / "dataset" / "all_classes.pkl").read_bytes()))
    assert all(label in allowed for label in output.values())

    OUT.mkdir(parents=True, exist_ok=True)
    prediction = OUT / "prediction.json"
    prediction.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = OUT / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction, "prediction.json")

    audit = {
        "base": str(BASE.relative_to(ROOT)),
        "seen_rule": "DINO metric seeds agree; same genus; alternate train count >= current; exclude 7.2 seen protected; top300 by DINO margin",
        "seen_candidates_available": len(rows),
        "seen_changes": len(selected_seen),
        "unseen_rule": "reuse v17 unseen 106-row low-coverage lineage probe; Dirichlet public result rejected as too broad",
        "unseen_changes": len(unseen_changes),
        "protected_seen_overwrite": sum(str(row["image_id"]) in protected_seen for row in selected_seen),
        "protected_unseen_overwrite": 0,
        "total_changes": sum(1 for key in base if output[key] != base[key]),
        "zip_members": ["prediction.json"],
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "selected_seen_top10": selected_seen[:10],
    }
    (OUT / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "seen_changes.csv").open("w", encoding="utf-8") as fp:
        fp.write("image_id,current,alternate,margin,current_count,alternate_count\n")
        for row in selected_seen:
            fp.write(
                f"{row['image_id']},{row['current']},{row['alternate']},"
                f"{row['margin']},{row['current_count']},{row['alternate_count']}\n"
            )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import zipfile
import argparse
from pathlib import Path

import torch


BASE_PATH = Path("runs/submission_20260702_seen_router_unseen_pair_o70species_avg_letterbox/prediction.json")
PAYLOAD_PATH = Path("runs/cloud_20260714/unseen_structured_public/fixed/predictions.pt")
OUT = Path("runs/cloud_20260714/unseen_structured_public/packages")


def genus(name: str) -> str:
    return name.split(maxsplit=1)[0]


def write_package(base: dict[str, str], changes: dict[str, str], name: str) -> int:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = dict(base)
    actual = sum(merged.get(key) != value for key, value in changes.items())
    merged.update(changes)
    out_json = out_dir / "prediction.json"
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, arcname="prediction.json")
    return actual


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=PAYLOAD_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    base_json = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    candidates = list(payload["candidates"])
    ids = list(payload["image_ids"])
    class_to_idx = {name: idx for idx, name in enumerate(candidates)}
    current = torch.tensor([class_to_idx[base_json[image_id]] for image_id in ids])
    tool_base = payload["base_pred_indices"].long()
    proposed = payload["best_pred_indices"].long()
    seen_map = json.loads(Path("work/full_manifests/seen_class_to_idx.json").read_text(encoding="utf-8"))
    known_genera = {genus(name) for name in seen_map}
    is_novel = torch.tensor([genus(name) not in known_genera for name in candidates])
    move = (~is_novel[current]) & is_novel[proposed] & proposed.ne(current)
    tool_changed = proposed.ne(tool_base)
    masks = {
        "strict_current_equals_toolbase": move & tool_changed & current.eq(tool_base),
        "tool_changed_move_novel": move & tool_changed,
        "all_move_novel": move,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, mask in masks.items():
        changes = {ids[i]: candidates[int(proposed[i])] for i in torch.where(mask)[0].tolist()}
        summary[name] = {"eligible": int(mask.sum()), "changed": write_package(base_json, changes, name)}
    summary["diagnostics"] = {
        "tool_changed": int(tool_changed.sum()),
        "current_equals_toolbase": int(current.eq(tool_base).sum()),
        "proposed_differs_current": int(proposed.ne(current).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

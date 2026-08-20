from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def load_logits(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = {
        "logits": payload["logits"].float(),
        "image_ids": list(payload["image_ids"]),
        "classes": list(payload["classes"]),
    }
    if "class_ids" in payload:
        out["class_ids"] = payload["class_ids"].long()
    if "labels" in payload:
        out["labels"] = list(payload["labels"])
    return out


def save_logits(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def write_pred_csv(path: Path, image_ids: list[str], pred: torch.Tensor, classes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["image_id", "prediction"])
        writer.writeheader()
        for image_id, idx in zip(image_ids, pred.tolist()):
            writer.writerow({"image_id": image_id, "prediction": classes[int(idx)]})


def assert_aligned(*items: dict[str, Any]) -> None:
    first = items[0]
    for item in items[1:]:
        if first["image_ids"] != item["image_ids"]:
            raise RuntimeError("image_ids are not aligned")
        if first["classes"] != item["classes"]:
            raise RuntimeError("classes are not aligned")
        if "class_ids" in first and "class_ids" in item and not torch.equal(first["class_ids"], item["class_ids"]):
            raise RuntimeError("class_ids are not aligned")


def command_build_current_logits(args: argparse.Namespace) -> None:
    base = load_logits(args.base_logits)
    joint = load_logits(args.joint_logits)
    dino = load_logits(args.dino_logits)
    assert_aligned(base, joint, dino)
    bpred = base["logits"].argmax(dim=1)
    dino_pred = dino["logits"].argmax(dim=1)
    jtop2 = joint["logits"].topk(2, dim=1)
    jpred = jtop2.indices[:, 0]
    jmargin = jtop2.values[:, 0] - jtop2.values[:, 1]
    selected = jpred.ne(bpred) & ((jmargin >= args.margin_threshold) | jpred.eq(dino_pred))
    logits = base["logits"].clone()
    logits[selected] = joint["logits"][selected]
    payload = {
        "logits": logits.half(),
        "image_ids": base["image_ids"],
        "classes": base["classes"],
        "source": {
            "base_logits": str(args.base_logits),
            "joint_logits": str(args.joint_logits),
            "dino_logits": str(args.dino_logits),
            "margin_threshold": args.margin_threshold,
            "rule": "joint_pred!=base_pred AND (joint_margin>=threshold OR joint_pred==dino_pred)",
            "selected": int(selected.sum().item()),
        },
    }
    if "class_ids" in base:
        payload["class_ids"] = base["class_ids"]
    if "labels" in base:
        payload["labels"] = base["labels"]
    save_logits(args.out_logits, payload)
    if args.out_csv:
        write_pred_csv(args.out_csv, base["image_ids"], logits.argmax(dim=1), base["classes"])
    print(json.dumps(payload["source"], indent=2), flush=True)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def split_dev_outer(image_ids: list[str], y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idx, (image_id, cls) in enumerate(zip(image_ids, y.tolist())):
        groups[int(cls)].append((stable_hash(image_id), idx))
    dev = torch.zeros(len(image_ids), dtype=torch.bool)
    for rows in groups.values():
        rows.sort()
        for j, (_h, idx) in enumerate(rows):
            if j % 2 == 0:
                dev[idx] = True
    return dev, ~dev


def stats(name: str, mask: torch.Tensor, bpred: torch.Tensor, cpred: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    bc = bpred.eq(y)
    cc = cpred.eq(y)
    changed = mask & cpred.ne(bpred)
    wins = changed & (~bc) & cc
    losses = changed & bc & (~cc)
    return {
        "name": name,
        "rows": int(mask.sum().item()),
        "base_acc": float((bc & mask).sum().item() / max(1, int(mask.sum().item()))),
        "cand_acc": float((cc & mask).sum().item() / max(1, int(mask.sum().item()))),
        "raw_net": int((cc & mask).sum().item() - (bc & mask).sum().item()),
        "raw_gain": float(((cc & mask).sum().item() - (bc & mask).sum().item()) / max(1, int(mask.sum().item()))),
        "changed": int(changed.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net_changed": int(wins.sum().item() - losses.sum().item()),
        "efficiency": float((wins.sum().item() - losses.sum().item()) / max(1, int(changed.sum().item()))),
        "oracle_acc": float(((bc | cc) & mask).sum().item() / max(1, int(mask.sum().item()))),
        "oracle_complement": float((((bc | cc) & mask).sum().item() - (bc & mask).sum().item()) / max(1, int(mask.sum().item()))),
    }


def selected_stats(
    name: str,
    selected: torch.Tensor,
    split_mask: torch.Tensor,
    bpred: torch.Tensor,
    cpred: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, Any]:
    bc = bpred.eq(y)
    cc = cpred.eq(y)
    sel = selected & split_mask & cpred.ne(bpred)
    wins = sel & (~bc) & cc
    losses = sel & bc & (~cc)
    return {
        "name": name,
        "rows": int(split_mask.sum().item()),
        "selected": int(sel.sum().item()),
        "wins": int(wins.sum().item()),
        "losses": int(losses.sum().item()),
        "net": int(wins.sum().item() - losses.sum().item()),
        "efficiency": float((wins.sum().item() - losses.sum().item()) / max(1, int(sel.sum().item()))),
        "acc_after_gate": float(((bc & split_mask).sum().item() + wins.sum().item() - losses.sum().item()) / max(1, int(split_mask.sum().item()))),
    }


def align_logits(base: dict[str, Any], cand: dict[str, Any]) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    if base["classes"] != cand["classes"]:
        raise RuntimeError("class order mismatch")
    bpos = {image_id: idx for idx, image_id in enumerate(base["image_ids"])}
    cpos = {image_id: idx for idx, image_id in enumerate(cand["image_ids"])}
    ids = [image_id for image_id in base["image_ids"] if image_id in cpos]
    bi = torch.tensor([bpos[x] for x in ids], dtype=torch.long)
    ci = torch.tensor([cpos[x] for x in ids], dtype=torch.long)
    if "class_ids" not in base:
        raise RuntimeError("base logits need class_ids for validation")
    y = base["class_ids"][bi]
    if "class_ids" in cand and not torch.equal(y, cand["class_ids"][ci]):
        raise RuntimeError("candidate class_ids mismatch")
    return ids, base["logits"][bi], cand["logits"][ci], y


def score_bank(blogits: torch.Tensor, clogits: torch.Tensor) -> dict[str, torch.Tensor]:
    bprob = F.softmax(blogits, dim=1)
    cprob = F.softmax(clogits, dim=1)
    btop2 = blogits.topk(2, dim=1)
    ctop2 = clogits.topk(2, dim=1)
    bmargin = btop2.values[:, 0] - btop2.values[:, 1]
    cmargin = ctop2.values[:, 0] - ctop2.values[:, 1]
    bconf = bprob.max(dim=1).values
    cconf = cprob.max(dim=1).values
    return {
        "cand_margin": cmargin,
        "cand_conf": cconf,
        "delta_margin": cmargin - bmargin,
        "delta_conf": cconf - bconf,
        "hybrid_margin_conf": (cmargin - bmargin) + 2.0 * (cconf - bconf),
        "hybrid_cand_delta": cmargin + 0.5 * (cmargin - bmargin),
    }


def write_zip(pred_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")


def read_seen_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return [row["image_id"] for row in csv.DictReader(fp)]


def protected_seen_ids(current: dict[str, str], old_base: dict[str, str], fixed_ref: dict[str, str], seen_ids: list[str]) -> set[str]:
    protected = set()
    for image_id in seen_ids:
        if current.get(image_id) != old_base.get(image_id):
            protected.add(image_id)
        if old_base.get(image_id) != fixed_ref.get(image_id):
            protected.add(image_id)
    return protected


def build_package(
    *,
    out_dir: Path,
    name: str,
    selected: torch.Tensor,
    test_ids: list[str],
    test_logits: torch.Tensor,
    classes: list[str],
    current_public: dict[str, str],
    protected: set[str],
) -> dict[str, Any]:
    pred = dict(current_public)
    cand_pred = test_logits.argmax(dim=1)
    selected_idx = torch.nonzero(selected, as_tuple=False).flatten().tolist()
    changed: list[str] = []
    skipped_protected = 0
    same_as_current = 0
    for idx in selected_idx:
        image_id = test_ids[idx]
        if image_id in protected:
            skipped_protected += 1
            continue
        new_label = classes[int(cand_pred[idx])]
        if pred.get(image_id) == new_label:
            same_as_current += 1
            continue
        pred[image_id] = new_label
        changed.append(image_id)
    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    pred_path = variant_dir / "prediction.json"
    zip_path = variant_dir / "submission.zip"
    pred_path.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_zip(pred_path, zip_path)
    audit = {
        "variant": name,
        "selected_before_protection": int(selected.sum().item()),
        "protected_count": len(protected),
        "skipped_protected": skipped_protected,
        "same_as_current": same_as_current,
        "public_seen_changed": len(changed),
        "zip": str(zip_path),
        "prediction_json": str(pred_path),
    }
    (variant_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def command_eval_build(args: argparse.Namespace) -> None:
    base_val = load_logits(args.base_val_logits)
    cand_val = load_logits(args.cand_val_logits)
    ids, blogits, clogits, y = align_logits(base_val, cand_val)
    bpred = blogits.argmax(dim=1)
    cpred = clogits.argmax(dim=1)
    changed = cpred.ne(bpred)
    dev, outer = split_dev_outer(ids, y)
    scores = score_bank(blogits, clogits)
    targets = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000, 5000]
    trials = []
    dev_changed_idx = torch.nonzero(dev & changed, as_tuple=False).flatten()
    for score_name, score in scores.items():
        order = dev_changed_idx[torch.argsort(score[dev_changed_idx], descending=True)]
        for k in targets:
            kk = min(k, int(order.numel()))
            if kk < args.min_dev_selected:
                continue
            threshold = float(score[order[kk - 1]].item())
            selected = changed & (score >= threshold)
            trial = {
                "score": score_name,
                "target_dev": k,
                "threshold": threshold,
                "dev": selected_stats("dev", selected, dev, bpred, cpred, y),
                "outer": selected_stats("outer", selected, outer, bpred, cpred, y),
                "all": selected_stats("all", selected, torch.ones(len(ids), dtype=torch.bool), bpred, cpred, y),
            }
            trials.append(trial)
    trials.sort(key=lambda t: (t["outer"]["net"], t["outer"]["efficiency"], t["dev"]["net"]), reverse=True)
    out = {
        "base_val_logits": str(args.base_val_logits),
        "cand_val_logits": str(args.cand_val_logits),
        "overall": stats("all", torch.ones(len(ids), dtype=torch.bool), bpred, cpred, y),
        "dev_raw": stats("dev", dev, bpred, cpred, y),
        "outer_raw": stats("outer", outer, bpred, cpred, y),
        "best_trials_by_outer": trials[:50],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["overall"], indent=2), flush=True)
    if trials:
        print(json.dumps(trials[0], indent=2), flush=True)

    if args.cand_test_logits is None or args.base_test_logits is None:
        return
    base_test = load_logits(args.base_test_logits)
    cand_test = load_logits(args.cand_test_logits)
    assert_aligned(base_test, cand_test)
    current_public = json.loads(args.current_public_json.read_text(encoding="utf-8"))
    old_base = json.loads(args.old_base_json.read_text(encoding="utf-8"))
    fixed_ref = json.loads(args.fixed_ref_json.read_text(encoding="utf-8"))
    seen_ids = read_seen_ids(args.test_seen_manifest)
    protected = protected_seen_ids(current_public, old_base, fixed_ref, seen_ids)
    test_scores = score_bank(base_test["logits"], cand_test["logits"])
    current_idx = {label: idx for idx, label in enumerate(cand_test["classes"])}
    current_pred_idx = torch.tensor([current_idx[current_public[image_id]] for image_id in cand_test["image_ids"]], dtype=torch.long)
    cand_test_pred = cand_test["logits"].argmax(dim=1)
    test_changed = cand_test_pred.ne(current_pred_idx)
    package_audits = {}
    package_trials = [t for t in trials if t["outer"]["net"] > 0 and t["outer"]["selected"] > 0]
    for rank, trial in enumerate(package_trials[: args.max_packages], start=1):
        score = test_scores[trial["score"]]
        selected = test_changed & (score >= float(trial["threshold"]))
        name = f"rank{rank:02d}_{trial['score']}_thr_{str(round(float(trial['threshold']), 6)).replace('-', 'm').replace('.', 'p')}"
        audit = build_package(
            out_dir=args.out_dir / "packages",
            name=name,
            selected=selected,
            test_ids=cand_test["image_ids"],
            test_logits=cand_test["logits"],
            classes=cand_test["classes"],
            current_public=current_public,
            protected=protected,
        )
        audit["trial"] = trial
        package_audits[name] = audit
    (args.out_dir / "packages_summary.json").write_text(json.dumps(package_audits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-current-logits")
    build.add_argument("--base-logits", type=Path, required=True)
    build.add_argument("--joint-logits", type=Path, required=True)
    build.add_argument("--dino-logits", type=Path, required=True)
    build.add_argument("--out-logits", type=Path, required=True)
    build.add_argument("--out-csv", type=Path, default=None)
    build.add_argument("--margin-threshold", type=float, default=0.28125)
    build.set_defaults(func=command_build_current_logits)

    ev = sub.add_parser("eval-build")
    ev.add_argument("--base-val-logits", type=Path, required=True)
    ev.add_argument("--cand-val-logits", type=Path, required=True)
    ev.add_argument("--base-test-logits", type=Path, default=None)
    ev.add_argument("--cand-test-logits", type=Path, default=None)
    ev.add_argument("--current-public-json", type=Path, required=True)
    ev.add_argument("--old-base-json", type=Path, required=True)
    ev.add_argument("--fixed-ref-json", type=Path, required=True)
    ev.add_argument("--test-seen-manifest", type=Path, required=True)
    ev.add_argument("--out-dir", type=Path, required=True)
    ev.add_argument("--min-dev-selected", type=int, default=50)
    ev.add_argument("--max-packages", type=int, default=12)
    ev.set_defaults(func=command_eval_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

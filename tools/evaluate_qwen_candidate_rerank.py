from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confidence_rank(value: str) -> int:
    value = str(value or "").lower()
    if value == "high":
        return 3
    if value == "medium":
        return 2
    if value == "low":
        return 1
    return 0


def evaluate(rows: list[dict[str, Any]], min_confidence: str, only_changed: bool) -> dict[str, Any]:
    min_rank = confidence_rank(min_confidence)
    known = [row for row in rows if row.get("label")]
    wins = 0
    losses = 0
    changed = 0
    triggered = 0
    base_correct = 0
    final_correct = 0
    for row in known:
        base = row.get("base_prediction", "")
        cand = row.get("prediction", base)
        label = row.get("label", "")
        trigger = confidence_rank(row.get("confidence", "")) >= min_rank
        if only_changed:
            trigger = trigger and cand != base
        final = cand if trigger else base
        base_ok = base == label
        final_ok = final == label
        base_correct += int(base_ok)
        final_correct += int(final_ok)
        changed += int(final != base)
        triggered += int(trigger)
        wins += int((not base_ok) and final_ok)
        losses += int(base_ok and (not final_ok))
    return {
        "min_confidence": min_confidence,
        "only_changed": only_changed,
        "known": len(known),
        "base_top1": base_correct / max(1, len(known)),
        "new_top1": final_correct / max(1, len(known)),
        "triggered": triggered,
        "changed": changed,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "win_loss_ratio": wins / max(1, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.qwen_jsonl) if "error" not in row]
    sweep = []
    for min_confidence in ["unknown", "low", "medium", "high"]:
        for only_changed in [False, True]:
            sweep.append(evaluate(rows, min_confidence, only_changed))
    best = max(sweep, key=lambda row: (row["new_top1"], row["net_wins"], -row["losses"], -row["changed"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "sweep.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)
    summary = {
        "qwen_jsonl": str(args.qwen_jsonl),
        "valid_rows": len(rows),
        "best": best,
        "sweep_csv": str(args.out_dir / "sweep.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

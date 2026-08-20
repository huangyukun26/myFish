from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-score", type=float, required=True)
    parser.add_argument("--unseen-score", type=float, required=True)
    parser.add_argument("--overall-score", type=float, default=None)
    parser.add_argument("--seen-count", type=int, default=20097)
    parser.add_argument("--unseen-count", type=int, default=15568)
    parser.add_argument("--baseline-seen-score", type=float, default=None)
    parser.add_argument("--baseline-unseen-score", type=float, default=None)
    parser.add_argument("--baseline-overall-score", type=float, default=None)
    args = parser.parse_args()

    total = args.seen_count + args.unseen_count
    seen_correct = args.seen_score * args.seen_count
    unseen_correct = args.unseen_score * args.unseen_count
    implied_overall = (seen_correct + unseen_correct) / total
    result = {
        "seen_count": args.seen_count,
        "unseen_count": args.unseen_count,
        "total": total,
        "seen_weight": args.seen_count / total,
        "unseen_weight": args.unseen_count / total,
        "seen_score": args.seen_score,
        "unseen_score": args.unseen_score,
        "reported_overall_score": args.overall_score,
        "implied_overall_score": implied_overall,
        "seen_correct_estimate": seen_correct,
        "unseen_correct_estimate": unseen_correct,
        "total_correct_estimate": seen_correct + unseen_correct,
    }
    if args.baseline_seen_score is not None and args.baseline_unseen_score is not None:
        base_seen = args.baseline_seen_score * args.seen_count
        base_unseen = args.baseline_unseen_score * args.unseen_count
        base_total = base_seen + base_unseen
        base_overall = base_total / total
        result["baseline"] = {
            "seen_score": args.baseline_seen_score,
            "unseen_score": args.baseline_unseen_score,
            "reported_overall_score": args.baseline_overall_score,
            "implied_overall_score": base_overall,
            "seen_correct_estimate": base_seen,
            "unseen_correct_estimate": base_unseen,
            "total_correct_estimate": base_total,
        }
        result["delta"] = {
            "seen_score": args.seen_score - args.baseline_seen_score,
            "unseen_score": args.unseen_score - args.baseline_unseen_score,
            "overall_score_implied": implied_overall - base_overall,
            "seen_correct_estimate": seen_correct - base_seen,
            "unseen_correct_estimate": unseen_correct - base_unseen,
            "total_correct_estimate": (seen_correct + unseen_correct) - base_total,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

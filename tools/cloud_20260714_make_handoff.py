from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / "runs" / "cloud_20260714_return"
FILES = [
    ROOT / "docs" / "CLOUD_EXPERIMENT_REPORT_20260714.md",
    RETURN / "README_FINAL.md",
    RETURN / "FINAL_PACKAGE_AUDIT.json",
    RETURN / "FINAL_CANDIDATES.csv",
    RETURN / "FINAL_DECISION.json",
    RETURN / "V15_REPRO_RECIPE.json",
    RETURN / "v12_balanced_q25_submission.zip",
    RETURN / "v13_balanced_q35_subset_submission.zip",
    RETURN / "v15_balanced_current_genus_adv_q30_consensus_submission.zip",
    RETURN / "v10_balanced_strict_submission.zip",
    RETURN / "v12_maxnet_q25_submission.zip",
    RETURN / "v12_balanced_q50_submission.zip",
    RETURN / "v12_balanced_q0_aggressive_submission.zip",
    RETURN / "v12_FINAL_SUMMARY.json",
    RETURN / "v13_summary.json",
    RETURN / "v14_summary.json",
    RETURN / "v15_summary.json",
    RETURN / "v12_seed_correlation_stress.json",
    RETURN / "v15_cluster_bootstrap.json",
    RETURN / "current_genus_advantage_oof.json",
    RETURN / "current_genus_advantage_ensemble_oof.json",
    RETURN / "current_genus_consensus_oof.json",
    RETURN / "unseen_q25_loss_cases.json",
    RETURN / "models" / "genus_gate_avg3_seed2050_best_model.pt",
    RETURN / "models" / "genus_gate_avg3_seed2051_best_model.pt",
    RETURN / "models" / "genus_gate_avg3_seed2052_best_model.pt",
]


def main() -> None:
    output = RETURN / "cloud_20260714_handoff.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in FILES:
            if not path.exists():
                raise FileNotFoundError(path)
            if path.is_relative_to(RETURN):
                arcname = Path("return") / path.relative_to(RETURN)
            else:
                arcname = path.relative_to(ROOT)
            zf.write(path, arcname.as_posix())
    print(output, output.stat().st_size)


if __name__ == "__main__":
    main()

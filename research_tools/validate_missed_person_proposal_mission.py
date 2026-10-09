"""Validate the artifact-gated missed-person proposal mission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASELINE_RECALL = 0.0602191533


def validate(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "joint_proposal_recall_at_least_0_15": float(metrics.get("joint_proposal_recall", -1)) >= 0.15,
        "joint_proposal_recall_at_least_2x_baseline": float(metrics.get("joint_proposal_recall", -1)) >= 2 * BASELINE_RECALL,
        "mean_roi_area_at_most_0_10": float(metrics.get("mean_roi_area", 2)) <= 0.10,
        "max_roi_area_at_most_0_15": float(metrics.get("max_roi_area", 2)) <= 0.15,
        "sequence_split_has_no_leakage": metrics.get("sequence_split_leakage") is False,
        "gt_not_used_online": metrics.get("gt_used_online") is False,
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": (
            "joint proposal recall and ROI budget gates passed"
            if passed else "missed-person proposal mission gates not yet satisfied"
        ),
        "baseline_joint_proposal_recall": BASELINE_RECALL,
        "checks": checks,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = validate(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the fixed development gates for detector-conditioned residual queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline = float(metrics.get("static_raw_head_joint_proposal_recall", 0.0))
    recall = float(metrics.get("mean_joint_proposal_recall", 0.0))
    relative = recall / baseline if baseline > 0 else 0.0
    checks = {
        "mean_recall_at_least_0_15": recall >= .15,
        "relative_to_static_at_least_1_20": relative >= 1.20,
        "worst_camera_recall_at_least_0_08": float(metrics.get("worst_camera_joint_proposal_recall", 0.0)) >= .08,
        "mean_roi_area_at_most_0_10": float(metrics.get("mean_roi_area", 1.0)) <= .10,
        "max_roi_area_at_most_0_15": float(metrics.get("max_roi_area", 1.0)) <= .15,
        "camera_split_has_no_leakage": not bool(metrics.get("camera_split_leakage", True)),
        "gt_not_used_online": not bool(metrics.get("gt_used_online", True)),
        "strong_model_not_called": not bool(metrics.get("strong_model_called", True)),
        "sealed_camera_not_used": not bool(metrics.get("sealed_camera_used", True)),
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "summary": "detector-conditioned residual-query development gates satisfied" if passed else "detector-conditioned residual-query development gates not satisfied",
        "checks": checks,
        "metrics": metrics,
        "relative_to_static_raw_head": relative,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(json.loads(args.metrics.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

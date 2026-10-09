"""Validate decisive closure of the DCRQ research mission.

This validator distinguishes research completion from algorithm promotion.  A
fully audited negative experiment completes the mission with outcome=rejected;
it never changes failed performance gates into successful performance claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "mean_joint_proposal_recall",
    "worst_camera_joint_proposal_recall",
    "static_raw_head_joint_proposal_recall",
    "mean_roi_area",
    "max_roi_area",
    "camera_split_leakage",
    "gt_used_online",
    "strong_model_called",
    "sealed_camera_used",
)


def validate_closure(
    metrics: dict[str, Any], evidence: dict[str, Any], provenance: dict[str, Any], *, complete: bool
) -> dict[str, Any]:
    baseline = float(metrics.get("static_raw_head_joint_proposal_recall", 0.0))
    recall = float(metrics.get("mean_joint_proposal_recall", 0.0))
    relative = recall / baseline if baseline > 0 else 0.0
    performance_checks = {
        "mean_recall_at_least_0_15": recall >= .15,
        "relative_to_static_at_least_1_20": relative >= 1.20,
        "worst_camera_recall_at_least_0_08": float(metrics.get("worst_camera_joint_proposal_recall", 0.0)) >= .08,
    }
    audit_checks = {
        "experiment_complete_marker_present": complete,
        "evidence_scope_is_fixed_development_gate": evidence.get("evidence_status") == "BOUNDED_FOUR_CAMERA_ENGINEERING_DEVELOPMENT_GATE",
        "metrics_record_decisive_run": metrics.get("evidence_status") == "BOUNDED_FOUR_CAMERA_ENGINEERING_DEVELOPMENT_GATE_FAILED",
        "evidence_matches_closure_metrics": all(evidence.get(key) == metrics.get(key) for key in METRIC_KEYS),
        "four_leave_one_camera_out_folds_present": len(evidence.get("folds", ())) == 4,
        "configuration_search_disabled": provenance.get("configuration_search") is False,
        "mean_roi_area_at_most_0_10": float(metrics.get("mean_roi_area", 1.0)) <= .10,
        "max_roi_area_at_most_0_15": float(metrics.get("max_roi_area", 1.0)) <= .15,
        "camera_split_has_no_leakage": metrics.get("camera_split_leakage") is False,
        "gt_not_used_online": metrics.get("gt_used_online") is False and provenance.get("gt_used_online") is False,
        "strong_model_not_called": metrics.get("strong_model_called") is False and provenance.get("strong_model_called") is False,
        "sealed_camera_not_used": metrics.get("sealed_camera_used") is False and provenance.get("sealed_camera_used") is False,
    }
    algorithm_promoted = all(performance_checks.values())
    research_complete = all(audit_checks.values())
    outcome = "promoted" if algorithm_promoted else "rejected"
    return {
        "status": "passed" if research_complete else "failed",
        "passed": research_complete,
        "research_outcome": outcome if research_complete else "incomplete",
        "algorithm_promoted": algorithm_promoted if research_complete else False,
        "summary": (
            f"DCRQ research mission complete; algorithm {outcome} by preregistered gates"
            if research_complete else "DCRQ research closure evidence is incomplete or violates scope"
        ),
        "audit_checks": audit_checks,
        "performance_checks": performance_checks,
        "relative_to_static_raw_head": relative,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_closure(
        json.loads(args.metrics.read_text(encoding="utf-8")),
        json.loads(args.evidence.read_text(encoding="utf-8")),
        json.loads(args.provenance.read_text(encoding="utf-8")),
        complete=args.complete.read_text(encoding="utf-8").strip() == "COMPLETE",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

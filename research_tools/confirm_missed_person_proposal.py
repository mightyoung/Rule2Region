"""Evaluate one preregistered uncertainty-only proposal head on new cameras."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.missed_person_proposal import (
    audit_sequence_split,
    build_residual_targets,
    fit_pointwise_ridge,
    predict_pointwise_ridge,
)
from research_tools.run_missed_person_proposal_baseline import (
    FrameRecord,
    _features,
    evaluate,
    load_records,
)


def run_frozen_confirmation(
    fit_records: list[FrameRecord],
    confirmation_records: list[FrameRecord],
    *,
    l2: float = 1e-3,
    peak_count: int = 1,
    total_area: float = .095,
    budget: float = .20,
    output_width: int = 40,
    output_height: int = 23,
) -> dict[str, Any]:
    if not fit_records or not confirmation_records:
        raise ValueError("fit_and_confirmation_records_required")
    fit_sequences = {record.sequence_id for record in fit_records}
    confirmation_sequences = {record.sequence_id for record in confirmation_records}
    fit_cameras = {record.camera_id for record in fit_records}
    confirmation_cameras = {record.camera_id for record in confirmation_records}
    sequence_audit = audit_sequence_split(fit_sequences, confirmation_sequences, {"sentinel_unused"})
    camera_audit = audit_sequence_split(fit_cameras, confirmation_cameras, {"sentinel_unused"})
    if sequence_audit["sequence_split_leakage"] or camera_audit["sequence_split_leakage"]:
        raise ValueError("fit_confirmation_leakage")

    fit_features = np.stack([
        _features(record, output_width, output_height)["uncertainty"]
        for record in fit_records
    ])
    fit_targets = np.stack([
        build_residual_targets(
            record.truth,
            tuple(box for box, _ in record.predictions),
            image_width=record.width,
            image_height=record.height,
            output_width=output_width,
            output_height=output_height,
        ).heatmap
        for record in fit_records
    ])
    confirmation_features = np.stack([
        _features(record, output_width, output_height)["uncertainty"]
        for record in confirmation_records
    ])
    weights = fit_pointwise_ridge(fit_features, fit_targets, l2=l2)
    scores = predict_pointwise_ridge(confirmation_features, weights)
    confirmation = evaluate(
        confirmation_records,
        scores,
        peak_count=peak_count,
        total_area=total_area,
        budget=budget,
    )
    return {
        "schema_version": "fieldpack.missed-person-proposal-confirmation.v1",
        "evidence_status": "FROZEN_CONFIG_INDEPENDENT_CONFIRMATION_ENGINEERING_PILOT",
        "method": "uncertainty_only_pointwise_ridge",
        "configuration_source": "PREREGISTERED_FROM_DISJOINT_CALIBRATION_NOT_CONFIRMATION",
        "frozen_configuration": {
            "l2": l2,
            "peak_count": peak_count,
            "total_area": total_area,
            "frame_budget": budget,
            "grid": [output_height, output_width],
        },
        "online_inputs": ["archived_yolo11s_predictions"],
        "gt_used_online": False,
        "fit_gt_used_offline": True,
        "confirmation_gt_used_only_for_evaluation": True,
        "strong_model_called": False,
        "fit": {
            "cameras": sorted(fit_cameras),
            "sequences": sorted(fit_sequences),
            "frames": len(fit_records),
        },
        "confirmation_split": {
            "cameras": sorted(confirmation_cameras),
            "sequences": sorted(confirmation_sequences),
            "frames": len(confirmation_records),
        },
        "sequence_split_audit": sequence_audit,
        "camera_split_audit": camera_audit,
        "sequence_split_leakage": False,
        "confirmation": confirmation,
        "joint_proposal_recall": confirmation["joint_proposal_recall"],
        "mean_roi_area": confirmation["mean_roi_area"],
        "max_roi_area": confirmation["max_roi_area"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen uncertainty-only confirmation")
    parser.add_argument("--fit-receipt-root", type=Path, required=True)
    parser.add_argument("--fit-pilot-root", type=Path, required=True)
    parser.add_argument("--fit-camera", required=True)
    parser.add_argument("--confirmation-receipt-root", type=Path, required=True)
    parser.add_argument("--confirmation-pilot-root", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--peak-count", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.confirmation_manifest.read_text(encoding="utf-8"))
    if manifest.get("selection_policy") != "label_blind_sorted_camera_and_video_equal_bin_midpoints":
        raise ValueError("confirmation_selection_not_label_blind")
    excluded = set(manifest.get("excluded_camera_ids", ()))
    if args.fit_camera not in excluded:
        raise ValueError("fit_camera_not_excluded_from_confirmation_selection")

    fit_records = [
        record
        for record in load_records(args.fit_receipt_root, args.fit_pilot_root)
        if record.camera_id == args.fit_camera
    ]
    confirmation_records = load_records(args.confirmation_receipt_root, args.confirmation_pilot_root)
    payload = run_frozen_confirmation(
        fit_records,
        confirmation_records,
        l2=args.l2,
        peak_count=args.peak_count,
    )
    payload["confirmation_selection_policy"] = manifest["selection_policy"]
    payload["excluded_camera_ids"] = sorted(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("joint_proposal_recall", "mean_roi_area", "max_roi_area", "sequence_split_leakage")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

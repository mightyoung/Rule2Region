"""Fixed learned dense-head residual proposal development gate."""
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
from research_tools.run_dense_residual_proposal import mask_existing_predictions
from research_tools.run_missed_person_proposal_baseline import FrameRecord, evaluate, load_records
from research_tools.yolo_dense_person_score import FrozenYoloDensePersonScore


def detection_occupancy(record: FrameRecord, *, output_width: int = 40, output_height: int = 23) -> np.ndarray:
    occupancy = np.zeros((output_height, output_width), dtype=np.float32)
    for box, _confidence in record.predictions:
        x1 = max(0, min(output_width - 1, int(np.floor(box.x1 / record.width * output_width))))
        x2 = max(x1 + 1, min(output_width, int(np.ceil(box.x2 / record.width * output_width))))
        y1 = max(0, min(output_height - 1, int(np.floor(box.y1 / record.height * output_height))))
        y2 = max(y1 + 1, min(output_height, int(np.ceil(box.y2 / record.height * output_height))))
        occupancy[y1:y2, x1:x2] = 1.0
    return occupancy


def assemble_features(records: list[FrameRecord], multiscale: np.ndarray) -> np.ndarray:
    if multiscale.shape != (len(records), 3, 23, 40):
        raise ValueError("multiscale_dense_scores_shape_mismatch")
    occupancy = np.stack([detection_occupancy(record) for record in records])[:, None]
    return np.concatenate((multiscale, occupancy), axis=1)


def run_fixed_gate(
    fit_records: list[FrameRecord],
    evaluation_records: list[FrameRecord],
    fit_multiscale: np.ndarray,
    evaluation_multiscale: np.ndarray,
) -> dict[str, Any]:
    fit_sequences = {record.sequence_id for record in fit_records}
    evaluation_sequences = {record.sequence_id for record in evaluation_records}
    fit_cameras = {record.camera_id for record in fit_records}
    evaluation_cameras = {record.camera_id for record in evaluation_records}
    sequence_audit = audit_sequence_split(fit_sequences, evaluation_sequences, {"sentinel_unused"})
    camera_audit = audit_sequence_split(fit_cameras, evaluation_cameras, {"sentinel_unused"})
    if sequence_audit["sequence_split_leakage"] or camera_audit["sequence_split_leakage"]:
        raise ValueError("learned_dense_gate_split_leakage")
    fit_features = assemble_features(fit_records, fit_multiscale)
    evaluation_features = assemble_features(evaluation_records, evaluation_multiscale)
    targets = np.stack([
        build_residual_targets(
            record.truth,
            tuple(box for box, _ in record.predictions),
            image_width=record.width,
            image_height=record.height,
            output_width=40,
            output_height=23,
        ).heatmap
        for record in fit_records
    ])
    weights = fit_pointwise_ridge(fit_features, targets, l2=.001)
    scores = predict_pointwise_ridge(evaluation_features, weights)
    scores = np.stack([
        mask_existing_predictions(score, record, expansion=.10)
        for score, record in zip(scores, evaluation_records)
    ])
    evaluation = evaluate(
        evaluation_records,
        scores,
        peak_count=1,
        total_area=.095,
        budget=.20,
    )
    return {
        "schema_version": "fieldpack.learned-dense-residual-development-gate.v1",
        "evidence_status": "FIXED_DEVELOPMENT_GATE_NOT_INDEPENDENT_CONFIRMATION",
        "method": "pointwise_ridge_on_three_raw_person_scales_plus_detection_occupancy",
        "fixed_configuration": {
            "l2": .001,
            "positive_weight": 20.0,
            "mask_expansion": .10,
            "peak_count": 1,
            "frame_budget": .20,
            "total_area": .095,
            "grid": [23, 40],
        },
        "fit": {"cameras": sorted(fit_cameras), "sequences": sorted(fit_sequences), "frames": len(fit_records)},
        "evaluation_split": {"cameras": sorted(evaluation_cameras), "sequences": sorted(evaluation_sequences), "frames": len(evaluation_records)},
        "sequence_split_audit": sequence_audit,
        "camera_split_audit": camera_audit,
        "sequence_split_leakage": False,
        "online_inputs": ["frozen_yolo11s_raw_person_logits", "archived_yolo11s_predictions"],
        "gt_used_online": False,
        "strong_model_called": False,
        "evaluation": evaluation,
        "joint_proposal_recall": evaluation["joint_proposal_recall"],
        "mean_roi_area": evaluation["mean_roi_area"],
        "max_roi_area": evaluation["max_roi_area"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed learned dense residual development gate")
    parser.add_argument("--fit-receipt-root", type=Path, required=True)
    parser.add_argument("--fit-pilot-root", type=Path, required=True)
    parser.add_argument("--evaluation-receipt-root", type=Path, required=True)
    parser.add_argument("--evaluation-pilot-root", type=Path, required=True)
    parser.add_argument("--small-model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_records = load_records(args.fit_receipt_root, args.fit_pilot_root)
    evaluation_records = load_records(args.evaluation_receipt_root, args.evaluation_pilot_root)
    with FrozenYoloDensePersonScore(args.small_model, device=args.device) as extractor:
        fit_multiscale = extractor.extract_multiscale(
            (record.rgb for record in fit_records), output_width=40, output_height=23
        )
        evaluation_multiscale = extractor.extract_multiscale(
            (record.rgb for record in evaluation_records), output_width=40, output_height=23
        )
    payload = run_fixed_gate(fit_records, evaluation_records, fit_multiscale, evaluation_multiscale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("joint_proposal_recall", "mean_roi_area", "max_roi_area")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

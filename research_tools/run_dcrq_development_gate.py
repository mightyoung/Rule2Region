"""Run the preregistered four-camera DCRQ development gate.

Ground truth is used only to fit residual targets and score held-out cameras.
The online score path consumes cached frozen-detector observations only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, parse_mot17_gt
from research_tools.missed_person_proposal import (
    audit_sequence_split,
    build_residual_targets,
    fit_pointwise_ridge,
    predict_pointwise_ridge,
)
from research_tools.run_missed_person_proposal_baseline import FrameRecord, _camera, evaluate


SEALED_CAMERA = "VIRAT_S_010113"
EXPECTED_CAMERAS = (
    "VIRAT_S_010004",
    "VIRAT_S_010111",
    "VIRAT_S_010112",
    "VIRAT_S_010204",
)


@dataclass(frozen=True)
class LoadedObservations:
    records: tuple[FrameRecord, ...]
    online_features: np.ndarray
    occupancy: np.ndarray
    static_scores: np.ndarray
    sequence_slices: dict[str, tuple[int, int]]


def causal_channels(base: np.ndarray, *, alpha: float = 0.60) -> np.ndarray:
    """Add causal EMA and signed innovation without looking at future frames."""
    if base.ndim != 4 or not 0.0 < alpha <= 1.0:
        raise ValueError("invalid_causal_feature_input")
    ema = np.empty_like(base, dtype=np.float32)
    difference = np.zeros_like(base, dtype=np.float32)
    ema[0] = base[0]
    for index in range(1, len(base)):
        difference[index] = base[index] - base[index - 1]
        ema[index] = alpha * base[index] + (1.0 - alpha) * ema[index - 1]
    return np.concatenate((base, ema, difference), axis=1).astype(np.float32)


def _mean3x3(features: np.ndarray) -> np.ndarray:
    padded = np.pad(features, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    result = np.zeros_like(features, dtype=np.float32)
    for y in range(3):
        for x in range(3):
            result += padded[:, :, y:y + features.shape[2], x:x + features.shape[3]]
    return result / 9.0


def fixed_depthwise_head_features(causal: np.ndarray) -> np.ndarray:
    """Fixed depthwise 3x3 context followed later by learned pointwise fusion."""
    local_mean = _mean3x3(causal)
    center_surround = causal - local_mean
    return np.concatenate((causal, local_mean, center_surround), axis=1).astype(np.float32)


def _prediction_boxes(receipt: dict[str, Any]) -> tuple[tuple[Box, float], ...]:
    return tuple(
        (Box(**item["box"]), float(item["confidence"]))
        for item in receipt["predictions"]
    )


def load_observations(cache_root: Path, pilot_root: Path) -> LoadedObservations:
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("gt_used_online") is not False or manifest.get("strong_model_called") is not False:
        raise ValueError("observation_manifest_scope_violation")
    records: list[FrameRecord] = []
    feature_rows: list[np.ndarray] = []
    occupancy_rows: list[np.ndarray] = []
    static_rows: list[np.ndarray] = []
    sequence_slices: dict[str, tuple[int, int]] = {}
    cameras: set[str] = set()
    for row in manifest["sequences"]:
        sequence = str(row["sequence_id"])
        camera = _camera(sequence)
        if camera == SEALED_CAMERA or SEALED_CAMERA in sequence:
            raise ValueError("sealed_camera_present_in_observations")
        cameras.add(camera)
        arrays = np.load(cache_root / row["cache"])
        receipt = json.loads((cache_root / row["receipt"]).read_text(encoding="utf-8"))
        if receipt.get("gt_used_online") is not False or receipt.get("strong_model_called") is not False:
            raise ValueError("observation_receipt_scope_violation")
        receipt_by_frame = {int(item["frame"]): item for item in receipt["receipts"]["yolo11s_global"]}
        sequence_root = pilot_root / "sequences" / sequence
        truth_by_frame: dict[int, list[Box]] = {}
        for item in parse_mot17_gt(sequence_root / "gt.txt"):
            truth_by_frame.setdefault(item.frame, []).append(item.box)
        frame_ids = arrays["frame_ids"].astype(int)
        normalized = arrays["person_robust_normalized"].astype(np.float32)
        entropy = arrays["dfl_entropy"].astype(np.float32)
        occupancy = arrays["detection_occupancy"].astype(np.float32)
        if normalized.shape[0] != len(frame_ids) or entropy.shape != normalized.shape or occupancy.shape[0] != len(frame_ids):
            raise ValueError("observation_cache_shape_mismatch")
        base = np.concatenate((normalized, entropy, occupancy), axis=1)
        causal = causal_channels(base)
        start = len(records)
        for index, frame in enumerate(frame_ids):
            image_path = sequence_root / "img1" / f"{frame:06d}.jpg"
            with Image.open(image_path) as image:
                width, height = image.size
            predictions = _prediction_boxes(receipt_by_frame[int(frame)])
            records.append(FrameRecord(
                sequence, camera, int(frame), width, height, predictions,
                tuple(truth_by_frame.get(int(frame), ())),
                np.empty((0, 0, 3), dtype=np.uint8), np.empty((0, 0, 3), dtype=np.uint8),
            ))
            feature_rows.append(causal[index])
            occupancy_rows.append(occupancy[index])
            static_rows.append(np.max(normalized[index], axis=0))
        sequence_slices[sequence] = (start, len(records))
    if tuple(sorted(cameras)) != EXPECTED_CAMERAS:
        raise ValueError("unexpected_development_camera_set")
    online = fixed_depthwise_head_features(np.stack(feature_rows))
    occupancy_array = np.stack(occupancy_rows)[:, 0]
    static = np.stack(static_rows)
    static = np.where(occupancy_array < 0.5, static, float(static.min()) - 1.0).astype(np.float32)
    return LoadedObservations(tuple(records), online, occupancy_array, static, sequence_slices)


def _indices(records: Iterable[FrameRecord], camera: str, *, equal: bool) -> np.ndarray:
    return np.asarray([index for index, record in enumerate(records) if (record.camera_id == camera) is equal], dtype=np.int64)


def _targets(records: tuple[FrameRecord, ...], height: int, width: int) -> np.ndarray:
    return np.stack([
        build_residual_targets(
            record.truth, tuple(box for box, _ in record.predictions),
            image_width=record.width, image_height=record.height,
            output_width=width, output_height=height,
        ).heatmap
        for record in records
    ])


def _standardize_fit(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    scale = train.std(axis=(0, 2, 3), keepdims=True)
    scale = np.maximum(scale, 1e-4)
    return ((train - mean) / scale).astype(np.float32), ((evaluation - mean) / scale).astype(np.float32)


def run_gate(loaded: LoadedObservations) -> dict[str, Any]:
    records = loaded.records
    height, width = loaded.online_features.shape[-2:]
    targets = _targets(records, height, width)
    folds: list[dict[str, Any]] = []
    dcrq_recalls: list[float] = []
    static_recalls: list[float] = []
    mean_areas: list[float] = []
    max_areas: list[float] = []
    for held_out in EXPECTED_CAMERAS:
        train_indices = _indices(records, held_out, equal=False)
        evaluation_indices = _indices(records, held_out, equal=True)
        train_records = [records[index] for index in train_indices]
        evaluation_records = [records[index] for index in evaluation_indices]
        sequence_audit = audit_sequence_split(
            {row.sequence_id for row in train_records}, {"calibration_not_used"},
            {row.sequence_id for row in evaluation_records},
        )
        camera_audit = audit_sequence_split(
            {row.camera_id for row in train_records}, {"calibration_not_used"}, {held_out},
        )
        if sequence_audit["sequence_split_leakage"] or camera_audit["sequence_split_leakage"]:
            raise ValueError("leave_one_camera_out_leakage")
        fit_features, evaluation_features = _standardize_fit(
            loaded.online_features[train_indices], loaded.online_features[evaluation_indices]
        )
        weights = fit_pointwise_ridge(
            fit_features, targets[train_indices], l2=1e-2, positive_weight=20.0
        )
        scores = predict_pointwise_ridge(evaluation_features, weights)
        scores = np.where(loaded.occupancy[evaluation_indices] < 0.5, scores, 0.0)
        dcrq = evaluate(evaluation_records, scores, peak_count=1, total_area=.095, budget=.20)
        static = evaluate(
            evaluation_records, loaded.static_scores[evaluation_indices],
            peak_count=1, total_area=.095, budget=.20,
        )
        dcrq_recalls.append(float(dcrq["joint_proposal_recall"]))
        static_recalls.append(float(static["joint_proposal_recall"]))
        mean_areas.append(float(dcrq["mean_roi_area"]))
        max_areas.append(float(dcrq["max_roi_area"]))
        folds.append({
            "held_out_camera": held_out,
            "fit_cameras": sorted({row.camera_id for row in train_records}),
            "fit_sequences": sorted({row.sequence_id for row in train_records}),
            "evaluation_sequences": sorted({row.sequence_id for row in evaluation_records}),
            "fit_frames": len(train_records),
            "evaluation_frames": len(evaluation_records),
            "sequence_split_audit": sequence_audit,
            "camera_split_audit": camera_audit,
            "dcrq": dcrq,
            "static_raw_head": static,
        })
    mean_recall = float(np.mean(dcrq_recalls))
    static_recall = float(np.mean(static_recalls))
    return {
        "schema_version": "fieldpack.dcrq-development-gate.v1",
        "evidence_status": "BOUNDED_FOUR_CAMERA_ENGINEERING_DEVELOPMENT_GATE",
        "preregistered_configuration": {
            "input_channels": ["person_robust_normalized_3", "dfl_entropy_3", "detection_occupancy_1"],
            "causal_features": {"ema_alpha": .60, "signed_frame_difference": True},
            "head": "fixed_depthwise_mean3x3_and_center_surround_then_pointwise_weighted_ridge",
            "l2": 1e-2,
            "positive_weight": 20.0,
            "frame_budget": .20,
            "frame_count_rule": "ceil(sequence_frames*0.20)",
            "peak_count": 1,
            "roi_total_area": .095,
            "iou_miss_threshold": .50,
        },
        "online_inputs": ["cached_frozen_yolo11s_head_observations"],
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
        "camera_split_leakage": False,
        "development_cameras": list(EXPECTED_CAMERAS),
        "folds": folds,
        "per_camera_joint_proposal_recall": dict(zip(EXPECTED_CAMERAS, dcrq_recalls)),
        "per_camera_static_raw_head_joint_proposal_recall": dict(zip(EXPECTED_CAMERAS, static_recalls)),
        "mean_joint_proposal_recall": mean_recall,
        "worst_camera_joint_proposal_recall": min(dcrq_recalls),
        "static_raw_head_joint_proposal_recall": static_recall,
        "relative_to_static_raw_head": mean_recall / static_recall if static_recall > 0 else 0.0,
        "mean_roi_area": float(np.mean(mean_areas)),
        "max_roi_area": max(max_areas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed DCRQ four-camera development gate")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    payload = run_gate(load_observations(args.cache_root, args.pilot_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "mean_joint_proposal_recall", "worst_camera_joint_proposal_recall",
        "static_raw_head_joint_proposal_recall", "relative_to_static_raw_head",
        "mean_roi_area", "max_roi_area",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a CPU-only, label-blind proposal-head sentinel on archived YOLO receipts."""
from __future__ import annotations

import argparse
import json
import math
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
    missed_ground_truth,
    predict_pointwise_ridge,
)


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    camera_id: str
    frame: int
    width: int
    height: int
    predictions: tuple[tuple[Box, float], ...]
    truth: tuple[Box, ...]
    rgb: np.ndarray
    reference_rgb: np.ndarray


def _camera(sequence_id: str) -> str:
    parts = sequence_id.split("_")
    if len(parts) < 3:
        raise ValueError("virat_sequence_id_invalid")
    return "_".join(parts[:3])


def _resize_gray(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(rgb).convert("L").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def _rasterize_detections(
    predictions: tuple[tuple[Box, float], ...],
    image_width: int,
    image_height: int,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    channels = np.zeros((4, output_height, output_width), dtype=np.float32)
    for box, confidence in predictions:
        x0 = max(0, min(output_width - 1, int(math.floor(box.x1 / image_width * output_width))))
        x1 = max(x0 + 1, min(output_width, int(math.ceil(box.x2 / image_width * output_width))))
        y0 = max(0, min(output_height - 1, int(math.floor(box.y1 / image_height * output_height))))
        y1 = max(y0 + 1, min(output_height, int(math.ceil(box.y2 / image_height * output_height))))
        channels[0, y0:y1, x0:x1] = np.maximum(channels[0, y0:y1, x0:x1], confidence)
        channels[1, y0:y1, x0:x1] = np.maximum(channels[1, y0:y1, x0:x1], 1.0 - confidence)
        area = max(1.0, (box.x2 - box.x1) * (box.y2 - box.y1))
        smallness = min(1.0, math.sqrt(image_width * image_height / area) / 50.0)
        channels[2, y0:y1, x0:x1] += smallness
        channels[3, y0, x0:x1] += 1.0
        channels[3, y1 - 1, x0:x1] += 1.0
        channels[3, y0:y1, x0] += 1.0
        channels[3, y0:y1, x1 - 1] += 1.0
    channels[2:] = np.clip(channels[2:], 0.0, 1.0)
    return channels


def _features(record: FrameRecord, output_width: int, output_height: int) -> dict[str, np.ndarray]:
    uncertainty = _rasterize_detections(
        record.predictions, record.width, record.height, output_width, output_height
    )
    current = _resize_gray(record.rgb, output_width, output_height)
    reference = _resize_gray(record.reference_rgb, output_width, output_height)
    rgb_diff = np.abs(current - reference)[None, ...]
    return {
        "uncertainty": uncertainty,
        "rgb_diff": rgb_diff,
        "fused": np.concatenate((uncertainty, rgb_diff), axis=0),
    }


def load_records(receipt_root: Path, pilot_root: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for receipt_path in sorted(receipt_root.glob("VIRAT*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        sequence = str(receipt["sequence_id"])
        sequence_root = pilot_root / "sequences" / sequence
        gt_rows = parse_mot17_gt(sequence_root / "gt.txt")
        truth_by_frame: dict[int, list[Box]] = {}
        for item in gt_rows:
            truth_by_frame.setdefault(item.frame, []).append(item.box)
        prediction_rows = receipt["receipts"]["yolo11s_global"]
        available = sorted(int(path.stem) for path in (sequence_root / "img1").glob("*.jpg"))
        positions = {frame: index for index, frame in enumerate(available)}
        for row in prediction_rows:
            frame = int(row["frame"])
            position = positions[frame]
            reference_frame = available[position - 1] if position > 0 else available[min(1, len(available) - 1)]
            rgb = np.asarray(Image.open(sequence_root / "img1" / f"{frame:06d}.jpg").convert("RGB"))
            reference = np.asarray(Image.open(sequence_root / "img1" / f"{reference_frame:06d}.jpg").convert("RGB"))
            predictions = tuple(
                (Box(**item["box"]), float(item["confidence"])) for item in row["predictions"]
            )
            records.append(FrameRecord(
                sequence, _camera(sequence), frame, rgb.shape[1], rgb.shape[0], predictions,
                tuple(truth_by_frame.get(frame, ())), rgb, reference,
            ))
    if not records:
        raise ValueError("no_receipt_records")
    return records


def _split_records(records: Iterable[FrameRecord]) -> dict[str, list[FrameRecord]]:
    camera_order = sorted({record.camera_id for record in records})
    if len(camera_order) != 3:
        raise ValueError("expected_exactly_three_camera_groups")
    mapping = dict(zip(camera_order, ("fit", "calibration", "evaluation")))
    return {
        split: [record for record in records if mapping[record.camera_id] == split]
        for split in ("fit", "calibration", "evaluation")
    }


def _peak_centers(score: np.ndarray, count: int) -> list[tuple[float, float]]:
    working = score.copy()
    centers: list[tuple[float, float]] = []
    radius = max(1, min(score.shape) // 10)
    for _ in range(count):
        flat = int(np.argmax(working))
        y, x = np.unravel_index(flat, working.shape)
        centers.append(((x + 0.5) / score.shape[1], (y + 0.5) / score.shape[0]))
        working[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = -1
    return centers


def _roi_boxes(score: np.ndarray, count: int, total_area: float, width: int, height: int) -> tuple[Box, ...]:
    per_roi_area = total_area / count
    roi_width_fraction = math.sqrt(per_roi_area * height / width)
    roi_height_fraction = per_roi_area / roi_width_fraction
    boxes = []
    for cx, cy in _peak_centers(score, count):
        x1 = min(max(0.0, cx - roi_width_fraction / 2), 1.0 - roi_width_fraction)
        y1 = min(max(0.0, cy - roi_height_fraction / 2), 1.0 - roi_height_fraction)
        boxes.append(Box(x1 * width, y1 * height, (x1 + roi_width_fraction) * width, (y1 + roi_height_fraction) * height))
    return tuple(boxes)


def _union_area_ratio(boxes: tuple[Box, ...], width: int, height: int) -> float:
    mask = np.zeros((180, 320), dtype=bool)
    for box in boxes:
        x0 = max(0, min(mask.shape[1] - 1, int(box.x1 / width * mask.shape[1])))
        x1 = max(x0 + 1, min(mask.shape[1], int(math.ceil(box.x2 / width * mask.shape[1]))))
        y0 = max(0, min(mask.shape[0] - 1, int(box.y1 / height * mask.shape[0])))
        y1 = max(y0 + 1, min(mask.shape[0], int(math.ceil(box.y2 / height * mask.shape[0]))))
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def _contains_center(roi: Box, target: Box) -> bool:
    cx = (target.x1 + target.x2) / 2.0
    cy = (target.y1 + target.y2) / 2.0
    return roi.x1 <= cx <= roi.x2 and roi.y1 <= cy <= roi.y2


def evaluate(records: list[FrameRecord], scores: np.ndarray, *, peak_count: int, total_area: float, budget: float) -> dict[str, Any]:
    by_sequence: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_sequence.setdefault(record.sequence_id, []).append(index)
    selected: set[int] = set()
    for sequence, indices in by_sequence.items():
        selected_count = max(1, int(math.ceil(len(indices) * budget)))
        order = sorted(indices, key=lambda index: (-float(scores[index].max()), records[index].frame))
        selected.update(order[:selected_count])
    total_missed = 0
    covered = 0
    roi_areas: list[float] = []
    selected_by_sequence: dict[str, int] = {}
    sequence_counts = {
        sequence: {"total_missed_persons": 0, "missed_persons_on_selected_frames": 0, "covered_missed_persons": 0}
        for sequence in by_sequence
    }
    missed_on_selected = 0
    for index, record in enumerate(records):
        missed = missed_ground_truth(record.truth, tuple(box for box, _ in record.predictions))
        total_missed += len(missed)
        sequence_counts[record.sequence_id]["total_missed_persons"] += len(missed)
        if index not in selected:
            continue
        missed_on_selected += len(missed)
        sequence_counts[record.sequence_id]["missed_persons_on_selected_frames"] += len(missed)
        selected_by_sequence[record.sequence_id] = selected_by_sequence.get(record.sequence_id, 0) + 1
        rois = _roi_boxes(scores[index], peak_count, total_area, record.width, record.height)
        roi_areas.append(_union_area_ratio(rois, record.width, record.height))
        frame_covered = sum(any(_contains_center(roi, box) for roi in rois) for box in missed)
        covered += frame_covered
        sequence_counts[record.sequence_id]["covered_missed_persons"] += frame_covered
    sequence_rows = {}
    for sequence, counts in sequence_counts.items():
        total = counts["total_missed_persons"]
        selected_missed = counts["missed_persons_on_selected_frames"]
        sequence_rows[sequence] = {
            **counts,
            "temporal_capture_recall": selected_missed / total if total else 0.0,
            "spatial_recall_given_selected": counts["covered_missed_persons"] / selected_missed if selected_missed else 0.0,
            "joint_proposal_recall": counts["covered_missed_persons"] / total if total else 0.0,
        }
    sequence_recalls = [row["joint_proposal_recall"] for row in sequence_rows.values()]
    return {
        "frames": len(records),
        "selected_frames": len(selected),
        "selected_by_sequence": selected_by_sequence,
        "total_missed_persons": total_missed,
        "covered_missed_persons": covered,
        "missed_persons_on_selected_frames": missed_on_selected,
        "pooled_joint_proposal_recall": covered / total_missed if total_missed else 0.0,
        "joint_proposal_recall": float(np.mean(sequence_recalls)) if sequence_recalls else 0.0,
        "worst_sequence_joint_proposal_recall": min(sequence_recalls, default=0.0),
        "temporal_capture_recall": float(np.mean([row["temporal_capture_recall"] for row in sequence_rows.values()])),
        "spatial_recall_given_selected": float(np.mean([row["spatial_recall_given_selected"] for row in sequence_rows.values()])),
        "mean_roi_area": float(np.mean(roi_areas)) if roi_areas else 0.0,
        "max_roi_area": max(roi_areas, default=0.0),
        "sequences": sequence_rows,
    }


def run(
    records: list[FrameRecord],
    output_width: int = 40,
    output_height: int = 23,
    feature_difference: np.ndarray | None = None,
) -> dict[str, Any]:
    splits = _split_records(records)
    sequence_audit = audit_sequence_split(
        {r.sequence_id for r in splits["fit"]},
        {r.sequence_id for r in splits["calibration"]},
        {r.sequence_id for r in splits["evaluation"]},
    )
    camera_audit = audit_sequence_split(
        {r.camera_id for r in splits["fit"]},
        {r.camera_id for r in splits["calibration"]},
        {r.camera_id for r in splits["evaluation"]},
    )
    split_features: dict[str, dict[str, np.ndarray]] = {}
    split_targets: dict[str, np.ndarray] = {}
    for split, rows in splits.items():
        feature_rows = [_features(row, output_width, output_height) for row in rows]
        split_features[split] = {
            mode: np.stack([item[mode] for item in feature_rows])
            for mode in ("uncertainty", "rgb_diff", "fused")
        }
        split_targets[split] = np.stack([
            build_residual_targets(
                row.truth, tuple(box for box, _ in row.predictions),
                image_width=row.width, image_height=row.height,
                output_width=output_width, output_height=output_height,
            ).heatmap
            for row in rows
        ])
    if feature_difference is not None:
        if feature_difference.shape != (len(records), 3, output_height, output_width):
            raise ValueError("frozen_feature_difference_shape_mismatch")
        record_index = {id(record): index for index, record in enumerate(records)}
        for split, rows in splits.items():
            frozen = np.stack([feature_difference[record_index[id(record)]] for record in rows])
            split_features[split]["feature_diff"] = frozen
            split_features[split]["uncertainty_feature"] = np.concatenate(
                (split_features[split]["uncertainty"], frozen), axis=1
            )
            split_features[split]["all"] = np.concatenate(
                (split_features[split]["fused"], frozen), axis=1
            )
    candidates = []
    trained: dict[tuple[str, float], np.ndarray] = {}
    modes = tuple(split_features["fit"])
    for mode in modes:
        for l2 in (1e-3, 1e-2, 1e-1):
            weights = fit_pointwise_ridge(split_features["fit"][mode], split_targets["fit"], l2=l2)
            trained[(mode, l2)] = weights
            calibration_scores = predict_pointwise_ridge(split_features["calibration"][mode], weights)
            for peak_count in (1, 2, 3):
                result = evaluate(splits["calibration"], calibration_scores, peak_count=peak_count, total_area=.095, budget=.20)
                candidates.append({"mode": mode, "l2": l2, "peak_count": peak_count, **result})
    best = max(candidates, key=lambda row: (row["worst_sequence_joint_proposal_recall"], row["joint_proposal_recall"], -row["mean_roi_area"], -row["peak_count"], row["mode"]))
    evaluation_ablations = {}
    for mode in modes:
        mode_best = max(
            (row for row in candidates if row["mode"] == mode),
            key=lambda row: (row["worst_sequence_joint_proposal_recall"], row["joint_proposal_recall"], -row["mean_roi_area"], -row["peak_count"]),
        )
        mode_scores = predict_pointwise_ridge(
            split_features["evaluation"][mode], trained[(mode, mode_best["l2"])]
        )
        evaluation_ablations[mode] = {
            "calibration": mode_best,
            "evaluation": evaluate(
                splits["evaluation"], mode_scores, peak_count=mode_best["peak_count"],
                total_area=.095, budget=.20,
            ),
        }
    evaluation = evaluation_ablations[best["mode"]]["evaluation"]
    return {
        "schema_version": "fieldpack.missed-person-proposal-baseline.v1",
        "evidence_status": "BOUNDED_ENGINEERING_PILOT_NOT_FORMAL_STATISTICS",
        "online_inputs": ["archived_yolo11s_predictions", "current_rgb", "previous_available_rgb"],
        "gt_used_online": False,
        "budget": .20,
        "grid": [output_height, output_width],
        "split": {
            name: {"sequences": sorted({r.sequence_id for r in rows}), "cameras": sorted({r.camera_id for r in rows}), "frames": len(rows)}
            for name, rows in splits.items()
        },
        "sequence_split_audit": sequence_audit,
        "camera_split_audit": camera_audit,
        "sequence_split_leakage": bool(sequence_audit["sequence_split_leakage"] or camera_audit["sequence_split_leakage"]),
        "calibration_winner": best,
        "evaluation": evaluation,
        "evaluation_ablations": evaluation_ablations,
        "joint_proposal_recall": evaluation["joint_proposal_recall"],
        "mean_roi_area": evaluation["mean_roi_area"],
        "max_roi_area": evaluation["max_roi_area"],
        "calibration_candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-model")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    records = load_records(args.receipt_root, args.pilot_root)
    frozen = None
    if args.small_model:
        from research_tools.yolo_frozen_feature_difference import FrozenYoloFeatureDifference
        with FrozenYoloFeatureDifference(args.small_model, device=args.device) as extractor:
            frozen = extractor.extract_pairs(
                ((record.rgb, record.reference_rgb) for record in records),
                output_width=40, output_height=23,
            )
    payload = run(records, feature_difference=frozen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("joint_proposal_recall", "mean_roi_area", "max_roi_area", "sequence_split_leakage")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

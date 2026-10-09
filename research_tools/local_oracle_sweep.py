"""Bounded Oracle sweep for local input scale, confidence, and fusion."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from evidencegate.cascade import Box as CascadeBox, boundary_aware_wbf, primary_ownership
from research_tools.budgeted_cascade import (
    _ambiguity_score,
    _batched_predict,
    _crop,
    _missed_score,
    deterministic_budget,
    tight_candidate_rois,
)
from research_tools.cascade_experiment import (
    Box,
    GroundTruth,
    Prediction,
    UltralyticsYOLOAdapter,
    _nms,
    compute_metrics,
    parse_mot17_gt,
    select_frames,
)
from research_tools.run_cascade_successor import _heuristic_rois, _oracle_rois, write_experiment_receipt


def merge_sweep_predictions(
    frame_index: int,
    global_predictions: Sequence[Prediction],
    local_predictions: Sequence[Prediction],
    roi: Any,
    crop_bounds: tuple[int, int, int, int],
    *,
    local_confidence: float,
    fusion: str,
) -> tuple[Prediction, ...]:
    if not 0 <= local_confidence <= 1:
        raise ValueError("local_confidence_out_of_range")
    if fusion not in {"nms", "wbf"}:
        raise ValueError("unsupported_fusion")
    x1, y1, _, _ = crop_bounds
    local_boxes = [
        CascadeBox(
            pred.box.x1 + x1,
            pred.box.y1 + y1,
            pred.box.x2 + x1,
            pred.box.y2 + y1,
            score=pred.confidence,
            source=pred.source,
            roi_id=roi.roi_id,
        )
        for pred in local_predictions
        if pred.confidence >= local_confidence
    ]
    global_boxes = [CascadeBox(*pred.box.coords, score=pred.confidence, source=pred.source) for pred in global_predictions]
    combined = global_boxes + primary_ownership(local_boxes, (roi,))
    fused = boundary_aware_wbf(combined, (roi,), iou_threshold=.55) if fusion == "wbf" else _nms(combined, .5)
    return tuple(
        Prediction(frame_index, Box(item.x1, item.y1, item.x2, item.y2), item.score, item.source)
        for item in fused
    )


def motion_residual_boxes(
    frame: np.ndarray,
    reference: np.ndarray,
    *,
    block_size: int = 16,
    percentile: float = 90.0,
    minimum_difference: float = 8.0,
) -> tuple[tuple[Box, ...], float]:
    """Return label-blind change components and normalized residual energy."""
    if frame.shape != reference.shape or frame.ndim != 3:
        raise ValueError("motion_reference_shape_mismatch")
    if block_size < 1 or not 0 < percentile < 100 or minimum_difference < 0:
        raise ValueError("motion_parameters_invalid")
    gray = frame.astype(np.float32).mean(axis=2)
    prior = reference.astype(np.float32).mean(axis=2)
    residual = np.abs(gray - prior)
    height, width = residual.shape
    rows = (height + block_size - 1) // block_size
    columns = (width + block_size - 1) // block_size
    block_means = np.zeros((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            cell = residual[row * block_size : min(height, (row + 1) * block_size), column * block_size : min(width, (column + 1) * block_size)]
            block_means[row, column] = float(cell.mean()) if cell.size else 0.0
    threshold = max(minimum_difference, float(np.percentile(block_means, percentile)))
    active = block_means > threshold
    visited: set[tuple[int, int]] = set()
    boxes: list[Box] = []
    for row in range(rows):
        for column in range(columns):
            if not active[row, column] or (row, column) in visited:
                continue
            stack = [(row, column)]; visited.add((row, column)); component = []
            while stack:
                current = stack.pop(); component.append(current)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbor = (current[0] + dr, current[1] + dc)
                        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < columns and active[neighbor] and neighbor not in visited:
                            visited.add(neighbor); stack.append(neighbor)
            y_values = [item[0] for item in component]; x_values = [item[1] for item in component]
            boxes.append(Box(
                min(x_values) * block_size,
                min(y_values) * block_size,
                min(width, (max(x_values) + 1) * block_size),
                min(height, (max(y_values) + 1) * block_size),
            ))
    normalized_energy = float(np.mean(residual) / 255.0)
    return tuple(boxes), normalized_energy


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    numeric = ("precision", "recall", "f1", "ap50", "count_mae", "small_object_recall")
    return {key: float(np.mean([row[key] for row in rows])) for key in numeric}


def proposal_coverage(
    selected_frames: Sequence[int],
    selected_rois: Mapping[int, Any],
    missed_truth: Mapping[int, Sequence[Box]],
) -> dict[str, float]:
    missed_total = 0
    covered_total = 0
    selected_with_miss = 0
    for frame in selected_frames:
        missed = tuple(missed_truth.get(frame, ()))
        missed_total += len(missed)
        covered = sum(
            selected_rois[frame].contains_center(CascadeBox(*box.coords))
            for box in missed
        )
        covered_total += covered
        selected_with_miss += int(bool(missed))
    return {
        "selected_missed_persons": float(missed_total),
        "covered_missed_persons": float(covered_total),
        "proposal_recall_on_selected_misses": covered_total / missed_total if missed_total else 0.0,
        "selected_frames_with_miss_fraction": selected_with_miss / len(selected_frames) if selected_frames else 0.0,
    }


def run_local_oracle_sweep(
    frames: Mapping[int, np.ndarray],
    truth: Sequence[GroundTruth],
    small: Any,
    local_detectors: Mapping[int, Any],
    *,
    roi_sizes: Sequence[int],
    confidence_thresholds: Sequence[float],
    fusions: Sequence[str],
    budget: float,
    batch_size: int,
    seed: int,
    sequence_id: str,
    global_imgsz: int = 640,
    route: str = "oracle",
    reference_frames: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    frame_ids = tuple(sorted(frames))
    if not frame_ids:
        raise ValueError("frames_must_not_be_empty")
    if set(roi_sizes) != set(local_detectors):
        raise ValueError("roi_sizes_and_detectors_mismatch")
    if len(set(roi_sizes)) != len(tuple(roi_sizes)) or any(size < 1 for size in roi_sizes):
        raise ValueError("invalid_roi_sizes")
    if any(not 0 <= value <= 1 for value in confidence_thresholds):
        raise ValueError("invalid_confidence_thresholds")
    if set(fusions) - {"nms", "wbf"}:
        raise ValueError("invalid_fusions")
    if route not in {"oracle", "heuristic", "motion"}:
        raise ValueError("invalid_route")
    if route == "motion" and (reference_frames is None or set(reference_frames) != set(frame_ids)):
        raise ValueError("motion_reference_frames_mismatch")
    by_truth: dict[int, list[GroundTruth]] = {}
    for item in truth:
        by_truth.setdefault(item.frame, []).append(item)

    began = time.perf_counter()
    global_batches = small.predict_batch([frames[index] for index in frame_ids], frame_indices=frame_ids)
    global_ms = (time.perf_counter() - began) * 1000.0
    global_outputs = {frame: tuple(rows) for frame, rows in zip(frame_ids, global_batches)}
    missed_truth = {
        frame: tuple(
            item.box for item in by_truth.get(frame, ())
            if not any(item.box.iou(pred.box) >= .5 for pred in global_outputs[frame])
        )
        for frame in frame_ids
    }
    motion_energy: dict[int, float] = {}
    proposal_boxes: dict[int, tuple[Box, ...]] = {}
    for frame in frame_ids:
        if route == "oracle":
            proposal_boxes[frame] = missed_truth[frame]
        elif route == "heuristic":
            proposal_boxes[frame] = tuple(pred.box for pred in global_outputs[frame])
        else:
            proposal_boxes[frame], motion_energy[frame] = motion_residual_boxes(frames[frame], reference_frames[frame])  # type: ignore[index]
    candidates = {
        frame: tight_candidate_rois(
            frames[frame].shape[1], frames[frame].shape[0], proposal_boxes[frame],
            prefix={"oracle": "o", "heuristic": "h", "motion": "m"}[route], expansion=0.0,
        )
        for frame in frame_ids
    }
    if route == "oracle":
        selected_roi = {
            frame: _oracle_rois(global_outputs[frame], by_truth.get(frame, ()), candidates[frame], 1)[0]
            for frame in frame_ids
        }
        scores = {frame: _missed_score(global_outputs[frame], by_truth.get(frame, ())) for frame in frame_ids}
    elif route == "heuristic":
        selected_roi = {
            frame: _heuristic_rois(global_outputs[frame], candidates[frame], 1)[0]
            for frame in frame_ids
        }
        scores = {frame: _ambiguity_score(global_outputs[frame], candidates[frame]) for frame in frame_ids}
    else:
        selected_roi = {
            frame: max(
                candidates[frame],
                key=lambda roi: (
                    float(np.mean(np.abs(frames[frame][int(roi.y1):int(roi.y2), int(roi.x1):int(roi.x2)].astype(np.float32) - reference_frames[frame][int(roi.y1):int(roi.y2), int(roi.x1):int(roi.x2)].astype(np.float32)))),  # type: ignore[index]
                    roi.roi_id,
                ),
            )
            for frame in frame_ids
        }
        scores = motion_energy
    selected_ids = deterministic_budget(frame_ids, budget, scores, seed=seed, namespace=f"{sequence_id}:{route}-local-sweep")
    coverage = proposal_coverage(selected_ids, selected_roi, missed_truth)
    crops: list[np.ndarray] = []
    crop_bounds: dict[int, tuple[int, int, int, int]] = {}
    for frame in selected_ids:
        crop, bounds = _crop(frames[frame], selected_roi[frame])
        crops.append(crop)
        crop_bounds[frame] = bounds

    conditions: dict[str, dict[str, Any]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    baseline_rows = []
    for frame in frame_ids:
        baseline_rows.append({
            **compute_metrics(global_outputs[frame], by_truth.get(frame, ())),
            "frame": frame,
            "predictions": [asdict(item) for item in global_outputs[frame]],
        })
    conditions["yolo11s_global"] = {
        **_summarize(baseline_rows),
        "budget_fraction": 0.0,
        "selected_frames": 0,
        "model_input_tensor_ratio_to_small_global": 1.0,
        "model_compute_total_ms": global_ms,
        "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END",
    }
    receipts["yolo11s_global"] = baseline_rows

    cache_receipts: dict[str, Any] = {}
    for roi_size in sorted(roi_sizes):
        local_outputs, local_ms, batches = _batched_predict(
            local_detectors[roi_size], crops, selected_ids, batch_size,
        )
        cache_key = f"{route}20_context0_imgsz{roi_size}"
        cache_receipts[cache_key] = {
            "selected_frames": list(selected_ids),
            "local_model_compute_ms": local_ms,
            "batches": batches,
        }
        for confidence in sorted(confidence_thresholds):
            for fusion in sorted(fusions):
                name = f"{route}_i{roi_size}_c{int(round(confidence * 100)):02d}_{fusion}"
                rows = []
                for frame in frame_ids:
                    predictions = global_outputs[frame]
                    if frame in selected_ids:
                        predictions = merge_sweep_predictions(
                            frame, predictions, local_outputs[frame], selected_roi[frame], crop_bounds[frame],
                            local_confidence=confidence, fusion=fusion,
                        )
                    rows.append({
                        **compute_metrics(predictions, by_truth.get(frame, ())),
                        "frame": frame,
                        "upgraded": frame in selected_ids,
                        "selected_roi": asdict(selected_roi[frame]) if frame in selected_ids else None,
                        "predictions": [asdict(item) for item in predictions],
                    })
                tensor_ratio = 1.0 + budget * (roi_size * roi_size) / (global_imgsz * global_imgsz)
                conditions[name] = {
                    **_summarize(rows),
                    "budget_fraction": len(selected_ids) / len(frame_ids),
                    "selected_frames": len(selected_ids),
                    "roi_imgsz": roi_size,
                    "local_confidence": confidence,
                    "fusion": fusion,
                    "model_input_tensor_ratio_to_small_global": tensor_ratio,
                    "model_compute_total_ms": global_ms + local_ms,
                    "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END",
                    "local_prediction_cache_key": cache_key,
                    **coverage,
                }
                receipts[name] = rows
    return {
        "schema_version": "fieldpack.local-oracle-sweep.v1" if route == "oracle" else "fieldpack.local-route-transfer.v1",
        "evidence_status": "ENGINEERING_PILOT_NOT_FORMAL_STATISTICS",
        "route": route,
        "route_scope": "GT_SELECTED_UPPER_BOUND_NOT_DEPLOYABLE" if route == "oracle" else "LABEL_BLIND_DEPLOYABLE_CANDIDATE",
        "sequence_id": sequence_id,
        "frames": len(frame_ids),
        "seed": seed,
        "budget": budget,
        "batch_size": batch_size,
        "context_expansion": 0.0,
        "global_imgsz": global_imgsz,
        "roi_sizes": sorted(roi_sizes),
        "confidence_thresholds": sorted(confidence_thresholds),
        "fusions": sorted(fusions),
        "selected_frames": list(selected_ids),
        "conditions": conditions,
        "receipts": receipts,
        "local_prediction_caches": cache_receipts,
    }


def select_disjoint_sample(
    available: Sequence[int],
    sample_size: int,
    *,
    seed: int,
    exclude_seed: int | None = None,
    exclude_sample_size: int | None = None,
) -> tuple[int, ...]:
    candidates = tuple(sorted(set(int(value) for value in available)))
    excluded: set[int] = set()
    if exclude_seed is not None:
        excluded = set(select_frames(candidates, exclude_sample_size or sample_size, seed=exclude_seed))
    remaining = tuple(value for value in candidates if value not in excluded)
    if len(remaining) < sample_size:
        raise ValueError("insufficient_disjoint_frames")
    return select_frames(remaining, sample_size, seed=seed)


def _load_sample(
    frames_dir: Path,
    gt: Sequence[GroundTruth],
    sample_size: int,
    seed: int,
    *,
    exclude_seed: int | None = None,
    exclude_sample_size: int | None = None,
) -> tuple[dict[int, np.ndarray], tuple[GroundTruth, ...]]:
    available = [int(path.stem) for path in frames_dir.glob("*.jpg") if path.stem.isdigit()]
    selected = set(select_disjoint_sample(available, sample_size, seed=seed, exclude_seed=exclude_seed, exclude_sample_size=exclude_sample_size))
    frames = {frame: np.asarray(Image.open(frames_dir / f"{frame:06d}.jpg").convert("RGB")) for frame in sorted(selected)}
    return frames, tuple(item for item in gt if item.frame in selected)


def load_reference_frames(frames_dir: Path, frame_ids: Sequence[int]) -> dict[int, np.ndarray]:
    available = sorted(int(path.stem) for path in frames_dir.glob("*.jpg") if path.stem.isdigit())
    positions = {frame: index for index, frame in enumerate(available)}
    references = {}
    for frame in frame_ids:
        position = positions[frame]
        reference_id = available[position - 1] if position > 0 else available[min(1, len(available) - 1)]
        references[frame] = np.asarray(Image.open(frames_dir / f"{reference_id:06d}.jpg").convert("RGB"))
    return references


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded local Oracle factorial sweep")
    parser.add_argument("--frames-dir", type=Path, required=True); parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--small-model", required=True); parser.add_argument("--local-model", required=True)
    parser.add_argument("--sequence-id", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20); parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--exclude-seed", type=int); parser.add_argument("--exclude-sample-size", type=int)
    parser.add_argument("--roi-sizes", default="384,512,640"); parser.add_argument("--confidence-thresholds", default="0.10,0.25,0.50")
    parser.add_argument("--fusions", default="nms,wbf"); parser.add_argument("--budget", type=float, default=.20)
    parser.add_argument("--route", choices=("oracle", "heuristic", "motion"), default="oracle")
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--device", default="0"); parser.add_argument("--global-imgsz", type=int, default=640)
    args = parser.parse_args()
    roi_sizes = tuple(int(value) for value in args.roi_sizes.split(",")); thresholds = tuple(float(value) for value in args.confidence_thresholds.split(",")); fusions = tuple(value.strip() for value in args.fusions.split(","))
    truth = parse_mot17_gt(args.gt); frames, truth = _load_sample(args.frames_dir, truth, args.sample_size, args.seed, exclude_seed=args.exclude_seed, exclude_sample_size=args.exclude_sample_size)
    small = UltralyticsYOLOAdapter(args.small_model, device=args.device, name="yolo11s", imgsz=args.global_imgsz)
    locals_by_size = {size: UltralyticsYOLOAdapter(args.local_model, device=args.device, name=f"yolo11m_roi_{size}", imgsz=size, conf=min(thresholds)) for size in roi_sizes}
    if frames:
        first = frames[min(frames)]; small.predict(first, frame_index=-1)
        for detector in locals_by_size.values(): detector.predict(first, frame_index=-1)
    reference_frames = load_reference_frames(args.frames_dir, tuple(frames)) if args.route == "motion" else None
    result = run_local_oracle_sweep(frames, truth, small, locals_by_size, roi_sizes=roi_sizes, confidence_thresholds=thresholds, fusions=fusions, budget=args.budget, batch_size=args.batch_size, seed=args.seed, sequence_id=args.sequence_id, global_imgsz=args.global_imgsz, route=args.route, reference_frames=reference_frames)
    result["sample_exclusion"] = {"exclude_seed": args.exclude_seed, "exclude_sample_size": args.exclude_sample_size}
    write_experiment_receipt(args.output, result)
    print(json.dumps({"output": str(args.output), "frames": len(frames), "conditions": len(result["conditions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

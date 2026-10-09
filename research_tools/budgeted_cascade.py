"""Sparse frame-budget and micro-batched ROI cascade experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from evidencegate.cascade import Box as CascadeBox, ROI, iou as cascade_iou, primary_ownership
from research_tools.cascade_experiment import (
    Box,
    GroundTruth,
    Prediction,
    UltralyticsYOLOAdapter,
    _nms,
    candidate_rois,
    compute_metrics,
    parse_mot17_gt,
    router_features,
    select_frames,
)
from research_tools.run_cascade_successor import _heuristic_rois, _oracle_rois, write_experiment_receipt


def deterministic_budget(frames: Sequence[int], fraction: float, scores: Mapping[int, float], *, seed: int, namespace: str) -> tuple[int, ...]:
    if not 0 <= fraction <= 1:
        raise ValueError("budget_fraction_out_of_range")
    count = int(math.ceil(len(frames) * fraction)) if fraction else 0
    ranked = sorted(
        frames,
        key=lambda frame: (
            -float(scores.get(frame, 0.0)),
            hashlib.sha256(f"{seed}|{namespace}|{frame}".encode()).hexdigest(),
        ),
    )
    return tuple(sorted(ranked[:count]))


def rotated_condition_order(conditions: Sequence[str], *, seed: int, sequence_id: str, batch_index: int) -> tuple[str, ...]:
    if not conditions:
        return ()
    digest = hashlib.sha256(f"{seed}|{sequence_id}|{batch_index}".encode()).digest()
    offset = int.from_bytes(digest[:4], "big") % len(conditions)
    return tuple(conditions[offset:]) + tuple(conditions[:offset])


def _bounded_roi(
    width: int,
    height: int,
    boxes: Sequence[Box],
    *,
    roi_id: str,
    min_area_ratio: float = 0.05,
    max_expanded_area_ratio: float = 0.15,
    expansion: float = 0.15,
) -> ROI:
    if expansion < 0:
        raise ValueError("context_expansion_must_be_nonnegative")
    if min_area_ratio <= 0 or max_expanded_area_ratio <= 0:
        raise ValueError("roi_area_ratio_must_be_positive")
    if min_area_ratio * ((1.0 + expansion) ** 2) > max_expanded_area_ratio + 1e-12:
        raise ValueError("context_expansion_incompatible_with_area_bounds")
    image_area = width * height
    min_area = image_area * min_area_ratio
    max_core_area = image_area * max_expanded_area_ratio / ((1.0 + expansion) ** 2)
    x1 = min(box.x1 for box in boxes); y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes); y2 = max(box.y2 for box in boxes)
    cluster_width, cluster_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    target_area = min(max_core_area, max(min_area, cluster_width * cluster_height * 2.25))
    aspect = min(2.0, max(0.75, cluster_width / cluster_height))
    roi_width = min(float(width), math.sqrt(target_area * aspect))
    roi_height = min(float(height), target_area / max(1.0, roi_width))
    if roi_height > height:
        roi_height = float(height); roi_width = min(float(width), target_area / roi_height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0.0, cx - roi_width / 2), width - roi_width)
    top = min(max(0.0, cy - roi_height / 2), height - roi_height)
    return ROI(left, top, left + roi_width, top + roi_height, roi_id, expansion)


def _cluster_boxes(boxes: Sequence[Box], width: int, height: int) -> tuple[tuple[Box, ...], ...]:
    if not boxes:
        return ()
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]; index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    margin_x, margin_y = width * 0.04, height * 0.04
    for left, first in enumerate(boxes):
        for right in range(left + 1, len(boxes)):
            second = boxes[right]
            separated = (
                first.x2 + margin_x < second.x1
                or second.x2 + margin_x < first.x1
                or first.y2 + margin_y < second.y1
                or second.y2 + margin_y < first.y1
            )
            if not separated:
                union(left, right)
    grouped: dict[int, list[Box]] = {}
    for index, box in enumerate(boxes):
        grouped.setdefault(find(index), []).append(box)
    return tuple(tuple(grouped[key]) for key in sorted(grouped))


def tight_candidate_rois(
    width: int,
    height: int,
    boxes: Sequence[Box],
    *,
    prefix: str,
    min_area_ratio: float = 0.05,
    max_expanded_area_ratio: float = 0.15,
    expansion: float = 0.15,
) -> tuple[ROI, ...]:
    clusters = _cluster_boxes(boxes, width, height)
    if not clusters:
        # Label-free fallback is still bounded: nine small windows, not half-frame tiles.
        fallback: list[Box] = []
        for row in range(3):
            for column in range(3):
                cx, cy = width * (column + 0.5) / 3, height * (row + 0.5) / 3
                fallback.append(Box(cx - 1, cy - 1, cx + 1, cy + 1))
        clusters = tuple((box,) for box in fallback)
    candidates = [
        _bounded_roi(
            width,
            height,
            cluster,
            roi_id=f"{prefix}{index:04d}",
            min_area_ratio=min_area_ratio,
            max_expanded_area_ratio=max_expanded_area_ratio,
            expansion=expansion,
        )
        for index, cluster in enumerate(clusters)
    ]
    deduplicated: list[ROI] = []
    for roi in candidates:
        roi_box = CascadeBox(roi.x1, roi.y1, roi.x2, roi.y2)
        if all(cascade_iou(roi_box, CascadeBox(item.x1, item.y1, item.x2, item.y2)) < 0.75 for item in deduplicated):
            deduplicated.append(roi)
    return tuple(deduplicated)


def _ambiguity_score(predictions: Sequence[Prediction], rois: Sequence[Any]) -> float:
    values = []
    for roi in rois:
        density, mean_conf, small_fraction, boundary_fraction = router_features(predictions, roi)
        values.append((1.0 - mean_conf) + small_fraction + boundary_fraction + min(1.0, density * roi.pixel_area))
    return max(values, default=0.0)


def _missed_score(predictions: Sequence[Prediction], truth: Sequence[GroundTruth]) -> float:
    return float(sum(not any(item.box.iou(pred.box) >= 0.5 for pred in predictions) for item in truth))


def _crop(frame: np.ndarray, roi: Any) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    ex = roi.expanded
    x1, y1 = max(0, int(ex[0])), max(0, int(ex[1]))
    x2, y2 = min(width, int(ex[2])), min(height, int(ex[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty_roi_crop")
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def merge_local_predictions(
    frame_index: int,
    global_predictions: Sequence[Prediction],
    local_predictions: Sequence[Prediction],
    roi: Any,
    crop_bounds: tuple[int, int, int, int],
) -> tuple[Prediction, ...]:
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
    ]
    global_boxes = [CascadeBox(*pred.box.coords, score=pred.confidence, source=pred.source) for pred in global_predictions]
    combined = global_boxes + primary_ownership(local_boxes, (roi,))
    fused = _nms(combined, 0.5)
    return tuple(Prediction(frame_index, Box(item.x1, item.y1, item.x2, item.y2), item.score, item.source) for item in fused)


def _batched_predict(detector: Any, frames: Sequence[np.ndarray], indices: Sequence[int], batch_size: int) -> tuple[dict[int, tuple[Prediction, ...]], float, list[dict[str, float]]]:
    output: dict[int, tuple[Prediction, ...]] = {}
    total_ms = 0.0
    receipts = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        batch_indices = indices[start : start + batch_size]
        began = time.perf_counter()
        predictions = detector.predict_batch(batch_frames, frame_indices=batch_indices)
        elapsed_ms = (time.perf_counter() - began) * 1000.0
        total_ms += elapsed_ms
        for frame_index, rows in zip(batch_indices, predictions):
            output[int(frame_index)] = tuple(rows)
        receipts.append({"batch_size": float(len(batch_frames)), "elapsed_ms": elapsed_ms, "amortized_ms": elapsed_ms / len(batch_frames)})
    return output, total_ms, receipts


def run_budgeted_phase(
    frames: Mapping[int, np.ndarray],
    truth: Sequence[GroundTruth],
    small: Any,
    large_global: Any,
    local: Any,
    *,
    budgets: Sequence[float],
    batch_size: int,
    seed: int,
    sequence_id: str,
    global_imgsz: int = 640,
    roi_imgsz: int = 384,
    context_expansion: float = 0.15,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size_must_be_positive")
    frame_ids = tuple(sorted(frames))
    if not frame_ids:
        raise ValueError("frames_must_not_be_empty")
    frame_values = tuple(frames[index] for index in frame_ids)
    by_truth: dict[int, list[GroundTruth]] = {}
    for item in truth:
        by_truth.setdefault(item.frame, []).append(item)

    global_outputs: dict[str, dict[int, tuple[Prediction, ...]]] = {}
    global_times: dict[str, float] = {}
    global_batches: dict[str, list[dict[str, float]]] = {}
    for batch_index, start in enumerate(range(0, len(frame_ids), batch_size)):
        ids = frame_ids[start : start + batch_size]
        values = frame_values[start : start + batch_size]
        for condition in rotated_condition_order(("yolo11s_global", "yolo11m_global"), seed=seed, sequence_id=sequence_id, batch_index=batch_index):
            detector = small if condition == "yolo11s_global" else large_global
            began = time.perf_counter()
            result = detector.predict_batch(values, frame_indices=ids)
            elapsed = (time.perf_counter() - began) * 1000.0
            global_times[condition] = global_times.get(condition, 0.0) + elapsed
            global_batches.setdefault(condition, []).append({"batch_size": float(len(ids)), "elapsed_ms": elapsed, "amortized_ms": elapsed / len(ids)})
            store = global_outputs.setdefault(condition, {})
            store.update({int(frame): tuple(rows) for frame, rows in zip(ids, result)})

    heuristic_rois = {
        frame: tight_candidate_rois(
            frames[frame].shape[1], frames[frame].shape[0],
            tuple(pred.box for pred in global_outputs["yolo11s_global"][frame]), prefix="h",
            expansion=context_expansion,
        )
        for frame in frame_ids
    }
    missed_truth = {
        frame: tuple(
            item.box for item in by_truth.get(frame, ())
            if not any(item.box.iou(pred.box) >= 0.5 for pred in global_outputs["yolo11s_global"][frame])
        )
        for frame in frame_ids
    }
    oracle_rois = {
        frame: tight_candidate_rois(
            frames[frame].shape[1], frames[frame].shape[0], missed_truth[frame], prefix="o",
            expansion=context_expansion,
        )
        for frame in frame_ids
    }
    heuristic_roi = {frame: _heuristic_rois(global_outputs["yolo11s_global"][frame], heuristic_rois[frame], 1)[0] for frame in frame_ids}
    oracle_roi = {frame: _oracle_rois(global_outputs["yolo11s_global"][frame], by_truth.get(frame, ()), oracle_rois[frame], 1)[0] for frame in frame_ids}
    heuristic_scores = {frame: _ambiguity_score(global_outputs["yolo11s_global"][frame], heuristic_rois[frame]) for frame in frame_ids}
    oracle_scores = {frame: _missed_score(global_outputs["yolo11s_global"][frame], by_truth.get(frame, ())) for frame in frame_ids}

    conditions: dict[str, dict[str, Any]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}

    def record_condition(
        name: str,
        predictions: Mapping[int, tuple[Prediction, ...]],
        selected: set[int],
        selected_rois: Mapping[int, Any],
        actual_crop_bounds: Mapping[int, tuple[int, int, int, int]],
        local_ms: float,
        local_batches: list[dict[str, float]],
    ) -> None:
        rows = []
        small_amortized = global_times["yolo11s_global"] / len(frame_ids)
        local_amortized = local_ms / max(1, len(selected))
        for frame in frame_ids:
            metrics = compute_metrics(predictions[frame], by_truth.get(frame, ()))
            upgraded = frame in selected
            roi = selected_rois.get(frame) if upgraded else None
            selected_roi = None
            if roi is not None:
                image_height, image_width = frames[frame].shape[:2]
                crop_xyxy = list(actual_crop_bounds[frame])
                core_area = (roi.x2 - roi.x1) * (roi.y2 - roi.y1)
                crop_area = (crop_xyxy[2] - crop_xyxy[0]) * (crop_xyxy[3] - crop_xyxy[1])
                selected_roi = {
                    "roi_id": roi.roi_id,
                    "core_xyxy": [roi.x1, roi.y1, roi.x2, roi.y2],
                    "crop_xyxy": crop_xyxy,
                    "core_area_ratio": core_area / (image_width * image_height),
                    "crop_area_ratio": crop_area / (image_width * image_height),
                    "resize_scale_xy": [roi_imgsz / max(1.0, crop_xyxy[2] - crop_xyxy[0]), roi_imgsz / max(1.0, crop_xyxy[3] - crop_xyxy[1])],
                    "context_expansion": roi.expansion,
                }
            rows.append(
                {
                    **metrics,
                    "frame": frame,
                    "upgraded": upgraded,
                    "selected_roi": selected_roi,
                    "amortized_model_compute_ms": small_amortized + (local_amortized if upgraded else 0.0),
                    "amortized_compute_ms": small_amortized + (local_amortized if upgraded else 0.0),
                    "model_input_tensor_ratio_to_small_global": 1.0 + ((roi_imgsz * roi_imgsz) / (global_imgsz * global_imgsz) if upgraded else 0.0),
                    "predictions": [{"frame": p.frame, "box": asdict(p.box), "confidence": p.confidence, "source": p.source} for p in predictions[frame]],
                }
            )
        model_compute_times = [row["amortized_model_compute_ms"] for row in rows]
        numeric = ("precision", "recall", "f1", "ap50", "count_mae", "small_object_recall", "model_input_tensor_ratio_to_small_global")
        summary = {key: float(np.mean([row[key] for row in rows])) for key in numeric}
        summary.update(
            {
                "budget_fraction": len(selected) / len(frame_ids),
                "selected_frames": len(selected),
                "model_compute_p50_ms": float(np.percentile(model_compute_times, 50)),
                "model_compute_p95_ms": float(np.percentile(model_compute_times, 95)),
                "latency_p50_ms": float(np.percentile(model_compute_times, 50)),
                "latency_p95_ms": float(np.percentile(model_compute_times, 95)),
                "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END",
                "small_global_total_ms": global_times["yolo11s_global"],
                "local_batch_total_ms": local_ms,
                "total_compute_ms": global_times["yolo11s_global"] + local_ms,
                "throughput_fps": 1000.0 * len(frame_ids) / max(1e-9, global_times["yolo11s_global"] + local_ms),
            }
        )
        conditions[name] = summary
        receipts[name] = rows
        conditions[name]["local_batches"] = local_batches

    record_condition("yolo11s_global", global_outputs["yolo11s_global"], set(), {}, {}, 0.0, [])
    large_rows = global_outputs["yolo11m_global"]
    large_metrics = []
    for frame in frame_ids:
        metrics = compute_metrics(large_rows[frame], by_truth.get(frame, ()))
        large_metrics.append({**metrics, "frame": frame, "upgraded": True, "selected_roi": None, "amortized_model_compute_ms": global_times["yolo11m_global"] / len(frame_ids), "amortized_compute_ms": global_times["yolo11m_global"] / len(frame_ids), "model_input_tensor_ratio_to_small_global": 1.0, "predictions": [{"frame": p.frame, "box": asdict(p.box), "confidence": p.confidence, "source": p.source} for p in large_rows[frame]]})
    numeric = ("precision", "recall", "f1", "ap50", "count_mae", "small_object_recall", "model_input_tensor_ratio_to_small_global")
    conditions["yolo11m_global"] = {key: float(np.mean([row[key] for row in large_metrics])) for key in numeric}
    conditions["yolo11m_global"].update({"budget_fraction": 1.0, "selected_frames": len(frame_ids), "model_compute_p50_ms": global_times["yolo11m_global"] / len(frame_ids), "model_compute_p95_ms": global_times["yolo11m_global"] / len(frame_ids), "latency_p50_ms": global_times["yolo11m_global"] / len(frame_ids), "latency_p95_ms": global_times["yolo11m_global"] / len(frame_ids), "timing_scope": "MODEL_CALLS_ONLY_NOT_END_TO_END", "local_batch_total_ms": 0.0, "total_compute_ms": global_times["yolo11m_global"], "throughput_fps": 1000.0 * len(frame_ids) / global_times["yolo11m_global"], "local_batches": global_batches["yolo11m_global"]})
    receipts["yolo11m_global"] = large_metrics

    for route_name, scores, roi_map in (("heuristic", heuristic_scores, heuristic_roi), ("oracle", oracle_scores, oracle_roi)):
        for budget in budgets:
            selected_ids = deterministic_budget(frame_ids, budget, scores, seed=seed, namespace=f"{sequence_id}:{route_name}:{budget}")
            crops, crop_ids, crop_bounds = [], [], {}
            for frame in selected_ids:
                crop, bounds = _crop(frames[frame], roi_map[frame])
                crops.append(crop); crop_ids.append(frame); crop_bounds[frame] = bounds
            local_predictions, local_ms, local_batches = _batched_predict(local, crops, crop_ids, batch_size)
            merged = dict(global_outputs["yolo11s_global"])
            for frame in selected_ids:
                merged[frame] = merge_local_predictions(frame, merged[frame], local_predictions[frame], roi_map[frame], crop_bounds[frame])
            suffix = int(round(budget * 100))
            record_condition(f"{route_name}_budget_{suffix:02d}", merged, set(selected_ids), roi_map, crop_bounds, local_ms, local_batches)

    return {
        "schema_version": "fieldpack.budgeted-cascade.v1",
        "phase": "budgeted_screen",
        "sequence_id": sequence_id,
        "frames": len(frame_ids),
        "seed": seed,
        "budgets": list(budgets),
        "batch_size": batch_size,
        "global_imgsz": global_imgsz,
        "roi_imgsz": roi_imgsz,
        "context_expansion": context_expansion,
        "max_expanded_area_ratio": 0.15,
        "condition_order_policy": "deterministic_batch_rotation",
        "route_comparability": {
            "heuristic": "LABEL_BLIND_DEPLOYABLE_SCREEN",
            "oracle": "GT_SELECTED_UPPER_BOUND_NOT_PAIRED_WITH_HEURISTIC",
        },
        "conditions": conditions,
        "receipts": receipts,
        "global_batches": global_batches,
    }


def _load_sample(frames_dir: Path, gt: Sequence[GroundTruth], sample_size: int, seed: int) -> tuple[dict[int, np.ndarray], tuple[GroundTruth, ...]]:
    available = [int(path.stem) for path in frames_dir.glob("*.jpg") if path.stem.isdigit()]
    selected = set(select_frames(available, sample_size, seed=seed))
    frames = {frame: np.asarray(Image.open(frames_dir / f"{frame:06d}.jpg").convert("RGB")) for frame in sorted(selected)}
    return frames, tuple(item for item in gt if item.frame in selected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sparse micro-batched VIRAT cascade")
    parser.add_argument("--frames-dir", type=Path, required=True); parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--small-model", required=True); parser.add_argument("--local-model", required=True)
    parser.add_argument("--sequence-id", required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20); parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--budgets", default="0.05,0.10,0.20"); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0"); parser.add_argument("--global-imgsz", type=int, default=640); parser.add_argument("--roi-imgsz", type=int, default=384)
    parser.add_argument("--context-expansion", type=float, default=0.15)
    args = parser.parse_args()
    budgets = tuple(float(value) for value in args.budgets.split(","))
    truth = parse_mot17_gt(args.gt); frames, truth = _load_sample(args.frames_dir, truth, args.sample_size, args.seed)
    small = UltralyticsYOLOAdapter(args.small_model, device=args.device, name="yolo11s", imgsz=args.global_imgsz)
    large = UltralyticsYOLOAdapter(args.local_model, device=args.device, name="yolo11m_global", imgsz=args.global_imgsz)
    local = UltralyticsYOLOAdapter(args.local_model, device=args.device, name="yolo11m_roi", imgsz=args.roi_imgsz)
    if frames:
        first = frames[min(frames)]; small.predict(first, frame_index=-1); large.predict(first, frame_index=-1); local.predict(first, frame_index=-1)
    result = run_budgeted_phase(frames, truth, small, large, local, budgets=budgets, batch_size=args.batch_size, seed=args.seed, sequence_id=args.sequence_id, global_imgsz=args.global_imgsz, roi_imgsz=args.roi_imgsz, context_expansion=args.context_expansion)
    write_experiment_receipt(args.output, result)
    print(json.dumps({"output": str(args.output), "frames": len(frames), "conditions": len(result["conditions"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

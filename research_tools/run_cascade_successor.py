"""Bounded orchestration for the benefit-aware YOLO11s/YOLO11m cascade."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evidencegate.cascade import BenefitRouter, ROI, calibration_metrics
from .cascade_experiment import (
    Box, GroundTruth, Prediction, UltralyticsYOLOAdapter, candidate_rois,
    compute_metrics, immutable_json_write, load_router, parse_mot17_gt,
    router_features, run_cascade_frame, save_router, select_frames,
)

CONDITIONS = ("yolo11s_global", "yolo11m_global", "heuristic_nms", "benefit_nms", "benefit_wbf")
SCREEN_CONDITIONS = ("yolo11s_global", "yolo11m_global", "heuristic_nms", "random_nms", "oracle_nms")


class _FixedDetector:
    """Return already measured global predictions without another model call."""
    def __init__(self, predictions: Sequence[Prediction], name: str = "cached") -> None:
        self.predictions, self.name = tuple(predictions), name

    def predict(self, frame: Any, *, frame_index: int) -> tuple[Prediction, ...]:
        return self.predictions


def _frame_rois(frame: Any) -> tuple[ROI, ...]:
    height, width = frame.shape[:2]
    return candidate_rois(width, height, cell_width=max(64, width // 2), cell_height=max(64, height // 2))


def _inside_roi(predictions: Sequence[Prediction], roi: ROI) -> tuple[Prediction, ...]:
    return tuple(p for p in predictions if roi.x1 <= (p.box.x1 + p.box.x2) / 2 <= roi.x2 and roi.y1 <= (p.box.y1 + p.box.y2) / 2 <= roi.y2)


def _truth_inside(truth: Sequence[GroundTruth], roi: ROI) -> tuple[GroundTruth, ...]:
    return tuple(g for g in truth if roi.x1 <= (g.box.x1 + g.box.x2) / 2 <= roi.x2 and roi.y1 <= (g.box.y1 + g.box.y2) / 2 <= roi.y2)


def _local_predictions(frame: np.ndarray, frame_index: int, roi: ROI, detector: Any) -> tuple[Prediction, ...]:
    height, width = frame.shape[:2]
    ex = roi.expanded
    x1, y1 = max(0, int(ex[0])), max(0, int(ex[1]))
    x2, y2 = min(width, int(ex[2])), min(height, int(ex[3]))
    if x2 <= x1 or y2 <= y1:
        return ()
    result = []
    for pred in detector.predict(frame[y1:y2, x1:x2], frame_index=frame_index):
        mapped = Prediction(frame_index, Box(pred.box.x1 + x1, pred.box.y1 + y1, pred.box.x2 + x1, pred.box.y2 + y1), pred.confidence, pred.source)
        if roi.x1 <= (mapped.box.x1 + mapped.box.x2) / 2 <= roi.x2 and roi.y1 <= (mapped.box.y1 + mapped.box.y2) / 2 <= roi.y2:
            result.append(mapped)
    return tuple(result)


def _roi_examples(frames: Mapping[int, np.ndarray], ground_truth: Sequence[GroundTruth], small: Any, local: Any, *, delta: float) -> tuple[list[tuple[float, ...]], list[float], list[dict[str, Any]]]:
    by_frame: dict[int, list[GroundTruth]] = {}
    for gt in ground_truth:
        by_frame.setdefault(gt.frame, []).append(gt)
    features: list[tuple[float, ...]] = []
    targets: list[float] = []
    receipts: list[dict[str, Any]] = []
    for frame_index, frame in sorted(frames.items()):
        global_predictions = tuple(small.predict(frame, frame_index=frame_index))
        for roi in _frame_rois(frame):
            truth = _truth_inside(by_frame.get(frame_index, ()), roi)
            base = compute_metrics(_inside_roi(global_predictions, roi), truth)
            strong = compute_metrics(_local_predictions(frame, frame_index, roi, local), truth)
            benefit = float(strong["ap50"] - base["ap50"])
            target = float(benefit > delta)
            features.append(router_features(global_predictions, roi))
            targets.append(target)
            receipts.append({"frame": frame_index, "roi": asdict(roi), "base_ap50": base["ap50"], "local_ap50": strong["ap50"], "benefit": benefit, "target": bool(target)})
    return features, targets, receipts


def fit_phase(frames: Mapping[int, np.ndarray], ground_truth: Sequence[GroundTruth], small: Any, local: Any, *, router_path: str | Path, delta: float = 0.0, sequence_id: str = "MOT17-02") -> dict[str, Any]:
    features, targets, receipts = _roi_examples(frames, ground_truth, small, local, delta=delta)
    router = BenefitRouter().fit(features, targets)
    digest = save_router(router_path, router)
    return {"phase": "fit", "sequence_partition": "fit", "sequence_id": sequence_id, "frames": len(frames), "roi_samples": len(targets), "positive_labels": int(sum(targets)), "delta": delta, "router_sha256": digest, "roi_receipts": receipts}


def calibration_phase(frames: Mapping[int, np.ndarray], ground_truth: Sequence[GroundTruth], small: Any, local: Any, *, router_path: str | Path, calibrated_router_path: str | Path, delta: float = 0.0, sequence_id: str = "MOT17-09") -> dict[str, Any]:
    features, targets, receipts = _roi_examples(frames, ground_truth, small, local, delta=delta)
    router = load_router(router_path)
    before_probabilities = router.predict(features)
    before = calibration_metrics(before_probabilities, targets)
    router.calibrate(features, targets)
    after_probabilities = router.predict(features)
    after = calibration_metrics(after_probabilities, targets)
    router.decision_threshold = _select_threshold(after_probabilities, targets)
    digest = save_router(calibrated_router_path, router)
    return {"phase": "calibrate", "sequence_partition": "calibrate", "sequence_id": sequence_id, "frames": len(frames), "roi_samples": len(targets), "positive_labels": int(sum(targets)), "router_sha256": digest, "before": before, "after": after, "decision_threshold": router.decision_threshold, "roi_receipts": receipts}


def _select_threshold(probabilities: Sequence[float], targets: Sequence[float]) -> float:
    """Select an F1-optimal gate on calibration data; prefer stricter ties."""
    if len(probabilities) != len(targets) or not targets:
        raise ValueError("threshold_calibration_data_invalid")
    candidates = sorted({float(p) for p in probabilities} | {1.0})
    best = (-1.0, 0.5)
    for threshold in candidates:
        predicted = [p >= threshold for p in probabilities]
        tp = sum(bool(y) and p for p, y in zip(predicted, targets))
        fp = sum(not bool(y) and p for p, y in zip(predicted, targets))
        fn = sum(bool(y) and not p for p, y in zip(predicted, targets))
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        best = max(best, (f1, threshold))
    return float(best[1])


def _heuristic_rois(global_predictions: Sequence[Prediction], rois: Sequence[ROI], max_rois: int) -> tuple[ROI, ...]:
    # Fixed, label-free ambiguity score: low confidence + small/boundary fractions.
    ranked = []
    for roi in rois:
        density, mean_conf, small_fraction, boundary_fraction = router_features(global_predictions, roi)
        score = (1.0 - mean_conf) + small_fraction + boundary_fraction + min(1.0, density * roi.pixel_area)
        ranked.append((score, roi.roi_id, roi))
    return tuple(item[2] for item in sorted(ranked, key=lambda x: (-x[0], x[1]))[:max_rois])


def _random_rois(rois: Sequence[ROI], max_rois: int, *, seed: int, sequence_id: str, frame_index: int) -> tuple[ROI, ...]:
    ranked = sorted(
        rois,
        key=lambda roi: hashlib.sha256(
            f"{seed}|{sequence_id}|{frame_index}|{roi.roi_id}".encode("utf-8")
        ).hexdigest(),
    )
    return tuple(ranked[:max_rois])


def _oracle_rois(global_predictions: Sequence[Prediction], truth: Sequence[GroundTruth], rois: Sequence[ROI], max_rois: int) -> tuple[ROI, ...]:
    missed = [item for item in truth if not any(item.box.iou(pred.box) >= 0.5 for pred in global_predictions)]
    ranked = []
    for roi in rois:
        inside = [item for item in missed if roi.x1 <= (item.box.x1 + item.box.x2) / 2 <= roi.x2 and roi.y1 <= (item.box.y1 + item.box.y2) / 2 <= roi.y2]
        small = sum(item.box.area <= 32 * 32 for item in inside)
        ranked.append((len(inside), small, roi.roi_id, roi))
    return tuple(item[3] for item in sorted(ranked, key=lambda item: (-item[0], -item[1], item[2]))[:max_rois])


def _prediction_payload(predictions: Sequence[Prediction]) -> list[dict[str, Any]]:
    return [{"frame": p.frame, "box": asdict(p.box), "confidence": p.confidence, "source": p.source} for p in predictions]


def evaluate_phase(frames: Mapping[int, np.ndarray], ground_truth: Sequence[GroundTruth], small: Any, local: Any, *, router: BenefitRouter, threshold: float | None = None, max_rois: int = 2, sequence_id: str = "evaluation") -> dict[str, Any]:
    threshold = float(router.decision_threshold if threshold is None else threshold)
    by_frame: dict[int, list[GroundTruth]] = {}
    for gt in ground_truth:
        by_frame.setdefault(gt.frame, []).append(gt)
    cells: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for frame_index, frame in sorted(frames.items()):
        started = time.perf_counter()
        global_predictions = tuple(small.predict(frame, frame_index=frame_index))
        small_ms = (time.perf_counter() - started) * 1000.0
        rois = _frame_rois(frame)
        probabilities = tuple(float(v) for v in router.predict([router_features(global_predictions, roi) for roi in rois]))
        benefit_rois = tuple(roi for _, roi in sorted(((score, roi) for score, roi in zip(probabilities, rois) if score >= threshold), key=lambda x: (-x[0], x[1].roi_id))[:max_rois])
        heuristic_rois = _heuristic_rois(global_predictions, rois, max_rois)
        fixed = _FixedDetector(global_predictions, "yolo11s")
        for condition in CONDITIONS:
            if condition == "yolo11s_global":
                predictions, engineering, latency_ms, selected = global_predictions, {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}, small_ms, ()
            elif condition == "yolo11m_global":
                started = time.perf_counter(); predictions = tuple(local.predict(frame, frame_index=frame_index)); latency_ms = (time.perf_counter() - started) * 1000.0
                engineering, selected = {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}, ()
            else:
                selected = heuristic_rois if condition == "heuristic_nms" else benefit_rois
                started = time.perf_counter(); predictions, engineering = run_cascade_frame(frame, frame_index, fixed, local, selected_rois=selected, condition=condition); cascade_ms = (time.perf_counter() - started) * 1000.0
                latency_ms = small_ms + cascade_ms
            metrics = compute_metrics(predictions, by_frame.get(frame_index, ()))
            cells[condition].append({**metrics, **engineering, "frame": frame_index, "latency_ms": latency_ms, "selected_roi_ids": [r.roi_id for r in selected], "router_scores": [{"roi_id": r.roi_id, "probability": p} for r, p in zip(rois, probabilities)], "predictions": _prediction_payload(predictions)})
    summaries: dict[str, dict[str, float]] = {}
    numeric_keys = ("precision", "recall", "f1", "ap50", "count_mae", "small_object_recall", "processed_pixel_ratio", "extra_roi_pixel_ratio", "roi_count", "latency_ms")
    for condition, rows in cells.items():
        summaries[condition] = {key: float(sum(float(row[key]) for row in rows) / len(rows)) if rows else 0.0 for key in numeric_keys}
    return {"phase": "evaluate", "sequence_partition": "evaluate", "sequence_id": sequence_id, "frames": len(frames), "threshold": threshold, "max_rois": max_rois, "conditions": summaries, "receipts": cells}


def screen_phase(
    frames: Mapping[int, np.ndarray],
    ground_truth: Sequence[GroundTruth],
    small: Any,
    large_global: Any,
    local: Any,
    *,
    max_rois: int = 2,
    sequence_id: str = "screen",
    seed: int = 42,
    global_imgsz: int = 640,
    roi_imgsz: int = 512,
) -> dict[str, Any]:
    """Run the label-blind controls and a GT missed-person ROI upper bound."""
    by_frame: dict[int, list[GroundTruth]] = {}
    for gt in ground_truth:
        by_frame.setdefault(gt.frame, []).append(gt)
    cells: dict[str, list[dict[str, Any]]] = {condition: [] for condition in SCREEN_CONDITIONS}
    for frame_index, frame in sorted(frames.items()):
        started = time.perf_counter()
        global_predictions = tuple(small.predict(frame, frame_index=frame_index))
        small_ms = (time.perf_counter() - started) * 1000.0
        rois = _frame_rois(frame)
        truth = by_frame.get(frame_index, ())
        selections = {
            "heuristic_nms": _heuristic_rois(global_predictions, rois, max_rois),
            "random_nms": _random_rois(rois, max_rois, seed=seed, sequence_id=sequence_id, frame_index=frame_index),
            "oracle_nms": _oracle_rois(global_predictions, truth, rois, max_rois),
        }
        fixed = _FixedDetector(global_predictions, "yolo11s")
        height, width = frame.shape[:2]
        source_pixels = max(1, height * width)
        for condition in SCREEN_CONDITIONS:
            if condition == "yolo11s_global":
                predictions = global_predictions
                engineering = {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}
                latency_ms, selected = small_ms, ()
                tensor_pixels = global_imgsz * global_imgsz
            elif condition == "yolo11m_global":
                started = time.perf_counter()
                predictions = tuple(large_global.predict(frame, frame_index=frame_index))
                latency_ms = (time.perf_counter() - started) * 1000.0
                engineering, selected = {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}, ()
                tensor_pixels = global_imgsz * global_imgsz
            else:
                selected = selections[condition]
                started = time.perf_counter()
                predictions, engineering = run_cascade_frame(
                    frame, frame_index, fixed, local, selected_rois=selected, condition=condition
                )
                latency_ms = small_ms + (time.perf_counter() - started) * 1000.0
                tensor_pixels = global_imgsz * global_imgsz + len(selected) * roi_imgsz * roi_imgsz
            metrics = compute_metrics(predictions, truth)
            cells[condition].append(
                {
                    **metrics,
                    **engineering,
                    "frame": frame_index,
                    "latency_ms": latency_ms,
                    "model_input_pixels": tensor_pixels,
                    "model_input_pixel_ratio": tensor_pixels / source_pixels,
                    "model_input_tensor_ratio_to_small_global": tensor_pixels / (global_imgsz * global_imgsz),
                    "selected_roi_ids": [roi.roi_id for roi in selected],
                    "predictions": _prediction_payload(predictions),
                }
            )
    summaries: dict[str, dict[str, float]] = {}
    numeric_keys = ("precision", "recall", "f1", "ap50", "count_mae", "small_object_recall", "processed_pixel_ratio", "extra_roi_pixel_ratio", "roi_count", "model_input_pixels", "model_input_pixel_ratio", "model_input_tensor_ratio_to_small_global")
    for condition, rows in cells.items():
        summary = {key: float(sum(float(row[key]) for row in rows) / len(rows)) if rows else 0.0 for key in numeric_keys}
        latencies = [float(row["latency_ms"]) for row in rows]
        summary.update(
            {
                "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
                "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
                "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            }
        )
        summaries[condition] = summary
    return {"phase": "screen", "sequence_partition": "subset_pilot", "sequence_id": sequence_id, "frames": len(frames), "seed": seed, "max_rois": max_rois, "global_imgsz": global_imgsz, "roi_imgsz": roi_imgsz, "conditions": summaries, "receipts": cells}


def write_experiment_receipt(path: str | Path, payload: Mapping[str, Any]) -> str:
    return immutable_json_write(path, {"schema_version": "cascade-experiment.run.v2", **payload})


def _load_sample(frames_dir: Path, ground_truth: Sequence[GroundTruth], sample_size: int, seed: int) -> tuple[dict[int, np.ndarray], tuple[GroundTruth, ...]]:
    from PIL import Image
    available = [int(p.stem) for p in frames_dir.glob("*.jpg") if p.stem.isdigit()]
    selected = set(select_frames(available, sample_size, seed=seed))
    frames = {index: np.asarray(Image.open(frames_dir / f"{index:06d}.jpg").convert("RGB")) for index in sorted(selected)}
    return frames, tuple(gt for gt in ground_truth if gt.frame in selected)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded benefit-aware cascade phase")
    parser.add_argument("--phase", choices=("fit", "calibrate", "evaluate", "screen"), required=True)
    parser.add_argument("--frames-dir", type=Path, required=True); parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--small-model", required=True); parser.add_argument("--local-model", required=True)
    parser.add_argument("--router", type=Path); parser.add_argument("--calibrated-router", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--sample-size", type=int, default=2); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0"); parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--threshold", type=float); parser.add_argument("--max-rois", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--global-imgsz", type=int, default=640)
    parser.add_argument("--roi-imgsz", type=int, default=512)
    args = parser.parse_args(argv)
    if args.sample_size < 1 or args.max_rois < 1:
        parser.error("sample-size and max-rois must be positive")
    all_truth = parse_mot17_gt(args.gt)
    frames, truth = _load_sample(args.frames_dir, all_truth, args.sample_size, args.seed)
    small = UltralyticsYOLOAdapter(args.small_model, device=args.device, name="yolo11s", imgsz=args.global_imgsz)
    local = UltralyticsYOLOAdapter(args.local_model, device=args.device, name="yolo11m_roi", imgsz=args.roi_imgsz)
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if frames:
        warmup_frame = frames[min(frames)]
        for _ in range(args.warmup):
            small.predict(warmup_frame, frame_index=-1)
            local.predict(warmup_frame, frame_index=-1)
    if args.phase == "screen":
        large_global = UltralyticsYOLOAdapter(args.local_model, device=args.device, name="yolo11m_global", imgsz=args.global_imgsz)
        if frames and args.warmup:
            large_global.predict(frames[min(frames)], frame_index=-1)
        result = screen_phase(frames, truth, small, large_global, local, max_rois=args.max_rois, sequence_id=args.sequence_id, seed=args.seed, global_imgsz=args.global_imgsz, roi_imgsz=args.roi_imgsz)
    elif args.phase == "fit":
        if args.router is None: parser.error("--router is required for fit")
        result = fit_phase(frames, truth, small, local, router_path=args.router, delta=args.delta, sequence_id=args.sequence_id)
    elif args.phase == "calibrate":
        if args.router is None or args.calibrated_router is None: parser.error("--router and --calibrated-router are required for calibrate")
        result = calibration_phase(frames, truth, small, local, router_path=args.router, calibrated_router_path=args.calibrated_router, delta=args.delta, sequence_id=args.sequence_id)
    else:
        if args.router is None: parser.error("--router is required for evaluate")
        active_router = load_router(args.calibrated_router or args.router)
        result = evaluate_phase(frames, truth, small, local, router=active_router, threshold=args.threshold, max_rois=args.max_rois, sequence_id=args.sequence_id)
    digest = write_experiment_receipt(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest, "frames": len(frames)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen YOLO11s vs YOLO26 detect/depth spatial ablation on fixed VIRAT frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, parse_mot17_gt
from research_tools.missed_person_proposal import missed_ground_truth
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA
from research_tools.run_missed_person_proposal_baseline import (
    FrameRecord,
    _camera,
    _contains_center,
    _roi_boxes,
    _union_area_ratio,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_edge_support(depth: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    """Return robust [0,1] depth discontinuity support on the proposal grid."""
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError("invalid_depth_map")
    log_depth = np.log(np.clip(depth.astype(np.float32), 1e-4, None))
    gy, gx = np.gradient(log_depth)
    edge = np.hypot(gx, gy)
    resized = np.asarray(
        Image.fromarray(edge, mode="F").resize((output_width, output_height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    q10, q90 = np.quantile(resized, (.10, .90))
    return np.clip((resized - q10) / max(float(q90 - q10), 1e-6), 0.0, 1.0).astype(np.float32)


def depth_rerank_semantic(semantic: np.ndarray, depth_support: np.ndarray) -> np.ndarray:
    """Fixed semantic-dominant fusion; depth cannot create a candidate alone."""
    if semantic.shape != depth_support.shape or semantic.ndim != 2:
        raise ValueError("semantic_depth_shape_mismatch")
    finite = np.isfinite(semantic)
    if not finite.any():
        raise ValueError("semantic_map_has_no_finite_values")
    low, high = np.quantile(semantic[finite], (.05, .95))
    normalized = np.clip((semantic - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return (normalized * (.75 + .25 * depth_support)).astype(np.float32)


def greedy_counts(truth: Iterable[Box], predictions: Iterable[tuple[Box, float]], iou: float = .50) -> tuple[int, int, int]:
    truth_rows = list(truth)
    matched: set[int] = set()
    tp = 0
    predictions_rows = sorted(predictions, key=lambda row: -row[1])
    for box, _confidence in predictions_rows:
        candidates = [(target.iou(box), index) for index, target in enumerate(truth_rows) if index not in matched]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou:
            matched.add(best_index)
            tp += 1
    return tp, len(predictions_rows) - tp, len(truth_rows) - tp


def _predictions(result: Any) -> tuple[tuple[Box, float], ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    return tuple(
        (Box(*map(float, coords)), float(confidence))
        for coords, confidence, cls in zip(
            boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()
        ) if int(cls) == 0
    )


def ultralytics_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert decoded PIL RGB to the BGR contract Ultralytics assigns to NumPy inputs."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected_rgb_hwc3")
    return np.ascontiguousarray(rgb[..., ::-1])


def _timed_predict(model: Any, images: list[np.ndarray], *, device: str, imgsz: int, end2end: bool | None = None) -> tuple[list[Any], float]:
    import torch

    kwargs: dict[str, Any] = {"source": images, "device": device, "imgsz": imgsz, "batch": len(images), "verbose": False, "conf": .25}
    if end2end is not None:
        kwargs["end2end"] = end2end
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    results = list(model.predict(**kwargs))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return results, (time.perf_counter() - start) * 1000.0 / len(images)


def _load_fixed(cache_root: Path, pilot_root: Path) -> tuple[list[FrameRecord], np.ndarray, dict[int, dict[str, Any]]]:
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("numpy_input_color") != "BGR_CONVERTED_FROM_PIL_RGB":
        raise ValueError("observation_cache_violates_ultralytics_numpy_color_contract")
    records: list[FrameRecord] = []
    static_rows: list[np.ndarray] = []
    image_rows: dict[int, dict[str, Any]] = {}
    cameras: set[str] = set()
    for sequence_row in manifest["sequences"]:
        sequence = sequence_row["sequence_id"]
        camera = _camera(sequence)
        if camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        cameras.add(camera)
        arrays = np.load(cache_root / sequence_row["cache"])
        receipt = json.loads((cache_root / sequence_row["receipt"]).read_text(encoding="utf-8"))
        receipt_by_frame = {int(row["frame"]): row for row in receipt["receipts"]["yolo11s_global"]}
        sequence_root = pilot_root / "sequences" / sequence
        truth_by_frame: dict[int, list[Box]] = defaultdict(list)
        for item in parse_mot17_gt(sequence_root / "gt.txt"):
            truth_by_frame[item.frame].append(item.box)
        normalized = arrays["person_robust_normalized"]
        occupancy = arrays["detection_occupancy"][:, 0]
        for offset, frame in enumerate(arrays["frame_ids"].astype(int)):
            path = sequence_root / "img1" / f"{frame:06d}.jpg"
            image = np.asarray(Image.open(path).convert("RGB"))
            old = tuple(
                (Box(**row["box"]), float(row["confidence"])) for row in receipt_by_frame[frame]["predictions"]
            )
            index = len(records)
            records.append(FrameRecord(sequence, camera, frame, image.shape[1], image.shape[0], old, tuple(truth_by_frame[frame]), image, image))
            score = np.max(normalized[offset], axis=0)
            static_rows.append(np.where(occupancy[offset] < .5, score, float(score.min()) - 1.0))
            image_rows[index] = {"path": str(path)}
    if tuple(sorted(cameras)) != EXPECTED_CAMERAS or len(records) != 256:
        raise ValueError("fixed_pilot_scope_mismatch")
    return records, np.stack(static_rows).astype(np.float32), image_rows


def _detection_summary(records: list[FrameRecord], predictions: dict[str, list[tuple[tuple[Box, float], ...]]], latency: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name, rows in predictions.items():
        cameras: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for record, predicted in zip(records, rows):
            counts = greedy_counts(record.truth, predicted)
            bucket = cameras[record.camera_id]
            for index, value in enumerate(counts):
                bucket[index] += value
        per_camera = {}
        for camera, (tp, fp, fn) in sorted(cameras.items()):
            per_camera[camera] = {
                "tp": tp, "fp": fp, "fn": fn,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
            }
        recalls = [row["recall"] for row in per_camera.values()]
        precisions = [row["precision"] for row in per_camera.values()]
        result[model_name] = {
            "per_camera": per_camera,
            "equal_camera_recall": float(np.mean(recalls)),
            "worst_camera_recall": min(recalls),
            "equal_camera_precision": float(np.mean(precisions)),
            "latency_ms_per_frame": latency[model_name],
        }
    return result


def _proposal_summary(records: list[FrameRecord], static: np.ndarray, depth_maps: list[np.ndarray]) -> dict[str, Any]:
    fused = np.stack([depth_rerank_semantic(score, depth_edge_support(depth, *score.shape)) for score, depth in zip(static, depth_maps)])
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_sequence[record.sequence_id].append(index)
    selected = set()
    for indices in by_sequence.values():
        selected.update(sorted(indices, key=lambda index: (-float(static[index].max()), records[index].frame))[:math.ceil(len(indices) * .20)])
    modes = {"static_semantic": static, "semantic_75_depth_25": fused}
    output: dict[str, Any] = {}
    for mode, scores in modes.items():
        camera_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        areas: list[float] = []
        for index, record in enumerate(records):
            missed = missed_ground_truth(record.truth, tuple(box for box, _ in record.predictions))
            camera_counts[record.camera_id][0] += len(missed)
            if index not in selected:
                continue
            roi = _roi_boxes(scores[index], 1, .095, record.width, record.height)
            areas.append(_union_area_ratio(roi, record.width, record.height))
            camera_counts[record.camera_id][1] += len(missed)
            camera_counts[record.camera_id][2] += sum(any(_contains_center(box, target) for box in roi) for target in missed)
        per_camera = {}
        for camera, (total, selected_missed, covered) in sorted(camera_counts.items()):
            per_camera[camera] = {
                "total_yolo11s_missed": int(total),
                "selected_missed": int(selected_missed),
                "covered": int(covered),
                "joint_proposal_recall": covered / total if total else 0.0,
                "spatial_recall_given_selected": covered / selected_missed if selected_missed else 0.0,
            }
        output[mode] = {
            "per_camera": per_camera,
            "equal_camera_joint_proposal_recall": float(np.mean([row["joint_proposal_recall"] for row in per_camera.values()])),
            "worst_camera_joint_proposal_recall": min(row["joint_proposal_recall"] for row in per_camera.values()),
            "mean_roi_area": float(np.mean(areas)),
            "max_roi_area": max(areas),
        }
    return output


def detection_gate(detection: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate each preregistered YOLO26 head without post-hoc head substitution."""
    baseline = detection["yolo11s"]
    variants: dict[str, Any] = {}
    for name in sorted(key for key in detection if key.startswith("yolo26s_")):
        candidate = detection[name]
        absolute_gain = candidate["equal_camera_recall"] - baseline["equal_camera_recall"]
        relative_gain = (
            candidate["equal_camera_recall"] / baseline["equal_camera_recall"]
            if baseline["equal_camera_recall"] > 0 else None
        )
        checks = {
            "recall_gain": absolute_gain >= .02 or (relative_gain is not None and relative_gain >= 1.10),
            "worst_camera": candidate["worst_camera_recall"] >= baseline["worst_camera_recall"] - .02,
            "precision": candidate["equal_camera_precision"] >= baseline["equal_camera_precision"] - .05,
            "latency": candidate["latency_ms_per_frame"] <= 1.50 * baseline["latency_ms_per_frame"],
        }
        variants[name] = {
            "recommended": all(checks.values()),
            "checks": checks,
            "absolute_recall_gain": absolute_gain,
            "relative_recall_gain": relative_gain,
            "latency_ratio": candidate["latency_ms_per_frame"] / baseline["latency_ms_per_frame"],
        }
    passing = [name for name, row in variants.items() if row["recommended"]]
    selected = max(passing, key=lambda name: detection[name]["equal_camera_recall"], default=None)
    return {"recommended": bool(passing), "selected_variant": selected, "variants": variants}


def depth_gate(proposals: dict[str, dict[str, Any]], depth_latency_ms: float) -> dict[str, Any]:
    """Evaluate the fixed auxiliary and fail closed on an undefined zero baseline."""
    baseline = proposals["static_semantic"]
    candidate = proposals["semantic_75_depth_25"]
    baseline_recall = baseline["equal_camera_joint_proposal_recall"]
    relative_gain = candidate["equal_camera_joint_proposal_recall"] / baseline_recall if baseline_recall > 0 else None
    per_camera_drop = max(
        baseline["per_camera"][camera]["joint_proposal_recall"]
        - candidate["per_camera"][camera]["joint_proposal_recall"]
        for camera in EXPECTED_CAMERAS
    )
    checks = {
        "positive_defined_baseline": baseline_recall > 0,
        "relative_recall_gain": relative_gain is not None and relative_gain >= 1.10,
        "worst_camera_drop": per_camera_drop <= .02,
        "mean_roi_area": candidate["mean_roi_area"] <= .10,
        "max_roi_area": candidate["max_roi_area"] <= .15,
    }
    return {
        "recommended": all(checks.values()),
        "checks": checks,
        "relative_recall_gain": relative_gain,
        "maximum_camera_drop": per_camera_drop,
        "added_latency_ms_per_frame": depth_latency_ms,
    }


def run(cache_root: Path, pilot_root: Path, yolo26_detect: Path, yolo26_depth: Path, yolo11s: Path, *, device: str = "0", batch_size: int = 8) -> dict[str, Any]:
    from ultralytics import YOLO

    records, static, _images = _load_fixed(cache_root, pilot_root)
    models = {
        "yolo11s": YOLO(yolo11s),
        "yolo26s_one_to_one": YOLO(yolo26_detect),
        "yolo26s_one_to_many": YOLO(yolo26_detect),
        "yolo26s_depth": YOLO(yolo26_depth),
    }
    detect_head = models["yolo26s_one_to_many"].model.model[-1]
    one_to_many_available = detect_head.cv2 is not None and detect_head.cv3 is not None
    predictions: dict[str, list[tuple[tuple[Box, float], ...]]] = {"yolo11s": [], "yolo26s_one_to_one": []}
    if one_to_many_available:
        predictions["yolo26s_one_to_many"] = []
    depth_maps: list[np.ndarray] = []
    latency_total = defaultdict(float)
    depth_has_boxes = False

    # Symmetric unmeasured warm-up prevents the shared YOLO26 detector's
    # one-to-many branch from receiving an unfair steady-state latency benefit.
    warmup_images = [ultralytics_bgr(row.rgb) for row in records[:batch_size]]
    _timed_predict(models["yolo11s"], warmup_images, device=device, imgsz=640)
    _timed_predict(models["yolo26s_one_to_one"], warmup_images, device=device, imgsz=640, end2end=True)
    if one_to_many_available:
        _timed_predict(models["yolo26s_one_to_many"], warmup_images, device=device, imgsz=640, end2end=False)
    _timed_predict(models["yolo26s_depth"], warmup_images, device=device, imgsz=768)

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images = [ultralytics_bgr(row.rgb) for row in batch]
        old_results, old_latency = _timed_predict(models["yolo11s"], images, device=device, imgsz=640)
        e2e_results, e2e_latency = _timed_predict(models["yolo26s_one_to_one"], images, device=device, imgsz=640, end2end=True)
        o2m_results, o2m_latency = (None, None)
        if one_to_many_available:
            o2m_results, o2m_latency = _timed_predict(models["yolo26s_one_to_many"], images, device=device, imgsz=640, end2end=False)
        depth_results, depth_latency = _timed_predict(models["yolo26s_depth"], images, device=device, imgsz=768)
        timing_rows = [("yolo11s", old_latency), ("yolo26s_one_to_one", e2e_latency), ("yolo26s_depth", depth_latency)]
        if o2m_latency is not None:
            timing_rows.append(("yolo26s_one_to_many", o2m_latency))
        for key, value in timing_rows:
            latency_total[key] += value * len(images)
        predictions["yolo11s"].extend(_predictions(result) for result in old_results)
        predictions["yolo26s_one_to_one"].extend(_predictions(result) for result in e2e_results)
        if o2m_results is not None:
            predictions["yolo26s_one_to_many"].extend(_predictions(result) for result in o2m_results)
        for result in depth_results:
            depth_has_boxes = depth_has_boxes or getattr(result, "boxes", None) is not None
            depth_maps.append(result.depth.data.detach().cpu().numpy().astype(np.float32))
    latency = {key: value / len(records) for key, value in latency_total.items()}
    detection = _detection_summary(records, predictions, latency)
    proposals = _proposal_summary(records, static, depth_maps)
    detect_gate_result = detection_gate(detection)
    depth_gate_result = depth_gate(proposals, latency["yolo26s_depth"])
    return {
        "schema_version": "fieldpack.yolo26-depth-spatial-ablation.v1",
        "evidence_status": "BOUNDED_256_FRAME_YOLO26_DETECT_DEPTH_ABLATION",
        "comparison_complete": True,
        "development_camera_count": 4,
        "frame_count": 256,
        "configuration_search": False,
        "camera_split_leakage": False,
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
        "checkpoint_hashes_recorded": True,
        "numpy_input_color": "BGR_CONVERTED_FROM_PIL_RGB",
        "checkpoint_sha256": {"yolo11s": _sha256(yolo11s), "yolo26s": _sha256(yolo26_detect), "yolo26s_depth": _sha256(yolo26_depth)},
        "latency_reported": True,
        "one_to_many_available": one_to_many_available,
        "one_to_many_unavailable_reason": None if one_to_many_available else "OFFICIAL_DEPLOY_CHECKPOINT_CV2_CV3_ABSENT",
        "latency_ms_per_frame": latency,
        "yolo26_depth_has_boxes": depth_has_boxes,
        "fixed_protocol": {"confidence": .25, "detection_iou": .50, "detect_imgsz": 640, "depth_imgsz": 768, "frame_budget": .20, "roi_area": .095, "fusion": "semantic_normalized*(0.75+0.25*depth_edge_support)", "unmeasured_warmup_batches_per_mode": 1, "numpy_input_color": "BGR_CONVERTED_FROM_PIL_RGB"},
        "detection": detection,
        "proposal": proposals,
        "detect_gate": detect_gate_result,
        "depth_gate": depth_gate_result,
        "detect_replacement_recommended": detect_gate_result["recommended"],
        "depth_aux_recommended": depth_gate_result["recommended"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26-detect", type=Path, required=True)
    parser.add_argument("--yolo26-depth", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    payload = run(args.cache_root, args.pilot_root, args.yolo26_detect, args.yolo26_depth, args.yolo11s, device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"detect_replacement_recommended": payload["detect_replacement_recommended"], "depth_aux_recommended": payload["depth_aux_recommended"], "yolo26_depth_has_boxes": payload["yolo26_depth_has_boxes"], "latency_ms_per_frame": payload["latency_ms_per_frame"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scene-adaptive YOLO11s + high-resolution YOLO26 + one-gap recovery ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, compute_metrics, parse_mot17_gt
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA
from research_tools.run_missed_person_proposal_baseline import _camera
from research_tools.run_yolo26_depth_spatial_ablation import _sha256, ultralytics_bgr
from research_tools.qwen_scene_route_teacher import load_profile


@dataclass(frozen=True)
class SceneFrame:
    sequence_id: str
    camera_id: str
    frame: int
    width: int
    height: int
    image_path: Path
    global_predictions: tuple[Prediction, ...]
    truth: tuple[GroundTruth, ...]


@dataclass(frozen=True)
class Tile:
    tile_id: str
    core: Box
    crop: Box


def fixed_tiles(width: int, height: int, overlap_px: int = 64) -> tuple[Tile, ...]:
    """Return four stable ownership tiles with expanded inference crops."""
    if width < 2 or height < 2 or overlap_px < 0:
        raise ValueError("invalid_tile_geometry")
    x_mid, y_mid = width // 2, height // 2
    cores = (
        Box(0, 0, x_mid, y_mid), Box(x_mid, 0, width, y_mid),
        Box(0, y_mid, x_mid, height), Box(x_mid, y_mid, width, height),
    )
    output = []
    for index, core in enumerate(cores):
        crop = Box(
            max(0, core.x1 - overlap_px), max(0, core.y1 - overlap_px),
            min(width, core.x2 + overlap_px), min(height, core.y2 + overlap_px),
        )
        output.append(Tile(f"t{index}", core, crop))
    return tuple(output)


def _center(box: Box) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


def _owned(box: Box, core: Box) -> bool:
    x, y = _center(box)
    return core.x1 <= x < core.x2 and core.y1 <= y < core.y2


def nms(rows: Iterable[Prediction], iou: float = .50) -> tuple[Prediction, ...]:
    kept: list[Prediction] = []
    for row in sorted(rows, key=lambda item: (-item.confidence, item.source, item.box.coords)):
        if all(row.box.iou(existing.box) < iou for existing in kept):
            kept.append(row)
    return tuple(kept)


def fit_camera_scale_prior(
    history: Iterable[SceneFrame], *, bands: int = 4, small_height_px: float = 48.0, minimum_samples: int = 4
) -> dict[str, Any]:
    """Fit label-blind per-camera vertical scale tables from historical detections."""
    by_camera: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(bands)])
    for record in history:
        for prediction in record.global_predictions:
            _x, y = _center(prediction.box)
            band = min(bands - 1, max(0, int(y / record.height * bands)))
            by_camera[record.camera_id][band].append(prediction.box.y2 - prediction.box.y1)
    output: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        rows = []
        for index, heights in enumerate(by_camera[camera]):
            median = float(np.median(heights)) if heights else None
            rows.append({
                "band": index,
                "samples": len(heights),
                "median_height_px": median,
                "route_high_resolution": len(heights) < minimum_samples or (median is not None and median <= small_height_px),
            })
        output[camera] = rows
    return output


def selected_tiles(prior: dict[str, Any], camera: str, width: int, height: int) -> tuple[Tile, ...]:
    tiles = fixed_tiles(width, height)
    risky = {row["band"] for row in prior[camera] if row["route_high_resolution"]}
    selected = []
    for tile in tiles:
        first = min(3, int(tile.core.y1 / height * 4))
        last = min(3, int(max(tile.core.y1, tile.core.y2 - 1) / height * 4))
        if any(band in risky for band in range(first, last + 1)):
            selected.append(tile)
    return tuple(selected)


def recover_one_frame_gaps(by_frame: dict[int, tuple[Prediction, ...]], *, link_iou: float = .30) -> dict[int, tuple[Prediction, ...]]:
    """Fill only t-1/t+1 supported gaps, equivalent to a one-frame online delay."""
    output = {frame: tuple(rows) for frame, rows in by_frame.items()}
    frames = sorted(output)
    for frame in frames[1:-1]:
        if frame - 1 not in output or frame + 1 not in output:
            continue
        additions: list[Prediction] = []
        used_next: set[int] = set()
        for previous in output[frame - 1]:
            candidates = [
                (previous.box.iou(future.box), index, future)
                for index, future in enumerate(output[frame + 1]) if index not in used_next
            ]
            overlap, index, future = max(candidates, default=(0.0, -1, None), key=lambda row: (row[0], -row[1]))
            if future is None or overlap < link_iou:
                continue
            interpolated = Box(*((np.asarray(previous.box.coords) + np.asarray(future.box.coords)) / 2.0).tolist())
            if any(interpolated.iou(current.box) >= link_iou for current in output[frame]):
                continue
            used_next.add(index)
            additions.append(Prediction(frame, interpolated, min(previous.confidence, future.confidence) * .8, "track_gap_fill"))
        output[frame] = nms((*output[frame], *additions))
    return output


def _load_records(cache_root: Path, pilot_root: Path) -> tuple[list[SceneFrame], list[SceneFrame], dict[str, Any]]:
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("numpy_input_color") != "BGR_CONVERTED_FROM_PIL_RGB":
        raise ValueError("bgr_corrected_cache_required")
    by_camera: dict[str, list[str]] = defaultdict(list)
    sequence_rows = {row["sequence_id"]: row for row in manifest["sequences"]}
    for sequence in sequence_rows:
        camera = _camera(sequence)
        if camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        by_camera[camera].append(sequence)
    if tuple(sorted(by_camera)) != EXPECTED_CAMERAS or any(len(rows) != 2 for rows in by_camera.values()):
        raise ValueError("expected_two_sequences_per_development_camera")
    split = {
        camera: {"evaluation": sorted(rows)[0], "history": sorted(rows)[1]}
        for camera, rows in sorted(by_camera.items())
    }
    history: list[SceneFrame] = []
    evaluation: list[SceneFrame] = []
    for camera, parts in split.items():
        for partition, sequence in parts.items():
            row = sequence_rows[sequence]
            arrays = np.load(cache_root / row["cache"])
            receipt = json.loads((cache_root / row["receipt"]).read_text(encoding="utf-8"))
            receipt_by_frame = {int(item["frame"]): item for item in receipt["receipts"]["yolo11s_global"]}
            root = pilot_root / "sequences" / sequence
            truth_by_frame: dict[int, list[GroundTruth]] = defaultdict(list)
            if partition == "evaluation":
                for gt in parse_mot17_gt(root / "gt.txt"):
                    truth_by_frame[gt.frame].append(gt)
            target = history if partition == "history" else evaluation
            for frame in arrays["frame_ids"].astype(int):
                image_path = root / "img1" / f"{frame:06d}.jpg"
                with Image.open(image_path) as image:
                    width, height = image.size
                predictions = tuple(
                    Prediction(frame, Box(**item["box"]), float(item["confidence"]), "yolo11s_global")
                    for item in receipt_by_frame[frame]["predictions"]
                )
                target.append(SceneFrame(sequence, camera, frame, width, height, image_path, predictions, tuple(truth_by_frame[frame])))
    if len(history) != 128 or len(evaluation) != 128:
        raise ValueError("fixed_history_evaluation_scope_mismatch")
    return history, evaluation, split


def _tile_predictions(
    model: Any,
    records: list[SceneFrame],
    tiles_by_camera: dict[str, tuple[Tile, ...]],
    *,
    device: str,
    batch_size: int,
    confidence: float = .25,
    rect: bool = True,
) -> tuple[dict[tuple[str, int], tuple[Prediction, ...]], float, int]:
    import torch

    tasks: list[tuple[SceneFrame, Tile, np.ndarray]] = []
    for record in records:
        rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
        for tile in tiles_by_camera[record.camera_id]:
            crop = tile.crop
            array = rgb[int(crop.y1):int(crop.y2), int(crop.x1):int(crop.x2)]
            tasks.append((record, tile, ultralytics_bgr(array)))
    output: dict[tuple[str, int], list[Prediction]] = defaultdict(list)
    elapsed = 0.0
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start:start + batch_size]
        images = [row[2] for row in chunk]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        began = time.perf_counter()
        results = model.predict(
            source=images,
            device=device,
            imgsz=640,
            batch=len(images),
            verbose=False,
            conf=confidence,
            end2end=False,
            rect=rect,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed += (time.perf_counter() - began) * 1000.0
        for (record, tile, _image), result in zip(chunk, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()):
                if int(cls) != 0:
                    continue
                local = Box(*map(float, coords))
                global_box = Box(local.x1 + tile.crop.x1, local.y1 + tile.crop.y1, local.x2 + tile.crop.x1, local.y2 + tile.crop.y1)
                if _owned(global_box, tile.core):
                    output[(record.sequence_id, record.frame)].append(Prediction(record.frame, global_box, float(confidence), f"yolo26_hr_tile:{tile.tile_id}"))
    return {key: tuple(value) for key, value in output.items()}, elapsed, len(tasks)


def _global_latency(model: Any, records: list[SceneFrame], *, device: str, batch_size: int) -> float:
    import torch

    elapsed = 0.0
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        images = [ultralytics_bgr(np.asarray(Image.open(record.image_path).convert("RGB"))) for record in chunk]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        began = time.perf_counter()
        model.predict(source=images, device=device, imgsz=640, batch=len(images), verbose=False, conf=.25)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed += (time.perf_counter() - began) * 1000.0
    return elapsed


def _condition_metrics(records: list[SceneFrame], rows: dict[tuple[str, int], tuple[Prediction, ...]]) -> dict[str, Any]:
    per_camera: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        selected = [record for record in records if record.camera_id == camera]
        predictions = [prediction for record in selected for prediction in rows[(record.sequence_id, record.frame)]]
        truth = [gt for record in selected for gt in record.truth]
        per_camera[camera] = compute_metrics(predictions, truth)
    keys = ("precision", "recall", "small_object_recall", "count_mae", "ap50")
    return {
        "per_camera": per_camera,
        **{f"equal_camera_{key}": float(np.mean([row[key] for row in per_camera.values()])) for key in keys},
        "worst_camera_recall": min(row["recall"] for row in per_camera.values()),
    }


def static_two_tile_oracle(records: list[SceneFrame], all_rows: dict[tuple[str, int], tuple[Prediction, ...]]) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, int], tuple[Prediction, ...]]]:
    """Evaluation-only upper bound for whether sparse static routing is viable."""
    from itertools import combinations

    chosen: dict[str, tuple[str, ...]] = {}
    output: dict[tuple[str, int], tuple[Prediction, ...]] = {}
    for camera in EXPECTED_CAMERAS:
        camera_records = [record for record in records if record.camera_id == camera]
        best_score: tuple[float, float, float, int, tuple[str, ...]] | None = None
        best_subset: tuple[str, ...] = ()
        for count in (0, 1, 2):
            for subset in combinations(("t0", "t1", "t2", "t3"), count):
                predictions = []
                for record in camera_records:
                    key = (record.sequence_id, record.frame)
                    selected = tuple(row for row in all_rows.get(key, ()) if row.source.rsplit(":", 1)[-1] in subset)
                    predictions.extend(nms((*record.global_predictions, *selected)))
                metrics = compute_metrics(predictions, [truth for record in camera_records for truth in record.truth])
                score = (metrics["small_object_recall"], metrics["recall"], metrics["precision"], -count, subset)
                if best_score is None or score > best_score:
                    best_score, best_subset = score, subset
        chosen[camera] = best_subset
        for record in camera_records:
            key = (record.sequence_id, record.frame)
            selected = tuple(row for row in all_rows.get(key, ()) if row.source.rsplit(":", 1)[-1] in best_subset)
            output[key] = nms((*record.global_predictions, *selected))
    return chosen, output


def run(cache_root: Path, pilot_root: Path, yolo11s: Path, yolo26: Path, *, device: str = "0", batch_size: int = 8, qwen_profile_path: Path | None = None) -> dict[str, Any]:
    from ultralytics import YOLO

    history, evaluation, split = _load_records(cache_root, pilot_root)
    prior = fit_camera_scale_prior(history)
    all_tiles = {camera: fixed_tiles(1280, 720) for camera in EXPECTED_CAMERAS}
    prior_tiles = {camera: selected_tiles(prior, camera, 1280, 720) for camera in EXPECTED_CAMERAS}
    qwen_profile = load_profile(qwen_profile_path) if qwen_profile_path is not None else None
    qwen_tiles = None if qwen_profile is None else {
        camera: tuple(tile for tile in fixed_tiles(1280, 720) if tile.tile_id in qwen_profile["cameras"][camera]["complex_tiles"])
        for camera in EXPECTED_CAMERAS
    }

    all_model = YOLO(yolo26)
    prior_model = YOLO(yolo26)
    baseline_model = YOLO(yolo11s)
    qwen_model = YOLO(yolo26) if qwen_tiles is not None else None
    # Each one-to-many predictor is built on a separate model instance.
    warmup = ultralytics_bgr(np.asarray(Image.open(evaluation[0].image_path).convert("RGB")))
    all_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=.25, end2end=False)
    prior_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=.25, end2end=False)
    baseline_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=.25)
    if qwen_model is not None:
        qwen_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=.25, end2end=False)
    baseline_ms = _global_latency(baseline_model, evaluation, device=device, batch_size=batch_size)
    all_rows, all_ms, all_calls = _tile_predictions(all_model, evaluation, all_tiles, device=device, batch_size=batch_size)
    prior_rows, prior_ms, prior_calls = _tile_predictions(prior_model, evaluation, prior_tiles, device=device, batch_size=batch_size)
    oracle_tiles, oracle_rows = static_two_tile_oracle(evaluation, all_rows)
    qwen_rows, qwen_ms, qwen_calls = ({}, 0.0, 0)
    if qwen_model is not None and qwen_tiles is not None:
        qwen_rows, qwen_ms, qwen_calls = _tile_predictions(qwen_model, evaluation, qwen_tiles, device=device, batch_size=batch_size)

    baseline: dict[tuple[str, int], tuple[Prediction, ...]] = {}
    full_hr: dict[tuple[str, int], tuple[Prediction, ...]] = {}
    scaled: dict[tuple[str, int], tuple[Prediction, ...]] = {}
    qwen_routed: dict[tuple[str, int], tuple[Prediction, ...]] = {}
    for record in evaluation:
        key = (record.sequence_id, record.frame)
        baseline[key] = record.global_predictions
        full_hr[key] = nms((*record.global_predictions, *all_rows.get(key, ())))
        scaled[key] = nms((*record.global_predictions, *prior_rows.get(key, ())))
        if qwen_profile is not None:
            qwen_routed[key] = nms((*record.global_predictions, *qwen_rows.get(key, ())))
    recovered = dict(scaled)
    for sequence in sorted({record.sequence_id for record in evaluation}):
        subset = {record.frame: scaled[(sequence, record.frame)] for record in evaluation if record.sequence_id == sequence}
        for frame, predictions in recover_one_frame_gaps(subset).items():
            recovered[(sequence, frame)] = predictions

    conditions = {
        "b0_yolo11s_global": _condition_metrics(evaluation, baseline),
        "b1_full_high_resolution_yolo26": _condition_metrics(evaluation, full_hr),
        "b2_camera_scale_routed_yolo26": _condition_metrics(evaluation, scaled),
        "b3_scale_route_plus_gap_recovery": _condition_metrics(evaluation, recovered),
    }
    if qwen_profile is not None:
        conditions["b4_qwen_history_routed_yolo26"] = _condition_metrics(evaluation, qwen_routed)
    b0 = conditions["b0_yolo11s_global"]
    b1 = conditions["b1_full_high_resolution_yolo26"]
    b2 = conditions["b2_camera_scale_routed_yolo26"]
    b3 = conditions["b3_scale_route_plus_gap_recovery"]
    b1_gain = b1["equal_camera_small_object_recall"] - b0["equal_camera_small_object_recall"]
    baseline_per_frame = baseline_ms / len(evaluation)
    full_hr_per_frame = baseline_per_frame + all_ms / len(evaluation)
    routed_per_frame = baseline_per_frame + prior_ms / len(evaluation)
    qwen_per_frame = baseline_per_frame + qwen_ms / len(evaluation)
    retained = (
        (b2["equal_camera_small_object_recall"] - b0["equal_camera_small_object_recall"]) / b1_gain
        if b1_gain > 0 else None
    )
    gates = {
        "high_resolution_branch": {
            "passed": (b1_gain >= .02 or b1["equal_camera_small_object_recall"] >= 1.10 * b0["equal_camera_small_object_recall"])
            and b1["equal_camera_precision"] >= b0["equal_camera_precision"] - .05
            and full_hr_per_frame <= 3.0 * baseline_per_frame,
            "small_recall_absolute_gain": b1_gain,
            "precision_change": b1["equal_camera_precision"] - b0["equal_camera_precision"],
            "latency_ratio": full_hr_per_frame / baseline_per_frame,
        },
        "camera_scale_prior": {
            "passed": retained is not None and retained >= .80 and prior_calls <= .60 * all_calls and routed_per_frame <= 2.0 * baseline_per_frame,
            "retained_high_resolution_gain": retained,
            "tile_call_ratio_to_full_hr": prior_calls / all_calls if all_calls else 0.0,
            "latency_ratio": routed_per_frame / baseline_per_frame,
        },
        "gap_recovery": {
            "passed": (b3["equal_camera_recall"] - b2["equal_camera_recall"] >= .01)
            and b3["equal_camera_precision"] >= b2["equal_camera_precision"] - .02
            and b3["equal_camera_count_mae"] <= b2["equal_camera_count_mae"],
            "recall_change": b3["equal_camera_recall"] - b2["equal_camera_recall"],
            "precision_change": b3["equal_camera_precision"] - b2["equal_camera_precision"],
            "count_mae_change": b3["equal_camera_count_mae"] - b2["equal_camera_count_mae"],
        },
    }
    if qwen_profile is not None:
        b4 = conditions["b4_qwen_history_routed_yolo26"]
        qwen_retained = ((b4["equal_camera_small_object_recall"] - b0["equal_camera_small_object_recall"]) / b1_gain) if b1_gain > 0 else None
        gates["qwen_history_scene_router"] = {
            "passed": qwen_retained is not None and qwen_retained >= .80
            and qwen_calls <= .60 * all_calls and qwen_per_frame <= 2.0 * baseline_per_frame
            and b4["equal_camera_precision"] >= b0["equal_camera_precision"] - .05,
            "retained_high_resolution_gain": qwen_retained,
            "tile_call_ratio_to_full_hr": qwen_calls / all_calls if all_calls else 0.0,
            "latency_ratio": qwen_per_frame / baseline_per_frame,
            "precision_change_vs_baseline": b4["equal_camera_precision"] - b0["equal_camera_precision"],
            "offline_teacher_ms_excluded_from_online_latency": float(qwen_profile["offline_teacher_ms"]),
        }
    return {
        "schema_version": "fieldpack.scene-adaptive-small-track-ablation.v2" if qwen_profile is not None else "fieldpack.scene-adaptive-small-track-ablation.v1",
        "evidence_status": "BOUNDED_FOUR_CAMERA_HISTORY_ONLINE_ABLATION",
        "history_frames": len(history),
        "evaluation_frames": len(evaluation),
        "camera_count": 4,
        "split": split,
        "split_policy": "LEXICOGRAPHIC_SECOND_SEQUENCE_HISTORY_FIRST_SEQUENCE_EVALUATION",
        "history_gt_used_for_configuration": False,
        "evaluation_gt_used_online": False,
        "sealed_camera_used": False,
        "configuration_search": False,
        "qwen_called": qwen_profile is not None,
        "qwen_online_called": False,
        "depth_called": False,
        "checkpoint_sha256": {"yolo11s": _sha256(yolo11s), "yolo26s": _sha256(yolo26)},
        "fixed_protocol": {"global_conf": .25, "tile_conf": .25, "tile_imgsz": 640, "tile_grid": "2x2_core_plus_64px_overlap", "scale_bands": 4, "small_height_px": 48, "minimum_history_samples": 4, "nms_iou": .50, "gap_link_iou": .30},
        "camera_scale_prior": prior,
        "evaluation_only_static_two_tile_oracle": {
            "uses_evaluation_gt": True,
            "deployable": False,
            "selection_policy": "PER_CAMERA_ENUMERATE_ZERO_TO_TWO_TILES_LEXICOGRAPHIC_SMALL_RECALL_RECALL_PRECISION_COST",
            "tiles": oracle_tiles,
            "metrics": _condition_metrics(evaluation, oracle_rows),
            "tile_call_ratio_to_full_hr": sum(len(value) for value in oracle_tiles.values()) / (4 * 4),
        },
        "qwen_profile_sha256": _sha256(qwen_profile_path) if qwen_profile_path is not None else None,
        "qwen_scene_profile": qwen_profile["cameras"] if qwen_profile is not None else None,
        "tile_counts": {"full_high_resolution": all_calls, "camera_scale_routed": prior_calls, **({"qwen_history_routed": qwen_calls} if qwen_profile is not None else {})},
        "tile_inference_ms_total": {"full_high_resolution": all_ms, "camera_scale_routed": prior_ms, **({"qwen_history_routed": qwen_ms} if qwen_profile is not None else {})},
        "latency_ms_per_evaluation_frame": {"yolo11s_global": baseline_per_frame, "full_high_resolution": full_hr_per_frame, "camera_scale_routed": routed_per_frame, **({"qwen_history_routed": qwen_per_frame} if qwen_profile is not None else {})},
        "conditions": conditions,
        "gates": gates,
        "research_outcome": "_".join(name for name, gate in gates.items() if gate["passed"]) or "all_rejected",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--qwen-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    payload = run(args.cache_root, args.pilot_root, args.yolo11s, args.yolo26, device=args.device, batch_size=args.batch_size, qwen_profile_path=args.qwen_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"research_outcome": payload["research_outcome"], "gates": payload["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

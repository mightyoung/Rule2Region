"""Development-only Top-6 exact replay with a sealed 10/22 calibration split.

The module intentionally does not call ``_load_records`` because that helper opens
evaluation GT while constructing records.  This runner freezes the frame split,
candidate trace, complete predictions, latency observations, and operator
fingerprints before it reads any GT file.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, compute_metrics, parse_mot17_gt
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS, SEALED_CAMERA, _camera
from research_tools.run_scene_adaptive_small_track_ablation import _owned, fixed_tiles, nms
from research_tools.run_yolo26_depth_spatial_ablation import ultralytics_bgr
from research_tools.sceneconfig_e2_proxy_validation import _supported_fraction

FAILED_PROXY_UTILITY = -1.0e30
FAILED_METRIC = 1.0e30

@dataclass(frozen=True)
class ScopeFrame:
    sequence_id: str
    camera_id: str
    frame: int
    width: int
    height: int
    image_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def temporal_calibration_frames(frames: Iterable[int], count: int = 10) -> tuple[int, ...]:
    """Choose deterministic temporal anchors without inspecting labels."""
    ordered = tuple(sorted(set(int(frame) for frame in frames)))
    if count < 1 or count >= len(ordered):
        raise ValueError("calibration_count_must_leave_an_evaluation_partition")
    indices = [min(len(ordered) - 1, int((index + .5) * len(ordered) / count)) for index in range(count)]
    selected = tuple(ordered[index] for index in indices)
    if len(set(selected)) != count:
        raise ValueError("calibration_anchor_collision")
    return selected


def select_top_k(proxy: dict[str, Any], k: int = 6) -> dict[str, list[dict[str, Any]]]:
    if proxy.get("holdout_opened") is not False or proxy.get("evaluation_gt_used_for_proxy") is not False:
        raise ValueError("proxy_not_zero_label_safe")
    cards = {row["config_id"]: row for row in proxy["panel"]}
    selected: dict[str, list[dict[str, Any]]] = {}
    for camera in EXPECTED_CAMERAS:
        ranked = sorted(proxy["per_camera"][camera], key=lambda row: (-float(row["proxy_utility"]), row["config_id"]))
        if len(ranked) < k:
            raise ValueError(f"insufficient_proxy_candidates:{camera}")
        selected[camera] = [
            {**cards[row["config_id"]], "approx_proxy_utility": float(row["proxy_utility"]), "proxy_rank": rank}
            for rank, row in enumerate(ranked[:k], 1)
        ]
    return selected


def _load_scope_without_gt(cache_root: Path, pilot_root: Path, proxy_split: dict[str, Any]) -> tuple[list[ScopeFrame], dict[str, Any]]:
    manifest = json.loads((cache_root / "cache-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("numpy_input_color") != "BGR_CONVERTED_FROM_PIL_RGB":
        raise ValueError("bgr_corrected_cache_required")
    sequence_rows = {row["sequence_id"]: row for row in manifest["sequences"]}
    by_camera: dict[str, list[str]] = defaultdict(list)
    for sequence in sequence_rows:
        camera = _camera(sequence)
        if camera == SEALED_CAMERA:
            raise ValueError("sealed_camera_present")
        by_camera[camera].append(sequence)
    split = {camera: {"evaluation": sorted(rows)[0], "history": sorted(rows)[1]} for camera, rows in sorted(by_camera.items())}
    if tuple(split) != EXPECTED_CAMERAS or split != proxy_split:
        raise ValueError("proxy_and_runtime_split_mismatch")
    records: list[ScopeFrame] = []
    for camera in EXPECTED_CAMERAS:
        sequence = split[camera]["evaluation"]
        row = sequence_rows[sequence]
        arrays = np.load(cache_root / row["cache"])
        root = pilot_root / "sequences" / sequence
        for frame in arrays["frame_ids"].astype(int):
            image_path = root / "img1" / f"{frame:06d}.jpg"
            with Image.open(image_path) as image:
                width, height = image.size
            records.append(ScopeFrame(sequence, camera, int(frame), width, height, image_path))
    if len(records) != 128 or any(sum(row.camera_id == camera for row in records) != 32 for camera in EXPECTED_CAMERAS):
        raise ValueError("fixed_32_frame_development_scope_mismatch")
    return records, split


def make_split_manifest(records: list[ScopeFrame], split: dict[str, Any], calibration_count: int) -> dict[str, Any]:
    cameras = {}
    for camera in EXPECTED_CAMERAS:
        frames = [row.frame for row in records if row.camera_id == camera]
        calibration = temporal_calibration_frames(frames, calibration_count)
        evaluation = tuple(frame for frame in sorted(frames) if frame not in set(calibration))
        cameras[camera] = {"sequence_id": split[camera]["evaluation"], "calibration_frames": calibration, "evaluation_frames": evaluation}
    return {
        "schema_version": "sceneconfig.top6-frame-split.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "selection_rule": "deterministic_temporal_midpoint_anchors_without_gt",
        "calibration_frames_per_camera": calibration_count,
        "evaluation_frames_per_camera": 32 - calibration_count,
        "holdout_opened": False,
        "cameras": cameras,
    }


def _serialize_predictions(rows: Iterable[Prediction]) -> list[dict[str, Any]]:
    return [{"frame": row.frame, "box": asdict(row.box), "confidence": row.confidence, "source": row.source} for row in rows]


def _deserialize_predictions(rows: Iterable[dict[str, Any]]) -> tuple[Prediction, ...]:
    return tuple(Prediction(int(row["frame"]), Box(**row["box"]), float(row["confidence"]), row["source"]) for row in rows)


def _predict_global(model: Any, rgb: np.ndarray, frame: int, device: str) -> tuple[Prediction, ...]:
    result = model.predict(source=[ultralytics_bgr(rgb)], device=device, imgsz=640, batch=1, verbose=False, conf=.25)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    return tuple(
        Prediction(frame, Box(*map(float, coords)), float(confidence), "yolo11s_global")
        for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist())
        if int(cls) == 0
    )


def _predict_tiles(model: Any, rgb: np.ndarray, frame: int, card: dict[str, Any], device: str, roi_batch_size: int) -> tuple[Prediction, ...]:
    by_id = {tile.tile_id: tile for tile in fixed_tiles(rgb.shape[1], rgb.shape[0])}
    tiles = [by_id[value] for value in card["tiles"]]
    output: list[Prediction] = []
    crops = [ultralytics_bgr(rgb[int(tile.crop.y1):int(tile.crop.y2), int(tile.crop.x1):int(tile.crop.x2)]) for tile in tiles]
    for start in range(0, len(crops), roi_batch_size):
        real_crops, real_tiles = crops[start:start + roi_batch_size], tiles[start:start + roi_batch_size]
        inference = list(real_crops)
        if inference:
            inference.extend(np.zeros_like(inference[0]) for _ in range(roi_batch_size - len(inference)))
        results = model.predict(source=inference, device=device, imgsz=640, batch=len(inference), verbose=False, conf=.05, end2end=False, rect=False)
        for tile, result in zip(real_tiles, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()):
                if int(cls) != 0 or float(confidence) < float(card["tile_confidence"]):
                    continue
                local = Box(*map(float, coords))
                mapped = Box(local.x1 + tile.crop.x1, local.y1 + tile.crop.y1, local.x2 + tile.crop.x1, local.y2 + tile.crop.y1)
                if _owned(mapped, tile.core):
                    output.append(Prediction(frame, mapped, float(confidence), f"yolo26_hr_tile:{tile.tile_id}"))
    return tuple(output)


def exact_proxy_metrics(frame_rows: list[dict[str, Any]], tile_count: int) -> dict[str, float]:
    raw_added: list[Prediction] = []
    retained_added: list[Prediction] = []
    global_counts: list[int] = []
    merged_counts: list[int] = []
    temporal: dict[int, tuple[Prediction, ...]] = {}
    for row in frame_rows:
        raw = _deserialize_predictions(row["tile_raw"])
        merged = _deserialize_predictions(row["merged"])
        retained = tuple(item for item in merged if item.source.startswith("yolo26_hr_tile:"))
        raw_added.extend(raw)
        retained_added.extend(retained)
        temporal[int(row["frame"])] = retained
        global_counts.append(len(row["global"]))
        merged_counts.append(len(merged))
    support = _supported_fraction(temporal)
    duplicate_rate = 1.0 - len(retained_added) / len(raw_added) if raw_added else 0.0
    small_rate = sum((row.box.y2 - row.box.y1) <= 48 for row in retained_added) / len(retained_added) if retained_added else 0.0
    added_per_frame = len(retained_added) / max(1, len(frame_rows))
    cost_ratio = tile_count / 4.0
    utility = added_per_frame * (1.5 * support + .5 * small_rate - (1.0 - support)) - .5 * duplicate_rate - .15 * cost_ratio
    return {
        "added_per_frame": added_per_frame,
        "count_disagreement": float(np.mean(np.abs(np.asarray(merged_counts) - np.asarray(global_counts)))),
        "small_added_rate": small_rate,
        "temporal_support_rate": support,
        "duplicate_suppression_rate": duplicate_rate,
        "tile_cost_ratio": cost_ratio,
        "exact_proxy_utility": utility,
    }


def _run_operator(
    records: list[ScopeFrame], card: dict[str, Any], yolo11s: Path, yolo26: Path, *, device: str, repeats: int, roi_batch_size: int
) -> tuple[list[dict[str, Any]], list[float], bool, str | None]:
    """Run one complete configuration on fresh model instances."""
    from ultralytics import YOLO
    import torch

    global_model = YOLO(yolo11s)
    expert_model = YOLO(yolo26) if card["tiles"] else None
    warmup = np.asarray(Image.open(records[0].image_path).convert("RGB"))
    _predict_global(global_model, warmup, records[0].frame, device)
    if expert_model is not None:
        _predict_tiles(expert_model, warmup, records[0].frame, {**card, "tiles": [card["tiles"][0]]}, device, roi_batch_size)
    frame_rows: list[dict[str, Any]] = []
    latency: list[float] = []
    try:
        for repeat in range(repeats):
            for record in records:
                rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                began = time.perf_counter()
                global_rows = _predict_global(global_model, rgb, record.frame, device)
                tile_rows = _predict_tiles(expert_model, rgb, record.frame, card, device, roi_batch_size) if expert_model is not None else ()
                merged = nms((*global_rows, *tile_rows))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - began) * 1000.0
                latency.append(elapsed_ms)
                if repeat == 0:
                    frame_rows.append({
                        "sequence_id": record.sequence_id,
                        "camera_id": record.camera_id,
                        "frame": record.frame,
                        "global": _serialize_predictions(global_rows),
                        "tile_raw": _serialize_predictions(tile_rows),
                        "merged": _serialize_predictions(merged),
                    })
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return frame_rows, latency, True, str(exc)
    finally:
        del global_model, expert_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return frame_rows, latency, False, None


def _p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if values else FAILED_METRIC


def _metrics_for_frames(frame_rows: list[dict[str, Any]], truth: list[GroundTruth], frames: tuple[int, ...]) -> dict[str, float]:
    allowed = set(frames)
    predictions = [item for row in frame_rows if int(row["frame"]) in allowed for item in _deserialize_predictions(row["merged"])]
    selected_truth = [row for row in truth if row.frame in allowed]
    metrics = compute_metrics(predictions, selected_truth)
    pred_counts = defaultdict(int)
    gt_counts = defaultdict(int)
    for row in predictions:
        pred_counts[row.frame] += 1
    for row in selected_truth:
        gt_counts[row.frame] += 1
    metrics["count_mae"] = float(np.mean([abs(pred_counts[frame] - gt_counts[frame]) for frame in frames]))
    metrics["frame_count"] = float(len(frames))
    return metrics


def choose_calibrated_candidate(
    candidates: list[dict[str, Any]], baseline_metrics: dict[str, float], *, precision_floor: float = -.05
) -> tuple[str, list[dict[str, Any]]]:
    diagnostics = []
    for row in candidates:
        metrics = row["calibration_metrics"]
        precision_change = metrics["precision"] - baseline_metrics["precision"]
        admissible = not row["oom"] and row["latency_ratio_p95"] <= 3.0 and precision_change >= precision_floor
        diagnostics.append({
            "config_id": row["config_id"],
            "precision_change": precision_change,
            "empirical_precision_noninferiority": precision_change >= precision_floor,
            "latency_admissible": row["latency_ratio_p95"] <= 3.0,
            "oom": row["oom"],
            "admissible": admissible,
            "calibration_count_mae": metrics["count_mae"],
        })
    admissible_rows = [row for row, diag in zip(candidates, diagnostics) if diag["admissible"]]
    if not admissible_rows:
        return "B0", diagnostics
    chosen = min(admissible_rows, key=lambda row: (row["calibration_metrics"]["count_mae"], -row["calibration_metrics"]["precision"], -row["exact_proxy_utility"], row["config_id"]))
    return chosen["config_id"], diagnostics


def evaluate_gate(per_camera: dict[str, dict[str, Any]], *, deployment_oom: bool) -> dict[str, Any]:
    complete = set(per_camera) == set(EXPECTED_CAMERAS)
    ratios = [row["c10"]["latency_ratio_p95"] for row in per_camera.values()]
    improvements = [row["c10"]["relative_count_mae_improvement"] for row in per_camera.values()]
    precision_changes = [row["c10"]["precision_change"] for row in per_camera.values()]
    gate = {
        "all_four_cameras_retained": complete,
        "all_camera_p95_latency_ratio_le_3": complete and all(value <= 3.0 for value in ratios),
        "max_camera_p95_latency_ratio": max(ratios, default=float("inf")),
        "macro_precision_change": float(np.mean(precision_changes)) if precision_changes else float("-inf"),
        "macro_precision_change_ge_minus_0_05": bool(precision_changes) and float(np.mean(precision_changes)) >= -.05,
        "median_camera_relative_count_mae_improvement": float(np.median(improvements)) if improvements else float("-inf"),
        "median_camera_relative_count_mae_improvement_ge_0_10": bool(improvements) and float(np.median(improvements)) >= .10,
        "deployment_oom": deployment_oom,
    }
    gate["passed"] = all((gate["all_four_cameras_retained"], gate["all_camera_p95_latency_ratio_le_3"], gate["macro_precision_change_ge_minus_0_05"], gate["median_camera_relative_count_mae_improvement_ge_0_10"], not deployment_oom))
    return gate


def run(
    proxy_path: Path, cache_root: Path, pilot_root: Path, yolo11s: Path, yolo26: Path, output_dir: Path, *,
    device: str = "0", repeats: int = 5, roi_batch_size: int = 1, calibration_count: int = 10, top_k: int = 6,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    top = select_top_k(proxy, top_k)
    records, split = _load_scope_without_gt(cache_root, pilot_root, proxy["split"])
    frame_manifest = make_split_manifest(records, split, calibration_count)
    frame_manifest_path = output_dir / "frame-split-manifest.json"
    frame_manifest_path.write_text(json.dumps(frame_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    query_trace = {
        "schema_version": "sceneconfig.top6-query-trace.v1",
        "selection_source": "sealed_approximate_proxy_only",
        "gt_used": False,
        "top_k": top_k,
        "cameras": top,
    }
    query_path = output_dir / "top6-query-trace.json"
    query_path.write_text(json.dumps(query_trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model_hashes = {"yolo11s": _sha256(yolo11s), "yolo26": _sha256(yolo26)}
    operator_base = {
        "schema_version": "sceneconfig.operator-fingerprint.v1",
        "global": {"model_sha256": model_hashes["yolo11s"], "imgsz": 640, "confidence": .25, "batch": 1},
        "roi": {"model_sha256": model_hashes["yolo26"], "imgsz": 640, "atomic_confidence": .05, "compiler_threshold_per_card": True, "batch": roi_batch_size, "padding": "zero_like", "rect": False, "end2end": False},
        "order": ["global", "roi_crop", "roi_inference", "ownership_mapping", "nms"],
        "nms_iou": .50,
        "tracker": "none",
        "warmup": "one_global_and_one_first_roi_per_fresh_configuration_model_instance",
        "input_color": "BGR_CONVERTED_FROM_PIL_RGB",
    }
    fingerprint_path = output_dir / "operator-fingerprints.json"
    fingerprints: dict[str, Any] = {}
    prediction_path = output_dir / "predictions.jsonl"
    summary: dict[str, Any] = {"cameras": {}}
    with prediction_path.open("x", encoding="utf-8") as ledger:
        for camera in EXPECTED_CAMERAS:
            camera_records = [row for row in records if row.camera_id == camera]
            baseline_card = {"config_id": "B0", "policy": "yolo11s_global", "tiles": [], "tile_confidence": .25}
            baseline_frames, baseline_latency, baseline_oom, baseline_error = _run_operator(camera_records, baseline_card, yolo11s, yolo26, device=device, repeats=repeats, roi_batch_size=roi_batch_size)
            ledger.write(json.dumps({"camera_id": camera, "config_id": "B0", "frames": baseline_frames}, sort_keys=True) + "\n")
            ledger.flush()
            print(f"TOP6_PROGRESS camera={camera} config=B0 oom={baseline_oom} frames={len(baseline_frames)}", flush=True)
            camera_summary = {
                "baseline": {"oom": baseline_oom, "error": baseline_error, "p95_ms": _p95(baseline_latency), "frame_count": len(baseline_frames)},
                "candidates": [],
            }
            fingerprints[f"{camera}:B0"] = {**operator_base, "camera_id": camera, "config": baseline_card, "fingerprint_sha256": _canonical_sha({**operator_base, "camera_id": camera, "config": baseline_card})}
            for card in top[camera]:
                frames, latency, oom, error = _run_operator(camera_records, card, yolo11s, yolo26, device=device, repeats=repeats, roi_batch_size=roi_batch_size)
                ledger.write(json.dumps({"camera_id": camera, "config_id": card["config_id"], "frames": frames}, sort_keys=True) + "\n")
                ledger.flush()
                print(f"TOP6_PROGRESS camera={camera} config={card['config_id']} oom={oom} frames={len(frames)}", flush=True)
                proxy_metrics = exact_proxy_metrics(frames, len(card["tiles"])) if not oom and len(frames) == 32 else {"exact_proxy_utility": FAILED_PROXY_UTILITY}
                candidate = {**card, **proxy_metrics, "oom": oom, "error": error, "p95_ms": _p95(latency), "latency_ratio_p95": _p95(latency) / camera_summary["baseline"]["p95_ms"], "frame_count": len(frames)}
                camera_summary["candidates"].append(candidate)
                fingerprints[f"{camera}:{card['config_id']}"] = {**operator_base, "camera_id": camera, "config": card, "fingerprint_sha256": _canonical_sha({**operator_base, "camera_id": camera, "config": card})}
            summary["cameras"][camera] = camera_summary
    fingerprint_path.write_text(json.dumps({"schema_version": "sceneconfig.operator-fingerprints.v1", "operators": fingerprints}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pre_gt_summary_path = output_dir / "pre-gt-execution-summary.json"
    pre_gt_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal_payload = {
        "schema_version": "sceneconfig.pre-gt-seal.v1",
        "stage": "predictions_complete_gt_unread",
        "holdout_opened": False,
        "files": {path.name: _sha256(path) for path in (frame_manifest_path, query_path, prediction_path, fingerprint_path, pre_gt_summary_path)},
    }
    seal_path = output_dir / "PRE_GT_SEAL.json"
    seal_path.write_text(json.dumps(seal_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # GT access begins only after all configuration outputs above are sealed.
    ledger_rows = {(row["camera_id"], row["config_id"]): row["frames"] for row in (json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines())}
    per_camera: dict[str, Any] = {}
    deployment_oom = False
    for camera in EXPECTED_CAMERAS:
        sequence = split[camera]["evaluation"]
        truth = list(parse_mot17_gt(pilot_root / "sequences" / sequence / "gt.txt"))
        cal_frames = tuple(frame_manifest["cameras"][camera]["calibration_frames"])
        eval_frames = tuple(frame_manifest["cameras"][camera]["evaluation_frames"])
        baseline_rows = ledger_rows[(camera, "B0")]
        baseline_cal = _metrics_for_frames(baseline_rows, truth, cal_frames)
        baseline_eval = _metrics_for_frames(baseline_rows, truth, eval_frames)
        baseline_p95 = summary["cameras"][camera]["baseline"]["p95_ms"]
        deployment_oom = deployment_oom or summary["cameras"][camera]["baseline"]["oom"]
        candidates = []
        for row in summary["cameras"][camera]["candidates"]:
            frames = ledger_rows[(camera, row["config_id"])]
            cal_metrics = _metrics_for_frames(frames, truth, cal_frames) if not row["oom"] and len(frames) == 32 else {"precision": 0.0, "count_mae": FAILED_METRIC, "frame_count": float(len(cal_frames))}
            eval_metrics = _metrics_for_frames(frames, truth, eval_frames) if not row["oom"] and len(frames) == 32 else {"precision": 0.0, "count_mae": FAILED_METRIC, "frame_count": float(len(eval_frames))}
            candidates.append({**row, "calibration_metrics": cal_metrics, "evaluation_metrics": eval_metrics})
        c10_id, selection_diagnostics = choose_calibrated_candidate(candidates, baseline_cal)
        approx_id = top[camera][0]["config_id"]
        exact_eligible = [row for row in candidates if not row["oom"] and row["latency_ratio_p95"] <= 3.0]
        exact_id = max(exact_eligible, key=lambda row: (row["exact_proxy_utility"], row["config_id"]))["config_id"] if exact_eligible else "B0"

        def condition(config_id: str) -> dict[str, Any]:
            if config_id == "B0":
                metrics, ratio, oom = baseline_eval, 1.0, summary["cameras"][camera]["baseline"]["oom"]
            else:
                selected = next(row for row in candidates if row["config_id"] == config_id)
                metrics, ratio, oom = selected["evaluation_metrics"], selected["latency_ratio_p95"], selected["oom"]
            base_mae = baseline_eval["count_mae"]
            improvement = (base_mae - metrics["count_mae"]) / base_mae if base_mae > 0 else 0.0
            return {
                "config_id": config_id,
                "metrics": metrics,
                "latency_ratio_p95": ratio,
                "precision_change": metrics["precision"] - baseline_eval["precision"],
                "relative_count_mae_improvement": improvement,
                "oom": oom,
            }

        camera_result = {
            "b0": condition("B0"),
            "a0_approximate_zero_label": condition(approx_id),
            "e0_exact_zero_label": condition(exact_id),
            "c10": condition(c10_id),
            "calibration_baseline": baseline_cal,
            "calibration_selection_diagnostics": selection_diagnostics,
            "candidate_metrics": candidates,
            "c10_fallback_to_b0": c10_id == "B0",
        }
        deployment_oom = deployment_oom or camera_result["c10"]["oom"]
        per_camera[camera] = camera_result
    gate = evaluate_gate(per_camera, deployment_oom=deployment_oom)
    result = {
        "schema_version": "sceneconfig.top6-calibrated-replay.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": gate["passed"],
        "validation_outcome": "MECHANISM_GATE_PASSED" if gate["passed"] else "VALIDATED_REJECTION_OR_FALLBACK",
        "holdout_opened": False,
        "sealed_camera_used": False,
        "qwen_called": False,
        "gt_access_order": "after_pre_gt_seal",
        "calibration_is_mechanism_only_not_statistical_guarantee": True,
        "top_k": top_k,
        "calibration_frames_per_camera": calibration_count,
        "evaluation_frames_per_camera": 32 - calibration_count,
        "repeats": repeats,
        "roi_batch_size": roi_batch_size,
        "pre_gt_seal_sha256": _sha256(seal_path),
        "gate": gate,
        "per_camera": per_camera,
    }
    (output_dir / "top6-calibrated-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--roi-batch-size", type=int, default=1)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    result = run(args.proxy, args.cache_root, args.pilot_root, args.yolo11s, args.yolo26, args.output_dir, device=args.device, repeats=args.repeats, roi_batch_size=args.roi_batch_size, calibration_count=args.calibration_count, top_k=args.top_k)
    print(json.dumps({"passed": result["passed"], "validation_outcome": result["validation_outcome"], "gate": result["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

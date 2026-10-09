"""Development-only end-to-end E4 deployability gate for deterministic M0."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, Prediction, compute_metrics
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.run_scene_adaptive_small_track_ablation import _load_records, _owned, fixed_tiles, nms
from research_tools.run_yolo26_depth_spatial_ablation import ultralytics_bgr


def select_m0_configs(proxy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Freeze one config per camera from proxy utility only, before GT access."""
    cards = {row["config_id"]: row for row in proxy["panel"]}
    selected = {}
    for camera in EXPECTED_CAMERAS:
        best = max(proxy["per_camera"][camera], key=lambda row: (row["proxy_utility"], row["config_id"]))
        selected[camera] = {**cards[best["config_id"]], "selection_score": best["proxy_utility"]}
    return selected


def evaluate_gate(per_camera: dict[str, dict[str, Any]], *, oom: bool) -> dict[str, Any]:
    ratios = [row["latency_ratio_p95"] for row in per_camera.values()]
    relative_mae = [row["relative_count_mae_improvement"] for row in per_camera.values()]
    macro_precision_change = float(np.mean([row["precision_change"] for row in per_camera.values()]))
    gate = {
        "all_camera_p95_latency_ratio_le_3": all(value <= 3.0 for value in ratios),
        "max_camera_p95_latency_ratio": max(ratios, default=float("inf")),
        "macro_precision_change": macro_precision_change,
        "macro_precision_change_ge_minus_0_05": macro_precision_change >= -.05,
        "median_camera_relative_count_mae_improvement": float(np.median(relative_mae)),
        "median_camera_relative_count_mae_improvement_ge_0_10": float(np.median(relative_mae)) >= .10,
        "deployment_oom": oom,
    }
    gate["passed"] = (
        gate["all_camera_p95_latency_ratio_le_3"]
        and gate["macro_precision_change_ge_minus_0_05"]
        and gate["median_camera_relative_count_mae_improvement_ge_0_10"]
        and not oom
    )
    return gate


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _predict_global(model: Any, rgb: np.ndarray, frame: int, device: str) -> tuple[Prediction, ...]:
    result = model.predict(source=[ultralytics_bgr(rgb)], device=device, imgsz=640, batch=1, verbose=False, conf=.25)[0]
    output = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()):
            if int(cls) == 0:
                output.append(Prediction(frame, Box(*map(float, coords)), float(confidence), "yolo11s_global"))
    return tuple(output)


def _predict_tiles(
    model: Any,
    rgb: np.ndarray,
    frame: int,
    card: dict[str, Any],
    device: str,
    *,
    fixed_batch_size: int | None = None,
) -> tuple[Prediction, ...]:
    by_id = {tile.tile_id: tile for tile in fixed_tiles(rgb.shape[1], rgb.shape[0])}
    tiles = [by_id[value] for value in card["tiles"]]
    if not tiles:
        return ()
    output = []
    crops = [ultralytics_bgr(rgb[int(tile.crop.y1):int(tile.crop.y2), int(tile.crop.x1):int(tile.crop.x2)]) for tile in tiles]
    chunk_size = fixed_batch_size or len(crops)
    for start in range(0, len(crops), chunk_size):
        real_crops = crops[start:start + chunk_size]
        real_tiles = tiles[start:start + chunk_size]
        inference_crops = list(real_crops)
        if fixed_batch_size is not None:
            inference_crops.extend(np.zeros_like(real_crops[0]) for _ in range(fixed_batch_size - len(real_crops)))
        results = model.predict(source=inference_crops, device=device, imgsz=640, batch=len(inference_crops), verbose=False, conf=.05, end2end=False, rect=False)
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


def run(
    proxy_path: Path,
    cache_root: Path,
    pilot_root: Path,
    yolo11s: Path,
    yolo26: Path,
    output_dir: Path,
    *,
    device: str = "0",
    repeats: int = 5,
    roi_batch_size: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    if proxy.get("holdout_opened") is not False or proxy.get("evaluation_gt_used_for_proxy") is not False:
        raise ValueError("proxy_not_zero_label_safe")
    selected = select_m0_configs(proxy)
    config_payload = {
        "schema_version": "sceneconfig.e4-m0-compiled-configs.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "selection_source": "proxy_utility_only",
        "evaluation_gt_used_for_selection": False,
        "holdout_opened": False,
        "cameras": selected,
    }
    config_path = output_dir / "compiled-configs.json"
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    (output_dir / "CONFIGS_FROZEN").write_text(config_sha + "  compiled-configs.json\n", encoding="utf-8")

    # Development evaluation GT is opened only after the M0 configs are frozen.
    _history, evaluation, split = _load_records(cache_root, pilot_root)
    from ultralytics import YOLO
    import torch

    global_model, expert_model = YOLO(yolo11s), YOLO(yolo26)
    warmup = np.asarray(Image.open(evaluation[0].image_path).convert("RGB"))
    _predict_global(global_model, warmup, evaluation[0].frame, device)
    _predict_tiles(expert_model, warmup, evaluation[0].frame, {"tiles": ["t0"], "tile_confidence": .25}, device, fixed_batch_size=roi_batch_size)
    per_camera: dict[str, Any] = {}
    oom = False
    for camera in EXPECTED_CAMERAS:
        records = [row for row in evaluation if row.camera_id == camera]
        baseline_ms: list[float] = []
        compiled_ms: list[float] = []
        baseline_first: list[Prediction] = []
        compiled_first: list[Prediction] = []
        try:
            for repeat in range(repeats):
                for record in records:
                    rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    began = time.perf_counter()
                    baseline = _predict_global(global_model, rgb, record.frame, device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    baseline_ms.append((time.perf_counter() - began) * 1000.0)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    began = time.perf_counter()
                    global_rows = _predict_global(global_model, rgb, record.frame, device)
                    tile_rows = _predict_tiles(expert_model, rgb, record.frame, selected[camera], device, fixed_batch_size=roi_batch_size)
                    merged = nms((*global_rows, *tile_rows))
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    compiled_ms.append((time.perf_counter() - began) * 1000.0)
                    if repeat == 0:
                        baseline_first.extend(baseline)
                        compiled_first.extend(merged)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            break
        truth = [gt for record in records for gt in record.truth]
        baseline_metrics = compute_metrics(baseline_first, truth)
        compiled_metrics = compute_metrics(compiled_first, truth)
        baseline_p95 = _percentile(baseline_ms, 95) if baseline_ms else float("inf")
        compiled_p95 = _percentile(compiled_ms, 95) if compiled_ms else float("inf")
        baseline_mae = baseline_metrics["count_mae"]
        relative_mae = (baseline_mae - compiled_metrics["count_mae"]) / baseline_mae if baseline_mae > 0 else 0.0
        per_camera[camera] = {
            "config_id": selected[camera]["config_id"],
            "baseline_p50_ms": _percentile(baseline_ms, 50) if baseline_ms else float("inf"),
            "baseline_p95_ms": baseline_p95,
            "compiled_p50_ms": _percentile(compiled_ms, 50) if compiled_ms else float("inf"),
            "compiled_p95_ms": compiled_p95,
            "latency_ratio_p95": compiled_p95 / baseline_p95,
            "baseline_count_mae": baseline_mae,
            "compiled_count_mae": compiled_metrics["count_mae"],
            "relative_count_mae_improvement": relative_mae,
            "baseline_precision": baseline_metrics["precision"],
            "compiled_precision": compiled_metrics["precision"],
            "precision_change": compiled_metrics["precision"] - baseline_metrics["precision"],
            "latency_observations_per_condition": len(baseline_ms),
        }
        if oom:
            break
    gate = evaluate_gate(per_camera, oom=oom)
    result = {
        "schema_version": "sceneconfig.e4-m0-deployability.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": gate["passed"],
        "holdout_opened": False,
        "sealed_camera_used": False,
        "qwen_called": False,
        "evaluation_gt_used_for_selection": False,
        "evaluation_gt_used_for_post_freeze_metrics": True,
        "repeats": repeats,
        "latency_boundary": "cpu_rgb_image_ready_through_gpu_synchronized_global_roi_mapping_and_nms",
        "roi_letterbox_rect": False,
        "roi_fixed_batch_size": roi_batch_size,
        "roi_atomic_model_confidence": .05,
        "roi_compiler_threshold_applied_after_inference": True,
        "compiled_configs_sha256": config_sha,
        "split": split,
        "gate": gate,
        "per_camera": per_camera,
        "decision": "M0_DEVELOPMENT_E4_PASS" if gate["passed"] else "M0_NOT_DEPLOYABLE_UNDER_FROZEN_E4",
    }
    (output_dir / "e4-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--roi-batch-size", type=int)
    args = parser.parse_args()
    result = run(args.proxy, args.cache_root, args.pilot_root, args.yolo11s, args.yolo26, args.output_dir, device=args.device, repeats=args.repeats, roi_batch_size=args.roi_batch_size)
    print(json.dumps({"passed": result["passed"], "gate": result["gate"], "decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded SceneConfig E0 contract smoke on two development cameras.

This is not the paper experiment.  It exercises the data/cache contract,
full-frame YOLO11s, one high-resolution YOLO26 ROI, coordinate restoration,
merge ownership, and end-to-end latency accounting on 64 frames per camera.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.run_scene_adaptive_small_track_ablation import Tile, fixed_tiles, nms
from research_tools.run_yolo26_depth_spatial_ablation import ultralytics_bgr


SMOKE_CAMERAS = ("VIRAT_S_010004", "VIRAT_S_010204")


@dataclass(frozen=True)
class SmokeFrame:
    camera_id: str
    sequence_id: str
    frame: int
    image_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera(sequence_id: str) -> str:
    return "_".join(sequence_id.split("_")[:3])


def build_smoke_scope(cache_root: Path, pilot_root: Path) -> tuple[list[SmokeFrame], dict[str, Any]]:
    manifest_path = cache_root / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "fieldpack.dcrq-observation-cache.v1":
        raise ValueError("unexpected_cache_schema")
    if manifest.get("numpy_input_color") != "BGR_CONVERTED_FROM_PIL_RGB":
        raise ValueError("bgr_corrected_cache_required")
    selected = [row for row in manifest["sequences"] if _camera(row["sequence_id"]) in SMOKE_CAMERAS]
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frames: list[SmokeFrame] = []
    cache_receipts: list[dict[str, Any]] = []
    for row in selected:
        by_camera[_camera(row["sequence_id"])].append(row)
        cache_path = cache_root / row["cache"]
        receipt_path = cache_root / row["receipt"]
        if _sha256(cache_path) != row["cache_sha256"] or _sha256(receipt_path) != row["receipt_sha256"]:
            raise ValueError(f"cache_hash_mismatch:{row['sequence_id']}")
        arrays = np.load(cache_path)
        frame_ids = arrays["frame_ids"].astype(int).tolist()
        if len(frame_ids) != 32:
            raise ValueError(f"expected_32_frames:{row['sequence_id']}")
        for frame_id in frame_ids:
            image_path = pilot_root / "sequences" / row["sequence_id"] / "img1" / f"{frame_id:06d}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"missing_frame:{image_path}")
            frames.append(SmokeFrame(_camera(row["sequence_id"]), row["sequence_id"], frame_id, image_path))
        cache_receipts.append({
            "sequence_id": row["sequence_id"],
            "cache_sha256": row["cache_sha256"],
            "receipt_sha256": row["receipt_sha256"],
            "frames": len(frame_ids),
        })
    if set(by_camera) != set(SMOKE_CAMERAS) or any(len(rows) != 2 for rows in by_camera.values()):
        raise ValueError("expected_two_sequences_for_each_smoke_camera")
    counts = Counter(frame.camera_id for frame in frames)
    if any(counts[camera] != 64 for camera in SMOKE_CAMERAS):
        raise ValueError("expected_64_frames_per_smoke_camera")
    return sorted(frames, key=lambda row: (row.camera_id, row.sequence_id, row.frame)), {
        "cache_manifest_sha256": _sha256(manifest_path),
        "cache_receipts": cache_receipts,
    }


def _global_predictions(model: Any, bgr: np.ndarray, frame_id: int, *, device: str) -> tuple[Prediction, ...]:
    result = model.predict(source=[bgr], device=device, imgsz=640, verbose=False, conf=0.25)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    return tuple(
        Prediction(frame_id, Box(*map(float, coords)), float(confidence), "yolo11s_global")
        for coords, confidence, cls in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        )
        if int(cls) == 0
    )


def _owned(box: Box, core: Box) -> bool:
    center_x = (box.x1 + box.x2) / 2.0
    center_y = (box.y1 + box.y2) / 2.0
    return core.x1 <= center_x < core.x2 and core.y1 <= center_y < core.y2


def _roi_predictions(
    model: Any,
    rgb: np.ndarray,
    tile: Tile,
    frame_id: int,
    *,
    device: str,
) -> tuple[Prediction, ...]:
    crop = tile.crop
    roi = rgb[int(crop.y1):int(crop.y2), int(crop.x1):int(crop.x2)]
    result = model.predict(
        source=[ultralytics_bgr(roi)],
        device=device,
        imgsz=640,
        verbose=False,
        conf=0.10,
        end2end=False,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    height, width = rgb.shape[:2]
    output: list[Prediction] = []
    for coords, confidence, cls in zip(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().tolist(),
    ):
        if int(cls) != 0:
            continue
        x1, y1, x2, y2 = map(float, coords)
        mapped = Box(
            max(0.0, min(float(width), x1 + crop.x1)),
            max(0.0, min(float(height), y1 + crop.y1)),
            max(0.0, min(float(width), x2 + crop.x1)),
            max(0.0, min(float(height), y2 + crop.y1)),
        )
        if mapped.x2 <= mapped.x1 or mapped.y2 <= mapped.y1:
            continue
        if _owned(mapped, tile.core):
            output.append(Prediction(frame_id, mapped, float(confidence), f"yolo26_roi:{tile.tile_id}"))
    return tuple(output)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _p(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run(
    cache_root: Path,
    pilot_root: Path,
    yolo11s: Path,
    yolo26: Path,
    *,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    frames, cache_evidence = build_smoke_scope(cache_root, pilot_root)
    baseline_model = YOLO(yolo11s)
    expert_model = YOLO(yolo26)
    warmup = ultralytics_bgr(np.asarray(Image.open(frames[0].image_path).convert("RGB")))
    baseline_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=0.25)
    expert_model.predict(source=[warmup], device=device, imgsz=640, verbose=False, conf=0.10, end2end=False)

    selected_tiles = {
        SMOKE_CAMERAS[0]: fixed_tiles(1280, 720)[1],
        SMOKE_CAMERAS[1]: fixed_tiles(1280, 720)[0],
    }
    baseline_latency: list[float] = []
    compiled_latency: list[float] = []
    baseline_counts: Counter[str] = Counter()
    compiled_counts: Counter[str] = Counter()
    coordinate_checks = 0
    for record in frames:
        rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
        bgr = ultralytics_bgr(rgb)
        _sync()
        began = time.perf_counter()
        baseline = _global_predictions(baseline_model, bgr, record.frame, device=device)
        _sync()
        baseline_latency.append((time.perf_counter() - began) * 1000.0)

        _sync()
        began = time.perf_counter()
        global_rows = _global_predictions(baseline_model, bgr, record.frame, device=device)
        roi_rows = _roi_predictions(expert_model, rgb, selected_tiles[record.camera_id], record.frame, device=device)
        merged = nms((*global_rows, *roi_rows), iou=0.50)
        _sync()
        compiled_latency.append((time.perf_counter() - began) * 1000.0)
        baseline_counts[record.camera_id] += len(baseline)
        compiled_counts[record.camera_id] += len(merged)
        for prediction in roi_rows:
            box = prediction.box
            if not (0 <= box.x1 < box.x2 <= rgb.shape[1] and 0 <= box.y1 < box.y2 <= rgb.shape[0]):
                raise ValueError("coordinate_mapping_out_of_bounds")
            coordinate_checks += 1

    ratio = _p(compiled_latency, 95) / max(_p(baseline_latency, 95), 1e-9)
    return {
        "schema_version": "sceneconfig.e0-contract-smoke.v1",
        "verification_status": "DEVELOPMENT_ONLY",
        "paper_evidence": False,
        "confirmation_camera_used": False,
        "camera_ids": list(SMOKE_CAMERAS),
        "camera_count": len(SMOKE_CAMERAS),
        "frames_per_camera": 64,
        "frame_count": len(frames),
        "cache_contract": cache_evidence,
        "compiled_configuration": {
            camera: {
                "config_id": f"e0-{tile.tile_id}",
                "global_detector": "yolo11s",
                "roi_detector": "yolo26s",
                "roi_id": tile.tile_id,
                "roi_core": tile.core.coords,
                "roi_crop": tile.crop.coords,
                "merge": "ownership+nms",
            }
            for camera, tile in selected_tiles.items()
        },
        "coordinate_mapping_checks": coordinate_checks,
        "latency": {
            "boundary": "cpu_image_ready_to_gpu_synchronized_merged_predictions",
            "baseline_p50_ms": _p(baseline_latency, 50),
            "baseline_p95_ms": _p(baseline_latency, 95),
            "compiled_p50_ms": _p(compiled_latency, 50),
            "compiled_p95_ms": _p(compiled_latency, 95),
            "compiled_to_baseline_p95_ratio": ratio,
            "sample_count_per_condition": len(frames),
        },
        "detection_counts": {
            "baseline": dict(sorted(baseline_counts.items())),
            "compiled": dict(sorted(compiled_counts.items())),
        },
        "contracts": {
            "schema_valid": True,
            "cache_hashes_valid": True,
            "coordinate_mapping_valid": True,
            "complete_timing_boundary_present": True,
            "holdout_gt_read": False,
            "qwen_called": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.cache_root, args.pilot_root, args.yolo11s, args.yolo26, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": "passed",
        "camera_count": payload["camera_count"],
        "frame_count": payload["frame_count"],
        "p95_ratio": payload["latency"]["compiled_to_baseline_p95_ratio"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

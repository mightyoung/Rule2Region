"""Development-only E2 validation for the SceneConfig label-blind proxy.

The proxy ledger is written and hashed before evaluation labels are consumed.
This module intentionally uses only the four development cameras and never
opens the sealed confirmation camera or H-holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, Prediction, compute_metrics
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.run_scene_adaptive_small_track_ablation import (
    SceneFrame,
    _load_records,
    _tile_predictions,
    fixed_tiles,
    nms,
)
from research_tools.run_yolo26_depth_spatial_ablation import ultralytics_bgr


@dataclass(frozen=True)
class ConfigCard:
    config_id: str
    tiles: tuple[str, ...]
    tile_confidence: float
    policy: str = "yolo11s_global_plus_yolo26_static_roi"


def frozen_panel() -> tuple[ConfigCard, ...]:
    """Build exactly 24 deterministic, functionally distinct config cards."""
    tile_ids = ("t0", "t1", "t2", "t3")
    subsets = [()] + [part for size in range(1, 5) for part in combinations(tile_ids, size)]
    low_threshold_subsets = [part for part in subsets if 1 <= len(part) <= 2][:8]
    cards = [ConfigCard(f"C{index:02d}", part, .25) for index, part in enumerate(subsets)]
    low_start = len(cards)
    cards.extend(
        ConfigCard(f"C{low_start + index:02d}", part, .10)
        for index, part in enumerate(low_threshold_subsets)
    )
    if len(cards) != 24 or len({(card.tiles, card.tile_confidence) for card in cards}) != 24:
        raise AssertionError("panel_must_have_24_distinct_cards")
    return tuple(cards)


def _tile_id(row: Prediction) -> str:
    return row.source.rsplit(":", 1)[-1]


def compile_predictions(
    records: Iterable[SceneFrame],
    tile_rows: dict[tuple[str, int], tuple[Prediction, ...]],
    card: ConfigCard,
) -> dict[tuple[str, int], tuple[Prediction, ...]]:
    output = {}
    selected = set(card.tiles)
    for record in records:
        key = (record.sequence_id, record.frame)
        local = tuple(
            row for row in tile_rows.get(key, ())
            if _tile_id(row) in selected and row.confidence >= card.tile_confidence
        )
        output[key] = nms((*record.global_predictions, *local))
    return output


def _supported_fraction(by_frame: dict[int, tuple[Prediction, ...]], *, iou: float = .20) -> float:
    supported = total = 0
    frames = sorted(by_frame)
    for index, frame in enumerate(frames):
        rows = by_frame[frame]
        for row in rows:
            total += 1
            neighbors = []
            if index:
                neighbors.extend(by_frame[frames[index - 1]])
            if index + 1 < len(frames):
                neighbors.extend(by_frame[frames[index + 1]])
            if any(row.box.iou(other.box) >= iou for other in neighbors):
                supported += 1
    return supported / total if total else 0.0


def proxy_row(
    camera_records: list[SceneFrame],
    tile_rows: dict[tuple[str, int], tuple[Prediction, ...]],
    card: ConfigCard,
) -> dict[str, Any]:
    selected = set(card.tiles)
    raw_added = []
    retained_added = []
    merged_counts = []
    global_counts = []
    temporal: dict[int, tuple[Prediction, ...]] = {}
    for record in camera_records:
        key = (record.sequence_id, record.frame)
        raw = tuple(
            row for row in tile_rows.get(key, ())
            if _tile_id(row) in selected and row.confidence >= card.tile_confidence
        )
        merged = nms((*record.global_predictions, *raw))
        retained = tuple(row for row in merged if row.source.startswith("yolo26_hr_tile:"))
        raw_added.extend(raw)
        retained_added.extend(retained)
        temporal[record.frame] = retained
        merged_counts.append(len(merged))
        global_counts.append(len(record.global_predictions))
    frame_count = max(1, len(camera_records))
    support = _supported_fraction(temporal)
    duplicate_rate = 1.0 - len(retained_added) / len(raw_added) if raw_added else 0.0
    small_rate = (
        sum((row.box.y2 - row.box.y1) <= 48 for row in retained_added) / len(retained_added)
        if retained_added else 0.0
    )
    added_per_frame = len(retained_added) / frame_count
    disagreement = float(np.mean(np.abs(np.asarray(merged_counts) - np.asarray(global_counts))))
    cost_ratio = len(card.tiles) / 4.0
    # Frozen before GT: reward persistent small-person evidence, penalize isolated
    # additions, duplicate pressure, and expert-detector cost.
    utility = (
        added_per_frame * (1.5 * support + 0.5 * small_rate - 1.0 * (1.0 - support))
        - 0.5 * duplicate_rate
        - 0.15 * cost_ratio
    )
    return {
        "config_id": card.config_id,
        "tiles": list(card.tiles),
        "tile_confidence": card.tile_confidence,
        "added_per_frame": added_per_frame,
        "count_disagreement": disagreement,
        "small_added_rate": small_rate,
        "temporal_support_rate": support,
        "duplicate_suppression_rate": duplicate_rate,
        "tile_cost_ratio": cost_ratio,
        "proxy_utility": utility,
    }


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    a, b = _ranks(left), _ranks(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_global_predictions(model: Any, records: list[SceneFrame], *, device: str) -> list[SceneFrame]:
    """Replace legacy cached boxes with the same batch=1 operator used by E4."""
    output = []
    for record in records:
        rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
        result = model.predict(source=[ultralytics_bgr(rgb)], device=device, imgsz=640, batch=1, verbose=False, conf=.25)[0]
        predictions = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for coords, confidence, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist()):
                if int(cls) == 0:
                    predictions.append(Prediction(record.frame, Box(*map(float, coords)), float(confidence), "yolo11s_global"))
        output.append(replace(record, global_predictions=tuple(predictions)))
    return output


def run(
    cache_root: Path,
    pilot_root: Path,
    yolo11s: Path,
    yolo26: Path,
    output_dir: Path,
    *,
    device: str = "0",
    batch_size: int = 8,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    history, evaluation, split = _load_records(cache_root, pilot_root)
    if tuple(sorted({row.camera_id for row in evaluation})) != EXPECTED_CAMERAS:
        raise ValueError("development_camera_scope_mismatch")
    from ultralytics import YOLO

    global_model, model = YOLO(yolo11s), YOLO(yolo26)
    warmup = np.asarray(Image.open(evaluation[0].image_path).convert("RGB"))
    global_model.predict(source=[ultralytics_bgr(warmup)], device=device, imgsz=640, batch=1, verbose=False, conf=.25)
    warmup_tile = fixed_tiles(warmup.shape[1], warmup.shape[0])[0]
    warmup_crop = warmup[int(warmup_tile.crop.y1):int(warmup_tile.crop.y2), int(warmup_tile.crop.x1):int(warmup_tile.crop.x2)]
    model.predict(source=[ultralytics_bgr(warmup_crop)], device=device, imgsz=640, batch=1, verbose=False, conf=.05, end2end=False, rect=False)
    evaluation = canonical_global_predictions(global_model, evaluation, device=device)
    tiles = {camera: fixed_tiles(1280, 720) for camera in EXPECTED_CAMERAS}
    tile_rows, tile_ms, tile_calls = _tile_predictions(
        model, evaluation, tiles, device=device, batch_size=batch_size, confidence=.05, rect=False
    )
    cards = frozen_panel()
    proxy = {
        camera: [
            proxy_row([row for row in evaluation if row.camera_id == camera], tile_rows, card)
            for card in cards
        ]
        for camera in EXPECTED_CAMERAS
    }
    proxy_payload = {
        "schema_version": "sceneconfig.e2-proxy-ledger.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "history_gt_used_for_configuration": False,
        "evaluation_gt_used_for_proxy": False,
        "legacy_cached_global_predictions_used": False,
        "global_operator": {"batch_size": 1, "confidence": .25, "imgsz": 640},
        "sealed_camera_used": False,
        "split": split,
        "panel": [asdict(card) for card in cards],
        "per_camera": proxy,
        "tile_inference": {"calls": tile_calls, "elapsed_ms": tile_ms, "minimum_confidence": .05, "rect": False},
    }
    proxy_path = output_dir / "proxy-ledger.json"
    proxy_path.write_text(json.dumps(proxy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proxy_sha = _sha256(proxy_path)
    (output_dir / "PROXY_SEALED").write_text(proxy_sha + "  proxy-ledger.json\n", encoding="utf-8")

    # GT is consumed only below this point, after proxy-ledger sealing.
    per_camera: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        camera_records = [row for row in evaluation if row.camera_id == camera]
        truth = [gt for record in camera_records for gt in record.truth]
        baseline_predictions = [p for record in camera_records for p in record.global_predictions]
        baseline = compute_metrics(baseline_predictions, truth)
        evaluated = []
        for card, proxy_metrics in zip(cards, proxy[camera]):
            compiled = compile_predictions(camera_records, tile_rows, card)
            predictions = [p for rows in compiled.values() for p in rows]
            metrics = compute_metrics(predictions, truth)
            precision_floor_violation = max(0.0, baseline["precision"] - .05 - metrics["precision"])
            true_utility = -metrics["count_mae"] - 10.0 * precision_floor_violation - .15 * len(card.tiles) / 4.0
            evaluated.append({
                "config_id": card.config_id,
                "proxy_utility": proxy_metrics["proxy_utility"],
                "true_utility": true_utility,
                "count_mae": metrics["count_mae"],
                "precision": metrics["precision"],
                "precision_change": metrics["precision"] - baseline["precision"],
                "tile_cost_ratio": len(card.tiles) / 4.0,
            })
        rho = spearman(
            [row["proxy_utility"] for row in evaluated],
            [row["true_utility"] for row in evaluated],
        )
        selected = max(evaluated, key=lambda row: (row["proxy_utility"], row["config_id"]))
        per_camera[camera] = {"baseline": baseline, "spearman": rho, "proxy_selected": selected, "configs": evaluated}
    rhos = [row["spearman"] for row in per_camera.values()]
    precision_safe = sum(row["proxy_selected"]["precision_change"] >= -.05 for row in per_camera.values())
    passed = float(np.median(rhos)) > 0 and sum(value > 0 for value in rhos) >= 3 and precision_safe >= 3
    result = {
        "schema_version": "sceneconfig.e2-proxy-validation.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": passed,
        "holdout_opened": False,
        "sealed_camera_used": False,
        "proxy_ledger_sha256": proxy_sha,
        "gate": {
            "median_camera_spearman": float(np.median(rhos)),
            "positive_camera_count": sum(value > 0 for value in rhos),
            "precision_safe_selected_camera_count": precision_safe,
            "requirements": {"median_spearman_gt": 0, "positive_cameras_at_least": 3, "precision_safe_cameras_at_least": 3},
        },
        "per_camera": per_camera,
        "next_gate": "E3" if passed else "REJECT_OR_REDESIGN_ZERO_SHOT_PROXY",
    }
    (output_dir / "e2-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run(args.cache_root, args.pilot_root, args.yolo11s, args.yolo26, args.output_dir, device=args.device, batch_size=args.batch_size)
    print(json.dumps({"passed": result["passed"], "gate": result["gate"], "next_gate": result["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

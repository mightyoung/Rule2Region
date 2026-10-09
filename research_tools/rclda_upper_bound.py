"""CPU-only, development-only separability upper bound for RC-LDA.

The rule family and feature ledger are sealed before development GT is opened.
The resulting oracle is a branch-admission diagnostic, never a deployable rule.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, compute_metrics, parse_mot17_gt
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.run_scene_adaptive_small_track_ablation import fixed_tiles, nms


FEATURE_NAMES = ("confidence", "global_iou", "temporal_support", "scale_consistency", "boundary_margin")
TARGET_CONFIGS = {"VIRAT_S_010004": "C23", "VIRAT_S_010204": "C01"}
INHERITED_CONFIGS = {"VIRAT_S_010111": "C16", "VIRAT_S_010112": "B0"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction(row: dict[str, Any]) -> Prediction:
    return Prediction(int(row["frame"]), Box(**row["box"]), float(row["confidence"]), row["source"])


def _serialize(row: Prediction) -> dict[str, Any]:
    return {"frame": row.frame, "box": asdict(row.box), "confidence": row.confidence, "source": row.source}


def _json_text(payload: Any, *, indent: int | None = None) -> str:
    """Serialize evidence strictly; NaN/Infinity are invalid experiment output."""
    return json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False) + "\n"


def frozen_rule_grid() -> dict[str, Any]:
    weights = [values for values in itertools.product((0, 1, 2), repeat=len(FEATURE_NAMES)) if any(values)]
    return {
        "schema_version": "sceneconfig.rclda-rule-grid.v1",
        "feature_names": FEATURE_NAMES,
        "weights": [{"weight_id": f"W{index:03d}", "values": values} for index, values in enumerate(weights)],
        "thresholds": [round(index / 20.0, 2) for index in range(21)] + [1.01],
        "monotonicity": "nonnegative_weighted_sum_accept_if_score_ge_threshold",
        "target_configs": TARGET_CONFIGS,
        "inherited_configs": INHERITED_CONFIGS,
    }


def _load_prediction_ledger(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ledger = {(row["camera_id"], row["config_id"]): row["frames"] for row in rows}
    if len(ledger) != 28:
        raise ValueError("expected_28_complete_operator_rows")
    return ledger


def _max_iou(box: Box, rows: Iterable[Prediction]) -> float:
    return max((box.iou(row.box) for row in rows), default=0.0)


def _tile_id(source: str) -> str:
    if not source.startswith("yolo26_hr_tile:"):
        raise ValueError("unexpected_local_source")
    return source.split(":", 1)[1]


def _scale_priors(frames: list[dict[str, Any]], height: int) -> tuple[dict[int, float], float]:
    bands: dict[int, list[float]] = defaultdict(list)
    all_heights = []
    for frame in frames:
        for raw in frame["global"]:
            row = _prediction(raw)
            center_y = (row.box.y1 + row.box.y2) / 2.0
            band = min(3, max(0, int(center_y / max(1.0, height) * 4.0)))
            box_height = max(1.0, row.box.y2 - row.box.y1)
            bands[band].append(box_height)
            all_heights.append(box_height)
    fallback = float(np.median(all_heights)) if all_heights else 32.0
    return {band: float(np.median(values)) for band, values in bands.items()}, fallback


def build_feature_rows(
    frames: list[dict[str, Any]], *, width: int, height: int, evaluation_frames: set[int]
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Build GT-free local-candidate features for one complete configuration."""
    selected = [row for row in frames if int(row["frame"]) in evaluation_frames]
    by_frame = {int(row["frame"]): row for row in selected}
    local_by_frame_tile: dict[tuple[int, str], tuple[Prediction, ...]] = {}
    for frame, row in by_frame.items():
        grouped: dict[str, list[Prediction]] = defaultdict(list)
        for raw in row["tile_raw"]:
            pred = _prediction(raw)
            grouped[_tile_id(pred.source)].append(pred)
        for tile, predictions in grouped.items():
            local_by_frame_tile[(frame, tile)] = tuple(predictions)
    priors, fallback = _scale_priors(selected, height)
    tiles = {tile.tile_id: tile for tile in fixed_tiles(width, height)}
    feature_rows: list[dict[str, Any]] = []
    globals_by_frame: dict[int, list[dict[str, Any]]] = {}
    ordered_frames = sorted(by_frame)
    frame_position = {frame: index for index, frame in enumerate(ordered_frames)}
    for frame in ordered_frames:
        row = by_frame[frame]
        globals_ = tuple(_prediction(raw) for raw in row["global"])
        globals_by_frame[frame] = list(row["global"])
        for candidate_index, raw in enumerate(row["tile_raw"]):
            pred = _prediction(raw)
            tile_id = _tile_id(pred.source)
            neighbors: list[Prediction] = []
            position = frame_position[frame]
            for neighbor_index in (position - 1, position + 1):
                if 0 <= neighbor_index < len(ordered_frames):
                    neighbors.extend(local_by_frame_tile.get((ordered_frames[neighbor_index], tile_id), ()))
            center_y = (pred.box.y1 + pred.box.y2) / 2.0
            band = min(3, max(0, int(center_y / max(1.0, height) * 4.0)))
            expected_height = priors.get(band, fallback)
            actual_height = max(1.0, pred.box.y2 - pred.box.y1)
            scale_consistency = math.exp(-abs(math.log(actual_height / max(1.0, expected_height))))
            tile = tiles[tile_id]
            margin = min(pred.box.x1 - tile.crop.x1, tile.crop.x2 - pred.box.x2, pred.box.y1 - tile.crop.y1, tile.crop.y2 - pred.box.y2)
            boundary_margin = min(1.0, max(0.0, margin) / max(1.0, .25 * min(tile.crop.x2 - tile.crop.x1, tile.crop.y2 - tile.crop.y1)))
            features = (
                pred.confidence,
                _max_iou(pred.box, globals_),
                _max_iou(pred.box, neighbors),
                scale_consistency,
                boundary_margin,
            )
            feature_rows.append({
                "frame": frame,
                "candidate_index": candidate_index,
                "tile_id": tile_id,
                "prediction": _serialize(pred),
                "features": {name: float(value) for name, value in zip(FEATURE_NAMES, features)},
            })
    return feature_rows, globals_by_frame


def _metrics(rows_by_frame: dict[int, tuple[Prediction, ...]], truth: list[GroundTruth], frames: tuple[int, ...]) -> dict[str, float]:
    predictions = [row for frame in frames for row in rows_by_frame.get(frame, ())]
    allowed = set(frames)
    selected_truth = [row for row in truth if row.frame in allowed]
    metrics = compute_metrics(predictions, selected_truth)
    pred_counts = {frame: len(rows_by_frame.get(frame, ())) for frame in frames}
    gt_counts: dict[int, int] = defaultdict(int)
    for row in selected_truth:
        gt_counts[row.frame] += 1
    metrics["count_mae"] = float(np.mean([abs(pred_counts[frame] - gt_counts[frame]) for frame in frames]))
    metrics["frame_count"] = float(len(frames))
    return metrics


def _rows_from_feature_ledger(
    features: list[dict[str, Any]], globals_by_frame: dict[int, list[dict[str, Any]]], frames: tuple[int, ...], weights: tuple[int, ...], threshold: float
) -> dict[int, tuple[Prediction, ...]]:
    denominator = float(sum(weights))
    accepted: dict[int, list[Prediction]] = defaultdict(list)
    for row in features:
        score = sum(weight * float(row["features"][name]) for weight, name in zip(weights, FEATURE_NAMES)) / denominator
        if score >= threshold:
            accepted[int(row["frame"])].append(_prediction(row["prediction"]))
    output = {}
    for frame in frames:
        global_rows = tuple(_prediction(row) for row in globals_by_frame.get(frame, ()))
        output[frame] = nms((*global_rows, *accepted.get(frame, ())))
    return output


def evaluate_oracle(
    features: list[dict[str, Any]], globals_by_frame: dict[int, list[dict[str, Any]]], truth: list[GroundTruth],
    frames: tuple[int, ...], rule_grid: dict[str, Any], baseline_metrics: dict[str, float], *, latency_ratio: float,
) -> dict[str, Any]:
    evaluated = []
    baseline_mae = baseline_metrics["count_mae"]
    for weight_row in rule_grid["weights"]:
        weights = tuple(int(value) for value in weight_row["values"])
        for threshold in rule_grid["thresholds"]:
            predictions = _rows_from_feature_ledger(features, globals_by_frame, frames, weights, float(threshold))
            metrics = _metrics(predictions, truth, frames)
            improvement = (baseline_mae - metrics["count_mae"]) / baseline_mae if baseline_mae > 0 else 0.0
            precision_change = metrics["precision"] - baseline_metrics["precision"]
            evaluated.append({
                "weight_id": weight_row["weight_id"],
                "weights": weights,
                "threshold": threshold,
                "metrics": metrics,
                "relative_count_mae_improvement": improvement,
                "precision_change": precision_change,
                "latency_ratio_lower_bound": latency_ratio,
                "retained_local_count": sum(
                    row.source.startswith("yolo26_hr_tile:")
                    for rows in predictions.values()
                    for row in rows
                ),
            })
    viable = [row for row in evaluated if row["relative_count_mae_improvement"] >= .10 and row["precision_change"] >= -.05 and latency_ratio <= 3.0]
    precision_safe = [row for row in evaluated if row["precision_change"] >= -.05]
    ranking = lambda row: (row["relative_count_mae_improvement"], row["precision_change"], -row["retained_local_count"], row["weight_id"], -float(row["threshold"]))
    best_viable = max(viable, key=ranking) if viable else None
    best_safe = max(precision_safe, key=ranking) if precision_safe else None
    top_rules = sorted(evaluated, key=ranking, reverse=True)[:50]
    return {
        "rules_evaluated": len(evaluated),
        "viable_rule_count": len(viable),
        "best_viable_rule": best_viable,
        "best_precision_safe_rule": best_safe,
        "top_50_by_mae_gain_then_precision": top_rules,
    }


def _ledger_condition_metrics(frames: list[dict[str, Any]], truth: list[GroundTruth], evaluation_frames: tuple[int, ...]) -> dict[str, float]:
    rows = {int(row["frame"]): tuple(_prediction(item) for item in row["merged"]) for row in frames if int(row["frame"]) in set(evaluation_frames)}
    return _metrics(rows, truth, evaluation_frames)


def _assert_all_accepted_reproduces_merged(
    frames: list[dict[str, Any]],
    features: list[dict[str, Any]],
    globals_by_frame: dict[int, list[dict[str, Any]]],
    evaluation_frames: tuple[int, ...],
) -> None:
    reconstructed = _rows_from_feature_ledger(
        features, globals_by_frame, evaluation_frames, (1, 0, 0, 0, 0), 0.0
    )
    expected = {
        int(row["frame"]): tuple(_prediction(item) for item in row["merged"])
        for row in frames
        if int(row["frame"]) in set(evaluation_frames)
    }
    for frame in evaluation_frames:
        if tuple(_serialize(row) for row in reconstructed.get(frame, ())) != tuple(
            _serialize(row) for row in expected.get(frame, ())
        ):
            raise ValueError(f"all_accepted_operator_mismatch:{frame}")


def run(input_dir: Path, pilot_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refuse_existing_output:{output_dir}")
    output_dir.mkdir(parents=True)
    input_result = json.loads((input_dir / "top6-calibrated-result.json").read_text(encoding="utf-8"))
    frame_manifest = json.loads((input_dir / "frame-split-manifest.json").read_text(encoding="utf-8"))
    if input_result.get("evidence_status") != "DEVELOPMENT_ONLY" or input_result.get("holdout_opened") is not False:
        raise ValueError("invalid_input_evidence_boundary")
    ledger = _load_prediction_ledger(input_dir / "predictions.jsonl")
    rule_grid = frozen_rule_grid()
    rule_path = output_dir / "frozen-rule-grid.json"
    rule_path.write_text(_json_text(rule_grid, indent=2), encoding="utf-8")
    target_manifest = {
        "schema_version": "sceneconfig.rclda-target-manifest.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "oracle_only": True,
        "prior_development_gt_informed_target_selection": True,
        "current_run_gt_used_before_feature_seal": False,
        "holdout_opened": False,
        "sealed_camera_used": False,
        "target_configs": TARGET_CONFIGS,
        "inherited_configs": INHERITED_CONFIGS,
        "evaluation_frames": {camera: frame_manifest["cameras"][camera]["evaluation_frames"] for camera in EXPECTED_CAMERAS},
        "input_files": {name: _sha256(input_dir / name) for name in ("predictions.jsonl", "frame-split-manifest.json", "top6-calibrated-result.json", "PRE_GT_SEAL.json")},
    }
    target_path = output_dir / "target-manifest.json"
    target_path.write_text(_json_text(target_manifest, indent=2), encoding="utf-8")
    feature_path = output_dir / "feature-ledger.jsonl"
    feature_payload: dict[str, Any] = {}
    with feature_path.open("x", encoding="utf-8") as stream:
        for camera, config_id in TARGET_CONFIGS.items():
            sequence = frame_manifest["cameras"][camera]["sequence_id"]
            first_frame = int(frame_manifest["cameras"][camera]["evaluation_frames"][0])
            image_path = pilot_root / "sequences" / sequence / "img1" / f"{first_frame:06d}.jpg"
            with Image.open(image_path) as image:
                width, height = image.size
            evaluation_frames = set(int(value) for value in frame_manifest["cameras"][camera]["evaluation_frames"])
            features, globals_by_frame = build_feature_rows(ledger[(camera, config_id)], width=width, height=height, evaluation_frames=evaluation_frames)
            row = {"camera_id": camera, "config_id": config_id, "width": width, "height": height, "features": features, "globals_by_frame": globals_by_frame}
            stream.write(_json_text(row))
            feature_payload[camera] = row
    seal = {
        "schema_version": "sceneconfig.rclda-pre-gt-seal.v1",
        "stage": "rule_and_feature_ledgers_complete_current_run_gt_unread",
        "holdout_opened": False,
        "files": {path.name: _sha256(path) for path in (rule_path, target_path, feature_path)},
    }
    seal_path = output_dir / "PRE_GT_SEAL.json"
    seal_path.write_text(_json_text(seal, indent=2), encoding="utf-8")

    # Re-read the sealed feature bytes. Development GT access in this run is
    # prohibited before this point; prior stages already used development GT.
    feature_payload = {
        row["camera_id"]: {
            **row,
            "globals_by_frame": {int(frame): values for frame, values in row["globals_by_frame"].items()},
        }
        for row in (json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines() if line)
    }
    per_camera: dict[str, Any] = {}
    selected_improvements = []
    selected_precision_changes = []
    target_passes = []
    for camera in EXPECTED_CAMERAS:
        sequence = frame_manifest["cameras"][camera]["sequence_id"]
        evaluation_frames = tuple(int(value) for value in frame_manifest["cameras"][camera]["evaluation_frames"])
        truth = list(parse_mot17_gt(pilot_root / "sequences" / sequence / "gt.txt"))
        baseline = _ledger_condition_metrics(ledger[(camera, "B0")], truth, evaluation_frames)
        if camera in TARGET_CONFIGS:
            config_id = TARGET_CONFIGS[camera]
            _assert_all_accepted_reproduces_merged(
                ledger[(camera, config_id)],
                feature_payload[camera]["features"],
                feature_payload[camera]["globals_by_frame"],
                evaluation_frames,
            )
            inherited_latency = float(input_result["per_camera"][camera]["e0_exact_zero_label"]["latency_ratio_p95"])
            if input_result["per_camera"][camera]["e0_exact_zero_label"]["config_id"] != config_id:
                candidate = next(row for row in input_result["per_camera"][camera]["candidate_metrics"] if row["config_id"] == config_id)
                inherited_latency = float(candidate["latency_ratio_p95"])
            oracle = evaluate_oracle(
                feature_payload[camera]["features"], feature_payload[camera]["globals_by_frame"], truth,
                evaluation_frames, rule_grid, baseline, latency_ratio=inherited_latency,
            )
            unfiltered = _ledger_condition_metrics(ledger[(camera, config_id)], truth, evaluation_frames)
            best = oracle["best_viable_rule"]
            selected = best if best is not None else {
                "metrics": baseline,
                "relative_count_mae_improvement": 0.0,
                "precision_change": 0.0,
                "latency_ratio_lower_bound": 1.0,
                "fallback_to_b0": True,
            }
            passed = best is not None
            target_passes.append(passed)
            per_camera[camera] = {"role": "rclda_target", "base_config_id": config_id, "baseline_metrics": baseline, "unfiltered_metrics": unfiltered, "oracle": oracle, "selected_for_composite": selected, "target_gate_passed": passed}
            print(f"RCLDA_PROGRESS camera={camera} rules={oracle['rules_evaluated']} viable={oracle['viable_rule_count']}", flush=True)
        else:
            config_id = INHERITED_CONFIGS[camera]
            metrics = _ledger_condition_metrics(ledger[(camera, config_id)], truth, evaluation_frames)
            base_mae = baseline["count_mae"]
            improvement = (base_mae - metrics["count_mae"]) / base_mae if base_mae > 0 else 0.0
            precision_change = metrics["precision"] - baseline["precision"]
            selected = {"metrics": metrics, "relative_count_mae_improvement": improvement, "precision_change": precision_change, "latency_ratio_lower_bound": float(input_result["per_camera"][camera]["c10"]["latency_ratio_p95"])}
            per_camera[camera] = {"role": "inherited_safe_or_fallback", "config_id": config_id, "baseline_metrics": baseline, "selected_for_composite": selected}
        selected_improvements.append(float(per_camera[camera]["selected_for_composite"]["relative_count_mae_improvement"]))
        selected_precision_changes.append(float(per_camera[camera]["selected_for_composite"]["precision_change"]))
    composite = {
        "both_target_cameras_have_viable_rule": all(target_passes) and len(target_passes) == 2,
        "median_camera_relative_count_mae_improvement": float(np.median(selected_improvements)),
        "median_camera_relative_count_mae_improvement_ge_0_10": float(np.median(selected_improvements)) >= .10,
        "macro_precision_change": float(np.mean(selected_precision_changes)),
        "macro_precision_change_ge_minus_0_05": float(np.mean(selected_precision_changes)) >= -.05,
        "all_four_cameras_retained": set(per_camera) == set(EXPECTED_CAMERAS),
    }
    composite["passed"] = all((composite["both_target_cameras_have_viable_rule"], composite["median_camera_relative_count_mae_improvement_ge_0_10"], composite["macro_precision_change_ge_minus_0_05"], composite["all_four_cameras_retained"]))
    result = {
        "schema_version": "sceneconfig.rclda-upper-bound.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": composite["passed"],
        "validation_outcome": "RCLDA_GPU_REPLAY_ADMITTED" if composite["passed"] else "VALIDATED_REJECTION_OF_RCLDA_RULE_FAMILY",
        "oracle_only_not_deployable": True,
        "gpu_used": False,
        "gt_access_order": "after_pre_gt_seal",
        "holdout_opened": False,
        "sealed_camera_used": False,
        "all_accepted_operator_invariant_passed": True,
        "pre_gt_seal_sha256": _sha256(seal_path),
        "per_camera": per_camera,
        "composite_gate": composite,
    }
    (output_dir / "rclda-upper-bound-result.json").write_text(_json_text(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_dir, args.pilot_root, args.output_dir)
    print(json.dumps({"passed": result["passed"], "validation_outcome": result["validation_outcome"], "composite_gate": result["composite_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

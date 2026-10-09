from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Sequence, TypeAlias, assert_never

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, GroundTruth, Prediction, _match
from research_tools.execution_policy.compile import h1_pass
from research_tools.execution_policy.score import _arm_metrics
from research_tools.run_yolo26_depth_spatial_ablation import ultralytics_bgr

OUTSIDE_CONFS: Final = (0.15, 0.25, 0.35)
FOCUS_CONFS: Final = (0.05, 0.10, 0.15, 0.25, 0.35)
HORIZONS: Final = {"T1": ("H1",), "T2": ("H1", "H2"), "T4": ("H1", "H2", "H3", "H4")}
FUTURE_SEGMENTS: Final = ("F1", "F2")
BASELINE_CONF: Final = 0.15
DatasetName: TypeAlias = Literal["VIRAT", "ChokePoint"]


@dataclass(frozen=True, slots=True)
class ExperimentInputError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class FrameObservation:
    scene_id: str
    dataset: DatasetName
    segment_id: str
    frame: int
    width: int
    height: int
    predictions: tuple[Prediction, ...]
    truth: tuple[GroundTruth, ...]
    truth_count: int
    latency_ms: float
    truth_points: tuple[tuple[float, float], ...] = ()


def _cell_id(x: float, y: float, width: int, height: int) -> str:
    return f"g{min(3, max(0, int(4 * y / height)))}{min(3, max(0, int(4 * x / width)))}"


def _baseline(rows: Sequence[FrameObservation]) -> dict[int, tuple[Prediction, ...]]:
    return {row.frame: tuple(pred for pred in row.predictions if pred.confidence >= BASELINE_CONF) for row in rows}


def focus_cell_ids(rows: Sequence[FrameObservation]) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    baseline = _baseline(rows)
    for row in rows:
        match row.dataset:
            case "VIRAT":
                _, _, _, _, matched = _match(baseline[row.frame], row.truth, 0.5)
                points = tuple(((truth.box.x1 + truth.box.x2) / 2, (truth.box.y1 + truth.box.y2) / 2) for index, truth in enumerate(row.truth) if index not in matched)
            case "ChokePoint":
                points = row.truth_points if len(baseline[row.frame]) < row.truth_count else ()
            case unreachable:
                assert_never(unreachable)
        counts.update(_cell_id(x, y, row.width, row.height) for x, y in points)
    return tuple(cell for cell, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4])


def _filter(rows: Sequence[FrameObservation], cells: frozenset[str], outside: float, focus: float) -> dict[int, tuple[Prediction, ...]]:
    output: dict[int, tuple[Prediction, ...]] = {}
    for row in rows:
        output[row.frame] = tuple(
            pred for pred in row.predictions
            if pred.confidence >= (focus if _cell_id((pred.box.x1 + pred.box.x2) / 2, (pred.box.y1 + pred.box.y2) / 2, row.width, row.height) in cells else outside)
        )
    return output


def _metrics(rows: Sequence[FrameObservation], predictions: dict[int, tuple[Prediction, ...]]) -> dict[str, float]:
    if rows[0].dataset == "VIRAT":
        metrics = _arm_metrics([row.frame for row in rows], predictions, [truth for row in rows for truth in row.truth])
        return {"mae": metrics["mae"], "precision": metrics["precision"]}
    counts = [(len(predictions[row.frame]), row.truth_count) for row in rows]
    mae = sum(abs(predicted - truth) for predicted, truth in counts) / len(counts)
    precision = sum(1.0 if predicted == truth == 0 else min(predicted, truth) / predicted if predicted else 0.0 for predicted, truth in counts) / len(counts)
    return {"mae": mae, "precision": precision}


def _gate(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float | bool]:
    gate = h1_pass(compiled_mae=candidate["mae"], b11_mae=baseline["mae"], precision_change_vs_b11=candidate["precision"] - baseline["precision"], compiled_proxy=0.25, compiled_time_ms=0.0, b26_time_ms=1.0)
    zero_baseline_noninferior = baseline["mae"] > 0.0 or candidate["mae"] == 0.0
    return {
        "passed": bool(gate["passed"]) and zero_baseline_noninferior,
        "relative_mae_change_vs_b11": float(gate["mae_change_vs_b11"]),
        "precision_change_vs_b11": float(gate["precision_change_vs_b11"]),
        "zero_baseline_mae_noninferior": zero_baseline_noninferior,
    }


def _score_segment(rows: Sequence[FrameObservation], cells: frozenset[str], outside: float, focus: float) -> dict[str, float | bool]:
    baseline = _metrics(rows, _baseline(rows))
    candidate = _metrics(rows, _filter(rows, cells, outside, focus))
    return {**candidate, "baseline_mae": baseline["mae"], "baseline_precision": baseline["precision"], **_gate(candidate, baseline)}


def _candidate_rows(by_segment: dict[str, tuple[FrameObservation, ...]], history_ids: tuple[str, ...], cells: frozenset[str]) -> list[dict[str, Any]]:
    output = []
    for outside in OUTSIDE_CONFS:
        for focus in FOCUS_CONFS:
            history = {segment: _score_segment(by_segment[segment], cells, outside, focus) for segment in history_ids}
            history_pass = all(bool(score["passed"]) for score in history.values())
            output.append({
                "name": f"o{round(outside * 100):02d}_f{round(focus * 100):02d}", "outside_conf": outside, "focus_conf": focus,
                "threshold_change": abs(outside - BASELINE_CONF) + abs(focus - BASELINE_CONF), "history": history, "all_history_pass": history_pass,
                "history_mae_mean": sum(float(score["mae"]) for score in history.values()) / len(history),
                "history_precision_mean": sum(float(score["precision"]) for score in history.values()) / len(history),
            })
    return output


def _select(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in candidates if bool(row["all_history_pass"])]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row["history_mae_mean"]), -float(row["history_precision_mean"]), float(row["threshold_change"]), str(row["name"])))


def _summary(scenes: dict[str, dict[str, Any]], horizon: str) -> dict[str, Any]:
    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
        selected = [row[horizon] for row in rows if row[horizon]["selected"] is not None]
        return {
            "scene_count": len(rows),
            "history_selected_scene_count": len(selected),
            "stable_future_pass_scene_count": sum(bool(row["selected_future_pass"]) for row in selected),
            "false_activation_scene_count": sum(not bool(row["selected_future_pass"]) for row in selected),
            "oracle_intersection_scene_count": sum(bool(row[horizon]["oracle_intersection"]) for row in rows),
        }

    scene_rows = list(scenes.values())
    datasets = sorted({str(row["dataset"]) for row in scene_rows})
    return {**aggregate(scene_rows), "by_dataset": {dataset: aggregate([row for row in scene_rows if row["dataset"] == dataset]) for dataset in datasets}}


def run_experiment(observations: Sequence[FrameObservation]) -> dict[str, Any]:
    grouped: dict[str, list[FrameObservation]] = defaultdict(list)
    for row in observations:
        grouped[row.scene_id].append(row)
    scenes: dict[str, dict[str, Any]] = {}
    for scene_id, rows in sorted(grouped.items()):
        by_segment = {segment: tuple(row for row in rows if row.segment_id == segment) for segment in (*HORIZONS["T4"], *FUTURE_SEGMENTS)}
        if any(not segment_rows for segment_rows in by_segment.values()):
            raise ExperimentInputError(f"incomplete_scene_segments:{scene_id}")
        scene: dict[str, Any] = {
            "dataset": rows[0].dataset,
            "metric_family": "IOU_BOX_PRECISION" if rows[0].dataset == "VIRAT" else "COUNT_PRECISION",
            "metric_type": "BOX_MAE_AND_IOU_PRECISION" if rows[0].dataset == "VIRAT" else "COUNT_MAE_AND_COUNT_PRECISION",
            "mean_yolo11s_wall_ms": sum(row.latency_ms for row in rows) / len(rows),
        }
        for horizon, history_ids in HORIZONS.items():
            history_rows = tuple(row for segment in history_ids for row in by_segment[segment])
            cells = focus_cell_ids(history_rows)
            history_candidates = _candidate_rows(by_segment, history_ids, frozenset(cells))
            selected = _select(history_candidates)
            candidates = []
            for row in history_candidates:
                future = {
                    segment: _score_segment(by_segment[segment], frozenset(cells), float(row["outside_conf"]), float(row["focus_conf"]))
                    for segment in FUTURE_SEGMENTS
                }
                candidates.append({**row, "future": future, "all_future_pass": all(bool(score["passed"]) for score in future.values())})
            history_names = {str(row["name"]) for row in history_candidates if row["all_history_pass"]}
            future_names = {str(row["name"]) for row in candidates if row["all_future_pass"]}
            selected_future = next((row["future"] for row in candidates if selected is not None and row["name"] == selected["name"]), None)
            scene[horizon] = {
                "history_segment_ids": list(history_ids),
                "future_segment_ids": list(FUTURE_SEGMENTS),
                "focus_cell_ids": list(cells),
                "candidates": candidates,
                "history_eligible": sorted(history_names),
                "selected": selected,
                "selected_future": selected_future,
                "selected_future_pass": bool(selected and selected["name"] in future_names),
                "future_oracle": sorted(future_names),
                "oracle_intersection": sorted(history_names & future_names),
            }
        scenes[scene_id] = scene
    return {
        "schema_version": "fieldpack.expanded-history-stability.v1", "evidence_status": "FORMAL_INTERNAL_RESEARCH",
        "accuracy_gate_only_same_inference_cost": True, "future_used_for_selection": False, "model_calls_per_frame": 1, "precision_aggregation": "WITHIN_SCENE_AND_DATASET_ONLY",
        "candidate_count": len(OUTSIDE_CONFS) * len(FOCUS_CONFS), "scenes": scenes,
        "summary": {horizon: _summary(scenes, horizon) for horizon in HORIZONS},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_rows(manifest: Path, root: Path, model_path: Path, device: str) -> tuple[FrameObservation, ...]:
    from ultralytics import YOLO

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "fieldpack.prepared-fixed-camera.v1":
        raise ExperimentInputError("unexpected_prepared_manifest_schema")
    model = YOLO(str(model_path))
    output = []
    for scene in payload["scenes"]:
        scene_id = str(scene["scene_id"])
        dataset = str(scene["dataset"])
        width, height = int(scene["width"]), int(scene["height"])
        if dataset not in ("VIRAT", "ChokePoint"):
            raise ExperimentInputError(f"unsupported_dataset:{dataset}")
        if width <= 0 or height <= 0:
            raise ExperimentInputError(f"invalid_scene_dimensions:{scene_id}")
        truth_path = root / "scenes" / scene_id / "truth.json"
        if not truth_path.is_file() or _sha256(truth_path) != scene["truth_sha256"]:
            raise ExperimentInputError(f"truth_hash_mismatch:{scene_id}")
        truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
        if truth_payload.get("scene_id") != scene_id or truth_payload.get("dataset") != dataset:
            raise ExperimentInputError(f"scene_truth_identity_mismatch:{scene_id}")
        truth_by_frame = {int(row["frame_index"]): row for row in truth_payload["frames"]}
        prepared_indices = [int(row["frame_index"]) for row in scene["prepared_frames"]]
        if len(prepared_indices) != len(set(prepared_indices)) or set(prepared_indices) != set(truth_by_frame):
            raise ExperimentInputError(f"scene_truth_frame_set_mismatch:{scene_id}")
        for frame_row in scene["prepared_frames"]:
            frame = int(frame_row["frame_index"])
            truth_row = truth_by_frame.get(frame)
            if truth_row is None or truth_row["segment_id"] != frame_row["segment_id"] or truth_row["role"] != frame_row["role"]:
                raise ExperimentInputError(f"prepared_truth_alignment_mismatch:{scene_id}:{frame}")
            image_path = (root / str(frame_row["path"])).resolve()
            try:
                image_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ExperimentInputError(f"prepared_path_outside_root:{scene_id}:{frame}") from exc
            if not image_path.is_file() or _sha256(image_path) != frame_row["sha256"]:
                raise ExperimentInputError(f"prepared_frame_hash_mismatch:{scene_id}:{frame}")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            if (int(rgb.shape[1]), int(rgb.shape[0])) != (width, height):
                raise ExperimentInputError(f"prepared_frame_dimensions_mismatch:{scene_id}:{frame}")
            began = time.perf_counter()
            result = model.predict(source=[ultralytics_bgr(rgb)], device=device, imgsz=640, batch=1, verbose=False, conf=0.05)[0]
            elapsed = (time.perf_counter() - began) * 1000.0
            boxes = getattr(result, "boxes", None)
            predictions = () if boxes is None else tuple(
                Prediction(frame, Box(*map(float, coords)), float(score), "yolo11s")
                for coords, score, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist())
                if int(cls) == 0
            )
            truth_boxes = tuple(Box(float(box["bbox_xywh"][0]), float(box["bbox_xywh"][1]), float(box["bbox_xywh"][0] + box["bbox_xywh"][2]), float(box["bbox_xywh"][1] + box["bbox_xywh"][3])) for box in truth_row["truth_boxes"])
            if int(truth_row["truth_count"]) < 0 or (dataset == "VIRAT" and len(truth_boxes) != int(truth_row["truth_count"])) or (dataset == "ChokePoint" and truth_boxes):
                raise ExperimentInputError(f"truth_metric_contract_mismatch:{scene_id}:{frame}")
            truths = tuple(GroundTruth(frame, index + 1, box) for index, box in enumerate(truth_boxes))
            truth_points = tuple((float(point["x"]), float(point["y"])) for point in truth_row["truth_points"])
            output.append(FrameObservation(scene_id, dataset, str(frame_row["segment_id"]), frame, width, height, predictions, truths, int(truth_row["truth_count"]), elapsed, truth_points))
    return tuple(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--yolo11s", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    rows = _prepared_rows(args.prepared_manifest, args.prepared_root, args.yolo11s, args.device)
    if len({row.scene_id for row in rows}) != 8 or any(sum(row.scene_id == scene and row.segment_id == segment for row in rows) != 64 for scene in {row.scene_id for row in rows} for segment in (*HORIZONS["T4"], *FUTURE_SEGMENTS)):
        raise ExperimentInputError("formal_scope_requires_8_scenes_and_64_frames_per_segment")
    result = run_experiment(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

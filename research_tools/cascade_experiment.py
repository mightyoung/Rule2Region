"""Pure data and I/O helpers for the benefit-aware cascade experiment.

This module deliberately knows nothing about a detector implementation.  A detector
adapter returns :class:`Prediction` objects and the rest of the experiment can be
tested with small, deterministic fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True, order=True)
class Box:
    """An immutable xyxy box in global image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box coordinates must be ordered")

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def coords(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class GroundTruth:
    frame: int
    track_id: int
    box: Box
    visibility: float | None = None


@dataclass(frozen=True)
class Prediction:
    frame: int
    box: Box
    confidence: float = 1.0
    source: str = "detector"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class DetectorAdapter(Protocol):
    """Minimal detector boundary; implementations may return mappings or Predictions."""

    name: str

    def predict(self, frame: Any, *, frame_index: int) -> Sequence[Prediction]: ...


class CallableDetectorAdapter:
    def __init__(self, fn: Callable[..., Sequence[Prediction]], name: str = "callable") -> None:
        self._fn, self.name = fn, name

    def predict(self, frame: Any, *, frame_index: int) -> Sequence[Prediction]:
        return tuple(self._fn(frame, frame_index=frame_index))


class UltralyticsYOLOAdapter:
    """Lazy Ultralytics adapter. Importing this module never imports torch/ultralytics."""

    def __init__(self, model_path: str | Path, *, device: str | int = "cpu", name: str | None = None, imgsz: int | None = None, conf: float | None = None) -> None:
        if conf is not None and not 0 <= conf <= 1:
            raise ValueError("detector_conf_must_be_between_zero_and_one")
        self.model_path, self.device, self.imgsz, self.conf = str(model_path), device, imgsz, conf
        self.name = name or Path(model_path).stem
        self._model = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore
            except ImportError as exc:
                raise RuntimeError("ultralytics_required_for_yolo_adapter") from exc
            self._model = YOLO(self.model_path)
        return self._model

    def predict(self, frame: Any, *, frame_index: int) -> Sequence[Prediction]:
        arguments = {"device": self.device, "verbose": False}
        if self.imgsz is not None:
            arguments["imgsz"] = self.imgsz
        if self.conf is not None:
            arguments["conf"] = self.conf
        result = self._load()(frame, **arguments)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()
        xyxy = boxes.xyxy.detach().cpu().tolist(); conf = boxes.conf.detach().cpu().tolist(); classes = boxes.cls.detach().cpu().tolist()
        return tuple(Prediction(frame_index, Box(*map(float, coords)), float(score), self.name) for coords, score, cls in zip(xyxy, conf, classes) if int(cls) == 0)

    def predict_batch(self, frames: Sequence[Any], *, frame_indices: Sequence[int]) -> tuple[tuple[Prediction, ...], ...]:
        if len(frames) != len(frame_indices):
            raise ValueError("batch_frames_and_indices_length_mismatch")
        if not frames:
            return ()
        arguments = {"device": self.device, "verbose": False}
        if self.imgsz is not None:
            arguments["imgsz"] = self.imgsz
        if self.conf is not None:
            arguments["conf"] = self.conf
        results = self._load()(list(frames), **arguments)
        output: list[tuple[Prediction, ...]] = []
        for result, frame_index in zip(results, frame_indices):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                output.append(())
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            conf = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            output.append(
                tuple(
                    Prediction(frame_index, Box(*map(float, coords)), float(score), self.name)
                    for coords, score, cls in zip(xyxy, conf, classes)
                    if int(cls) == 0
                )
            )
        return tuple(output)


def candidate_rois(width: int, height: int, *, cell_width: int = 320, cell_height: int | None = None, overlap: float = 0.2, expansion: float = 0.15) -> tuple[Any, ...]:
    """Create stable overlapping ROI cells; geometry import is lazy to preserve isolation."""
    if width < 1 or height < 1 or cell_width < 1 or not 0 <= overlap < 1:
        raise ValueError("invalid_roi_grid")
    cell_height = cell_height or cell_width; sx, sy = max(1, int(cell_width * (1 - overlap))), max(1, int(cell_height * (1 - overlap)))
    from evidencegate.cascade import ROI
    rois = []
    for y in range(0, height, sy):
        for x in range(0, width, sx):
            rois.append(ROI(x, y, min(width, x + cell_width), min(height, y + cell_height), f"r{len(rois):04d}", expansion))
            if x + cell_width >= width: break
        if y + cell_height >= height: break
    return tuple(rois)


def router_features(boxes: Sequence[Prediction], roi: Any) -> tuple[float, ...]:
    """Cheap, detector-only features used by the benefit router."""
    inside = [b for b in boxes if roi.x1 <= (b.box.x1 + b.box.x2) / 2 <= roi.x2 and roi.y1 <= (b.box.y1 + b.box.y2) / 2 <= roi.y2]
    area = max(1.0, (roi.x2 - roi.x1) * (roi.y2 - roi.y1)); small = sum(b.box.area <= 32 * 32 for b in inside)
    boundary = sum(min((b.box.x1 - roi.x1), (roi.x2 - b.box.x2), (b.box.y1 - roi.y1), (roi.y2 - b.box.y2)) < 10 for b in inside)
    return (len(inside) / area, sum(b.confidence for b in inside) / len(inside) if inside else 0.0, small / len(inside) if inside else 0.0, boundary / len(inside) if inside else 0.0)


def _nms(boxes: Sequence[Any], threshold: float = 0.5) -> list[Any]:
    from evidencegate.cascade import iou
    # A crop's context detections are not primary counting claims.  Keep one
    # only when a different source/ROI corroborates it; and never suppress two
    # detections produced by the same ROI (crowded local people).
    corroborated = []
    for box in boxes:
        if getattr(box, "primary", True):
            corroborated.append(box)
            continue
        if any(other is not box and other.source != box.source and other.roi_id != box.roi_id and iou(box, other) >= threshold for other in boxes):
            corroborated.append(box)
    result = []
    for box in sorted(corroborated, key=lambda b: (-b.score, b.x1, b.y1)):
        if all(box.roi_id == kept.roi_id or iou(box, kept) < threshold for kept in result): result.append(box)
    return result


def run_cascade_frame(frame: Any, frame_index: int, global_detector: DetectorAdapter, local_detector: DetectorAdapter | None = None, *, rois: Sequence[Any] = (), selected_rois: Sequence[Any] = (), condition: str = "yolo11s_global", nms_threshold: float = 0.5) -> tuple[tuple[Prediction, ...], dict[str, float]]:
    """Run one condition without prescribing an image backend or detector package."""
    from evidencegate.cascade import Box as CBox, boundary_aware_wbf, map_local_box_to_global, primary_ownership
    global_predictions = tuple(global_detector.predict(frame, frame_index=frame_index))
    if condition == "yolo11s_global" or local_detector is None and condition != "yolo11m_global":
        return global_predictions, {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}
    if condition == "yolo11m_global":
        return tuple(local_detector.predict(frame, frame_index=frame_index)), {"processed_pixel_ratio": 1.0, "extra_roi_pixel_ratio": 0.0, "roi_count": 0.0}
    selected = tuple(selected_rois)
    local_boxes: list[Any] = []
    processed_pixels = 0.0
    height, width = getattr(frame, "shape", (0, 0))[:2]
    for roi in selected:
        ex = roi.expanded
        crop_x1, crop_y1 = max(0, int(ex[0])), max(0, int(ex[1]))
        crop_x2, crop_y2 = min(width, int(ex[2])), min(height, int(ex[3]))
        processed_pixels += max(0, crop_x2 - crop_x1) * max(0, crop_y2 - crop_y1)
        try: crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        except (TypeError, IndexError): crop = frame
        for pred in local_detector.predict(crop, frame_index=frame_index):
            local_boxes.append(CBox(
                pred.box.x1 + crop_x1, pred.box.y1 + crop_y1,
                pred.box.x2 + crop_x1, pred.box.y2 + crop_y1,
                score=pred.confidence, source=pred.source, roi_id=roi.roi_id,
            ))
    global_boxes = [CBox(*p.box.coords, score=p.confidence, source=p.source) for p in global_predictions]
    owned = primary_ownership(local_boxes, selected)
    combined = global_boxes + owned
    fused = boundary_aware_wbf(combined, selected) if condition == "benefit_wbf" else _nms(combined, nms_threshold)
    extra_ratio = processed_pixels / max(1, height * width)
    return tuple(Prediction(frame_index, Box(b.x1, b.y1, b.x2, b.y2), b.score, b.source) for b in fused), {"processed_pixel_ratio": 1.0 + extra_ratio, "extra_roi_pixel_ratio": extra_ratio, "roi_count": float(len(selected))}


def _parse_row(row: str, line_number: int) -> GroundTruth | None:
    fields = row.split(",")
    if len(fields) < 7:
        raise ValueError(f"MOT17 row {line_number} has fewer than 7 fields")
    try:
        frame, track_id = int(fields[0]), int(fields[1])
        x, y, width, height = (float(v) for v in fields[2:6])
        confidence = float(fields[6])
        visibility = float(fields[8]) if len(fields) > 8 and fields[8] else None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MOT17 row {line_number}") from exc
    # MOTChallenge GT uses class 1 for pedestrians; confidence 0 marks ignored rows.
    cls = int(float(fields[7])) if len(fields) > 7 and fields[7] else 1
    if cls != 1 or confidence <= 0:
        return None
    if frame < 1 or track_id < 1 or width < 0 or height < 0:
        raise ValueError(f"invalid MOT17 geometry at row {line_number}")
    return GroundTruth(frame, track_id, Box(x, y, x + width, y + height), visibility)


def parse_mot17_gt(source: str | Path | Iterable[str]) -> tuple[GroundTruth, ...]:
    """Parse ``gt/gt.txt`` into sorted pedestrian records, excluding ignored classes."""
    if isinstance(source, Path):
        lines = source.read_text(encoding="utf-8").splitlines()
    elif isinstance(source, str):
        # Accept both inline fixture text and a convenient string path.
        candidate = Path(source)
        try:
            is_path = candidate.exists()
        except OSError:
            is_path = False
        lines = candidate.read_text(encoding="utf-8").splitlines() if is_path else source.splitlines()
    else:
        lines = source
    records = []
    for line_number, row in enumerate(lines, 1):
        row = row.strip()
        if row and not row.startswith("#"):
            parsed = _parse_row(row, line_number)
            if parsed is not None:
                records.append(parsed)
    return tuple(sorted(records, key=lambda item: (item.frame, item.track_id)))


def select_frames(frames: Iterable[int], sample_size: int, *, seed: int = 0) -> tuple[int, ...]:
    """Select a reproducible bounded sample without modifying the input."""
    candidates = sorted(set(int(frame) for frame in frames))
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size >= len(candidates):
        return tuple(candidates)
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(candidates, sample_size)))


def select_frame_indices(start: int, stop: int, sample_size: int, *, seed: int = 0) -> tuple[int, ...]:
    if start < 1 or stop < start:
        raise ValueError("invalid frame range")
    return select_frames(range(start, stop + 1), sample_size, seed=seed)


def _match(predictions: Sequence[Prediction], truth: Sequence[GroundTruth], iou_threshold: float) -> tuple[int, int, int, set[int], set[int]]:
    available = set(range(len(truth))); matched_pred: set[int] = set(); matched_truth: set[int] = set()
    for pi, pred in sorted(enumerate(predictions), key=lambda item: (-item[1].confidence, item[0])):
        candidates = [(pred.box.iou(gt.box), ti) for ti, gt in enumerate(truth) if ti in available and gt.frame == pred.frame]
        if candidates:
            overlap, ti = max(candidates, key=lambda item: (item[0], -item[1]))
            if overlap >= iou_threshold:
                available.remove(ti); matched_pred.add(pi); matched_truth.add(ti)
    return len(matched_pred), len(predictions) - len(matched_pred), len(truth) - len(matched_truth), matched_pred, matched_truth


def compute_metrics(predictions: Iterable[Prediction], truth: Iterable[GroundTruth], *, iou_threshold: float = 0.5, small_area: float = 32.0 * 32.0) -> dict[str, float]:
    preds, gts = tuple(predictions), tuple(truth)
    tp, fp, fn, matched_pred, matched_truth = _match(preds, gts, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    frames = sorted({p.frame for p in preds} | {g.frame for g in gts})
    count_errors = [sum(p.frame == frame for p in preds) - sum(g.frame == frame for g in gts) for frame in frames]
    small_truth = {i for i, gt in enumerate(gts) if gt.box.area <= small_area}
    small_recalled = len(small_truth & matched_truth)
    small_recall = small_recalled / len(small_truth) if small_truth else 0.0
    # Deterministic AP50-like ranking: area under the precision/recall envelope.
    ranked = sorted(enumerate(preds), key=lambda item: (-item[1].confidence, item[0]))
    used: set[int] = set(); tp_rank: list[int] = []
    for _, pred in ranked:
        choices = [(pred.box.iou(gt.box), ti) for ti, gt in enumerate(gts) if ti not in used and gt.frame == pred.frame]
        hit = bool(choices) and max(choices)[0] >= iou_threshold
        if hit: used.add(max(choices)[1])
        tp_rank.append(int(hit))
    ap = 0.0
    if gts:
        cum = 0
        for rank, hit in enumerate(tp_rank, 1):
            cum += hit
            ap += (cum / rank) * hit
        ap /= len(gts)
    return {"precision": precision, "recall": recall, "f1": f1, "ap50": ap, "count_mae": sum(abs(e) for e in count_errors) / len(count_errors) if count_errors else 0.0, "small_object_recall": small_recall}


def immutable_json_write(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write a canonical receipt once; existing artifacts are never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable output already exists: {target}")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    target.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


# Descriptive aliases kept small so callers can use terminology from the protocol.
parse_mot17_ground_truth = parse_mot17_gt
select_frame_sample = select_frames
write_immutable_json = immutable_json_write
evaluate_predictions = compute_metrics


def fit_benefit_router(features: Sequence[Sequence[float]], targets: Sequence[float], *, calibration_features: Sequence[Sequence[float]] | None = None, calibration_targets: Sequence[float] | None = None) -> Any:
    """Fit the repository's deterministic router without importing it at module load."""
    from evidencegate.cascade import BenefitRouter
    router = BenefitRouter().fit(features, targets)
    if calibration_features is not None and calibration_targets is not None:
        router.calibrate(calibration_features, calibration_targets)
    return router


def save_router(path: str | Path, router: Any) -> str:
    weights = getattr(router, "weights", None)
    if weights is None:
        raise ValueError("router_not_fitted")
    payload = {"schema_version": "cascade-experiment.router.v2", "weights": [float(v) for v in weights], "calibration": [float(v) for v in router.calibration], "decision_threshold": float(getattr(router, "decision_threshold", 0.5))}
    return immutable_json_write(path, payload)


def load_router(path: str | Path) -> Any:
    from evidencegate.cascade import BenefitRouter
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    router = BenefitRouter(); import numpy as np
    router.weights = np.asarray(payload["weights"], dtype=float); router.calibration = tuple(payload.get("calibration", (1.0, 0.0))); router.decision_threshold = float(payload.get("decision_threshold", 0.5))
    return router


@dataclass(frozen=True)
class PredictionReceipt:
    schema_version: str
    sequence_id: str
    condition: str
    predictions: tuple[Prediction, ...] = field(default_factory=tuple)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "sequence_id": self.sequence_id, "condition": self.condition, "predictions": [{"frame": p.frame, "box": asdict(p.box), "confidence": p.confidence, "source": p.source} for p in self.predictions], "metrics": dict(self.metrics)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse MOT17 GT and emit an immutable cascade input receipt")
    parser.add_argument("--gt", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    gt = parse_mot17_gt(args.gt); frames = select_frames((r.frame for r in gt), args.sample_size, seed=args.seed) if args.sample_size else tuple(sorted({r.frame for r in gt}))
    payload = {"schema_version": "cascade-experiment.input.v1", "gt_records": len(gt), "frames": frames, "seed": args.seed, "source": str(args.gt)}
    digest = immutable_json_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "sha256": digest, "frames": len(frames)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

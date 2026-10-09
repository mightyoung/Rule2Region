"""Residual commissioning arm R2: frozen causal query on B0 dropouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.residual_compiler import PRIMARY_EVAL, apply_reject, box_height
from research_tools.route_a_clip_commissioning import load_clips, score_condition


RULE_ID = "residual.query.v1"
SUPPORT_IOU = 0.30
EXPAND = 0.50
MIN_SIDE = 64.0
MAX_QUERIES_PER_FRAME = 2
NMS_IOU = 0.50
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
QUERY_CONF = 0.25
QUERY_IMGSZ = 640
SOURCE = "yolo26_query"


DetectFn = Callable[[int, Box, Any], Sequence[Prediction]]


def box_center(box: Box) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


def expand_roi(
    box: Box,
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    expand: float = EXPAND,
    min_side: float = MIN_SIDE,
) -> Box:
    box_w = max(1.0, box.x2 - box.x1)
    box_h = max(1.0, box.y2 - box.y1)
    target_w = min(float(width), max(box_w * (1.0 + 2.0 * expand), min_side))
    target_h = min(float(height), max(box_h * (1.0 + 2.0 * expand), min_side))
    cx, cy = box_center(box)
    x1, y1 = cx - target_w / 2.0, cy - target_h / 2.0
    x2, y2 = x1 + target_w, y1 + target_h
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > width:
        x1 -= x2 - width
        x2 = float(width)
    if y2 > height:
        y1 -= y2 - height
        y2 = float(height)
    return Box(max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))


def unmatched_previous(
    previous: Sequence[Prediction],
    current: Sequence[Prediction],
    *,
    iou: float = SUPPORT_IOU,
) -> tuple[Prediction, ...]:
    return tuple(
        row for row in previous if all(row.box.iou(other.box) < iou for other in current)
    )


def propose_queries(
    previous: Sequence[Prediction],
    current: Sequence[Prediction],
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    max_queries: int = MAX_QUERIES_PER_FRAME,
) -> tuple[Box, ...]:
    seeds = unmatched_previous(previous, current)
    ranked = sorted(seeds, key=lambda row: (-row.confidence, -box_height(row.box), row.box.coords))
    return tuple(expand_roi(row.box, width=width, height=height) for row in ranked[:max_queries])


def nms(rows: Sequence[Prediction], *, iou: float = NMS_IOU) -> tuple[Prediction, ...]:
    kept: list[Prediction] = []
    for row in sorted(rows, key=lambda item: (-item.confidence, item.source, item.box.coords)):
        if all(row.box.iou(existing.box) < iou for existing in kept):
            kept.append(row)
    return tuple(kept)


def map_crop_box(local: Box, roi: Box) -> Box:
    return Box(local.x1 + roi.x1, local.y1 + roi.y1, local.x2 + roi.x1, local.y2 + roi.y1)


def keep_owned(pred: Prediction, roi: Box) -> bool:
    x, y = box_center(pred.box)
    return roi.x1 <= x < roi.x2 and roi.y1 <= y < roi.y2


def apply_queries(
    by_frame: Mapping[int, Sequence[Prediction]],
    frames: Sequence[int],
    validation_frames: Sequence[int],
    detect: DetectFn,
    images: Mapping[int, Any],
) -> tuple[dict[int, tuple[Prediction, ...]], dict[str, int]]:
    val_set = set(validation_frames)
    output: dict[int, tuple[Prediction, ...]] = {}
    query_count = 0
    added = 0
    for index, frame in enumerate(frames):
        current = tuple(by_frame.get(frame, ()))
        if frame not in val_set or index == 0:
            output[frame] = current
            continue
        previous = tuple(by_frame.get(frames[index - 1], ()))
        rois = propose_queries(previous, current)
        extras: list[Prediction] = []
        image = images[frame]
        for roi in rois:
            query_count += 1
            extras.extend(row for row in detect(frame, roi, image) if keep_owned(row, roi))
        added += len(extras)
        output[frame] = nms((*current, *extras)) if extras else current
    return output, {"val_queries": query_count, "val_boxes_added": added}


def _load_bgr(path: Path) -> Any:
    import numpy as np

    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb, np.ascontiguousarray(rgb[..., ::-1])


def _crop_bgr(bgr: Any, roi: Box) -> Any:
    return bgr[int(roi.y1) : int(roi.y2), int(roi.x1) : int(roi.x2)]


def _yolo26_detect_fn(model_path: Path, device: str) -> DetectFn:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))

    def detect(frame_id: int, roi: Box, image: Any) -> tuple[Prediction, ...]:
        crop = _crop_bgr(image, roi)
        if crop.size == 0:
            return ()
        result = model.predict(
            source=crop,
            device=device,
            imgsz=QUERY_IMGSZ,
            verbose=False,
            conf=QUERY_CONF,
            end2end=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return ()
        mapped = []
        for coords, score, cls in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            if int(cls) != 0:
                continue
            mapped.append(
                Prediction(frame_id, map_crop_box(Box(*map(float, coords)), roi), float(score), SOURCE)
            )
        return tuple(mapped)

    return detect


def _score_clip(
    clip: Mapping[str, Any],
    compiled: Mapping[int, Sequence[Prediction]],
    *,
    query_count: int,
) -> dict[str, Any]:
    val = tuple(clip["validation_frames"])
    val_only = {frame: tuple(compiled.get(frame, ())) for frame in val}
    ratio = query_count / (len(val) * 4.0) if val else 0.0
    scored = score_condition(clip, val_only, tile_cost_ratio=ratio)
    scored["val_queries"] = query_count
    scored["tile_cost_ratio"] = ratio
    return scored


def run(
    *,
    cache_root: Path,
    pilot_root: Path,
    output_path: Path,
    yolo26_path: Path | None = None,
    device: str = "0",
    detect: DetectFn | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refuse_existing_output:{output_path}")
    if detect is None:
        if yolo26_path is None:
            raise ValueError("yolo26_or_detect_required")
        detect = _yolo26_detect_fn(yolo26_path, device)
    clips = load_clips(cache_root, pilot_root)
    per_clip: dict[str, Any] = {}
    for sequence, clip in sorted(clips.items()):
        images: dict[int, Any] = {}
        for frame in clip["validation_frames"]:
            path = pilot_root / "sequences" / sequence / "img1" / f"{frame:06d}.jpg"
            _rgb, bgr = _load_bgr(path)
            images[frame] = bgr
        queried, query_stats = apply_queries(
            clip["global_by_frame"], clip["frames"], clip["validation_frames"], detect, images
        )
        rejected = apply_reject(clip["global_by_frame"], clip["frames"])
        combined, combined_stats = apply_queries(
            rejected, clip["frames"], clip["validation_frames"], detect, images
        )
        query_score = _score_clip(clip, queried, query_count=query_stats["val_queries"])
        combined_score = _score_clip(clip, combined, query_count=combined_stats["val_queries"])
        camera_clips = sorted(
            item["sequence_id"] for item in clips.values() if item["camera_id"] == clip["camera_id"]
        )
        per_clip[sequence] = {
            "camera_id": clip["camera_id"],
            "sequence_id": sequence,
            "role": "development_eval" if sequence == camera_clips[0] else "transfer",
            "rule_id": RULE_ID,
            "query": {**query_score, **query_stats},
            "combined_r1_then_query": {**combined_score, **combined_stats},
            "verdict": query_score["verdict"],
            "intention": query_score["intention"],
            "mae": query_score["mae"],
            "baseline_mae": query_score["baseline_mae"],
            "compiled_mae": query_score["compiled_mae"],
            "mae_relative_gain": query_score["mae_relative_gain"],
            "compiled_mae_relative_gain": query_score["compiled_mae_relative_gain"],
            "precision_change": query_score["precision_change"],
            "compiled_precision_change": query_score["compiled_precision_change"],
        }
    primary = [
        per_clip[PRIMARY_EVAL["VIRAT_S_010004"]],
        per_clip[PRIMARY_EVAL["VIRAT_S_010204"]],
    ]
    both_reject = all(row["verdict"] != "pass" for row in primary)
    result = {
        "schema_version": "sceneconfig.residual-query.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "arm": "R2_ledger_query",
        "rule_id": RULE_ID,
        "rule": {
            "support_iou": SUPPORT_IOU,
            "expand": EXPAND,
            "min_side": MIN_SIDE,
            "max_queries_per_frame": MAX_QUERIES_PER_FRAME,
            "nms_iou": NMS_IOU,
            "detector": "yolo26s",
            "imgsz": QUERY_IMGSZ,
            "conf": QUERY_CONF,
            "causal": True,
            "seeds_from": "b0_global_only",
        },
        "qwen_called": False,
        "query_run": True,
        "r1_thresholds_changed": False,
        "holdout_opened": False,
        "confirm_opened": False,
        "sealed_camera_used": False,
        "clip_count": len(per_clip),
        "pass_count": sum(row["verdict"] == "pass" for row in per_clip.values()),
        "h1": {
            "primary_clips": list(PRIMARY_EVAL.values()),
            "both_legal_reject": both_reject,
            "passed": not both_reject and all(row["verdict"] == "pass" for row in primary),
            "reason": "both_primary_eval_clips_failed_or_held" if both_reject else "confirmation_closed",
            "scoring_arm": "QUERY_ONLY",
        },
        "per_clip": per_clip,
        "next_gate": "STOP_R2_LEGAL_REJECT" if both_reject else "OPTIONAL_R3_VLM_AFTER_RECEIPT",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo26", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run(
        cache_root=args.cache_root,
        pilot_root=args.pilot_root,
        output_path=args.output,
        yolo26_path=args.yolo26,
        device=args.device,
    )
    print(json.dumps({"pass_count": result["pass_count"], "h1": result["h1"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

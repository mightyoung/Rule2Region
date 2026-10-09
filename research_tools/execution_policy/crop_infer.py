"""Crop each count region and run only that region's model."""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.card import FRAME_HEIGHT, FRAME_WIDTH, PolicyCard, RegionCard
from research_tools.execution_policy.compile import NMS_IOU, PartitionedCount
from research_tools.residual_query import keep_owned, map_crop_box, nms


IMGSZ = 640
MIN_SIDE = 2
MOSAIC_PAD = 8
MOSAIC_WRAP = 640


def load_bgr(path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def crop_region(bgr: np.ndarray, box: Box) -> tuple[np.ndarray, Box] | None:
    height, width = bgr.shape[:2]
    x1 = max(0, min(width, int(np.floor(box.x1))))
    y1 = max(0, min(height, int(np.floor(box.y1))))
    x2 = max(0, min(width, int(np.ceil(box.x2))))
    y2 = max(0, min(height, int(np.ceil(box.y2))))
    if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
        return None
    return bgr[y1:y2, x1:x2], Box(float(x1), float(y1), float(x2), float(y2))


def _clip_mapped(box: Box) -> Box | None:
    x1 = max(0.0, min(FRAME_WIDTH, box.x1))
    y1 = max(0.0, min(FRAME_HEIGHT, box.y1))
    x2 = max(0.0, min(FRAME_WIDTH, box.x2))
    y2 = max(0.0, min(FRAME_HEIGHT, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x1, y1, x2, y2)


def adaptive_imgsz(box: Box, *, min_size: int = 32, max_size: int = IMGSZ, multiple: int = 32) -> int:
    side = max(box.x2 - box.x1, box.y2 - box.y1)
    raw = int(math.ceil(max(side, float(min_size)) / multiple) * multiple)
    return max(min_size, min(max_size, raw))


def pack_mosaic(bgr: np.ndarray, regions: Sequence[RegionCard]) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    chips: list[tuple[RegionCard, np.ndarray, Box]] = []
    for region in regions:
        if region.role != "count":
            continue
        cropped = crop_region(bgr, region.box)
        if cropped is None:
            continue
        crop, roi = cropped
        chips.append((region, crop, roi))
    if not chips:
        return None
    cursor_x = MOSAIC_PAD
    cursor_y = MOSAIC_PAD
    row_h = 0
    width = MOSAIC_PAD
    height = MOSAIC_PAD
    layout: list[tuple[RegionCard, np.ndarray, Box, int, int]] = []
    for region, crop, roi in chips:
        ch, cw = crop.shape[0], crop.shape[1]
        if cursor_x + cw + MOSAIC_PAD > MOSAIC_WRAP and cursor_x > MOSAIC_PAD:
            cursor_x = MOSAIC_PAD
            cursor_y += row_h + MOSAIC_PAD
            row_h = 0
        layout.append((region, crop, roi, cursor_x, cursor_y))
        row_h = max(row_h, ch)
        cursor_x += cw + MOSAIC_PAD
        width = max(width, cursor_x)
        height = max(height, cursor_y + ch + MOSAIC_PAD)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    placements: list[dict[str, Any]] = []
    for region, crop, roi, dx, dy in layout:
        ch, cw = crop.shape[0], crop.shape[1]
        canvas[dy : dy + ch, dx : dx + cw] = crop
        placements.append({"region": region, "roi": roi, "dx": dx, "dy": dy, "cw": cw, "ch": ch})
    return canvas, placements


def mosaic_to_preds(
    detections: Sequence[tuple[Box, float]],
    placements: Sequence[Mapping[str, Any]],
    frame: int,
    *,
    conf: float,
    source: str,
) -> list[Prediction]:
    rows: list[Prediction] = []
    for local, score in detections:
        if score < conf:
            continue
        cx = (local.x1 + local.x2) / 2
        cy = (local.y1 + local.y2) / 2
        hit = next(
            (
                item
                for item in placements
                if item["dx"] <= cx < item["dx"] + item["cw"] and item["dy"] <= cy < item["dy"] + item["ch"]
            ),
            None,
        )
        if hit is None:
            continue
        chip_local = Box(local.x1 - hit["dx"], local.y1 - hit["dy"], local.x2 - hit["dx"], local.y2 - hit["dy"])
        mapped = _clip_mapped(map_crop_box(chip_local, hit["roi"]))
        if mapped is None:
            continue
        pred = Prediction(frame, mapped, score, source)
        if keep_owned(pred, hit["region"].box):
            rows.append(pred)
    return rows


def exclusive_mix(
    full_11s: Sequence[Prediction],
    crop_26: Sequence[Prediction],
    regions: Sequence[RegionCard],
) -> PartitionedCount:
    foci = [row.box for row in regions if row.role == "count" and row.model == "yolo26"]
    kept_11: list[Prediction] = []
    for pred in full_11s:
        if any(keep_owned(pred, box) for box in foci):
            continue
        kept_11.append(pred)
    return PartitionedCount(nms(list(kept_11), iou=NMS_IOU), nms(list(crop_26), iou=NMS_IOU))


def predict_image(
    model: Any,
    image: np.ndarray,
    *,
    conf: float,
    device: str,
    imgsz: int = IMGSZ,
) -> tuple[list[tuple[Box, float]], float]:
    started = time.perf_counter()
    result = model.predict(
        source=image,
        device=device,
        imgsz=int(imgsz),
        verbose=False,
        conf=conf,
        end2end=False,
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], elapsed_ms
    rows: list[tuple[Box, float]] = []
    for coords, score, cls in zip(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().tolist(),
    ):
        if int(cls) != 0:
            continue
        rows.append((Box(*map(float, coords)), float(score)))
    return rows, elapsed_ms


def infer_card_frame(
    models: Mapping[str, Any],
    card: PolicyCard,
    bgr: np.ndarray,
    frame: int,
    *,
    device: str,
    imgsz_mode: str = "fixed",
) -> tuple[PartitionedCount, float]:
    kept_11: list[Prediction] = []
    kept_26: list[Prediction] = []
    time_ms = 0.0
    for region in card.regions:
        if region.role != "count":
            continue
        cropped = crop_region(bgr, region.box)
        if cropped is None:
            continue
        crop, roi = cropped
        model = models.get(region.model)
        if model is None:
            raise ValueError(f"model_not_loaded:{region.model}")
        imgsz = adaptive_imgsz(roi) if imgsz_mode == "adaptive" else IMGSZ
        detections, elapsed_ms = predict_image(model, crop, conf=region.conf, device=device, imgsz=imgsz)
        time_ms += elapsed_ms
        bucket = kept_26 if region.model == "yolo26" else kept_11
        source = "yolo26" if region.model == "yolo26" else "yolo11s"
        for local, score in detections:
            if score < region.conf:
                continue
            mapped = _clip_mapped(map_crop_box(local, roi))
            if mapped is None:
                continue
            pred = Prediction(frame, mapped, score, source)
            if keep_owned(pred, region.box):
                bucket.append(pred)
    return PartitionedCount(nms(kept_11, iou=NMS_IOU), nms(kept_26, iou=NMS_IOU)), time_ms


def infer_batched_yolo26_frame(
    model: Any,
    card: PolicyCard,
    bgr: np.ndarray,
    frame: int,
    *,
    device: str,
    imgsz: int = IMGSZ,
    conf: float | None = None,
) -> tuple[PartitionedCount, float]:
    """Run every count-region YOLO26 crop as one model-native batch."""
    batch: list[np.ndarray] = []
    regions: list[tuple[RegionCard, Box]] = []
    for region in card.regions:
        if region.role != "count" or region.model != "yolo26":
            continue
        cropped = crop_region(bgr, region.box)
        if cropped is None:
            continue
        crop, roi = cropped
        batch.append(crop)
        regions.append((region, roi))
    if not batch:
        return PartitionedCount((), ()), 0.0

    started = time.perf_counter()
    results = list(
        model.predict(
            source=batch,
            device=device,
            imgsz=int(imgsz),
            verbose=False,
            conf=float(conf) if conf is not None else min(region.conf for region, _ in regions),
            end2end=False,
        )
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if len(results) != len(regions):
        raise ValueError(f"batch_result_count:{len(results)}!={len(regions)}")

    kept_26: list[Prediction] = []
    for result, (region, roi) in zip(results, regions):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for coords, score, cls in zip(
            boxes.xyxy.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.cls.detach().cpu().tolist(),
        ):
            threshold = float(conf) if conf is not None else region.conf
            if int(cls) != 0 or float(score) < threshold:
                continue
            local = Box(*map(float, coords))
            mapped = _clip_mapped(map_crop_box(local, roi))
            if mapped is None:
                continue
            pred = Prediction(frame, mapped, float(score), "yolo26")
            if keep_owned(pred, region.box):
                kept_26.append(pred)
    return PartitionedCount((), nms(kept_26, iou=NMS_IOU)), elapsed_ms


def infer_full_frame(
    model: Any,
    bgr: np.ndarray,
    frame: int,
    *,
    conf: float,
    source: str,
    device: str,
    imgsz: int = IMGSZ,
) -> tuple[tuple[Prediction, ...], float]:
    detections, elapsed_ms = predict_image(model, bgr, conf=conf, device=device, imgsz=imgsz)
    rows = [Prediction(frame, box, score, source) for box, score in detections if score >= conf]
    return nms(rows, iou=NMS_IOU), elapsed_ms


def infer_mosaic_frame(
    model: Any,
    card: PolicyCard,
    bgr: np.ndarray,
    frame: int,
    *,
    device: str,
    conf: float,
    source: str = "yolo26",
) -> tuple[PartitionedCount, float]:
    packed = pack_mosaic(bgr, card.regions)
    if packed is None:
        return PartitionedCount((), ()), 0.0
    canvas, placements = packed
    detections, elapsed_ms = predict_image(model, canvas, conf=conf, device=device, imgsz=IMGSZ)
    rows = mosaic_to_preds(detections, placements, frame, conf=conf, source=source)
    if source == "yolo26":
        return PartitionedCount((), nms(rows, iou=NMS_IOU)), elapsed_ms
    return PartitionedCount(nms(rows, iou=NMS_IOU), ()), elapsed_ms

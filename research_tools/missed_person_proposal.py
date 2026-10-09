"""Label and lightweight baseline utilities for missed-person proposals.

Ground truth is used only to construct offline residual targets.  The online
proposal path must consume detector-derived features and must never call the
target builders in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from research_tools.cascade_experiment import Box


@dataclass(frozen=True)
class ResidualTargets:
    heatmap: np.ndarray
    scale: np.ndarray
    scale_mask: np.ndarray
    missed_boxes: tuple[Box, ...]


def missed_ground_truth(
    truth: Sequence[Box], predictions: Sequence[Box], iou_threshold: float = 0.5
) -> tuple[Box, ...]:
    """Return GT boxes not matched by any prediction at the frozen IoU gate."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold_out_of_range")
    return tuple(
        gt for gt in truth
        if not any(gt.iou(prediction) >= iou_threshold for prediction in predictions)
    )


def _draw_gaussian(heatmap: np.ndarray, cx: float, cy: float, sigma: float) -> None:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x0 = max(0, int(np.floor(cx)) - radius)
    x1 = min(heatmap.shape[1], int(np.floor(cx)) + radius + 1)
    y0 = max(0, int(np.floor(cy)) - radius)
    y1 = min(heatmap.shape[0], int(np.floor(cy)) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


def build_residual_targets(
    truth: Sequence[Box],
    predictions: Sequence[Box],
    *,
    image_width: int,
    image_height: int,
    output_width: int,
    output_height: int,
    iou_threshold: float = 0.5,
) -> ResidualTargets:
    """Build CenterNet-style heatmap and normalized size targets for missed GT."""
    if min(image_width, image_height, output_width, output_height) <= 0:
        raise ValueError("invalid_spatial_shape")
    missed = missed_ground_truth(truth, predictions, iou_threshold)
    heatmap = np.zeros((output_height, output_width), dtype=np.float32)
    scale = np.zeros((2, output_height, output_width), dtype=np.float32)
    scale_mask = np.zeros((output_height, output_width), dtype=bool)
    sx, sy = output_width / image_width, output_height / image_height
    for box in missed:
        cx = ((box.x1 + box.x2) / 2.0) * sx
        cy = ((box.y1 + box.y2) / 2.0) * sy
        object_w = max(1.0, (box.x2 - box.x1) * sx)
        object_h = max(1.0, (box.y2 - box.y1) * sy)
        _draw_gaussian(heatmap, cx, cy, max(1.0, min(object_w, object_h) / 6.0))
        ix = min(output_width - 1, max(0, int(round(cx))))
        iy = min(output_height - 1, max(0, int(round(cy))))
        scale[:, iy, ix] = (
            (box.x2 - box.x1) / image_width,
            (box.y2 - box.y1) / image_height,
        )
        scale_mask[iy, ix] = True
    return ResidualTargets(heatmap, scale, scale_mask, missed)


def audit_sequence_split(
    fit: Iterable[str], calibration: Iterable[str], evaluation: Iterable[str]
) -> dict[str, object]:
    """Fail closed when any sequence appears in more than one research split."""
    splits = {
        "fit": set(fit),
        "calibration": set(calibration),
        "evaluation": set(evaluation),
    }
    overlaps = {
        "fit_calibration": sorted(splits["fit"] & splits["calibration"]),
        "fit_evaluation": sorted(splits["fit"] & splits["evaluation"]),
        "calibration_evaluation": sorted(splits["calibration"] & splits["evaluation"]),
    }
    empty = [name for name, values in splits.items() if not values]
    return {
        "sequence_split_leakage": any(overlaps.values()),
        "overlaps": overlaps,
        "empty_splits": empty,
        "valid": not any(overlaps.values()) and not empty,
    }


def fit_pointwise_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    l2: float = 1e-3,
    positive_weight: float = 20.0,
) -> np.ndarray:
    """Fit a cheap pointwise residual-head baseline on precomputed channels.

    This is deliberately not the proposed FPN.  It is a pipeline sentinel used
    to prove sequence isolation, target construction, and metric computation
    before expensive YOLO feature hooks are added.
    """
    if features.ndim != 4 or targets.ndim != 3:
        raise ValueError("expected_nchw_features_and_nhw_targets")
    if features.shape[0] != targets.shape[0] or features.shape[2:] != targets.shape[1:]:
        raise ValueError("feature_target_shape_mismatch")
    if l2 < 0:
        raise ValueError("negative_l2")
    if positive_weight < 1:
        raise ValueError("positive_weight_below_one")
    x = np.moveaxis(features, 1, -1).reshape(-1, features.shape[1]).astype(np.float64)
    y = targets.reshape(-1).astype(np.float64)
    x = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    sample_weight = 1.0 + (positive_weight - 1.0) * (y >= 0.05)
    root_weight = np.sqrt(sample_weight)
    x_weighted = x * root_weight[:, None]
    y_weighted = y * root_weight
    penalty = np.eye(x.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    return np.linalg.solve(x_weighted.T @ x_weighted + penalty, x_weighted.T @ y_weighted)


def predict_pointwise_ridge(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.ndim != 4 or weights.shape != (features.shape[1] + 1,):
        raise ValueError("feature_weight_shape_mismatch")
    scores = np.einsum("nchw,c->nhw", features, weights[:-1]) + weights[-1]
    return np.clip(scores, 0.0, 1.0).astype(np.float32)

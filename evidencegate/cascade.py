"""Deterministic geometry and routing primitives for the global/local cascade.

The module deliberately has no detector dependencies.  Coordinates are always
pixel coordinates in ``x1, y1, x2, y2`` form and local detections are mapped
through the expanded crop while ownership is decided against its core.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    source: str = "global"
    roi_id: str | None = None
    primary: bool = True

    def __post_init__(self) -> None:
        vals = (self.x1, self.y1, self.x2, self.y2, self.score)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError("box_values_must_be_finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box_must_have_positive_area")
        if not 0 <= self.score <= 1:
            raise ValueError("score_must_be_between_zero_and_one")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source_required")

    @property
    def coords(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class ROI:
    """A crop with an unexpanded core and expanded context rectangle."""

    x1: float
    y1: float
    x2: float
    y2: float
    roi_id: str = "roi"
    expansion: float = 0.15

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("roi_must_have_positive_area")
        if self.expansion < 0 or not math.isfinite(self.expansion):
            raise ValueError("expansion_must_be_nonnegative")

    @property
    def core(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def expanded(self) -> tuple[float, float, float, float]:
        w, h = self.x2 - self.x1, self.y2 - self.y1
        dx, dy = w * self.expansion / 2, h * self.expansion / 2
        return self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy

    @property
    def pixel_area(self) -> float:
        x1, y1, x2, y2 = self.expanded
        return (x2 - x1) * (y2 - y1)

    def contains_center(self, box: Box) -> bool:
        cx, cy = box.center
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


def map_local_box_to_global(box: Box, roi: ROI) -> Box:
    """Translate crop-relative coordinates (whose origin is expanded crop)."""
    ox, oy, _, _ = roi.expanded
    return replace(box, x1=box.x1 + ox, x2=box.x2 + ox,
                   y1=box.y1 + oy, y2=box.y2 + oy, roi_id=roi.roi_id)


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (a.area + b.area - inter) if inter else 0.0


def primary_ownership(boxes: Iterable[Box], rois: Iterable[ROI]) -> list[Box]:
    """Annotate local boxes with their ROI; global boxes remain untouched.

    A local box is primary to a crop only when its center lies in the core.
    Context-only detections retain their ROI for matching and are marked
    ``primary=False``.
    """
    by_id = {r.roi_id: r for r in rois}
    result: list[Box] = []
    for box in boxes:
        if box.roi_id is None or box.roi_id not in by_id:
            result.append(box)
            continue
        roi = by_id[box.roi_id]
        result.append(box if roi.contains_center(box) else replace(box, primary=False))
    return result


def _boundary_weight(box: Box, roi: ROI | None) -> float:
    if roi is None or box.roi_id != roi.roi_id:
        return 1.0
    ex = roi.expanded
    cx, cy = box.center
    distances = (cx - ex[0], ex[2] - cx, cy - ex[1], ex[3] - cy)
    margin = max(1e-9, min(ex[2] - ex[0], ex[3] - ex[1]) * 0.2)
    return max(0.1, min(1.0, min(distances) / margin))


def boundary_aware_wbf(
    boxes: Sequence[Box], rois: Iterable[ROI] = (), iou_threshold: float = 0.55,
    center_distance_threshold: float = 0.75, area_ratio_range: tuple[float, float] = (0.4, 2.5),
) -> list[Box]:
    """Fuse only cross-source/cross-ROI duplicates; same-ROI boxes stay distinct."""
    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError("iou_threshold_must_be_between_zero_and_one")
    if center_distance_threshold <= 0 or area_ratio_range[0] <= 0 or area_ratio_range[1] < area_ratio_range[0]:
        raise ValueError("matching_guards_invalid")
    roi_map = {r.roi_id: r for r in rois}
    remaining = sorted(enumerate(boxes), key=lambda p: (-p[1].score, p[0]))
    output: list[Box] = []
    while remaining:
        _, seed = remaining.pop(0)
        group = [seed]
        keep: list[tuple[int, Box]] = []
        for idx, candidate in remaining:
            cx, cy = seed.center; dx, dy = candidate.center
            norm_dist = math.hypot(cx - dx, cy - dy) / max(math.hypot(seed.x2-seed.x1, seed.y2-seed.y1), math.hypot(candidate.x2-candidate.x1, candidate.y2-candidate.y1))
            ratio = candidate.area / seed.area
            compatible = (candidate.source != seed.source and candidate.roi_id != seed.roi_id and
                          iou(seed, candidate) >= iou_threshold and norm_dist <= center_distance_threshold and
                          area_ratio_range[0] <= ratio <= area_ratio_range[1])
            (group if compatible else keep).append(candidate if compatible else (idx, candidate))
        remaining = keep
        # A context-only local box has no primary ownership; it must not become
        # a count by itself.  It is retained only when another source/ROI
        # corroborates it and the group is fused below.
        if len(group) == 1 and not group[0].primary:
            continue
        if len(group) == 1:
            output.append(seed)
            continue
        weights = np.array([max(1e-9, b.score) * _boundary_weight(b, roi_map.get(b.roi_id)) for b in group])
        coords = np.average(np.array([b.coords for b in group]), axis=0, weights=weights)
        output.append(Box(*coords, score=float(max(b.score for b in group)), source="fused", roi_id=None))
    return output


def select_rois_by_budget(rois: Sequence[ROI], benefits: Sequence[float], pixel_budget: float) -> list[ROI]:
    """Greedily select highest benefit-per-expanded-pixel ROIs under budget."""
    if len(rois) != len(benefits) or pixel_budget < 0:
        raise ValueError("roi_benefits_and_budget_invalid")
    ranked = sorted(zip(rois, benefits), key=lambda rb: (-(rb[1] / rb[0].pixel_area if rb[0].pixel_area else 0), rb[0].roi_id))
    selected: list[ROI] = []
    spent = 0.0
    for roi, benefit in ranked:
        if benefit <= 0:
            continue
        if spent + roi.pixel_area <= pixel_budget + 1e-9:
            selected.append(roi); spent += roi.pixel_area
    return selected


class BenefitRouter:
    """Small deterministic logistic router with held-out Platt calibration."""
    def __init__(self, learning_rate: float = 0.2, epochs: int = 300, l2: float = 1e-3):
        self.learning_rate, self.epochs, self.l2 = learning_rate, epochs, l2
        self.weights: np.ndarray | None = None
        self.calibration: tuple[float, float] = (1.0, 0.0)
        self.decision_threshold: float = 0.5

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> "BenefitRouter":
        x = np.asarray(features, dtype=float); y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0: raise ValueError("training_data_invalid")
        x1 = np.c_[np.ones(len(x)), x]; w = np.zeros(x1.shape[1])
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-np.clip(x1 @ w, -40, 40)))
            w -= self.learning_rate * ((x1.T @ (p - y)) / len(y) + self.l2 * np.r_[0, w[1:]])
        self.weights = w; return self

    def predict_raw(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if self.weights is None: raise RuntimeError("router_not_fitted")
        x = np.asarray(features, dtype=float); x1 = np.c_[np.ones(len(x)), x]
        return 1 / (1 + np.exp(-np.clip(x1 @ self.weights, -40, 40)))

    def calibrate(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> "BenefitRouter":
        z = np.log(np.clip(self.predict_raw(features), 1e-6, 1 - 1e-6) / np.clip(1 - self.predict_raw(features), 1e-6, 1)); y = np.asarray(targets, dtype=float)
        cal = BenefitRouter(self.learning_rate, self.epochs, self.l2).fit(np.asarray(z)[:, None], y)
        self.calibration = (float(cal.weights[1]), float(cal.weights[0])); return self

    def predict(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        raw = self.predict_raw(features); a, b = self.calibration
        z = a * np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1)) + b
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def to_dict(self) -> dict[str, object]:
        if self.weights is None:
            raise RuntimeError("router_not_fitted")
        return {"weights": self.weights.tolist(), "calibration": list(self.calibration),
                "decision_threshold": self.decision_threshold,
                "learning_rate": self.learning_rate, "epochs": self.epochs, "l2": self.l2}

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "BenefitRouter":
        obj = cls(float(state.get("learning_rate", .2)), int(state.get("epochs", 300)), float(state.get("l2", 1e-3)))
        obj.weights = np.asarray(state["weights"], dtype=float)
        cal = state.get("calibration", [1.0, 0.0]); obj.calibration = (float(cal[0]), float(cal[1]))
        obj.decision_threshold = float(state.get("decision_threshold", 0.5))
        return obj


def calibration_metrics(probabilities: Sequence[float], targets: Sequence[float], bins: int = 10) -> dict[str, float]:
    """Return Brier score and expected calibration error for router reports."""
    p, y = np.asarray(probabilities, float), np.asarray(targets, float)
    if p.shape != y.shape or p.size == 0 or bins < 1:
        raise ValueError("calibration_data_invalid")
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if np.any(mask): ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"brier": float(np.mean((p - y) ** 2)), "ece": ece}

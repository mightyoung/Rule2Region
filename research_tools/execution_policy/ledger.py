"""Aligned per-frame stores: YOLO11s and YOLO26 never share a list."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from research_tools.cascade_experiment import Box, Prediction


@dataclass(frozen=True)
class FrameLedger:
    frame: int
    yolo11s: tuple[Prediction, ...]
    yolo26: tuple[Prediction, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "yolo11s": [_row(pred) for pred in self.yolo11s],
            "yolo26": [_row(pred) for pred in self.yolo26],
        }


def _row(pred: Prediction) -> dict[str, Any]:
    return {
        "x1": pred.box.x1,
        "y1": pred.box.y1,
        "x2": pred.box.x2,
        "y2": pred.box.y2,
        "confidence": pred.confidence,
        "source": pred.source,
    }


def _from_row(frame: int, row: dict[str, Any], default_source: str) -> Prediction:
    return Prediction(
        frame,
        Box(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
        float(row["confidence"]),
        str(row.get("source") or default_source),
    )


def frame_ledger_from_json(payload: dict[str, Any]) -> FrameLedger:
    if "merged" in payload or "predictions" in payload:
        raise ValueError("mixed_box_list_forbidden")
    frame = int(payload["frame"])
    return FrameLedger(
        frame,
        tuple(_from_row(frame, row, "yolo11s") for row in payload.get("yolo11s") or ()),
        tuple(_from_row(frame, row, "yolo26") for row in payload.get("yolo26") or ()),
    )


def align_ledgers(
    frames: Iterable[int],
    yolo11s: dict[int, tuple[Prediction, ...]],
    yolo26: dict[int, tuple[Prediction, ...]],
) -> tuple[FrameLedger, ...]:
    return tuple(FrameLedger(frame, yolo11s.get(frame, ()), yolo26.get(frame, ())) for frame in frames)

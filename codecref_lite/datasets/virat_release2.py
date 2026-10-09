from __future__ import annotations

import math
from typing import Iterable

from ..schemas import TrackObservation

OBJECT_TYPES = {
    1: "person",
    2: "car",
    3: "vehicle",
    4: "object",
    5: "bike",
}


def parse_object_lines(
    lines: Iterable[str],
    *,
    source_video_id: str,
    source_fps: float,
    timeline_offset_seconds: float = 0.0,
) -> tuple[TrackObservation, ...]:
    if not source_video_id:
        raise ValueError("source video id is required")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source FPS must be positive and finite")
    if not math.isfinite(timeline_offset_seconds):
        raise ValueError("timeline offset must be finite")

    observations: list[TrackObservation] = []
    declared_durations: dict[str, int] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 8:
            raise ValueError(f"line {line_number}: expected 8 columns")
        try:
            object_id = str(int(columns[0]))
            duration = int(columns[1])
            current_frame = int(columns[2])
            x, y, width, height = (float(value) for value in columns[3:7])
            object_type = int(columns[7])
        except ValueError as error:
            raise ValueError(f"line {line_number}: invalid numeric field") from error

        if duration <= 0 or current_frame < 0:
            raise ValueError(f"line {line_number}: invalid duration or frame")
        if object_type not in OBJECT_TYPES:
            raise ValueError(f"line {line_number}: unknown object type {object_type}")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"line {line_number}: non-finite bounding box")
        previous_duration = declared_durations.setdefault(object_id, duration)
        if previous_duration != duration:
            raise ValueError(f"line {line_number}: inconsistent track duration")

        observations.append(
            TrackObservation(
                source_video_id=source_video_id,
                track_id=object_id,
                category=OBJECT_TYPES[object_type],
                pts_seconds=timeline_offset_seconds + current_frame / source_fps,
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )

    observations.sort(key=lambda item: (item.pts_seconds, item.track_id))
    return tuple(observations)

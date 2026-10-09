from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Segment:
    index: int
    start_seconds: float
    end_seconds: float
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def plan_overlapping_segments(
    frame_count: int,
    fps: float,
    *,
    segment_seconds: float = 1.0,
    overlap_seconds: float = 0.2,
) -> tuple[Segment, ...]:
    if frame_count <= 0:
        raise ValueError("frame_count_must_be_positive")
    if fps <= 0:
        raise ValueError("fps_must_be_positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds_must_be_positive")
    if overlap_seconds < 0 or overlap_seconds >= segment_seconds:
        raise ValueError("overlap_must_be_nonnegative_and_less_than_segment")

    segment_frames = max(1, round(segment_seconds * fps))
    overlap_frames = round(overlap_seconds * fps)
    step_frames = max(1, segment_frames - overlap_frames)
    starts = list(range(0, frame_count, step_frames))
    if starts[-1] + segment_frames < frame_count:
        starts.append(frame_count - segment_frames)

    segments: list[Segment] = []
    for start in starts:
        end = min(frame_count, start + segment_frames)
        if segments and end <= segments[-1].end_frame:
            continue
        segments.append(
            Segment(
                index=len(segments),
                start_seconds=start / fps,
                end_seconds=end / fps,
                start_frame=start,
                end_frame=end,
            )
        )
        if end == frame_count:
            break
    return tuple(segments)


def aggregate_segment_counts(
    results: Iterable[Mapping[str, object]],
) -> dict[str, int | float | None]:
    counts = [int(item["count"]) for item in results if item.get("count") is not None]
    if not counts:
        return {
            "successful_segments": 0,
            "max_visible_count": None,
            "median_segment_count": None,
        }
    return {
        "successful_segments": len(counts),
        "max_visible_count": max(counts),
        "median_segment_count": float(median(counts)),
    }


def active_segment(
    segments: Sequence[Mapping[str, object]], time_seconds: float
) -> Mapping[str, object] | None:
    if time_seconds < 0:
        return None
    candidates = [
        item
        for item in segments
        if float(item["start_seconds"]) <= time_seconds < float(item["end_seconds"])
    ]
    if not candidates:
        return segments[-1] if segments else None
    # During overlap, show the most recently started inference window.
    return max(candidates, key=lambda item: float(item["start_seconds"]))

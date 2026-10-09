from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schemas import TrackObservation


@dataclass(frozen=True)
class LabelContract:
    person_categories: frozenset[str]
    frame_width: int
    frame_height: int
    frame_period_seconds: float
    timestamp_tolerance_seconds: float
    timeline_origin_seconds: float = 0.0


@dataclass(frozen=True)
class WindowLabel:
    source_video_id: str
    start_seconds: float
    end_seconds: float
    count: int
    track_ids: tuple[str, ...]
    excluded_observations: int
    peak_frame_pts_seconds: float | None = None


def _has_valid_frame_intersection(
    observation: TrackObservation, frame_width: int, frame_height: int
) -> bool:
    if observation.width <= 0 or observation.height <= 0:
        return False
    x1 = max(0.0, observation.x)
    y1 = max(0.0, observation.y)
    x2 = min(float(frame_width), observation.x + observation.width)
    y2 = min(float(frame_height), observation.y + observation.height)
    return x2 > x1 and y2 > y1


def label_window(
    observations: Iterable[TrackObservation],
    source_video_id: str,
    start_seconds: float,
    end_seconds: float,
    contract: LabelContract,
) -> WindowLabel:
    if end_seconds <= start_seconds:
        raise ValueError("window end must be greater than start")
    if (
        contract.frame_width <= 0
        or contract.frame_height <= 0
        or contract.frame_period_seconds <= 0
        or contract.timestamp_tolerance_seconds < 0
        or contract.timestamp_tolerance_seconds
        > contract.frame_period_seconds / 2 + 1e-12
    ):
        raise ValueError("invalid label contract")

    tracks_by_timestamp: dict[float, set[str]] = {}
    excluded = 0
    seen_at_timestamp: dict[tuple[str, float], tuple[float, float, float, float]] = {}

    for observation in observations:
        if observation.source_video_id != source_video_id:
            continue
        if not math.isfinite(observation.pts_seconds):
            raise ValueError("non-finite observation timestamp")
        frame_position = (
            observation.pts_seconds - contract.timeline_origin_seconds
        ) / contract.frame_period_seconds
        aligned_pts = (
            contract.timeline_origin_seconds
            + round(frame_position) * contract.frame_period_seconds
        )
        if (
            abs(aligned_pts - observation.pts_seconds)
            > contract.timestamp_tolerance_seconds + 1e-12
        ):
            raise ValueError("observation timestamp exceeds alignment tolerance")
        if not (start_seconds <= aligned_pts < end_seconds):
            continue
        if observation.category not in contract.person_categories:
            excluded += 1
            continue
        if observation.ignored or observation.absent:
            excluded += 1
            continue
        if not _has_valid_frame_intersection(
            observation, contract.frame_width, contract.frame_height
        ):
            excluded += 1
            continue

        if not observation.track_id:
            raise ValueError("empty track id")
        box_values = (
            observation.x,
            observation.y,
            observation.width,
            observation.height,
        )
        if not all(math.isfinite(value) for value in box_values):
            raise ValueError("non-finite bounding box")
        key = (observation.track_id, aligned_pts)
        box = (observation.x, observation.y, observation.width, observation.height)
        previous = seen_at_timestamp.get(key)
        if previous is not None and previous != box:
            raise ValueError(f"conflicting duplicate track observation: {key}")
        seen_at_timestamp[key] = box
        tracks_by_timestamp.setdefault(aligned_pts, set()).add(observation.track_id)

    peak_pts = None
    peak_tracks: set[str] = set()
    for pts, tracks in sorted(tracks_by_timestamp.items()):
        if len(tracks) > len(peak_tracks):
            peak_pts = pts
            peak_tracks = tracks
    track_ids = tuple(sorted(peak_tracks))
    return WindowLabel(
        source_video_id=source_video_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        count=len(track_ids),
        track_ids=track_ids,
        excluded_observations=excluded,
        peak_frame_pts_seconds=peak_pts,
    )


def generate_nonoverlapping_windows(
    duration_seconds: float, window_seconds: float = 4.0
) -> tuple[tuple[float, float], ...]:
    if duration_seconds < 0 or window_seconds <= 0:
        raise ValueError("invalid duration or window size")
    count = int(duration_seconds // window_seconds)
    return tuple(
        (index * window_seconds, (index + 1) * window_seconds) for index in range(count)
    )


def select_label_enriched_case_windows(
    labels: Sequence[WindowLabel],
    positive_limit: int = 5,
    include_empty: bool = True,
    *,
    acknowledge_label_enrichment: bool = False,
) -> tuple[WindowLabel, ...]:
    if not acknowledge_label_enrichment:
        raise ValueError(
            "label-enriched case selection requires explicit acknowledgement"
        )
    if positive_limit < 0:
        raise ValueError("positive limit must be non-negative")
    intervals = [(item.start_seconds, item.end_seconds) for item in labels]
    if len(set(intervals)) != len(intervals):
        raise ValueError("duplicate case window interval")
    ordered_labels = sorted(
        labels, key=lambda item: (item.start_seconds, item.end_seconds)
    )
    positives = [label for label in ordered_labels if label.count > 0]
    selected: list[WindowLabel] = []
    if positives:
        amount = min(positive_limit, len(positives))
        if amount == 1:
            selected.append(positives[0])
        else:
            indices = {
                round(index * (len(positives) - 1) / (amount - 1))
                for index in range(amount)
            }
            selected.extend(positives[index] for index in sorted(indices))
    if include_empty:
        empty = next((label for label in ordered_labels if label.count == 0), None)
        if empty is not None:
            selected.append(empty)
    return tuple(sorted(selected, key=lambda item: item.start_seconds))

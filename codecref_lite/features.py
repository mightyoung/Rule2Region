from __future__ import annotations

import math
from typing import Iterable

from .schemas import CandidateFeature, MotionVector


def motion_statistics(
    vectors: Iterable[MotionVector], frame_width: int, frame_height: int
) -> tuple[float, float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    frame_area = float(frame_width * frame_height)
    covered_area = 0.0
    weighted_magnitude = 0.0

    for vector in vectors:
        if vector.motion_scale <= 0 or vector.width <= 0 or vector.height <= 0:
            raise ValueError("invalid motion vector")
        left = max(0, vector.dst_x)
        top = max(0, vector.dst_y)
        right = min(frame_width, vector.dst_x + vector.width)
        bottom = min(frame_height, vector.dst_y + vector.height)
        area = float(max(0, right - left) * max(0, bottom - top))
        if area == 0:
            continue
        dx = vector.motion_x / vector.motion_scale
        dy = vector.motion_y / vector.motion_scale
        magnitude = math.hypot(dx, dy)
        covered_area += area
        weighted_magnitude += area * magnitude

    density = min(1.0, covered_area / frame_area)
    mean_magnitude = weighted_magnitude / covered_area if covered_area else 0.0
    return density, mean_magnitude, density * mean_magnitude


def build_candidate_feature(
    *,
    index: int,
    pts_seconds: float,
    packet_size_bytes: int,
    frame_width: int,
    frame_height: int,
    vectors: Iterable[MotionVector],
    frame_type: str,
    key_frame: bool = False,
    mv_missing: bool = False,
    packet_attribution_supported: bool = True,
    pts_supported: bool = True,
    mv_side_data_valid: bool = True,
) -> CandidateFeature:
    if packet_size_bytes < 0:
        raise ValueError("packet size must be non-negative")
    density, mean_magnitude, energy = motion_statistics(
        vectors, frame_width, frame_height
    )
    packet_bpp = 8.0 * packet_size_bytes / (frame_width * frame_height)
    return CandidateFeature(
        index=index,
        pts_seconds=pts_seconds,
        packet_bpp=packet_bpp,
        mv_density=density,
        mv_mean=mean_magnitude,
        mv_energy=energy,
        frame_type=frame_type,
        key_frame=key_frame,
        mv_missing=mv_missing,
        packet_attribution_supported=packet_attribution_supported,
        pts_supported=pts_supported,
        mv_side_data_valid=mv_side_data_valid,
    )

from __future__ import annotations

import hashlib
import math
import random
from typing import Sequence

import numpy as np

from .schemas import CandidateFeature


def _validate_budget(length: int, budget: int) -> None:
    if length <= 0 or budget <= 0 or budget > length:
        raise ValueError("budget must be between one and candidate count")


def dense_select(candidate_count: int) -> tuple[int, ...]:
    if candidate_count <= 0:
        raise ValueError("candidate count must be positive")
    return tuple(range(candidate_count))


def uniform_select(candidate_count: int, budget: int) -> tuple[int, ...]:
    _validate_budget(candidate_count, budget)
    if budget == 1:
        return (0,)
    indices = {
        round(index * (candidate_count - 1) / (budget - 1)) for index in range(budget)
    }
    if len(indices) != budget:
        raise RuntimeError("uniform selector produced duplicate indices")
    return tuple(sorted(indices))


def _derived_seed(global_seed: int, window_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{window_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def random_select(
    candidate_count: int, budget: int, global_seed: int, window_id: str
) -> tuple[int, ...]:
    _validate_budget(candidate_count, budget)
    generator = random.Random(_derived_seed(global_seed, window_id))
    return tuple(sorted(generator.sample(range(candidate_count), budget)))


def rgb_difference_scores(frames: Sequence[np.ndarray]) -> tuple[float, ...]:
    if not frames:
        raise ValueError("at least one frame is required")
    reference_shape = frames[0].shape
    scores = [0.0]
    previous = frames[0].astype(np.float32)
    for frame in frames[1:]:
        if frame.shape != reference_shape:
            raise ValueError("all frames must share a shape")
        current = frame.astype(np.float32)
        scores.append(float(np.mean(np.abs(current - previous))))
        previous = current
    return tuple(scores)


def rgb_difference_select(frames: Sequence[np.ndarray], budget: int) -> tuple[int, ...]:
    _validate_budget(len(frames), budget)
    if budget == 1:
        return (0,)
    scores = rgb_difference_scores(frames)
    ranked = sorted(range(1, len(frames)), key=lambda index: (-scores[index], index))
    return tuple(sorted((0, *ranked[: budget - 1])))


def _normalized_tied_ranks(values: Sequence[float]) -> tuple[float, ...]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return tuple(0.0 for _ in values)
    rank = {value: index / (len(unique) - 1) for index, value in enumerate(unique)}
    return tuple(rank[value] for value in values)


def codec_guided_scores(features: Sequence[CandidateFeature]) -> tuple[float, ...]:
    if not features:
        raise ValueError("at least one feature is required")
    if len({item.index for item in features}) != len(features):
        raise ValueError("candidate feature indices must be unique")
    if any(
        not math.isfinite(value)
        for item in features
        for value in (item.pts_seconds, item.packet_bpp, item.mv_energy)
    ):
        raise ValueError("candidate feature values must be finite")
    if any(
        left.pts_seconds >= right.pts_seconds
        for left, right in zip(features, features[1:])
    ):
        raise ValueError("candidate features must be strictly ordered by PTS")
    packet_ranks = _normalized_tied_ranks([item.packet_bpp for item in features])
    motion_ranks = _normalized_tied_ranks([item.mv_energy for item in features])
    return tuple(
        0.5 * packet_rank + 0.5 * motion_rank
        for packet_rank, motion_rank in zip(packet_ranks, motion_ranks, strict=True)
    )


def codec_guided_select(
    features: Sequence[CandidateFeature], budget: int
) -> tuple[int, ...]:
    _validate_budget(len(features), budget)
    if any(
        not item.packet_attribution_supported or not item.pts_supported
        for item in features
    ):
        raise ValueError("CODEC_FEATURE_ATTRIBUTION_UNAVAILABLE")
    first = features[0]
    if any(not item.mv_side_data_valid for item in features):
        raise ValueError("CODEC_GUIDED_MV_INVALID")
    if first.mv_missing and not (first.key_frame and first.frame_type.upper() == "I"):
        raise ValueError("CODEC_GUIDED_MV_UNAVAILABLE")
    if any(item.mv_missing for item in features[1:]):
        raise ValueError("CODEC_GUIDED_MV_UNAVAILABLE")
    scores = codec_guided_scores(features)
    if budget == 1:
        return (features[0].index,)
    ranked = sorted(
        range(1, len(features)),
        key=lambda position: (
            -scores[position],
            features[position].pts_seconds,
            features[position].index,
        ),
    )
    selected_positions = (0, *ranked[: budget - 1])
    return tuple(sorted(features[position].index for position in selected_positions))

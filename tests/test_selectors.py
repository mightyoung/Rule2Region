import numpy as np
import pytest

from codecref_lite.schemas import CandidateFeature
from codecref_lite.selectors import (
    codec_guided_select,
    random_select,
    rgb_difference_select,
    uniform_select,
)


def _feature(
    index: int, packet: float, motion: float, missing: bool = False
) -> CandidateFeature:
    return CandidateFeature(
        index, index * 0.5, packet, 1.0, motion, motion, "P", False, missing
    )


def test_uniform_selection_is_deterministic() -> None:
    assert uniform_select(8, 4) == (0, 2, 5, 7)


def test_random_selection_is_reproducible() -> None:
    first = random_select(8, 4, 17, "window-1")
    second = random_select(8, 4, 17, "window-1")
    assert first == second
    assert len(first) == 4


def test_rgb_difference_keeps_first_and_largest_changes() -> None:
    frames = [np.full((2, 2), value, dtype=np.uint8) for value in (0, 1, 20, 21, 60)]
    assert rgb_difference_select(frames, 3) == (0, 2, 4)


def test_codec_selection_uses_equal_rank_sum_and_stable_ties() -> None:
    features = [
        _feature(0, 10, 0),
        _feature(1, 20, 1),
        _feature(2, 30, 4),
        _feature(3, 40, 2),
        _feature(4, 50, 3),
    ]
    assert codec_guided_select(features, 3) == (0, 2, 4)


def test_codec_selection_fails_closed_when_mv_is_missing() -> None:
    features = [_feature(0, 10, 0), _feature(1, 20, 1, True)]
    with pytest.raises(ValueError, match="CODEC_GUIDED_MV_UNAVAILABLE"):
        codec_guided_select(features, 1)


def test_codec_selection_allows_missing_mv_only_for_first_keyframe() -> None:
    features = [
        CandidateFeature(0, 0.0, 10, 0, 0, 0, "I", True, True),
        _feature(1, 20, 1),
    ]
    assert codec_guided_select(features, 1) == (0,)


def test_codec_selection_rejects_nonfinite_feature() -> None:
    features = [_feature(0, 10, 0), _feature(1, float("nan"), 1)]
    with pytest.raises(ValueError, match="finite"):
        codec_guided_select(features, 1)


def test_codec_selection_rejects_invalid_first_frame_side_data() -> None:
    features = [
        CandidateFeature(
            0,
            0.0,
            10,
            0,
            0,
            0,
            "I",
            True,
            True,
            mv_side_data_valid=False,
        ),
        _feature(1, 20, 1),
    ]
    with pytest.raises(ValueError, match="CODEC_GUIDED_MV_INVALID"):
        codec_guided_select(features, 1)

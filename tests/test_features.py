import pytest

from codecref_lite.features import build_candidate_feature, motion_statistics
from codecref_lite.schemas import MotionVector


def test_motion_statistics_are_area_weighted() -> None:
    vectors = [
        MotionVector(4, 0, 2, 10, 10, 0, 0),
        MotionVector(0, 6, 2, 10, 10, 10, 0),
    ]
    density, mean, energy = motion_statistics(vectors, 20, 20)
    assert density == pytest.approx(0.5)
    assert mean == pytest.approx(2.5)
    assert energy == pytest.approx(1.25)


def test_candidate_feature_computes_packet_bpp() -> None:
    feature = build_candidate_feature(
        index=0,
        pts_seconds=0.0,
        packet_size_bytes=100,
        frame_width=20,
        frame_height=20,
        vectors=[],
        frame_type="I",
        key_frame=True,
    )
    assert feature.packet_bpp == pytest.approx(2.0)
    assert feature.mv_energy == 0.0

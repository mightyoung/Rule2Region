import pytest

from evidencegate.demo import (
    active_segment,
    aggregate_segment_counts,
    plan_overlapping_segments,
)


def test_segment_plan_covers_four_second_clip_with_overlap() -> None:
    segments = plan_overlapping_segments(120, 30, segment_seconds=1.0, overlap_seconds=0.2)
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == 120
    assert all(item.frame_count <= 30 for item in segments)
    assert all(left.end_frame >= right.start_frame for left, right in zip(segments, segments[1:]))


def test_segment_plan_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        plan_overlapping_segments(120, 30, segment_seconds=1.0, overlap_seconds=1.0)


def test_aggregate_does_not_sum_people_across_segments() -> None:
    aggregate = aggregate_segment_counts([{"count": 7}, {"count": 8}, {"count": 7}])
    assert aggregate == {
        "successful_segments": 3,
        "max_visible_count": 8,
        "median_segment_count": 7.0,
    }


def test_active_segment_prefers_latest_window_in_overlap() -> None:
    segments = [
        {"index": 0, "start_seconds": 0.0, "end_seconds": 1.0},
        {"index": 1, "start_seconds": 0.8, "end_seconds": 1.8},
    ]
    assert active_segment(segments, 0.9)["index"] == 1

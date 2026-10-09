import pytest

from codecref_lite.labels import (
    LabelContract,
    WindowLabel,
    generate_nonoverlapping_windows,
    label_window,
    select_label_enriched_case_windows,
)
from codecref_lite.schemas import TrackObservation


CONTRACT = LabelContract(frozenset({"person"}), 100, 100, 0.04, 0.02)


def _observation(track_id: str, pts: float, **kwargs: object) -> TrackObservation:
    values = {
        "source_video_id": "video-1",
        "track_id": track_id,
        "category": "person",
        "pts_seconds": pts,
        "x": 10.0,
        "y": 10.0,
        "width": 20.0,
        "height": 20.0,
    }
    values.update(kwargs)
    return TrackObservation(**values)


def test_label_counts_peak_concurrent_tracks_in_half_open_window() -> None:
    observations = [
        _observation("a", 0.0),
        _observation("a", 1.0),
        _observation("b", 1.0),
        _observation("c", 4.0),
    ]
    label = label_window(observations, "video-1", 0.0, 4.0, CONTRACT)
    assert label.count == 2
    assert label.track_ids == ("a", "b")
    assert label.peak_frame_pts_seconds == 1.0


def test_label_does_not_accumulate_fragmented_tracks_across_time() -> None:
    observations = [_observation("fragment-a", 1.0), _observation("fragment-b", 2.0)]
    label = label_window(observations, "video-1", 0.0, 4.0, CONTRACT)
    assert label.count == 1


def test_label_rejects_conflicting_duplicate_observation() -> None:
    observations = [_observation("a", 1.0), _observation("a", 1.0, x=11.0)]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        label_window(observations, "video-1", 0.0, 4.0, CONTRACT)


def test_generate_nonoverlapping_windows_drops_partial_tail() -> None:
    assert generate_nonoverlapping_windows(9.0) == ((0.0, 4.0), (4.0, 8.0))


def test_label_enriched_selection_requires_acknowledgement_and_sorts_input() -> None:
    labels = [
        WindowLabel("video-1", 4.0, 8.0, 1, ("a",), 0),
        WindowLabel("video-1", 0.0, 4.0, 0, (), 0),
    ]
    with pytest.raises(ValueError, match="acknowledgement"):
        select_label_enriched_case_windows(labels)
    selected = select_label_enriched_case_windows(
        labels, acknowledge_label_enrichment=True
    )
    assert [item.start_seconds for item in selected] == [0.0, 4.0]

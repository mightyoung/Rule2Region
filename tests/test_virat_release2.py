import pytest

from codecref_lite.datasets.virat_release2 import parse_object_lines


def test_parse_release2_object_rows() -> None:
    rows = [
        "2 3 1 10 20 30 40 1",
        "1 2 0 1 2 3 4 2",
        "2 3 2 11 21 30 40 1",
    ]
    observations = parse_object_lines(
        rows, source_video_id="VIRAT_S_fixture", source_fps=10.0
    )
    assert [item.track_id for item in observations] == ["1", "2", "2"]
    assert observations[1].category == "person"
    assert observations[1].pts_seconds == pytest.approx(0.1)


def test_parse_release2_rejects_inconsistent_track_duration() -> None:
    rows = ["2 3 1 10 20 30 40 1", "2 4 2 11 21 30 40 1"]
    with pytest.raises(ValueError, match="inconsistent track duration"):
        parse_object_lines(rows, source_video_id="fixture", source_fps=10.0)


def test_parse_release2_rejects_unknown_object_type() -> None:
    with pytest.raises(ValueError, match="unknown object type"):
        parse_object_lines(
            ["1 1 0 0 0 10 10 99"], source_video_id="fixture", source_fps=10.0
        )

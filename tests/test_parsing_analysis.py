from codecref_lite.analysis import evaluate_engineering_quality, summarize_predictions
from codecref_lite.parsing import parse_count_json


def test_count_parser_is_strict_and_nonnegative() -> None:
    assert parse_count_json('{"count": 3}').count == 3
    assert not parse_count_json('{"count": -1}').valid
    assert not parse_count_json('{"count": true}').valid
    assert not parse_count_json('{"count": 3, "extra": 1}').valid


def test_analysis_keeps_invalid_separate_from_mae() -> None:
    rows = [
        {
            "source_video_id": "v1",
            "method": "dense",
            "label": 2,
            "prediction": 3,
            "status": "passed",
        },
        {
            "source_video_id": "v1",
            "method": "dense",
            "label": 1,
            "prediction": None,
            "status": "invalid",
        },
    ]
    summary = summarize_predictions(rows)[0]
    assert summary["valid_coverage"] == 0.5
    assert summary["mae_valid"] == 1.0


def test_engineering_quality_requires_complete_paired_results() -> None:
    rows = [
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "dense",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "codec",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
    ]
    result = evaluate_engineering_quality(
        rows, expected_windows_by_video={"v1": ["w1"]}
    )
    assert result["passed"]
    assert result["median_delta_mae"] == 0.0


def test_engineering_quality_fails_on_invalid_cell() -> None:
    rows = [
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "dense",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "codec",
            "label": 2,
            "prediction": None,
            "status": "invalid",
        },
    ]
    result = evaluate_engineering_quality(
        rows, expected_windows_by_video={"v1": ["w1"]}
    )
    assert not result["passed"]
    assert result["median_delta_mae"] is None


def test_engineering_quality_detects_window_missing_from_both_methods() -> None:
    rows = [
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "dense",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "codec",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
    ]
    result = evaluate_engineering_quality(
        rows, expected_windows_by_video={"v1": ["w1", "w2"]}
    )
    assert not result["passed"]
    assert result["overall_completion"] == 0.5


def test_engineering_quality_rejects_label_mismatch() -> None:
    rows = [
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "dense",
            "label": 2,
            "prediction": 2,
            "status": "passed",
        },
        {
            "source_video_id": "v1",
            "window_id": "w1",
            "method": "codec",
            "label": 3,
            "prediction": 2,
            "status": "passed",
        },
    ]
    result = evaluate_engineering_quality(
        rows, expected_windows_by_video={"v1": ["w1"]}
    )
    assert not result["passed"]

from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.rclda_upper_bound import (
    FEATURE_NAMES,
    _rows_from_feature_ledger,
    _assert_all_accepted_reproduces_merged,
    evaluate_oracle,
    frozen_rule_grid,
)


def _feature(frame: int, box: Box, values: tuple[float, ...]) -> dict:
    prediction = Prediction(frame, box, .9, "yolo26_hr_tile:t0")
    return {
        "frame": frame,
        "candidate_index": 0,
        "tile_id": "t0",
        "prediction": {"frame": frame, "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}, "confidence": .9, "source": prediction.source},
        "features": dict(zip(FEATURE_NAMES, values)),
    }


def test_rule_grid_is_frozen_and_monotone() -> None:
    grid = frozen_rule_grid()
    assert len(grid["weights"]) == 242
    assert len(grid["thresholds"]) == 22
    assert all(all(value >= 0 for value in row["values"]) and any(row["values"]) for row in grid["weights"])


def test_higher_threshold_cannot_add_candidates() -> None:
    features = [_feature(1, Box(0, 0, 10, 10), (1, 0, 0, 0, 0)), _feature(1, Box(20, 0, 30, 10), (.2, 0, 0, 0, 0))]
    low = _rows_from_feature_ledger(features, {1: []}, (1,), (1, 0, 0, 0, 0), .1)
    high = _rows_from_feature_ledger(features, {1: []}, (1,), (1, 0, 0, 0, 0), .8)
    assert len(low[1]) == 2
    assert len(high[1]) == 1


def test_all_accepted_rule_reproduces_original_merged_operator() -> None:
    global_row = {"frame": 1, "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": .9, "source": "yolo11s_global"}
    feature = _feature(1, Box(20, 0, 30, 10), (.8, 0, 0, 0, 0))
    frames = [{
        "frame": 1,
        "merged": [global_row, feature["prediction"]],
    }]
    _assert_all_accepted_reproduces_merged(frames, [feature], {1: [global_row]}, (1,))


def test_oracle_admits_a_separable_rule_family() -> None:
    features = [
        _feature(1, Box(0, 0, 10, 10), (1, 1, 1, 1, 1)),
        _feature(2, Box(20, 0, 30, 10), (1, 1, 1, 1, 1)),
    ]
    truth = [GroundTruth(1, 1, Box(0, 0, 10, 10)), GroundTruth(2, 2, Box(20, 0, 30, 10))]
    oracle = evaluate_oracle(features, {1: [], 2: []}, truth, (1, 2), frozen_rule_grid(), {"count_mae": 1.0, "precision": 0.0}, latency_ratio=2.0)
    assert oracle["rules_evaluated"] == 5324
    assert oracle["viable_rule_count"] > 0
    assert oracle["best_viable_rule"]["relative_count_mae_improvement"] == 1.0


def test_oracle_rejects_when_only_baseline_is_safe() -> None:
    global_row = {"frame": 1, "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "confidence": .9, "source": "yolo11s_global"}
    features = [_feature(1, Box(20, 20, 30, 30), (1, 1, 1, 1, 1))]
    truth = [GroundTruth(1, 1, Box(0, 0, 10, 10))]
    oracle = evaluate_oracle(features, {1: [global_row]}, truth, (1,), frozen_rule_grid(), {"count_mae": 0.0, "precision": 1.0}, latency_ratio=2.0)
    assert oracle["best_viable_rule"] is None

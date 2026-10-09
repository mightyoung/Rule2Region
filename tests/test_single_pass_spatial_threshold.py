from pathlib import Path

from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.card import PolicyCard, RegionCard
from research_tools.execution_policy.run_single_pass_spatial_threshold import (
    candidate_name,
    pick_history_candidate,
    spatial_filter,
)


def test_spatial_filter_uses_focus_and_outside_thresholds() -> None:
    card = PolicyCard((RegionCard("focus", Box(0, 0, 100, 100), "yolo26", 0.25),))
    predictions = (
        Prediction(1, Box(10, 10, 20, 20), 0.08),
        Prediction(1, Box(110, 10, 120, 20), 0.08),
        Prediction(1, Box(110, 10, 120, 20), 0.30),
    )
    kept = spatial_filter(predictions, card, outside_conf=0.25, focus_conf=0.05)
    assert kept == (predictions[0], predictions[2])


def test_candidate_identity_and_history_selection() -> None:
    assert candidate_name(0.15, 0.05) == "o15_f05"
    rows = [
        {"name": "bad", "all_history_pass": False, "history_mae_mean": 0, "history_precision_mean": 1, "threshold_change": 0},
        {"name": "more-change", "all_history_pass": True, "history_mae_mean": 1, "history_precision_mean": .8, "threshold_change": .2},
        {"name": "less-change", "all_history_pass": True, "history_mae_mean": 1, "history_precision_mean": .8, "threshold_change": .1},
    ]
    assert pick_history_candidate(rows)["name"] == "less-change"


def test_launcher_contract() -> None:
    text = Path("research_tools/remote/launch_single_pass_spatial_threshold.sh").read_text(encoding="utf-8")
    assert "GPU=${GPU:-1}" in text
    assert "candidate_count\"] == 15" in text
    assert "second_model_called\"] is False" in text
    assert "DEVELOPMENT_ONLY" in text
    assert ".env" not in text

from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.execution_policy.loop import run_history_loop


def test_e3_loop_freezes_last_or_early_card_without_mixing() -> None:
    ledgers = {}
    truth = []
    for frame in range(1, 5):
        ledgers[frame] = FrameLedger(
            frame,
            (Prediction(frame, Box(10, 400, 40, 500), 0.90, "yolo11s"),),
            (Prediction(frame, Box(10, 10, 40, 80), 0.80, "yolo26"),),
        )
        truth.append(GroundTruth(frame, 1, Box(10, 400, 40, 500)))
    result = run_history_loop(ledgers, [1, 2, 3, 4], truth)
    assert result["k"] == 3
    assert result["scoring"] == "ledger_filter_s1"
    assert 1 <= len(result["rounds"]) <= 3
    models = {row["model"] for row in result["intention_card"]["regions"]}
    assert models == {"yolo11s", "yolo26"}

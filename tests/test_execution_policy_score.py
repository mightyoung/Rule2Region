from research_tools.cascade_experiment import Box, GroundTruth, Prediction
from research_tools.execution_policy.card import parse_policy_card
from research_tools.execution_policy.ledger import FrameLedger
from research_tools.execution_policy.score import compare_camera, compare_three


def test_e1_three_way_keeps_streams_and_reports_s1() -> None:
    card = parse_policy_card(
        {
            "regions": [
                {"region_id": "far", "xyxy": [0, 0, 200, 200], "model": "yolo26", "conf": 0.25},
                {"region_id": "near", "xyxy": [200, 0, 400, 200], "model": "yolo11s", "conf": 0.25},
            ]
        }
    )
    ledgers = {
        1: FrameLedger(
            1,
            (Prediction(1, Box(210, 10, 240, 80), 0.90, "yolo11s"),),
            (Prediction(1, Box(10, 10, 40, 80), 0.80, "yolo26"),),
        )
    }
    truth = (GroundTruth(1, 1, Box(210, 10, 240, 80)),)
    result = compare_three(card, ledgers, [1], truth, compiled_time_ms=1.0, b26_time_ms=2.0)
    assert result["scoring"] == "ledger_filter_s1"
    assert result["val_score"] == "crop_infer_not_run"
    assert result["b11"]["pred_box_count"] == 1
    assert result["b26"]["pred_box_count"] == 1
    assert result["compiled"]["pred_box_count"] == 2
    assert result["h1"]["compiled_proxy"] < 1.0
    camera = compare_camera(
        card,
        [
            {"sequence": "A", "history": [1], "ledgers": ledgers, "truth": truth},
            {"sequence": "B", "history": [1], "ledgers": ledgers, "truth": truth},
        ],
        compiled_time_ms=1.0,
        b26_time_ms=2.0,
    )
    assert camera["scope"] == "camera"
    assert camera["sequences"] == ["A", "B"]
    assert camera["compiled"]["pred_box_count"] == 4

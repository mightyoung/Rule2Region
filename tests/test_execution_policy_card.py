from research_tools.cascade_experiment import Box, Prediction
from research_tools.execution_policy.card import CardInvalid, apply_patch, merge_overlapping_regions, parse_policy_card
from research_tools.execution_policy.compile import apply_card, h1_pass
from research_tools.execution_policy.card import latency_proxy
from research_tools.execution_policy.ledger import FrameLedger, frame_ledger_from_json
from research_tools.execution_policy.rules import COUNTING_RULE_CARD, DOOR_RULE_CARD


def test_rule_cards_are_frozen_and_separated() -> None:
    assert "yolo26" in COUNTING_RULE_CARD
    assert "H1 不启用高峰" in COUNTING_RULE_CARD
    assert "不进 H1" in DOOR_RULE_CARD
    assert "YOLO26" not in DOOR_RULE_CARD or "不决定" in DOOR_RULE_CARD


def test_overlap_and_illegal_menu_rejected() -> None:
    try:
        parse_policy_card(
            {
                "regions": [
                    {"region_id": "a", "xyxy": [0, 0, 200, 200], "model": "yolo26", "conf": 0.25},
                    {"region_id": "b", "xyxy": [100, 100, 300, 300], "model": "yolo11s", "conf": 0.25},
                ]
            }
        )
    except CardInvalid as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlap_must_fail")
    try:
        parse_policy_card({"regions": [{"xyxy": [0, 0, 10, 10], "model": "yolo99", "conf": 0.25}]})
    except CardInvalid:
        pass
    else:
        raise AssertionError("illegal_model_must_fail")
    try:
        parse_policy_card(
            {"regions": [{"xyxy": [0, 0, 100, 100], "model": "yolo11s", "conf": 0.25, "role": "exclude"}]}
        )
    except CardInvalid as exc:
        assert "count" in str(exc)
    else:
        raise AssertionError("zero_count_region_must_fail")


def test_ledger_forbids_mixed_lists_and_keeps_streams_aligned() -> None:
    try:
        frame_ledger_from_json({"frame": 1, "merged": [], "yolo11s": [], "yolo26": []})
    except ValueError as exc:
        assert "mixed" in str(exc)
    else:
        raise AssertionError("mixed_list_must_fail")
    card = parse_policy_card(
        {
            "regions": [
                {"region_id": "far", "xyxy": [0, 0, 400, 400], "model": "yolo26", "conf": 0.25},
                {"region_id": "near", "xyxy": [400, 400, 1280, 720], "model": "yolo11s", "conf": 0.25},
            ]
        }
    )
    ledger = FrameLedger(
        1,
        (
            Prediction(1, Box(10, 10, 40, 80), 0.90, "yolo11s"),
            Prediction(1, Box(500, 500, 540, 600), 0.80, "yolo11s"),
        ),
        (Prediction(1, Box(20, 20, 50, 70), 0.70, "yolo26"),),
    )
    parted = apply_card(card, ledger)
    assert all(row.source == "yolo11s" for row in parted.yolo11s)
    assert all(row.source == "yolo26" for row in parted.yolo26)
    assert all(row.box.x1 == 500 for row in parted.yolo11s)
    assert not any(row.box.x1 == 10 for row in parted.yolo11s)
    assert parted.yolo26[0].box.x1 == 20


def test_h1_requires_cheaper_than_26_and_count_near_11s() -> None:
    ok = h1_pass(
        compiled_mae=6.0,
        b11_mae=6.0,
        precision_change_vs_b11=0.0,
        compiled_proxy=0.40,
        compiled_time_ms=10.0,
        b26_time_ms=20.0,
    )
    assert ok["passed"] is True
    bad = h1_pass(
        compiled_mae=6.0,
        b11_mae=6.0,
        precision_change_vs_b11=0.0,
        compiled_proxy=0.40,
        compiled_time_ms=30.0,
        b26_time_ms=20.0,
    )
    assert bad["passed"] is False
    card = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 128, 72], "model": "yolo26", "conf": 0.25}]}
    )
    assert latency_proxy(card) == (128 * 72) / (1280 * 720)


def test_merge_overlapping_regions_unions_and_prefers_yolo26() -> None:
    merged = merge_overlapping_regions(
        {
            "regions": [
                {"region_id": "a", "xyxy": [0, 0, 100, 100], "model": "yolo11s", "conf": 0.35, "role": "count"},
                {"region_id": "b", "xyxy": [50, 50, 150, 150], "model": "yolo26", "conf": 0.25, "role": "count"},
                {"region_id": "c", "xyxy": [400, 400, 500, 500], "model": "yolo26", "conf": 0.25, "role": "count"},
            ]
        }
    )
    assert len(merged["regions"]) == 2
    first = next(row for row in merged["regions"] if row["region_id"] == "a")
    assert first["model"] == "yolo26"
    assert first["xyxy"] == [0, 0, 150, 150]
    assert first["conf"] == 0.25
    card = parse_policy_card(merged)
    assert len(card.regions) == 2


def test_apply_patch_changes_one_field() -> None:
    card = parse_policy_card(
        {"regions": [{"region_id": "far", "xyxy": [0, 0, 100, 100], "model": "yolo26", "conf": 0.25}]}
    )
    patched = apply_patch(card, {"action": "set_conf", "region_id": "far", "conf": 0.15})
    assert patched.regions[0].conf == 0.15
    assert apply_patch(card, {"action": "stop"}).regions[0].conf == 0.25

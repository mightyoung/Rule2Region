import json

import pytest

from research_tools.sceneconfig_e3_qwen_validation import _normalized_regret, _queries_to_95, parse_query_response


def test_query_parser_counts_cross_round_duplicate_as_consumed_slot() -> None:
    legal = {"C00", "C01", "C02", "C03", "C04"}
    first = parse_query_response(json.dumps({"config_ids": ["C00", "C01", "C02"], "rationale": "x"}), legal)
    assert all(row["status"] == "legal" for row in first["slots"])
    second = parse_query_response(
        json.dumps({"config_ids": ["C02", "C03", "C04"], "rationale": "y"}), legal, {"C00", "C01", "C02"}
    )
    assert [row["status"] for row in second["slots"]] == ["duplicate", "legal", "legal"]


def test_query_parser_rejects_non_exact_contract() -> None:
    with pytest.raises(ValueError):
        parse_query_response(json.dumps({"config_ids": ["C00"], "rationale": "x"}), {"C00"})


def test_regret_and_query_threshold_are_registered_space_normalized() -> None:
    utilities = {"a": 0.0, "b": 5.0, "c": 10.0}
    assert _normalized_regret(utilities, ["b"]) == .5
    assert _normalized_regret(utilities, ["c"]) == 0
    assert _queries_to_95(utilities, ["a", "c"]) == 2
    assert _queries_to_95(utilities, ["a", "b"]) == 7

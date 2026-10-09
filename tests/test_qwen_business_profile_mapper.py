import json
from pathlib import Path

import pytest

from research_tools.qwen_business_profile_mapper import (
    compile_profile,
    load_cards,
    load_blind_cases,
    load_registry,
    parse_profile_response,
    rule_map,
)


ROOT = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2")
REGISTRY = ROOT / "preference-profile-v1/profile-registry.json"
CARDS = ROOT / "preference-profile-v1/business-cards.json"
CATALOG = ROOT / "camera-catalog-010204-v1/decision-catalog.json"
BLIND = ROOT / "preference-profile-blind-v1/blind-cards.json"
GOLD = ROOT / "preference-profile-blind-v1/annotator-gold.json"
AUTHOR = ROOT / "preference-profile-blind-v1/author-cards.json"


def test_rule_baseline_matches_all_development_cards() -> None:
    load_registry(REGISTRY)
    for card in load_cards(CARDS):
        result = rule_map(card)
        assert result["profile_code"] == card["gold_profile_code"]
        assert result["abstain"] == card["gold_abstain"]


def test_profile_response_contract_and_gold_label() -> None:
    card = load_cards(CARDS)[0]
    payload = {
        "role": "business_profile_mapper",
        "card_id": card["card_id"],
        "profile_code": card["gold_profile_code"],
        "supporting_span_ids": card["gold_supporting_span_ids"],
        "abstain": False,
        "abstain_reason_codes": [],
    }
    parse_profile_response(json.dumps(payload), card)
    payload["supporting_span_ids"] = ["UNKNOWN"]
    with pytest.raises(ValueError, match="span"):
        parse_profile_response(json.dumps(payload), card)


def test_profiles_compile_deterministically_on_only_eligible_camera() -> None:
    expected = {
        "FN_PRIORITY_WITH_PRECISION_GATE": "C04",
        "COUNT_ACCURACY_WITH_RECALL_TIEBREAK": "C04",
        "LATENCY_FIRST_WITH_QUALITY_FLOOR": "C00",
    }
    for code, selected in expected.items():
        result = compile_profile(CATALOG, REGISTRY, code)
        assert result["selected_config_id"] == selected
        assert result["holdout_opened"] is False


def test_blind_cards_are_label_free_and_independent_labels_agree() -> None:
    blind_text = BLIND.read_text(encoding="utf-8")
    assert "author_intended_label" not in blind_text and "gold_profile_code" not in blind_text
    cases = load_blind_cases(BLIND, GOLD)
    author = {row["card_id"]: row["author_intended_label"] for row in json.loads(AUTHOR.read_text(encoding="utf-8"))["cards"]}
    normalized_gold = {row["card_id"]: row["gold_profile_code"] or "ABSTAIN" for row in cases}
    assert author == normalized_gold
    assert len(cases) == 16


def test_no_registered_profile_abstention_is_legal() -> None:
    card = next(row for row in load_blind_cases(BLIND, GOLD) if row["card_id"] == "BLIND13")
    payload = {
        "role": "business_profile_mapper",
        "card_id": card["card_id"],
        "profile_code": None,
        "supporting_span_ids": card["gold_supporting_span_ids"],
        "abstain": True,
        "abstain_reason_codes": ["NO_REGISTERED_PROFILE"],
    }
    parse_profile_response(json.dumps(payload), card)

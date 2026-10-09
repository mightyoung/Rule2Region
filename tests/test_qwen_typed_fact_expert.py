import json
from pathlib import Path

import pytest

from research_tools.qwen_typed_fact_expert import (
    OBJECTIVE_ORDER,
    PREFERENCE_PROFILE_CODE,
    build_fact_bundle,
    expected_domain_semantics,
    parse_typed_domain_response,
    parse_typed_unified_response,
)


CATALOG = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/camera-catalog-010204-v1/decision-catalog.json")
ZERO_CATALOG = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/camera-catalog-010111-v1/decision-catalog.json")


def _domain_payload(role: str, bundle: dict) -> dict:
    facts = bundle["facts_by_role"][role]
    base = {"role": role, "consumed_fact_ids": [row["fact_id"] for row in facts], "abstain": False}
    expected = expected_domain_semantics(role, bundle)
    if role == "business_expert":
        base.update(expected)
    else:
        base["config_assessments"] = expected
    return base


def test_typed_fact_bundle_separates_business_semantics_and_holdout() -> None:
    bundle = build_fact_bundle(CATALOG)
    serialized = json.dumps(bundle)
    assert "evaluation_metrics" not in serialized and "evaluation_frames" not in serialized
    business = bundle["facts_by_role"]["business_expert"]
    fn = next(row for row in business if row["fact_id"] == "BUS:COST:FALSE_NEGATIVE_HIGH")
    precision = next(row for row in business if row["fact_id"] == "BUS:GATE:PRECISION")
    assert fn["values"]["mapped_objective"] == "recall"
    assert precision["values"]["origin"] == "engineering_admission_independent_of_false_negative_cost"
    assert {row["config_id"] for row in bundle["facts_by_role"]["algorithm_diagnostician"]} == {"C00", "C01", "C02", "C03", "C04", "C05"}
    assert next(row for row in bundle["facts_by_role"]["algorithm_diagnostician"] if row["config_id"] == "C05")["value_code"] == "replay_unavailable"


def test_typed_domain_parser_rejects_semantic_misclassification() -> None:
    bundle = build_fact_bundle(CATALOG)
    payload = _domain_payload("deployment_expert", bundle)
    parse_typed_domain_response(json.dumps(payload), "deployment_expert", bundle)
    payload["config_assessments"][0]["classification"] = "within_budget_higher_cost"
    with pytest.raises(ValueError, match="semantics"):
        parse_typed_domain_response(json.dumps(payload), "deployment_expert", bundle)


def test_business_parser_requires_recall_mapping_before_precision_gate() -> None:
    bundle = build_fact_bundle(CATALOG)
    payload = _domain_payload("business_expert", bundle)
    assert payload["objective_order"] == OBJECTIVE_ORDER
    parse_typed_domain_response(json.dumps(payload), "business_expert", bundle)
    payload["objective_order"] = ["precision", "count_accuracy", "recall", "latency"]
    with pytest.raises(ValueError, match="semantics"):
        parse_typed_domain_response(json.dumps(payload), "business_expert", bundle)


def test_typed_unified_requires_abstention_without_preregistered_utility() -> None:
    bundle = build_fact_bundle(CATALOG)
    all_fact_ids = [fact["fact_id"] for rows in bundle["facts_by_role"].values() for fact in rows]
    payload = {
        "role": "unified_synthesizer",
        "consumed_report_ids": ["algorithm_report", "business_report", "deployment_report", "typed_fact_bundle"],
        "accepted_fact_ids": all_fact_ids,
        "conflict_codes": ["COUNT_ACCURACY_VS_LATENCY_TRADEOFF"],
        "preference_profile_code": PREFERENCE_PROFILE_CODE,
        "ranked_config_ids": [],
        "abstain": True,
        "abstain_reason_codes": ["NO_PREREGISTERED_UTILITY_PROFILE"],
    }
    parse_typed_unified_response(json.dumps(payload), bundle, CATALOG)
    payload["ranked_config_ids"] = ["C04"]
    with pytest.raises(ValueError, match="ranking"):
        parse_typed_unified_response(json.dumps(payload), bundle, CATALOG)


def test_zero_precision_camera_is_uninformative_and_not_ranked() -> None:
    bundle = build_fact_bundle(ZERO_CATALOG)
    rows = expected_domain_semantics("algorithm_diagnostician", bundle)
    assert {row["classification"] for row in rows if row["config_id"] != "C05"} == {"uninformative"}
    assert next(row for row in rows if row["config_id"] == "C05")["classification"] == "replay_unavailable"
    assert next(row for row in bundle["facts_by_role"]["algorithm_diagnostician"] if row["config_id"] == "C00")["value_code"] == "uninformative"

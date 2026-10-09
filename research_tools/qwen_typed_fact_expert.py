from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROLES = ("algorithm_diagnostician", "business_expert", "deployment_expert")
OBJECTIVE_ORDER = ["count_accuracy", "recall", "precision", "latency"]
PREFERENCE_PROFILE_CODE = "FN_PRIORITY_WITH_PRECISION_GATE"


def _load_catalog(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if "evaluation_metrics" in text or "evaluation_frames" in text:
        raise ValueError("heldout_evaluation_leaked_into_typed_fact_path")
    payload = json.loads(text)
    if not {"camera_id", "configs", "eligible_config_ids", "catalog_sha256"} <= set(payload):
        raise ValueError("camera_catalog_invalid")
    return payload


def _resolve_pointer(payload: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("fact_source_pointer_invalid")
    current = payload
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            raise ValueError("fact_source_pointer_unresolved")
        current = current[key]
    return current


def build_fact_bundle(catalog_path: Path) -> dict[str, Any]:
    catalog = _load_catalog(catalog_path)
    algorithm = []
    deployment = []
    configs_by_id: dict[str, Any] = {}
    for config in catalog["configs"]:
        replay = config["camera_replay"]
        config_id = config["config_id"]
        registered = replay["status"] == "EXACT_END_TO_END_REPLAY_AVAILABLE"
        warning = replay.get("statistical_warning")
        precision_code = (
            "UNINFORMATIVE" if warning else
            "PASS" if registered and replay["empirical_precision_noninferiority"] else
            "FAIL" if registered else
            "NOT_AVAILABLE"
        )
        configs_by_id[config_id] = {
            "replay_status": "AVAILABLE" if registered else "REPLAY_UNAVAILABLE",
            "calibration_metrics": replay.get("calibration_metrics"),
            "observed_precision_floor_code": precision_code,
            "eligible_for_ranking": replay["eligible_for_ranking"],
            "calibration_frames": replay.get("calibration_frames"),
            "p95_ms": replay.get("p95_ms"),
            "latency_ratio_p95": replay.get("latency_ratio_p95"),
            "latency_admissible": replay.get("latency_admissible"),
            "oom_observation_code": "NO_OOM_OBSERVED" if registered and not replay.get("oom") else "OOM_OBSERVED" if registered else "NOT_AVAILABLE",
        }
        if replay["status"] != "EXACT_END_TO_END_REPLAY_AVAILABLE":
            algorithm.append({
                "fact_id": f"ALG:{config_id}:REPLAY",
                "config_id": config_id,
                "fact_type": "replay_availability",
                "source_field_paths": [f"/source_snapshot/configs_by_id/{config_id}/replay_status"],
                "value_code": "REPLAY_UNAVAILABLE",
            })
            deployment.append({
                "fact_id": f"DEP:{config_id}:REPLAY",
                "config_id": config_id,
                "fact_type": "replay_availability",
                "source_field_paths": [f"/source_snapshot/configs_by_id/{config_id}/replay_status"],
                "value_code": "REPLAY_UNAVAILABLE",
            })
            continue
        algorithm.append({
            "fact_id": f"ALG:{config_id}:CALIBRATION",
            "config_id": config_id,
            "fact_type": "calibration_quality",
            "source_field_paths": [
                f"/source_snapshot/configs_by_id/{config_id}/calibration_metrics",
                f"/source_snapshot/configs_by_id/{config_id}/observed_precision_floor_code",
            ],
            "values": {
                "metrics": replay["calibration_metrics"],
                "observed_precision_floor_code": precision_code,
                "eligible_for_ranking": replay["eligible_for_ranking"],
                "calibration_frames": replay["calibration_frames"],
            },
        })
        deployment.append({
            "fact_id": f"DEP:{config_id}:RESOURCE",
            "config_id": config_id,
            "fact_type": "measured_resource",
            "source_field_paths": [
                f"/source_snapshot/configs_by_id/{config_id}/p95_ms",
                f"/source_snapshot/configs_by_id/{config_id}/latency_ratio_p95",
                f"/source_snapshot/configs_by_id/{config_id}/latency_admissible",
                f"/source_snapshot/configs_by_id/{config_id}/oom_observation_code",
            ],
            "values": {
                "p95_ms": replay["p95_ms"],
                "latency_ratio_p95": replay["latency_ratio_p95"],
                "latency_admissible": replay["latency_admissible"],
                "oom_observation_code": "NO_OOM_OBSERVED" if not replay["oom"] else "OOM_OBSERVED",
            },
        })
    business = [
        {"fact_id": "BUS:PRIMARY:COUNT_ACCURACY", "fact_type": "business_objective", "source_field_paths": ["/source_snapshot/business_policy/primary_objective"], "values": {"objective": "count_accuracy", "metric": "count_mae", "direction": "minimize"}},
        {"fact_id": "BUS:COST:FALSE_NEGATIVE_HIGH", "fact_type": "error_cost", "source_field_paths": ["/source_snapshot/business_policy/false_negative_cost"], "values": {"error": "false_negative", "cost": "high", "mapped_objective": "recall"}},
        {"fact_id": "BUS:COST:FALSE_POSITIVE_MEDIUM", "fact_type": "error_cost", "source_field_paths": ["/source_snapshot/business_policy/false_positive_cost"], "values": {"error": "false_positive", "cost": "medium", "mapped_objective": "precision"}},
        {"fact_id": "BUS:GATE:PRECISION", "fact_type": "engineering_admission_gate", "source_field_paths": ["/source_snapshot/business_policy/precision_gate_origin"], "values": {"gate": "observed_precision_floor", "origin": "engineering_admission_independent_of_false_negative_cost"}},
        {"fact_id": "BUS:GATE:LATENCY", "fact_type": "engineering_admission_gate", "source_field_paths": ["/source_snapshot/business_policy/latency_cap_ratio"], "values": {"gate": "latency_ratio_p95", "maximum": 3.0}},
    ]
    bundle = {
        "schema_version": "sceneconfig.qwen-typed-fact-bundle.v1",
        "camera_id": catalog["camera_id"],
        "catalog_sha256": catalog["catalog_sha256"],
        "evidence_status": "DEVELOPMENT_ONLY_CALIBRATION_ONLY",
        "holdout_opened": False,
        "source_snapshot": {
            "configs_by_id": configs_by_id,
            "business_policy": {
                "primary_objective": "count_accuracy",
                "false_negative_cost": "high",
                "false_positive_cost": "medium",
                "precision_gate_origin": "engineering_admission_independent_of_false_negative_cost",
                "latency_cap_ratio": 3.0,
                "utility_profile_status": "NOT_PREREGISTERED",
            },
        },
        "facts_by_role": {
            "algorithm_diagnostician": algorithm,
            "business_expert": business,
            "deployment_expert": deployment,
        },
    }
    for role in ("algorithm_diagnostician", "deployment_expert"):
        codes = {row["config_id"]: row["classification"] for row in expected_domain_semantics(role, bundle)}
        for fact in bundle["facts_by_role"][role]:
            fact["value_code"] = codes[fact["config_id"]]
    for rows in bundle["facts_by_role"].values():
        for fact in rows:
            for pointer in fact["source_field_paths"]:
                _resolve_pointer(bundle, pointer)
    return bundle


def expected_domain_semantics(role: str, bundle: dict[str, Any]) -> Any:
    facts = bundle["facts_by_role"][role]
    if role == "business_expert":
        return {
            "objective_order": OBJECTIVE_ORDER,
            "precision_gate_origin": "engineering_admission_independent_of_false_negative_cost",
        }
    if role == "algorithm_diagnostician":
        baseline = next(row for row in facts if row["config_id"] == "C00")["values"]["metrics"]
        rows = []
        for fact in facts:
            if fact["fact_type"] == "replay_availability":
                rows.append({"config_id": fact["config_id"], "classification": "replay_unavailable", "supporting_fact_ids": [fact["fact_id"]]})
                continue
            values = fact["values"]
            metrics = values["metrics"]
            if values["observed_precision_floor_code"] == "UNINFORMATIVE":
                classification = "uninformative"
            elif values["observed_precision_floor_code"] != "PASS" or not values["eligible_for_ranking"]:
                classification = "inadmissible"
            elif fact["config_id"] == "C00":
                classification = "baseline"
            elif all(metrics[key] == baseline[key] for key in ("count_mae", "precision", "recall")):
                classification = "no_quality_gain"
            elif metrics["count_mae"] <= baseline["count_mae"] and metrics["precision"] >= baseline["precision"] and metrics["recall"] >= baseline["recall"]:
                classification = "quality_improves"
            else:
                classification = "quality_tradeoff"
            rows.append({"config_id": fact["config_id"], "classification": classification, "supporting_fact_ids": [fact["fact_id"]]})
        return rows
    available = [row for row in facts if row["fact_type"] != "replay_availability"]
    minimum = min(row["values"]["p95_ms"] for row in available if row["values"]["latency_admissible"] and row["values"]["oom_observation_code"] == "NO_OOM_OBSERVED")
    rows = []
    for fact in facts:
        if fact["fact_type"] == "replay_availability":
            rows.append({"config_id": fact["config_id"], "classification": "replay_unavailable", "supporting_fact_ids": [fact["fact_id"]]})
            continue
        values = fact["values"]
        if not values["latency_admissible"] or values["oom_observation_code"] == "OOM_OBSERVED":
            classification = "inadmissible"
        elif values["p95_ms"] == minimum:
            classification = "lowest_latency"
        else:
            classification = "within_budget_higher_cost"
        rows.append({"config_id": fact["config_id"], "classification": classification, "supporting_fact_ids": [fact["fact_id"]]})
    return rows


def typed_domain_prompt(role: str, bundle_path: Path) -> str:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    facts = bundle["facts_by_role"][role]
    fact_ids = [row["fact_id"] for row in facts]
    if role == "business_expert":
        contract = (
            'Return exactly: {"role":"business_expert","consumed_fact_ids":[...],'
            '"objective_order":["count_accuracy","recall","precision","latency"],'
            '"precision_gate_origin":"engineering_admission_independent_of_false_negative_cost","abstain":false}.'
        )
    else:
        allowed = (
            ["baseline", "inadmissible", "uninformative", "replay_unavailable", "no_quality_gain", "quality_improves", "quality_tradeoff"]
            if role == "algorithm_diagnostician"
            else ["inadmissible", "replay_unavailable", "lowest_latency", "within_budget_higher_cost"]
        )
        contract = (
            'Return exactly one JSON object with keys role, consumed_fact_ids, config_assessments, abstain. '
            'Each config_assessment must have exactly config_id, classification, supporting_fact_ids. '
            f'Legal classifications: {json.dumps(allowed)}. Set abstain=false; only the unified role may abstain. '
            'For every config, copy that fact card value_code exactly into classification.'
        )
        if role == "algorithm_diagnostician":
            contract += (
                " The deterministic adapter already applied replay, uncertainty, admission, baseline, and quality precedence. Do not recompute or override value_code."
            )
        else:
            contract += (
                " The deterministic adapter already applied replay, OOM observation, and latency rules. Do not recompute or override value_code."
            )
    return (
        f"You are the typed-role processor for a calibration-only fixed-camera smoke. Set the role field exactly to {json.dumps(role)}. Facts are immutable. "
        "Do not restate numbers, produce prose, infer missing data, or authorize deployment. "
        f"Consume every fact_id exactly once: {json.dumps(fact_ids)}. Facts: {json.dumps(facts, separators=(',', ':'), sort_keys=True)}. "
        f"{contract} Output JSON only."
    )


def parse_typed_domain_response(text: str, role: str, bundle: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(text.strip())
    facts = bundle["facts_by_role"][role]
    fact_ids = [row["fact_id"] for row in facts]
    if payload.get("role") != role or payload.get("consumed_fact_ids") != fact_ids or payload.get("abstain") is not False:
        raise ValueError("typed_domain_identity_or_coverage_invalid")
    expected = expected_domain_semantics(role, bundle)
    if role == "business_expert":
        if set(payload) != {"role", "consumed_fact_ids", "objective_order", "precision_gate_origin", "abstain"}:
            raise ValueError("typed_business_keys_invalid")
        if payload["objective_order"] != expected["objective_order"] or payload["precision_gate_origin"] != expected["precision_gate_origin"]:
            raise ValueError("typed_business_semantics_invalid")
    else:
        if set(payload) != {"role", "consumed_fact_ids", "config_assessments", "abstain"}:
            raise ValueError("typed_domain_keys_invalid")
        if payload["config_assessments"] != expected:
            raise ValueError("typed_domain_semantics_invalid")
    return payload


def typed_unified_prompt(report_paths: Iterable[Path], bundle_path: Path, catalog_path: Path) -> str:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reports = []
    for report_id, path in zip(("algorithm_report", "business_report", "deployment_report"), report_paths, strict=True):
        reports.append({"report_id": report_id, "payload": json.loads(Path(path).read_text(encoding="utf-8"))})
    eligible = _load_catalog(catalog_path)["eligible_config_ids"]
    reports.append({"report_id": "typed_fact_bundle", "payload": {"catalog_sha256": bundle["catalog_sha256"], "eligible_config_ids": eligible, "utility_profile_status": "NOT_PREREGISTERED"}})
    all_fact_ids = [fact["fact_id"] for rows in bundle["facts_by_role"].values() for fact in rows]
    return (
        "You are the typed unified-role processor. Set the role field exactly to \"unified_synthesizer\". No scalar utility profile was preregistered, so you must abstain from ranking. "
        "Accept the bounded fact IDs and emit only finite codes. Do not restate numbers or add prose. "
        f"Reports: {json.dumps(reports, separators=(',', ':'), sort_keys=True)}. Legal config_ids: {json.dumps(eligible)}. "
        f"Legal accepted_fact_ids: {json.dumps(all_fact_ids)}. Preference profile code must be {PREFERENCE_PROFILE_CODE}. "
        "Return exactly one JSON object with keys role, consumed_report_ids, accepted_fact_ids, conflict_codes, "
        "preference_profile_code, ranked_config_ids, abstain, abstain_reason_codes. Consume report IDs algorithm_report, "
        "business_report, deployment_report, typed_fact_bundle in that order. ranked_config_ids must be empty, abstain must be true, "
        "and abstain_reason_codes must be [\"NO_PREREGISTERED_UTILITY_PROFILE\"]. Output JSON only."
    )


def parse_typed_unified_response(text: str, bundle: dict[str, Any], catalog_path: Path) -> dict[str, Any]:
    payload = json.loads(text.strip())
    expected_keys = {"role", "consumed_report_ids", "accepted_fact_ids", "conflict_codes", "preference_profile_code", "ranked_config_ids", "abstain", "abstain_reason_codes"}
    if set(payload) != expected_keys or payload["role"] != "unified_synthesizer" or payload["abstain"] is not True:
        raise ValueError("typed_unified_shape_invalid")
    if payload["consumed_report_ids"] != ["algorithm_report", "business_report", "deployment_report", "typed_fact_bundle"]:
        raise ValueError("typed_unified_report_coverage_invalid")
    if payload["ranked_config_ids"] != [] or payload["abstain_reason_codes"] != ["NO_PREREGISTERED_UTILITY_PROFILE"]:
        raise ValueError("typed_unified_ranking_invalid")
    all_fact_ids = {fact["fact_id"] for rows in bundle["facts_by_role"].values() for fact in rows}
    if set(payload["accepted_fact_ids"]) != all_fact_ids:
        raise ValueError("typed_unified_fact_reference_invalid")
    if payload["preference_profile_code"] != PREFERENCE_PROFILE_CODE:
        raise ValueError("typed_unified_semantics_invalid")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-bundle")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    domain = sub.add_parser("domain-prompt")
    domain.add_argument("--role", choices=ROLES, required=True)
    domain.add_argument("--bundle", type=Path, required=True)
    unified = sub.add_parser("unified-prompt")
    unified.add_argument("--domain-report", type=Path, action="append", required=True)
    unified.add_argument("--bundle", type=Path, required=True)
    unified.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build-bundle":
        payload = build_fact_bundle(args.catalog)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"camera_id": payload["camera_id"], "holdout_opened": False}, sort_keys=True))
    elif args.action == "domain-prompt":
        print(typed_domain_prompt(args.role, args.bundle))
    else:
        print(typed_unified_prompt(args.domain_report, args.bundle, args.catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

import pytest

from research_tools.qwen_scene_expert_v2 import (
    evaluate_qctx,
    expected_qctx_case_contract,
    parse_qctx_domain_response,
    parse_qctx_response,
    parse_visual_report,
    parse_qctx_unified_response,
    qctx_domain_prompt,
    camera_role_inputs,
    qctx_camera_domain_prompt,
    qctx_camera_unified_prompt,
    qctx_prompt,
    qctx_unified_prompt,
    load_qwen_multirole_proposal,
    qwen_multirole_role_input,
    validate_synthesis,
)


def test_qctx_parser_rejects_hallucinated_evidence() -> None:
    payload = {"role": "visual_analyst", "evidence_ids": ["sheet_9"], "observations": [], "summary": "x"}
    with pytest.raises(ValueError, match="exactly_cover"):
        parse_qctx_response(json.dumps(payload), {"sheet_0"})


def test_qctx_parser_requires_all_input_evidence_ids() -> None:
    payload = {"role": "visual_analyst", "evidence_ids": ["sheet_0"], "observations": [], "summary": "x"}
    with pytest.raises(ValueError, match="exactly_cover"):
        parse_qctx_response(json.dumps(payload), {"sheet_0", "sheet_1"})


def test_qctx_domain_requires_all_inputs() -> None:
    payload = {"role": "algorithm_diagnostician", "consumed_input_ids": ["detector_ledger_0"], "findings": [], "constraints": [], "uncertainties": [], "summary": "x"}
    with pytest.raises(ValueError, match="exactly_cover"):
        parse_qctx_domain_response(json.dumps(payload), "algorithm_diagnostician", {"detector_ledger_0", "track_ledger_0"})


def test_qctx_unified_requires_all_reports() -> None:
    payload = {"role": "unified_synthesizer", "consumed_report_ids": ["visual_pool"], "agreements": [], "conflicts": [], "minority_report_ids": [], "ranked_config_ids": ["C00"], "abstain": False, "summary": "x"}
    with pytest.raises(ValueError, match="exactly_cover"):
        parse_qctx_unified_response(json.dumps(payload), allowed_report_ids={"visual_pool", "algorithm_report"}, legal_config_ids={"C00"})


def test_visual_report_is_window_and_evidence_bound() -> None:
    payload = {
        "role": "visual_analyst",
        "window_id": "w00",
        "observations": [{
            "evidence_id": "cam0_w00_f0001",
            "region_id": "t0",
            "risk_types": ["far_small_person"],
            "confidence": "medium",
            "description": "small silhouettes near the perspective apex",
        }],
        "uncertainties": [],
    }
    assert parse_visual_report(json.dumps(payload), {"cam0_w00_f0001"}, "w00") == payload
    with pytest.raises(ValueError, match="identity"):
        parse_visual_report(json.dumps(payload), {"cam0_w00_f0001"}, "w01")


def test_synthesis_cannot_duplicate_configs_or_invent_evidence() -> None:
    payload = {
        "role": "unified_synthesizer",
        "agreements": [],
        "conflicts": [],
        "ranked_configs": [
            {"config_id": "C00", "supporting_evidence_ids": ["e0"], "opposing_evidence_ids": [], "rationale": "x"},
            {"config_id": "C00", "supporting_evidence_ids": ["e0"], "opposing_evidence_ids": [], "rationale": "y"},
        ],
        "abstain": False,
        "abstain_reason": "",
    }
    with pytest.raises(ValueError, match="illegal_or_duplicate"):
        validate_synthesis(payload, allowed_evidence_ids={"e0"}, legal_config_ids={"C00"})


def test_qctx_prompt_declares_only_bounded_role_and_ids() -> None:
    prompt = qctx_prompt(["sheet_0", "sheet_1"])
    assert "visual_analyst" in prompt
    assert "sheet_0" in prompt and "sheet_1" in prompt
    assert "do not count people" in prompt


def test_qctx_domain_prompt_embeds_post_validation_contract() -> None:
    prompt = qctx_domain_prompt("algorithm_diagnostician")
    for field in ("role", "consumed_input_ids", "findings", "constraints", "uncertainties", "summary"):
        assert field in prompt
    assert "Do not emit analysis" in prompt
    assert "detector_ledger_0" in prompt and "track_ledger_0" in prompt


def test_unified_prompt_binds_candidate_catalog_and_requires_safe_abstention(tmp_path: Path) -> None:
    visual = tmp_path / "visual.json"
    visual.write_text(json.dumps({"evidence_status": "fixture_only"}), encoding="utf-8")
    domains = []
    for name in ("algorithm", "business", "deployment"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"role": name}), encoding="utf-8")
        domains.append(path)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"catalog_id": "catalog-v1", "configs": []}), encoding="utf-8")
    prompt = qctx_unified_prompt([visual], domains, catalog)
    assert "candidate_config_catalog" in prompt
    assert "abstain=true" in prompt
    assert "rank only immutable catalog entries" in prompt


def test_camera_role_inputs_are_domain_isolated_and_withhold_evaluation() -> None:
    catalog = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/camera-catalog-010204-v1/decision-catalog.json")
    algorithm = camera_role_inputs("algorithm_diagnostician", catalog)
    deployment = camera_role_inputs("deployment_expert", catalog)
    business = camera_role_inputs("business_expert", catalog)
    assert "p95_ms" not in json.dumps(algorithm)
    assert "calibration_metrics" not in json.dumps(deployment)
    assert "calibration_metrics" not in json.dumps(business)
    for payload in (algorithm, deployment, business):
        serialized = json.dumps(payload)
        assert "evaluation_metrics" not in serialized
        assert "evaluation_frames" not in serialized


def test_camera_prompts_bind_only_eligible_ids(tmp_path: Path) -> None:
    catalog = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/camera-catalog-010204-v1/decision-catalog.json")
    prompt = qctx_camera_domain_prompt("algorithm_diagnostician", catalog)
    assert "camera_calibration_ledger" in prompt
    assert "Held-out evaluation data is unavailable" in prompt
    reports = []
    roles = ("algorithm_diagnostician", "business_expert", "deployment_expert")
    for role in roles:
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({
            "role": role,
            "consumed_input_ids": [next(iter(camera_role_inputs(role, catalog)))],
            "findings": [], "constraints": [], "uncertainties": [], "summary": "ok",
        }), encoding="utf-8")
        reports.append(path)
    unified = qctx_camera_unified_prompt(reports, catalog)
    assert 'Legal ranked_config_ids: ["C00", "C02", "C03", "C04"]' in unified
    assert "C01" in unified and '"eligible_for_ranking":false' in unified
    assert "evaluation_metrics" not in unified


def test_multirole_router_gives_subagent_only_its_package_and_main_only_report_manifest() -> None:
    proposal_path = Path(
        ".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/"
        "evidence-router-010204-v2/proposal-evidence.json"
    )
    proposal = load_qwen_multirole_proposal(proposal_path)
    assert proposal["proposal_scope"] == "CALIBRATION_ONLY"

    algorithm = qwen_multirole_role_input("algorithm_diagnostician", proposal_path)
    serialized_algorithm = json.dumps(algorithm)
    assert "calibration_detection_ledger" in serialized_algorithm
    assert "resource_ledger" not in serialized_algorithm
    assert "profile_registry" not in serialized_algorithm

    deployment = qwen_multirole_role_input("deployment_resource_expert", proposal_path)
    serialized_deployment = json.dumps(deployment)
    assert "resource_ledger" in serialized_deployment
    assert "calibration_detection_ledger" not in serialized_deployment

    unified = qwen_multirole_role_input("unified_synthesizer", proposal_path)
    serialized_unified = json.dumps(unified)
    assert unified["raw_evidence_access"] is False
    assert unified["eligible_config_ids"] == ["C00", "C02", "C03", "C04"]
    assert "calibration_detection_ledger" not in serialized_unified
    assert "resource_ledger" not in serialized_unified


def test_full_capacity_notice_is_not_a_truncation_warning(tmp_path: Path) -> None:
    from research_tools.qwen_scene_expert_v2 import _scan_warnings
    warnings = _scan_warnings("llama_context: n_ctx_seq < n_ctx_train -- the full capacity of the model will not be utilized")
    assert warnings["truncation"] is False


def _make_complete_qctx_run(tmp_path: Path, *, omit_telemetry: bool = False) -> None:
    cases = []
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    report_ids = ["visual_pool", "algorithm_report", "business_report", "deployment_report"]
    role_inputs = {
        "algorithm_diagnostician": ["detector_ledger_0", "track_ledger_0"],
        "business_expert": ["business_card_0"],
        "deployment_expert": ["resource_ledger_0"],
    }
    for case_id, repeat_of in expected_qctx_case_contract().items():
        ctx = 32768
        row = {"case_id": case_id}
        if repeat_of:
            row["repeat_of"] = repeat_of
        if case_id.startswith("ctx"):
            ctx = int(case_id.split("_")[0][3:])
            image_count = int(case_id.split("_")[1][3:])
            evidence_ids = [f"sheet_{index}" for index in range(image_count)]
            row.update({"ctx_size": ctx, "image_count": image_count, "evidence_ids": evidence_ids})
            payload = {"role": "visual_analyst", "evidence_ids": evidence_ids, "observations": [], "summary": "ok"}
        else:
            row.update({"ctx_size": ctx, "map_member": True})
            if "_visual_" in case_id:
                index = int(case_id.rsplit("_", 1)[1])
                evidence_ids = [f"sheet_{index}"]
                row.update({"response_kind": "visual", "image_count": 1, "evidence_ids": evidence_ids})
                payload = {"role": "visual_analyst", "evidence_ids": evidence_ids, "observations": [], "summary": "ok"}
            elif case_id.endswith("_unified"):
                row.update({"response_kind": "unified", "image_count": 0, "consumed_report_ids": report_ids, "legal_config_ids": [f"C0{i}" for i in range(6)]})
                payload = {"role": "unified_synthesizer", "consumed_report_ids": report_ids, "agreements": [], "conflicts": [], "minority_report_ids": [], "ranked_config_ids": ["C00"], "abstain": False, "summary": "ok"}
            else:
                role = next(name for name in role_inputs if case_id.endswith(name))
                inputs = role_inputs[role]
                row.update({"response_kind": "domain", "role": role, "image_count": 0, "consumed_input_ids": inputs})
                payload = {"role": role, "consumed_input_ids": inputs, "findings": [], "constraints": [], "uncertainties": [], "summary": "ok"}
        cases.append(row)
        (case_dir / f"{case_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        log = "clean" if omit_telemetry else f"n_ctx = {ctx}\nllama_perf_context_print: prompt eval time = 1.0 ms / 42 tokens\n"
        (case_dir / f"{case_id}.log").write_text(log, encoding="utf-8")
        (case_dir / f"{case_id}.meta.json").write_text(json.dumps({"exit_code": 0, "wall_ms": 1, "peak_gpu_memory_mib": 2, "ctx_size_requested": ctx}), encoding="utf-8")
    (tmp_path / "case-manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")


def test_qctx_evaluator_requires_complete_matrix_telemetry_and_repeats(tmp_path: Path) -> None:
    _make_complete_qctx_run(tmp_path)
    result = evaluate_qctx(tmp_path, tmp_path / "result.json")
    assert result["passed"] is True
    assert result["gate"]["expected_case_count"] == 37
    assert result["next_gate"] == "G0_5_HEADROOM"


def test_qctx_evaluator_rejects_missing_token_telemetry(tmp_path: Path) -> None:
    _make_complete_qctx_run(tmp_path, omit_telemetry=True)
    result = evaluate_qctx(tmp_path, tmp_path / "result.json")
    assert result["passed"] is False
    assert result["gate"]["telemetry_failure_code"] == "TOKEN_TELEMETRY_UNAVAILABLE"


def test_qctx_evaluator_rejects_incomplete_case_matrix(tmp_path: Path) -> None:
    _make_complete_qctx_run(tmp_path)
    manifest_path = tmp_path / "case-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = evaluate_qctx(tmp_path, tmp_path / "result.json")
    assert result["passed"] is False
    assert result["gate"]["case_matrix_complete"] is False

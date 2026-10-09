"""Contracts and G0-QCTX validator for the multi-role Qwen scene expert.

This module deliberately does not call Qwen.  The remote launcher starts one
fresh llama.cpp process per role/case; this code validates the files exchanged
between those stateless processes and the deterministic orchestrator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from research_tools.build_qwen_multirole_evidence_packages import validate_proposal_package


ROLE_NAMES = (
    "visual_analyst",
    "algorithm_diagnostician",
    "business_expert",
    "deployment_expert",
    "unified_synthesizer",
)
RISK_TYPES = {
    "far_small_person",
    "occlusion",
    "perspective_scale_change",
    "background_confusion",
    "temporal_instability",
    "duplicate_pressure",
    "uncertain",
}
WARNING_PATTERNS = {
    "truncation": re.compile(r"truncat|context(?: window)?(?: is)? full|context[^\n]*exceed|too many tokens", re.I),
    "token_position": re.compile(r"token.position|position.*token|multimodal.*position", re.I),
    "oom": re.compile(r"out of memory|cuda.*alloc|failed to allocate", re.I),
}
NEW_TOKEN_PATTERN = re.compile(r"with\s+(\d+)\s+new tokens", re.I)
MTMD_CHUNK_PATTERN = re.compile(r"n_chunks\s*=\s*(\d+).*total\s*=\s*(\d+)", re.I)
CTX_PATTERN = re.compile(r"(?:n_ctx|ctx_size|context size)\s*[=:]\s*(\d+)", re.I)
PROMPT_TOKEN_PATTERN = re.compile(r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens", re.I)
IMAGE_TOKEN_PATTERN = re.compile(r"n_tokens_batch\s*=\s*(\d+)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_object(payload: Any, keys: set[str], error: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError(error)
    return payload


def parse_qctx_response(text: str, allowed_evidence_ids: set[str]) -> dict[str, Any]:
    """Parse the small G0 visual-agent response without repairing it."""
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("qctx_response_not_exact_json") from exc
    _exact_object(payload, {"role", "evidence_ids", "observations", "summary"}, "qctx_keys_invalid")
    if payload["role"] != "visual_analyst":
        raise ValueError("qctx_role_invalid")
    evidence_ids = payload["evidence_ids"]
    if not isinstance(evidence_ids, list) or not evidence_ids or not all(isinstance(v, str) for v in evidence_ids):
        raise ValueError("qctx_evidence_ids_invalid")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("qctx_evidence_ids_duplicate")
    if set(evidence_ids) != allowed_evidence_ids:
        raise ValueError("qctx_evidence_ids_must_exactly_cover_input")
    if not isinstance(payload["observations"], list) or not all(isinstance(v, str) for v in payload["observations"]):
        raise ValueError("qctx_observations_invalid")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("qctx_summary_required")
    return payload


def parse_qctx_domain_response(text: str, role: str, allowed_input_ids: set[str]) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("qctx_domain_not_exact_json") from exc
    _exact_object(payload, {"role", "consumed_input_ids", "findings", "constraints", "uncertainties", "summary"}, "qctx_domain_keys_invalid")
    if payload["role"] != role or role not in {"algorithm_diagnostician", "business_expert", "deployment_expert"}:
        raise ValueError("qctx_domain_role_invalid")
    ids = payload["consumed_input_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != allowed_input_ids:
        raise ValueError("qctx_domain_inputs_must_exactly_cover_input")
    for field in ("findings", "constraints", "uncertainties"):
        if not isinstance(payload[field], list) or not all(isinstance(v, str) for v in payload[field]):
            raise ValueError(f"qctx_domain_{field}_invalid")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("qctx_domain_summary_required")
    return payload


def parse_qctx_unified_response(text: str, allowed_report_ids: set[str], legal_config_ids: set[str]) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("qctx_unified_not_exact_json") from exc
    _exact_object(
        payload,
        {"role", "consumed_report_ids", "agreements", "conflicts", "minority_report_ids", "ranked_config_ids", "abstain", "summary"},
        "qctx_unified_keys_invalid",
    )
    if payload["role"] != "unified_synthesizer":
        raise ValueError("qctx_unified_role_invalid")
    consumed = payload["consumed_report_ids"]
    if not isinstance(consumed, list) or len(consumed) != len(set(consumed)) or set(consumed) != allowed_report_ids:
        raise ValueError("qctx_unified_reports_must_exactly_cover_input")
    minority = payload["minority_report_ids"]
    if not isinstance(minority, list) or len(minority) != len(set(minority)) or not set(minority) <= allowed_report_ids:
        raise ValueError("qctx_unified_minority_reports_invalid")
    for field in ("agreements", "conflicts"):
        if not isinstance(payload[field], list) or not all(isinstance(v, str) for v in payload[field]):
            raise ValueError(f"qctx_unified_{field}_invalid")
    configs = payload["ranked_config_ids"]
    if not isinstance(configs, list) or len(configs) != len(set(configs)) or not set(configs) <= legal_config_ids:
        raise ValueError("qctx_unified_configs_invalid")
    if not isinstance(payload["abstain"], bool) or not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise ValueError("qctx_unified_terminal_fields_invalid")
    if payload["abstain"] == bool(configs):
        raise ValueError("qctx_unified_abstention_inconsistent")
    return payload


def parse_visual_report(text: str, allowed_evidence_ids: set[str], window_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("visual_report_not_exact_json") from exc
    _exact_object(payload, {"role", "window_id", "observations", "uncertainties"}, "visual_report_keys_invalid")
    if payload["role"] != "visual_analyst" or payload["window_id"] != window_id:
        raise ValueError("visual_report_identity_invalid")
    if not isinstance(payload["uncertainties"], list) or not all(isinstance(v, str) for v in payload["uncertainties"]):
        raise ValueError("visual_uncertainties_invalid")
    if not isinstance(payload["observations"], list):
        raise ValueError("visual_observations_invalid")
    for row in payload["observations"]:
        _exact_object(
            row,
            {"evidence_id", "region_id", "risk_types", "confidence", "description"},
            "visual_observation_keys_invalid",
        )
        if row["evidence_id"] not in allowed_evidence_ids:
            raise ValueError("visual_evidence_id_hallucinated")
        if not isinstance(row["region_id"], str) or not row["region_id"]:
            raise ValueError("visual_region_invalid")
        risks = row["risk_types"]
        if not isinstance(risks, list) or not risks or not set(risks) <= RISK_TYPES:
            raise ValueError("visual_risk_types_invalid")
        if row["confidence"] not in {"low", "medium", "high"}:
            raise ValueError("visual_confidence_invalid")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError("visual_description_required")
    return payload


def validate_synthesis(
    payload: dict[str, Any], *, allowed_evidence_ids: set[str], legal_config_ids: set[str]
) -> dict[str, Any]:
    """Validate the final proposal; do not infer, de-duplicate, or repair fields."""
    _exact_object(
        payload,
        {"role", "agreements", "conflicts", "ranked_configs", "abstain", "abstain_reason"},
        "synthesis_keys_invalid",
    )
    if payload["role"] != "unified_synthesizer":
        raise ValueError("synthesis_role_invalid")
    if not isinstance(payload["agreements"], list) or not isinstance(payload["conflicts"], list):
        raise ValueError("synthesis_deliberation_invalid")
    configs = payload["ranked_configs"]
    if not isinstance(configs, list):
        raise ValueError("synthesis_configs_invalid")
    seen: set[str] = set()
    for row in configs:
        _exact_object(row, {"config_id", "supporting_evidence_ids", "opposing_evidence_ids", "rationale"}, "synthesis_config_keys_invalid")
        config_id = row["config_id"]
        if config_id not in legal_config_ids or config_id in seen:
            raise ValueError("synthesis_config_illegal_or_duplicate")
        seen.add(config_id)
        for field in ("supporting_evidence_ids", "opposing_evidence_ids"):
            values = row[field]
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ValueError("synthesis_evidence_list_invalid")
            if not set(values) <= allowed_evidence_ids:
                raise ValueError("synthesis_evidence_id_hallucinated")
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("synthesis_rationale_required")
    if not isinstance(payload["abstain"], bool) or not isinstance(payload["abstain_reason"], str):
        raise ValueError("synthesis_abstention_invalid")
    if payload["abstain"] and configs:
        raise ValueError("synthesis_abstention_must_have_no_configs")
    if not payload["abstain"] and not configs:
        raise ValueError("synthesis_non_abstention_requires_config")
    return payload


def qctx_prompt(evidence_ids: Iterable[str]) -> str:
    ids = list(evidence_ids)
    return (
        "You are the visual_analyst sub-agent in a fixed-camera scene expert. "
        "Treat each attached contact sheet as runtime evidence only; do not count people, propose detector settings, "
        "or claim ground truth. Describe visible small-target, occlusion, perspective, background-confusion, and temporal risks. "
        f"The only legal evidence_ids are {json.dumps(ids)}. Include every legal ID exactly once in evidence_ids and invent none. "
        "Return only the schema-conforming JSON object."
    )


def qctx_domain_prompt(role: str) -> str:
    inputs = {
        "algorithm_diagnostician": {
            "detector_ledger_0": {"yolo11_count": [2, 1, 2], "yolo26_roi_count": [3, 2, 3], "small_box_ratio": 0.42},
            "track_ledger_0": {"one_gap_recoveries": 2, "short_track_births": 4, "duplicate_pressure": 0.08},
        },
        "business_expert": {
            "business_card_0": {"source": "pre_registered_camera_contract", "false_negative_cost": "high", "false_positive_cost": "medium", "latency_cap_ratio": 3.0},
        },
        "deployment_expert": {
            "resource_ledger_0": {"yolo11_p95_ms": 18.0, "candidate_p95_ms": [25.0, 41.0, 57.0], "peak_vram_mib": [3200, 7100, 9800], "oom": [False, False, False]},
        },
    }
    if role not in inputs:
        raise ValueError("unsupported_qctx_domain_role")
    rows = inputs[role]
    return (
        f"You are the {role} role-isolated sub-agent in a fixed-camera scene expert smoke test. "
        "Use only your domain inputs; do not infer ground truth, read other roles, or authorize deployment. "
        f"Inputs: {json.dumps(rows, separators=(',', ':'), sort_keys=True)}. "
        f"Include every consumed_input_id exactly once: {json.dumps(list(rows))}. "
        "Output exactly one JSON object with exactly these top-level keys in this order: "
        "role, consumed_input_ids, findings, constraints, uncertainties, summary. "
        f"Set role to {json.dumps(role)}; consumed_input_ids must be the required ID array; "
        "findings, constraints, and uncertainties must each be arrays of strings; summary must be a non-empty string. "
        "Findings may only restate or compare supplied fields. Put hypotheses in uncertainties, label them unverified, "
        "and do not introduce mechanisms, sensors, metrics, or causal explanations absent from the role inputs. "
        "Do not emit analysis, reasoning, think tags, Markdown, or any text outside the JSON object."
    )


def _load_camera_decision_catalog(path: Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    if "evaluation_metrics" in raw or "evaluation_frames" in raw:
        raise ValueError("heldout_evaluation_leaked_into_camera_decision_catalog")
    payload = json.loads(raw)
    required = {"camera_id", "configs", "eligible_config_ids", "catalog_sha256", "evidence_boundary"}
    if not required <= set(payload):
        raise ValueError("camera_decision_catalog_fields_missing")
    eligible = payload["eligible_config_ids"]
    if not isinstance(eligible, list) or len(eligible) != len(set(eligible)):
        raise ValueError("camera_decision_catalog_eligible_ids_invalid")
    config_ids = {row.get("config_id") for row in payload["configs"]}
    if not set(eligible) <= config_ids:
        raise ValueError("camera_decision_catalog_eligible_ids_unknown")
    return payload


def camera_role_inputs(role: str, candidate_catalog_path: Path) -> dict[str, Any]:
    catalog = _load_camera_decision_catalog(candidate_catalog_path)
    configs = catalog["configs"]
    if role == "algorithm_diagnostician":
        rows = []
        for config in configs:
            replay = config["camera_replay"]
            if replay["status"] != "EXACT_END_TO_END_REPLAY_AVAILABLE":
                continue
            rows.append({
                "config_id": config["config_id"],
                "calibration_frames": replay["calibration_frames"],
                "calibration_metrics": replay["calibration_metrics"],
                "calibration_precision_change_vs_b0": replay["calibration_precision_change_vs_b0"],
                "empirical_precision_noninferiority": replay["empirical_precision_noninferiority"],
                "statistical_warning": replay["statistical_warning"],
            })
        return {"camera_calibration_ledger": {
            "camera_id": catalog["camera_id"],
            "evidence_boundary": catalog["evidence_boundary"],
            "rows": rows,
        }}
    if role == "business_expert":
        return {"camera_business_card": {
            "camera_id": catalog["camera_id"],
            "source": "pre_registered_engineering_policy",
            "false_negative_cost": "high",
            "false_positive_cost": "medium",
            "precision_noninferiority_required": True,
            "latency_cap_ratio": 3.0,
            "deployment_authority": "deterministic_compiler_only",
        }}
    if role == "deployment_expert":
        rows = []
        for config in configs:
            replay = config["camera_replay"]
            if replay["status"] != "EXACT_END_TO_END_REPLAY_AVAILABLE":
                continue
            rows.append({
                "config_id": config["config_id"],
                "p95_ms": replay["p95_ms"],
                "latency_ratio_p95": replay["latency_ratio_p95"],
                "latency_admissible": replay["latency_admissible"],
                "oom": replay["oom"],
                "operator_fingerprint_sha256": replay["operator_fingerprint_sha256"],
            })
        return {"camera_resource_ledger": {
            "camera_id": catalog["camera_id"],
            "latency_cap_ratio": 3.0,
            "rows": rows,
        }}
    raise ValueError("unsupported_qctx_domain_role")


def qctx_camera_domain_prompt(role: str, candidate_catalog_path: Path) -> str:
    rows = camera_role_inputs(role, candidate_catalog_path)
    return (
        f"You are the {role} role-isolated sub-agent for a fixed-camera calibration-only ranking smoke. "
        "Use only the supplied role input. Held-out evaluation data is unavailable and must not be inferred. "
        "Do not read other roles, invent measurements, infer causality, or authorize deployment. "
        f"Inputs: {json.dumps(rows, separators=(',', ':'), sort_keys=True)}. "
        f"Include every consumed_input_id exactly once: {json.dumps(list(rows))}. "
        "Output exactly one JSON object with exactly these top-level keys in this order: "
        "role, consumed_input_ids, findings, constraints, uncertainties, summary. "
        f"Set role to {json.dumps(role)}; consumed_input_ids must be the required ID array; "
        "findings, constraints, and uncertainties must each be arrays of strings; summary must be non-empty. "
        "Use at most four concise strings in each array and keep summary under 100 words. "
        "Separate supplied measurements from uncertainty caused by the small calibration sample. "
        "Do not emit analysis, reasoning, think tags, Markdown, or any text outside the JSON object."
    )


def qctx_camera_unified_prompt(domain_paths: Iterable[Path], candidate_catalog_path: Path) -> str:
    catalog = _load_camera_decision_catalog(candidate_catalog_path)
    reports = []
    for report_id, path in zip(("algorithm_report", "business_report", "deployment_report"), domain_paths, strict=True):
        reports.append({"report_id": report_id, "payload": json.loads(Path(path).read_text(encoding="utf-8"))})
    compact_configs = []
    for config in catalog["configs"]:
        compact_configs.append({
            "config_id": config["config_id"],
            "operator": config["operator"],
            "eligible_for_ranking": config["camera_replay"]["eligible_for_ranking"],
            "replay_status": config["camera_replay"]["status"],
        })
    reports.append({"report_id": "candidate_config_catalog", "payload": {
        "camera_id": catalog["camera_id"],
        "catalog_sha256": catalog["catalog_sha256"],
        "eligible_config_ids": catalog["eligible_config_ids"],
        "configs": compact_configs,
        "visual_role_status": "UNAVAILABLE_REJECTED_RUNTIME",
    }})
    report_ids = [row["report_id"] for row in reports]
    eligible = catalog["eligible_config_ids"]
    return (
        "You are the unified_synthesizer for a calibration-only, text-ledger fixed-camera ranking smoke. "
        "The visual role is unavailable because its runtime was rejected; do not treat that absence as visual evidence. "
        "Reconcile all three domain reports, preserve conflicts, and rank only empirically eligible immutable configs. "
        "This is a proposal, not deployment authorization, and held-out evaluation outcomes are unavailable. "
        f"Legal consumed_report_ids: {json.dumps(report_ids)}. Legal ranked_config_ids: {json.dumps(eligible)}. "
        f"Reports: {json.dumps(reports, separators=(',', ':'), sort_keys=True)}. "
        "Output exactly one JSON object with exactly these top-level keys in this order: role, consumed_report_ids, "
        "agreements, conflicts, minority_report_ids, ranked_config_ids, abstain, summary. Set role to unified_synthesizer. "
        "Consume every legal report ID exactly once. Use arrays for agreements, conflicts, and minority_report_ids. "
        "Use at most four concise strings in agreements and conflicts and keep summary under 120 words. "
        "If the eligible set is empty or the supplied reports cannot support a bounded ranking, set abstain=true and ranked_config_ids=[]. "
        "Otherwise set abstain=false and rank one or more legal IDs without duplicates. Do not emit reasoning, Markdown, or text outside the JSON object."
    )


def load_qwen_multirole_proposal(path: Path) -> dict[str, Any]:
    """Load the frozen proposal package and fail closed on holdout leakage."""
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    validate_proposal_package(payload)
    return payload


def qwen_multirole_role_input(role: str, proposal_path: Path) -> dict[str, Any]:
    """Return exactly one role's package; the synthesizer receives only a report manifest."""
    proposal = load_qwen_multirole_proposal(proposal_path)
    role_packages = proposal["role_packages"]
    if role not in role_packages:
        raise ValueError("unsupported_multirole_evidence_role")
    if role == "unified_synthesizer":
        report_sources = []
        for role_name, package in role_packages.items():
            if role_name in {"unified_synthesizer", "deterministic_compiler"}:
                continue
            report_sources.append(
                {
                    "role": role_name,
                    "evidence_id": package["evidence_id"],
                    "availability": package["availability"],
                    "required_response": package.get("required_response"),
                }
            )
        return {
            "stage": proposal["stage"],
            "camera_id": proposal["camera_id"],
            "proposal_sha256": proposal["proposal_sha256"],
            "role": role,
            "eligible_config_ids": role_packages[role]["eligible_config_ids"],
            "allowed_inputs": role_packages[role]["allowed_inputs"],
            "expected_validated_role_reports": report_sources,
            "raw_evidence_access": False,
            "deployment_authority": proposal["deployment_authority"],
        }
    return {
        "stage": proposal["stage"],
        "camera_id": proposal["camera_id"],
        "proposal_sha256": proposal["proposal_sha256"],
        "role": role,
        "role_package": role_packages[role],
        "deployment_authority": proposal["deployment_authority"],
        "qwen_deployment_authority": False,
    }


def qctx_unified_prompt(
    visual_paths: Iterable[Path],
    domain_paths: Iterable[Path],
    candidate_catalog_path: Path | None = None,
) -> str:
    reports = []
    visual_rows = []
    for index, path in enumerate(visual_paths):
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invalid_report_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        visual_rows.append({"window_id": index, "payload": payload})
    reports.append({"report_id": "visual_pool", "payload": visual_rows})
    for report_id, path in zip(("algorithm_report", "business_report", "deployment_report"), domain_paths, strict=True):
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invalid_report_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        reports.append({"report_id": report_id, "payload": payload})
    if candidate_catalog_path is not None:
        raw = Path(candidate_catalog_path).read_text(encoding="utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"invalid_report_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}
        reports.append({"report_id": "candidate_config_catalog", "payload": payload})
    ids = [row["report_id"] for row in reports]
    return (
        "You are the unified_synthesizer in a bounded role-isolated scene expert smoke test. "
        "Reconcile the deterministic visual pool and three domain reports. Preserve disagreements and minority reports; do not infer ground truth or authorize deployment. "
        f"Legal consumed_report_ids: {json.dumps(ids)}. Legal config_ids: [\"C00\",\"C01\",\"C02\",\"C03\",\"C04\",\"C05\"]. "
        f"Reports: {json.dumps(reports, separators=(',', ':'), sort_keys=True)}. "
        "Output exactly one JSON object with exactly these top-level keys in this order: role, consumed_report_ids, "
        "agreements, conflicts, minority_report_ids, ranked_config_ids, abstain, summary. Set role to unified_synthesizer. "
        "Consume every legal report ID exactly once. agreements, conflicts, and minority_report_ids must be arrays; "
        "ranked_config_ids must contain only legal config IDs without duplicates. If the candidate catalog is missing or invalid, "
        "if visual evidence is rejected or fixture-only, or if measured latency/VRAM evidence required by a forbidden_when rule is null, "
        "set abstain=true and ranked_config_ids=[]. Otherwise abstain=false and rank only immutable catalog entries. "
        "Do not modify parameters, infer ground truth, authorize deployment, emit reasoning, Markdown, or text outside the JSON object."
    )


def _scan_warnings(text: str) -> dict[str, bool]:
    return {name: bool(pattern.search(text)) for name, pattern in WARNING_PATTERNS.items()}


def _log_telemetry(text: str) -> dict[str, Any]:
    """Expose only token telemetry the experimental CLI actually prints."""
    new_token_batches = [int(value) for value in NEW_TOKEN_PATTERN.findall(text)]
    chunks = [[int(done), int(total)] for done, total in MTMD_CHUNK_PATTERN.findall(text)]
    return {
        "observed_new_token_batches": new_token_batches,
        "observed_mtmd_chunk_pairs": chunks,
        "observed_context_sizes": [int(value) for value in CTX_PATTERN.findall(text)],
        "observed_prompt_token_counts": [int(value) for value in PROMPT_TOKEN_PATTERN.findall(text)],
        "observed_image_token_batches": [int(value) for value in IMAGE_TOKEN_PATTERN.findall(text)],
        "exact_total_token_count_available": bool(PROMPT_TOKEN_PATTERN.search(text)),
        "telemetry_note": "exact total is the token count emitted by llama_perf_context_print prompt eval time",
    }


def expected_qctx_case_contract() -> dict[str, str | None]:
    """Return the exact registered case IDs and their required r0 binding."""
    contract: dict[str, str | None] = {}
    for ctx in (16384, 32768, 65536):
        for image_count in (1, 2, 4):
            repeats = (0, 1, 2) if image_count == 1 and ctx in {16384, 32768} else (0,)
            for repeat in repeats:
                case_id = f"ctx{ctx}_img{image_count}_r{repeat}"
                contract[case_id] = f"ctx{ctx}_img{image_count}_r0" if repeat else None
    for repeat in (0, 1, 2):
        members = [f"visual_{index}" for index in range(4)] + [
            "algorithm_diagnostician",
            "business_expert",
            "deployment_expert",
            "unified",
        ]
        for member in members:
            case_id = f"chain_r{repeat}_{member}"
            contract[case_id] = f"chain_r0_{member}" if repeat else None
    return contract


def expected_compat_case_ids() -> set[str]:
    return {
        "compat_visual_16k_0",
        "compat_visual_32k_0",
        "compat_visual_32k_1",
        "compat_visual_32k_2",
        "compat_visual_32k_3",
        "compat_algorithm_diagnostician",
        "compat_business_expert",
        "compat_deployment_expert",
        "compat_unified",
    }


def evaluate_compat(run_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    manifest = json.loads((run_dir / "case-manifest.json").read_text(encoding="utf-8"))
    rows = manifest["cases"]
    ids = [row["case_id"] for row in rows]
    matrix_ok = len(ids) == len(set(ids)) and set(ids) == expected_compat_case_ids()
    cases = []
    for registered in rows:
        case_id = registered["case_id"]
        stdout_path = run_dir / "cases" / f"{case_id}.json"
        log_path = run_dir / "cases" / f"{case_id}.log"
        meta = json.loads((run_dir / "cases" / f"{case_id}.meta.json").read_text(encoding="utf-8"))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        parsed = None
        parse_error = None
        try:
            if registered["response_kind"] == "visual":
                parsed = parse_qctx_response(stdout_path.read_text(encoding="utf-8"), set(registered["evidence_ids"]))
            elif registered["response_kind"] == "domain":
                parsed = parse_qctx_domain_response(
                    stdout_path.read_text(encoding="utf-8"), registered["role"], set(registered["consumed_input_ids"])
                )
            else:
                parsed = parse_qctx_unified_response(
                    stdout_path.read_text(encoding="utf-8"), set(registered["consumed_report_ids"]), set(registered["legal_config_ids"])
                )
        except (OSError, ValueError) as exc:
            parse_error = str(exc)
        telemetry = _log_telemetry(log)
        cases.append({
            **registered,
            **meta,
            "schema_valid": parsed is not None,
            "parse_error": parse_error,
            "warnings": _scan_warnings(log),
            "token_telemetry": telemetry,
            "stdout_sha256": sha256_file(stdout_path),
            "log_sha256": sha256_file(log_path),
        })
    visual = [row for row in cases if row["response_kind"] == "visual"]
    text = [row for row in cases if row["response_kind"] != "visual"]
    def clean(row: dict[str, Any]) -> bool:
        telemetry = row["token_telemetry"]
        return (
            row["exit_code"] == 0
            and row["schema_valid"]
            and row["ctx_size_requested"] in telemetry["observed_context_sizes"]
            and telemetry["exact_total_token_count_available"]
            and not any(row["warnings"].values())
        )
    visual_ok = len(visual) == 5 and all(clean(row) for row in visual)
    text_ok = len(text) == 4 and all(clean(row) for row in text)
    passed = matrix_ok and visual_ok and text_ok
    result = {
        "schema_version": "sceneconfig.qwen-qctx-compat-smoke.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": passed,
        "holdout_opened": False,
        "sealed_camera_used": False,
        "qwen_online_called": False,
        "gate": {
            "case_matrix_complete": matrix_ok,
            "visual_path_clean": visual_ok,
            "text_roles_clean": text_ok,
            "successful_cases": sum(row["exit_code"] == 0 for row in cases),
            "schema_valid_cases": sum(row["schema_valid"] for row in cases),
            "token_position_warning_cases": sum(row["warnings"]["token_position"] for row in cases),
        },
        "cases": cases,
        "next_gate": "FULL_G0_RERUN" if passed else "REJECT_CURRENT_LLAMA_MTMD_RUNTIME",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_qctx(run_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refuse_existing_output:{output}")
    manifest = json.loads((run_dir / "case-manifest.json").read_text(encoding="utf-8"))
    registered_rows = manifest["cases"]
    registered_ids = [row["case_id"] for row in registered_rows]
    expected_contract = expected_qctx_case_contract()
    matrix_ok = (
        len(registered_ids) == len(set(registered_ids))
        and set(registered_ids) == set(expected_contract)
        and all(row.get("repeat_of") == expected_contract[row["case_id"]] for row in registered_rows)
    )
    cases: list[dict[str, Any]] = []
    for registered in registered_rows:
        case_id = registered["case_id"]
        stdout_path = run_dir / "cases" / f"{case_id}.json"
        log_path = run_dir / "cases" / f"{case_id}.log"
        meta_path = run_dir / "cases" / f"{case_id}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        warnings = _scan_warnings(log)
        parse_error = None
        parsed = None
        try:
            if registered.get("response_kind") == "unified":
                parsed = parse_qctx_unified_response(
                    stdout_path.read_text(encoding="utf-8"),
                    set(registered["consumed_report_ids"]),
                    set(registered["legal_config_ids"]),
                )
            elif registered.get("response_kind") == "domain":
                parsed = parse_qctx_domain_response(
                    stdout_path.read_text(encoding="utf-8"), registered["role"], set(registered["consumed_input_ids"])
                )
            else:
                parsed = parse_qctx_response(stdout_path.read_text(encoding="utf-8"), set(registered["evidence_ids"]))
        except (OSError, ValueError) as exc:
            parse_error = str(exc)
        cases.append(
            {
                **registered,
                **meta,
                "schema_valid": parsed is not None,
                "parse_error": parse_error,
                "warnings": warnings,
                "token_telemetry": _log_telemetry(log),
                "stdout_sha256": sha256_file(stdout_path),
                "log_sha256": sha256_file(log_path),
            }
        )
    required = [
        row for row in cases
        if row.get("response_kind", "visual") == "visual"
        and row["case_id"].startswith("ctx")
        and row["ctx_size"] in {16384, 32768}
        and row["image_count"] == 1
    ]
    map_reduce = [row for row in cases if row.get("map_member") is True]
    required_ok = bool(required) and all(
        row["exit_code"] == 0
        and row["schema_valid"]
        and not row["warnings"]["truncation"]
        and not row["warnings"]["token_position"]
        and not row["warnings"]["oom"]
        for row in required
    )
    map_reduce_ok = bool(map_reduce) and all(
        row["exit_code"] == 0
        and row["schema_valid"]
        and not row["warnings"]["truncation"]
        and not row["warnings"]["token_position"]
        and not row["warnings"]["oom"]
        for row in map_reduce
    )
    telemetry_ok = bool(cases) and all(
        row["ctx_size_requested"] in row["token_telemetry"]["observed_context_sizes"]
        and row["token_telemetry"]["exact_total_token_count_available"]
        for row in cases
    )
    deterministic = True
    determinism_pairs = []
    by_id = {row["case_id"]: row for row in cases}
    for row in cases:
        repeat_of = row.get("repeat_of")
        if not repeat_of:
            continue
        base = by_id.get(repeat_of)
        same = bool(base) and row["schema_valid"] and base["schema_valid"] and row["stdout_sha256"] == base["stdout_sha256"]
        deterministic &= same
        determinism_pairs.append({"case_id": row["case_id"], "repeat_of": repeat_of, "exact_json_match": same})
    passed = matrix_ok and telemetry_ok and required_ok and map_reduce_ok and bool(determinism_pairs) and deterministic
    result = {
        "schema_version": "sceneconfig.qwen-qctx-g0.v1",
        "evidence_status": "DEVELOPMENT_ONLY",
        "passed": passed,
        "holdout_opened": False,
        "sealed_camera_used": False,
        "qwen_online_called": False,
        "gate": {
            "required_single_image_16k_32k_passed": required_ok,
            "map_reduce_32k_passed": map_reduce_ok,
            "deterministic_repeats_passed": bool(determinism_pairs) and deterministic,
            "determinism_pairs": determinism_pairs,
            "case_matrix_complete": matrix_ok,
            "expected_case_count": len(expected_contract),
            "runtime_token_and_context_telemetry_complete": telemetry_ok,
            "telemetry_failure_code": None if telemetry_ok else "TOKEN_TELEMETRY_UNAVAILABLE",
            "all_cases": len(cases),
            "schema_valid_cases": sum(row["schema_valid"] for row in cases),
            "successful_cases": sum(row["exit_code"] == 0 for row in cases),
        },
        "cases": cases,
        "next_gate": "G0_5_HEADROOM" if passed else "VALIDATED_REJECTION_OF_LOCAL_QWEN_MULTI_AGENT_RUNTIME",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prompt = sub.add_parser("qctx-prompt")
    prompt.add_argument("--evidence-id", action="append", required=True)
    domain = sub.add_parser("qctx-domain-prompt")
    domain.add_argument("--role", choices=("algorithm_diagnostician", "business_expert", "deployment_expert"), required=True)
    camera_domain = sub.add_parser("qctx-camera-domain-prompt")
    camera_domain.add_argument("--role", choices=("algorithm_diagnostician", "business_expert", "deployment_expert"), required=True)
    camera_domain.add_argument("--candidate-catalog", type=Path, required=True)
    unified = sub.add_parser("qctx-unified-prompt")
    unified.add_argument("--visual-report", type=Path, action="append", required=True)
    unified.add_argument("--domain-report", type=Path, action="append", required=True)
    unified.add_argument("--candidate-catalog", type=Path)
    camera_unified = sub.add_parser("qctx-camera-unified-prompt")
    camera_unified.add_argument("--domain-report", type=Path, action="append", required=True)
    camera_unified.add_argument("--candidate-catalog", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-qctx")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    compat = sub.add_parser("evaluate-compat")
    compat.add_argument("--run-dir", type=Path, required=True)
    compat.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "qctx-prompt":
        print(qctx_prompt(args.evidence_id))
    elif args.action == "qctx-domain-prompt":
        print(qctx_domain_prompt(args.role))
    elif args.action == "qctx-camera-domain-prompt":
        print(qctx_camera_domain_prompt(args.role, args.candidate_catalog))
    elif args.action == "qctx-unified-prompt":
        print(qctx_unified_prompt(args.visual_report, args.domain_report, args.candidate_catalog))
    elif args.action == "qctx-camera-unified-prompt":
        print(qctx_camera_unified_prompt(args.domain_report, args.candidate_catalog))
    elif args.action == "evaluate-qctx":
        result = evaluate_qctx(args.run_dir, args.output)
        print(json.dumps({"passed": result["passed"], "gate": result["gate"], "next_gate": result["next_gate"]}, sort_keys=True))
    else:
        result = evaluate_compat(args.run_dir, args.output)
        print(json.dumps({"passed": result["passed"], "gate": result["gate"], "next_gate": result["next_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from research_tools.validate_qwen_native_visual_runtime import validate_run


PROPOSAL_STAGE = "PROPOSAL_CONFIGURATION"
EVALUATION_STAGE = "POST_PROPOSAL_EVALUATION"
DEPLOYMENT_AUTHORITY = "DETERMINISTIC_COMPILER_AND_HUMAN"
PROPOSAL_ROLES = {
    "clean_visual_scene_analyst",
    "detection_overlay_visual_analyst",
    "algorithm_diagnostician",
    "tracking_temporal_analyst",
    "scene_context_expert",
    "business_profile_mapper",
    "deployment_resource_expert",
    "unified_synthesizer",
    "deterministic_compiler",
}
FORBIDDEN_PROPOSAL_KEYS = {
    "evaluation_audit",
    "evaluation_frames",
    "evaluation_metrics",
    "ground_truth",
    "gt_boxes",
    "holdout_results",
    "deployment_credentials",
}
VISUAL_ROLE_NAMES = ("clean_visual_scene_analyst", "detection_overlay_visual_analyst")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _compact_detection(detection: dict[str, Any]) -> dict[str, Any]:
    box = detection["box"]
    return {
        "xyxy": [round(float(box[key]), 3) for key in ("x1", "y1", "x2", "y2")],
        "confidence": round(float(detection["confidence"]), 6),
        "source": detection["source"],
    }


def _load_prediction_rows(path: Path, camera_id: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("camera_id") == camera_id:
            rows[str(row["config_id"])] = row
    if not rows:
        raise ValueError(f"no prediction rows found for {camera_id}")
    return rows


def _walk_keys(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield key, child_path
            yield from _walk_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def validate_proposal_package(package: dict[str, Any]) -> None:
    if package.get("stage") != PROPOSAL_STAGE:
        raise ValueError("proposal package has the wrong stage")
    if package.get("deployment_authority") != DEPLOYMENT_AUTHORITY:
        raise ValueError("deployment authority must remain deterministic and human controlled")
    if package.get("qwen_deployment_authority") is not False:
        raise ValueError("Qwen must not receive deployment authority")
    for key, path in _walk_keys(package):
        if key.lower() in FORBIDDEN_PROPOSAL_KEYS:
            raise ValueError(f"forbidden proposal field at {path}")
    roles = package.get("role_packages", {})
    if set(roles) != PROPOSAL_ROLES:
        raise ValueError("proposal role set is incomplete or unexpected")
    if roles["tracking_temporal_analyst"].get("availability") != "UNAVAILABLE":
        raise ValueError("tracking role must abstain when tracker evidence is unavailable")
    visual_gate = package.get("visual_runtime_gate")
    visual_available = all(
        roles[role_name].get("availability") == "EXECUTION_ELIGIBLE_NATIVE_VISUAL_RUNTIME"
        for role_name in VISUAL_ROLE_NAMES
    )
    visual_unavailable = all(
        roles[role_name].get("availability") == "UNAVAILABLE" for role_name in VISUAL_ROLE_NAMES
    )
    if not (visual_available or visual_unavailable):
        raise ValueError("visual roles must share one fail-closed availability state")
    if visual_available:
        if not isinstance(visual_gate, dict) or visual_gate.get("passed") is not True:
            raise ValueError("visual roles require a passed native visual runtime gate")
        if visual_gate.get("holdout_opened") is not False:
            raise ValueError("visual runtime gate violated the holdout boundary")
        if visual_gate.get("deployment_authority") != DEPLOYMENT_AUTHORITY:
            raise ValueError("visual runtime gate changed deployment authority")
    elif visual_gate is not None:
        raise ValueError("rejected visual roles must not carry an admitted runtime gate")
    if set(roles["unified_synthesizer"].get("allowed_inputs", [])) != {
        "validated_role_reports",
        "eligible_config_ids",
        "conflict_codes",
        "business_profile_code",
    }:
        raise ValueError("unified synthesizer input allowlist changed")


def validate_evaluation_package(package: dict[str, Any], proposal_sha256: str) -> None:
    if package.get("stage") != EVALUATION_STAGE:
        raise ValueError("evaluation package has the wrong stage")
    if package.get("proposal_sha256") != proposal_sha256:
        raise ValueError("evaluation package is not bound to the frozen proposal")
    if package.get("accessible_by") != ["holdout_evaluator"]:
        raise ValueError("holdout package access must be evaluator-only")
    if package.get("feedback_to_same_proposal_round") is not False:
        raise ValueError("holdout feedback into the same proposal round is forbidden")


def _frame_ledger(
    prediction_rows: dict[str, dict[str, Any]],
    config_ids: list[str],
    calibration_frames: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledgers: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for config_id in config_ids:
        row = prediction_rows.get(config_id)
        if row is None:
            continue
        selected = [frame for frame in row["frames"] if int(frame["frame"]) in calibration_frames]
        frame_records: list[dict[str, Any]] = []
        merged_confidences: list[float] = []
        merged_areas: list[float] = []
        sources: Counter[str] = Counter()
        count_deltas: list[int] = []
        for frame in selected:
            global_boxes = [_compact_detection(item) for item in frame.get("global", [])]
            tile_boxes = [_compact_detection(item) for item in frame.get("tile_raw", [])]
            merged_boxes = [_compact_detection(item) for item in frame.get("merged", [])]
            for item in merged_boxes:
                merged_confidences.append(float(item["confidence"]))
                x1, y1, x2, y2 = item["xyxy"]
                merged_areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                sources[item["source"]] += 1
            count_deltas.append(len(merged_boxes) - len(global_boxes))
            frame_records.append(
                {
                    "frame": int(frame["frame"]),
                    "sequence_id": frame["sequence_id"],
                    "global_count": len(global_boxes),
                    "tile_raw_count": len(tile_boxes),
                    "merged_count_prediction": len(merged_boxes),
                    "global_boxes": global_boxes,
                    "tile_raw_boxes": tile_boxes,
                    "merged_boxes": merged_boxes,
                }
            )
        ledgers.append({"config_id": config_id, "frames": frame_records})
        summaries[config_id] = {
            "calibration_frame_count": len(frame_records),
            "mean_global_count": sum(record["global_count"] for record in frame_records) / len(frame_records),
            "mean_merged_count_prediction": sum(record["merged_count_prediction"] for record in frame_records)
            / len(frame_records),
            "mean_added_count_vs_global": sum(count_deltas) / len(count_deltas),
            "frames_with_count_change": sum(delta != 0 for delta in count_deltas),
            "confidence_quantiles": {
                "p10": _quantile(merged_confidences, 0.10),
                "p50": _quantile(merged_confidences, 0.50),
                "p90": _quantile(merged_confidences, 0.90),
            },
            "small_box_proxy": {
                "definition": "merged box area < 32^2 pixels; not a ground-truth small-object label",
                "count": sum(area < 32 * 32 for area in merged_areas),
                "fraction": (sum(area < 32 * 32 for area in merged_areas) / len(merged_areas))
                if merged_areas
                else None,
            },
            "merged_source_distribution": dict(sorted(sources.items())),
        }
    return ledgers, summaries


def build_packages(
    *,
    camera_id: str,
    decision_catalog_path: Path,
    evaluation_audit_path: Path,
    predictions_path: Path,
    frame_split_path: Path,
    fingerprints_path: Path,
    output_dir: Path,
    remote_frame_root: str | None = None,
    visual_manifest_path: Path | None = None,
    business_profile_registry_path: Path | None = None,
    visual_runtime_run_dir: Path | None = None,
) -> dict[str, Path]:
    decision = _load_json(decision_catalog_path)
    split = _load_json(frame_split_path)["cameras"][camera_id]
    fingerprints = _load_json(fingerprints_path)["operators"]
    prediction_rows = _load_prediction_rows(predictions_path, camera_id)
    calibration_frames = {int(value) for value in split["calibration_frames"]}

    catalog_ids = [str(value) for value in decision["config_ids"]]
    prediction_ids = ["B0", *[value for value in catalog_ids if value in prediction_rows]]
    ledgers, summaries = _frame_ledger(prediction_rows, prediction_ids, calibration_frames)

    config_fingerprints = {
        key.split(":", 1)[1]: value
        for key, value in fingerprints.items()
        if key.startswith(f"{camera_id}:") and key.split(":", 1)[1] in set(prediction_ids)
    }
    tracker_values = {value.get("tracker") for value in config_fingerprints.values()}
    if tracker_values != {"none"}:
        raise ValueError(f"expected tracker=none, observed {sorted(tracker_values)}")

    remote_assets: list[dict[str, Any]] = []
    if visual_manifest_path is not None:
        visual_manifest = _load_json(visual_manifest_path)
        if visual_manifest.get("camera_id") != camera_id:
            raise ValueError("visual manifest camera mismatch")
        if visual_manifest.get("contains_ground_truth") is not False:
            raise ValueError("visual proposal manifest must not contain ground truth")
        remote_assets = visual_manifest["assets"]
    elif remote_frame_root:
        for frame in sorted(calibration_frames):
            remote_assets.append(
                {
                    "frame": frame,
                    "remote_path": f"{remote_frame_root.rstrip('/')}/{frame:06d}.jpg",
                    "content_status": "REGISTERED_PATH_NOT_LOCALLY_CONSUMED",
                }
            )

    profile_registry: Any = {"status": "NOT_PROVIDED", "required_action": "MAP_OR_CONFIRM_PROFILE"}
    if business_profile_registry_path is not None:
        profile_registry = _load_json(business_profile_registry_path)

    visual_runtime_gate: dict[str, Any] | None = None
    if visual_runtime_run_dir is not None:
        runtime_result = validate_run(visual_runtime_run_dir)
        runtime_gate = runtime_result.get("gate", {})
        runtime_provenance = runtime_result.get("provenance", {})
        if (
            runtime_result.get("schema_version") != "sceneconfig.qwen-native-visual-runtime-result.v1"
            or runtime_result.get("status") != "passed"
            or runtime_result.get("passed") is not True
            or runtime_result.get("evidence_status") != "DEVELOPMENT_ONLY"
            or runtime_gate.get("expected_cases") != 9
            or runtime_gate.get("observed_cases") != 9
            or runtime_gate.get("successful_cases") != 9
            or any(
                runtime_gate.get(name) != 0
                for name in (
                    "token_position_warning_cases",
                    "truncation_warning_cases",
                    "oom_warning_cases",
                )
            )
            or runtime_provenance.get("holdout_opened") is not False
            or runtime_provenance.get("deployment_authority") != DEPLOYMENT_AUTHORITY
        ):
            raise ValueError("visual runtime result did not satisfy the native 9-case admission gate")
        visual_runtime_gate = {
            "passed": True,
            "validation_result_sha256": _canonical_sha256(runtime_result),
            "run_artifact_sha256": runtime_result.get("artifact_sha256"),
            "runtime": runtime_provenance.get("runtime"),
            "runtime_version": runtime_provenance.get("runtime_version"),
            "model_id": runtime_provenance.get("model_id"),
            "model_content_manifest_sha256": runtime_provenance.get(
                "model_content_manifest_sha256"
            ),
            "successful_cases": runtime_gate["successful_cases"],
            "holdout_opened": False,
            "deployment_authority": DEPLOYMENT_AUTHORITY,
        }

    visual_available = visual_runtime_gate is not None
    visual_availability = (
        "EXECUTION_ELIGIBLE_NATIVE_VISUAL_RUNTIME" if visual_available else "UNAVAILABLE"
    )
    visual_required_response = (
        "RUN_ROLE_AND_VALIDATE_REPORT"
        if visual_available
        else "ABSTAIN_VISUAL_RUNTIME_REJECTED"
    )

    role_packages: dict[str, Any] = {
        "clean_visual_scene_analyst": {
            "evidence_id": f"{camera_id}:clean-visual:v1",
            "availability": visual_availability,
            "role_execution_status": "NOT_RUN" if visual_available else "BLOCKED_RUNTIME_REJECTED",
            "role_report_status": "ABSENT",
            "required_response": visual_required_response,
            "reason": (
                "native Qwen3.5 visual runtime passed the temporal-frame and compatibility gate"
                if visual_available
                else "current llama-mtmd visual runtime failed the non-consecutive-token-position gate"
            ),
            "registered_scene_assets": remote_assets,
            "visual_token_metadata": {
                "status": "NOT_PRODUCED_ROLE_NOT_RUN" if visual_available else "NOT_PRODUCED",
                "runtime_gate": (
                    visual_runtime_gate["validation_result_sha256"]
                    if visual_available
                    else "REJECTED_CURRENT_LLAMA_MTMD_VISUAL_RUNTIME"
                ),
            },
            "allowed_inputs": ["calibration_images", "camera_id", "roi_geometry"],
            "forbidden_inputs": ["detector_overlays", "calibration_metrics", "count_predictions"],
            "task_boundary": "describe scene risks without counting or diagnosing detector errors",
        },
        "detection_overlay_visual_analyst": {
            "evidence_id": f"{camera_id}:overlay-visual:v1",
            "availability": visual_availability,
            "role_execution_status": "NOT_RUN" if visual_available else "BLOCKED_RUNTIME_REJECTED",
            "role_report_status": "ABSENT",
            "required_response": visual_required_response,
            "reason": (
                "native Qwen3.5 visual runtime passed the temporal-frame and compatibility gate"
                if visual_available
                else "current llama-mtmd visual runtime failed the non-consecutive-token-position gate"
            ),
            "registered_scene_assets": remote_assets,
            "overlay_status": (
                "MATERIALIZATION_REQUIRED_PER_REGISTERED_ASSET"
                if visual_available
                else "NOT_MATERIALIZED_FOR_CURRENT_RUNTIME"
            ),
            "allowed_inputs": [
                "calibration_images_with_detector_overlays",
                "box_ids",
                "camera_id",
                "roi_geometry",
            ],
            "forbidden_inputs": ["calibration_metrics", "holdout", "ground_truth"],
            "task_boundary": "judge visual plausibility of detector disagreement without using truth labels",
        },
        "algorithm_diagnostician": {
            "evidence_id": f"{camera_id}:algorithm:v1",
            "availability": "AVAILABLE_TEXT_STRUCTURED",
            "camera_id": camera_id,
            "calibration_metrics": {
                item["config_id"]: item["camera_replay"]
                for item in decision["configs"]
                if item["camera_replay"].get("status") == "EXACT_END_TO_END_REPLAY_AVAILABLE"
            },
            "calibration_detection_ledger": ledgers,
            "calibration_detection_summary": summaries,
            "allowed_inputs": [
                "yolo11_yolo26_boxes",
                "confidence",
                "source_model",
                "count_prediction",
                "calibration_metrics",
            ],
        },
        "tracking_temporal_analyst": {
            "evidence_id": f"{camera_id}:tracking:v1",
            "availability": "UNAVAILABLE",
            "required_response": "ABSTAIN_TRACKER_NOT_EXECUTED",
            "reason": "every registered operator fingerprint records tracker=none",
            "track_ids": None,
            "track_births_deaths": None,
            "track_gaps": None,
            "allowed_inputs": ["tracks", "count_continuity"],
        },
        "scene_context_expert": {
            "evidence_id": f"{camera_id}:scene-context:v1",
            "availability": "AVAILABLE_TEXT_STRUCTURED",
            "camera_id": camera_id,
            "sequence_id": split["sequence_id"],
            "reference_geometry": decision["reference_geometry"],
            "roi_catalog": {
                item["config_id"]: item["operator"]["selected_tiles"] for item in decision["configs"]
            },
            "historical_scene_summary": {
                "scope": "CALIBRATION_ONLY_NO_GROUND_TRUTH_INFERENCE",
                "detector_count_and_confidence_summary": summaries,
                "warning": "detector disagreement is not scene truth",
            },
            "allowed_inputs": ["camera_id", "roi", "fixed_geometry", "historical_summary"],
        },
        "business_profile_mapper": {
            "evidence_id": f"{camera_id}:business-profile:v1",
            "availability": "AVAILABLE_IF_BUSINESS_TEXT_SUPPLIED",
            "profile_registry": profile_registry,
            "allowed_inputs": ["business_requirements", "frozen_profile_registry"],
            "forbidden_inference": "do not invent a business priority from detector metrics",
        },
        "deployment_resource_expert": {
            "evidence_id": f"{camera_id}:deployment-resource:v1",
            "availability": "AVAILABLE_TEXT_STRUCTURED",
            "resource_ledger": {
                item["config_id"]: {
                    "p95_ms": item["camera_replay"].get("p95_ms"),
                    "latency_ratio_p95": item["camera_replay"].get("latency_ratio_p95"),
                    "latency_admissible": item["camera_replay"].get("latency_admissible"),
                    "oom": item["camera_replay"].get("oom"),
                    "operator_fingerprint_sha256": item["camera_replay"].get(
                        "operator_fingerprint_sha256"
                    ),
                }
                for item in decision["configs"]
            },
            "operator_fingerprints": config_fingerprints,
            "deployment_status": "HUMAN_APPROVAL_REQUIRED",
            "deployment_authority": DEPLOYMENT_AUTHORITY,
            "qwen_may_deploy": False,
            "allowed_inputs": ["latency", "oom", "resource_budget", "deployment_constraints"],
        },
        "unified_synthesizer": {
            "evidence_id": f"{camera_id}:unified:v1",
            "availability": "AVAILABLE_AFTER_ROLE_VALIDATION",
            "eligible_config_ids": decision["eligible_config_ids"],
            "allowed_inputs": [
                "validated_role_reports",
                "eligible_config_ids",
                "conflict_codes",
                "business_profile_code",
            ],
            "raw_evidence_access": False,
            "must_preserve": ["agreements", "conflicts", "minority_reports", "evidence_ids"],
        },
        "deterministic_compiler": {
            "evidence_id": f"{camera_id}:compiler:v1",
            "availability": "AVAILABLE",
            "eligible_config_ids": decision["eligible_config_ids"],
            "authority": DEPLOYMENT_AUTHORITY,
            "responsibilities": [
                "schema and evidence-reference validation",
                "eligibility and hard-gate enforcement",
                "profile application and Pareto selection",
                "human-approval handoff",
            ],
        },
    }

    proposal = {
        "schema_version": "sceneconfig.qwen-multirole-proposal-evidence.v1",
        "stage": PROPOSAL_STAGE,
        "evidence_status": "DEVELOPMENT_ONLY",
        "camera_id": camera_id,
        "proposal_scope": "CALIBRATION_ONLY",
        "holdout_access": "DENIED_TO_ALL_PROPOSAL_ROLES",
        "deployment_authority": DEPLOYMENT_AUTHORITY,
        "qwen_deployment_authority": False,
        **({"visual_runtime_gate": visual_runtime_gate} if visual_runtime_gate is not None else {}),
        "source_sha256": {
            "decision_catalog": _sha256(decision_catalog_path),
            "predictions": _sha256(predictions_path),
            "frame_split_manifest": _sha256(frame_split_path),
            "operator_fingerprints": _sha256(fingerprints_path),
            **({"visual_manifest": _sha256(visual_manifest_path)} if visual_manifest_path else {}),
            **(
                {
                    "visual_runtime_case_manifest": _sha256(
                        visual_runtime_run_dir / "case-manifest.json"
                    ),
                    "visual_runtime_provenance": _sha256(
                        visual_runtime_run_dir / "provenance.json"
                    ),
                }
                if visual_runtime_run_dir is not None
                else {}
            ),
            **(
                {"business_profile_registry": _sha256(business_profile_registry_path)}
                if business_profile_registry_path is not None
                else {}
            ),
        },
        "role_packages": role_packages,
    }
    validate_proposal_package(proposal)
    proposal_sha256 = _canonical_sha256(proposal)
    proposal["proposal_sha256"] = proposal_sha256

    # The holdout audit is intentionally opened only after the proposal content is frozen.
    evaluation_audit = _load_json(evaluation_audit_path)
    evaluation = {
        "schema_version": "sceneconfig.qwen-post-freeze-evaluation-evidence.v1",
        "stage": EVALUATION_STAGE,
        "evidence_status": "DEVELOPMENT_ONLY",
        "camera_id": camera_id,
        "proposal_sha256": proposal_sha256,
        "accessible_by": ["holdout_evaluator"],
        "forbidden_to": sorted(PROPOSAL_ROLES),
        "feedback_to_same_proposal_round": False,
        "evaluation_audit_sha256": _sha256(evaluation_audit_path),
        "evaluation_audit": evaluation_audit,
    }
    validate_evaluation_package(evaluation, proposal_sha256)

    output_dir.mkdir(parents=True, exist_ok=False)
    proposal_path = output_dir / "proposal-evidence.json"
    evaluation_path = output_dir / "post-freeze-evaluation-evidence.json"
    proposal_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evaluation_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema_version": "sceneconfig.qwen-evidence-router-receipt.v1",
        "status": "passed",
        "passed": True,
        "camera_id": camera_id,
        "proposal_path": str(proposal_path),
        "proposal_file_sha256": _sha256(proposal_path),
        "proposal_content_sha256": proposal_sha256,
        "evaluation_path": str(evaluation_path),
        "evaluation_file_sha256": _sha256(evaluation_path),
        "holdout_isolation": "LOGICALLY_SEPARATE_POST_FREEZE_FILE_SAME_OUTPUT_ROOT",
        "visual_runtime_status": (
            "PASSED_NATIVE_QWEN35_MULTIMODAL_TRANSPORT_CAPABILITY"
            if visual_available
            else "REJECTED_CURRENT_LLAMA_MTMD_VISUAL_RUNTIME"
        ),
        "tracking_status": "UNAVAILABLE_TRACKER_NONE",
        "deployment_authority": DEPLOYMENT_AUTHORITY,
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"proposal": proposal_path, "evaluation": evaluation_path, "receipt": receipt_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe Qwen multi-role evidence packages.")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--decision-catalog", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--frame-split", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-frame-root")
    parser.add_argument("--visual-manifest", type=Path)
    parser.add_argument("--business-profile-registry", type=Path)
    parser.add_argument("--visual-runtime-run-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = build_packages(
        camera_id=args.camera_id,
        decision_catalog_path=args.decision_catalog,
        evaluation_audit_path=args.evaluation_audit,
        predictions_path=args.predictions,
        frame_split_path=args.frame_split,
        fingerprints_path=args.fingerprints,
        output_dir=args.output_dir,
        remote_frame_root=args.remote_frame_root,
        visual_manifest_path=args.visual_manifest,
        business_profile_registry_path=args.business_profile_registry,
        visual_runtime_run_dir=args.visual_runtime_run_dir,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

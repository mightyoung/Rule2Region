from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from research_tools.build_qwen_candidate_config_catalog import validate_catalog


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_camera_views(
    *,
    base_catalog_path: Path,
    replay_result_path: Path,
    fingerprint_path: Path,
    pre_gt_seal_path: Path,
    camera_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = json.loads(base_catalog_path.read_text(encoding="utf-8"))
    validate_catalog(base)
    replay = json.loads(replay_result_path.read_text(encoding="utf-8"))
    fingerprints = json.loads(fingerprint_path.read_text(encoding="utf-8"))["operators"]
    if camera_id not in replay.get("per_camera", {}):
        raise ValueError("camera_missing_from_exact_replay")
    camera = replay["per_camera"][camera_id]
    candidates = {row["config_id"]: row for row in camera["candidate_metrics"]}
    diagnostics = {row["config_id"]: row for row in camera["calibration_selection_diagnostics"]}
    baseline = camera["b0"]
    decision = copy.deepcopy(base)
    decision["schema_version"] = "sceneconfig.qwen-camera-decision-catalog.v1"
    decision["catalog_id"] = f"{base['catalog_id']}:{camera_id}:decision-v1"
    decision["camera_id"] = camera_id
    decision["evidence_boundary"] = "DEVELOPMENT_ONLY_10_FRAME_CALIBRATION"
    decision["source_sha256"].update({
        "base_catalog_file": _sha256(base_catalog_path),
        "exact_replay_result": _sha256(replay_result_path),
        "operator_fingerprints": _sha256(fingerprint_path),
        "pre_gt_seal": _sha256(pre_gt_seal_path),
    })
    eligible = []
    audit_rows = []
    for config in decision["configs"]:
        config_id = config["config_id"]
        candidate = candidates.get(config_id)
        diagnostic = diagnostics.get(config_id)
        fingerprint = fingerprints.get(f"{camera_id}:{config_id}")
        if candidate is None or diagnostic is None or fingerprint is None:
            config["camera_replay"] = {
                "status": "NOT_IN_REGISTERED_TOP6_EXACT_REPLAY",
                "eligible_for_ranking": False,
                "reason": "No exact end-to-end replay was registered for this camera/config pair.",
            }
            continue
        precision_zero_warning = (
            float(camera["calibration_baseline"]["precision"]) == 0.0
            and float(candidate["calibration_metrics"]["precision"]) == 0.0
        )
        admissible = bool(diagnostic["admissible"]) and not precision_zero_warning
        if admissible:
            eligible.append(config_id)
        config["camera_replay"] = {
            "status": "EXACT_END_TO_END_REPLAY_AVAILABLE",
            "eligible_for_ranking": admissible,
            "calibration_frames": int(candidate["calibration_metrics"]["frame_count"]),
            "calibration_metrics": candidate["calibration_metrics"],
            "calibration_precision_change_vs_b0": diagnostic["precision_change"],
            "latency_admissible": bool(diagnostic["latency_admissible"]),
            "empirical_precision_noninferiority": bool(diagnostic["empirical_precision_noninferiority"]),
            "p95_ms": candidate["p95_ms"],
            "latency_ratio_p95": candidate["latency_ratio_p95"],
            "oom": bool(candidate["oom"]),
            "operator_fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "statistical_warning": (
                "ZERO_PRECISION_BASE_RATE_MAKES_NONINFERIORITY_UNINFORMATIVE"
                if precision_zero_warning else None
            ),
        }
        evaluation = candidate["evaluation_metrics"]
        b0_metrics = baseline["metrics"]
        audit_rows.append({
            "config_id": config_id,
            "evaluation_frames": int(evaluation["frame_count"]),
            "evaluation_metrics": evaluation,
            "evaluation_precision_change_vs_b0": evaluation["precision"] - b0_metrics["precision"],
            "evaluation_relative_count_mae_improvement_vs_b0": (
                (b0_metrics["count_mae"] - evaluation["count_mae"]) / b0_metrics["count_mae"]
                if b0_metrics["count_mae"] else 0.0
            ),
        })
    decision["eligible_config_ids"] = eligible
    decision["decision_input_scope"] = "CALIBRATION_ONLY_WITH_HELDOUT_AUDIT_SEPARATE"
    old_sha = decision.pop("catalog_sha256")
    decision["base_catalog_canonical_sha256"] = old_sha
    decision["catalog_sha256"] = _canonical_sha(decision)
    audit: dict[str, Any] = {
        "schema_version": "sceneconfig.qwen-camera-evaluation-audit.v1",
        "camera_id": camera_id,
        "evidence_boundary": "DEVELOPMENT_ONLY_22_FRAME_EVALUATION_WITHHELD_FROM_QWEN",
        "decision_catalog_id": decision["catalog_id"],
        "decision_catalog_sha256": decision["catalog_sha256"],
        "baseline": baseline,
        "rows": audit_rows,
        "source_sha256": {
            "exact_replay_result": _sha256(replay_result_path),
            "pre_gt_seal": _sha256(pre_gt_seal_path),
        },
    }
    audit["audit_sha256"] = _canonical_sha(audit)
    validate_camera_views(decision, audit)
    return decision, audit


def validate_camera_views(decision: dict[str, Any], audit: dict[str, Any]) -> None:
    serialized = json.dumps(decision, sort_keys=True)
    if "evaluation_metrics" in serialized or "evaluation_frames" in serialized:
        raise ValueError("evaluation_data_leaked_into_qwen_decision_view")
    eligible = set(decision["eligible_config_ids"])
    registered = {
        row["config_id"]
        for row in decision["configs"]
        if row["camera_replay"]["eligible_for_ranking"]
    }
    if eligible != registered:
        raise ValueError("eligible_config_ids_mismatch")
    unsigned_decision = dict(decision)
    digest = unsigned_decision.pop("catalog_sha256")
    if digest != _canonical_sha(unsigned_decision):
        raise ValueError("camera_decision_catalog_sha256_invalid")
    unsigned_audit = dict(audit)
    audit_digest = unsigned_audit.pop("audit_sha256")
    if audit_digest != _canonical_sha(unsigned_audit):
        raise ValueError("camera_audit_sha256_invalid")
    if audit["decision_catalog_sha256"] != decision["catalog_sha256"]:
        raise ValueError("decision_audit_binding_invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--pre-gt-seal", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    decision, audit = build_camera_views(
        base_catalog_path=args.base_catalog,
        replay_result_path=args.replay_result,
        fingerprint_path=args.fingerprints,
        pre_gt_seal_path=args.pre_gt_seal,
        camera_id=args.camera_id,
    )
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"camera_id": args.camera_id, "eligible_config_ids": decision["eligible_config_ids"], "catalog_sha256": decision["catalog_sha256"], "audit_sha256": audit["audit_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

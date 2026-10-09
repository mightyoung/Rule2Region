"""Validate an RC-LDA CPU-only upper-bound evidence pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research_tools.rclda_upper_bound import INHERITED_CONFIGS, TARGET_CONFIGS
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json:{path.name}:{exc}")
        return {}


def validate(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "COMPLETE", "SHA256SUMS", "provenance.json", "resource-before.txt", "resource-final.txt",
        "frozen-rule-grid.json", "target-manifest.json", "feature-ledger.jsonl", "PRE_GT_SEAL.json",
        "rclda-upper-bound-result.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        return {"status": "failed", "passed": False, "errors": errors}
    if (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != "COMPLETE":
        errors.append("invalid_complete_marker")
    sums = {}
    for line in (run_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            sums[parts[1].lstrip("*./")] = parts[0]
        else:
            errors.append("invalid_sha256sums_line")
    for name in required:
        if name == "SHA256SUMS":
            continue
        if name not in sums:
            errors.append(f"unhashed:{name}")
        elif _sha256(run_dir / name) != sums[name]:
            errors.append(f"sha256_mismatch:{name}")

    grid = _load(run_dir / "frozen-rule-grid.json", errors)
    target = _load(run_dir / "target-manifest.json", errors)
    seal = _load(run_dir / "PRE_GT_SEAL.json", errors)
    result = _load(run_dir / "rclda-upper-bound-result.json", errors)
    provenance = _load(run_dir / "provenance.json", errors)
    if len(grid.get("weights", ())) != 242 or len(grid.get("thresholds", ())) != 22:
        errors.append("unexpected_frozen_rule_grid_size")
    if grid.get("target_configs") != TARGET_CONFIGS or grid.get("inherited_configs") != INHERITED_CONFIGS:
        errors.append("frozen_config_contract_mismatch")
    if target.get("prior_development_gt_informed_target_selection") is not True or target.get("current_run_gt_used_before_feature_seal") is not False:
        errors.append("target_gt_history_disclosure_missing")
    if target.get("holdout_opened") is not False or target.get("sealed_camera_used") is not False:
        errors.append("target_boundary_violation")
    if set(target.get("evaluation_frames", {})) != set(EXPECTED_CAMERAS) or any(len(frames) != 22 for frames in target.get("evaluation_frames", {}).values()):
        errors.append("target_evaluation_frame_contract_mismatch")
    if seal.get("stage") != "rule_and_feature_ledgers_complete_current_run_gt_unread" or seal.get("holdout_opened") is not False:
        errors.append("invalid_pre_gt_seal")
    for name, expected in seal.get("files", {}).items():
        if not (run_dir / name).is_file() or _sha256(run_dir / name) != expected:
            errors.append(f"pre_gt_hash_mismatch:{name}")
    try:
        feature_rows = [json.loads(line) for line in (run_dir / "feature-ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as exc:
        feature_rows = []
        errors.append(f"invalid_feature_jsonl:{exc}")
    if {(row.get("camera_id"), row.get("config_id")) for row in feature_rows} != set(TARGET_CONFIGS.items()):
        errors.append("feature_ledger_target_set_mismatch")

    if result.get("schema_version") != "sceneconfig.rclda-upper-bound.v1":
        errors.append("invalid_result_schema")
    if result.get("evidence_status") != "DEVELOPMENT_ONLY" or result.get("oracle_only_not_deployable") is not True or result.get("gpu_used") is not False:
        errors.append("result_scope_violation")
    if result.get("gt_access_order") != "after_pre_gt_seal" or result.get("holdout_opened") is not False or result.get("sealed_camera_used") is not False:
        errors.append("result_gt_or_holdout_boundary_violation")
    if result.get("all_accepted_operator_invariant_passed") is not True:
        errors.append("all_accepted_operator_invariant_missing")
    if set(result.get("per_camera", {})) != set(EXPECTED_CAMERAS) or result.get("composite_gate", {}).get("all_four_cameras_retained") is not True:
        errors.append("result_camera_retention_failure")
    for camera in TARGET_CONFIGS:
        row = result.get("per_camera", {}).get(camera, {})
        if row.get("oracle", {}).get("rules_evaluated") != 5324:
            errors.append(f"target_rule_count_mismatch:{camera}")
    if provenance.get("evidence_status") != "DEVELOPMENT_ONLY" or provenance.get("gpu_used") is not False or provenance.get("holdout_opened") is not False:
        errors.append("invalid_provenance")

    method_passed = result.get("composite_gate", {}).get("passed") is True
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "method_gate_passed": method_passed,
        "validation_outcome": ("RCLDA_GPU_REPLAY_ADMITTED" if method_passed else "VALIDATED_REJECTION_OF_RCLDA_RULE_FAMILY") if not errors else "INVALID_EVIDENCE_PACK",
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "errors": errors,
        "summary": "The oracle evidence pack is valid; branch admission is reported separately." if not errors else "The oracle evidence pack is invalid.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

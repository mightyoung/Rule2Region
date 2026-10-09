"""Validate a completed development-only Top-6 calibrated replay evidence pack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json:{path.name}:{exc}")
        return {}


def validate(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "COMPLETE", "SHA256SUMS", "provenance.json", "gpu-admission.txt", "gpu-final.csv",
        "PRE_GT_SEAL.json", "frame-split-manifest.json", "top6-query-trace.json",
        "predictions.jsonl", "operator-fingerprints.json", "pre-gt-execution-summary.json",
        "top6-calibrated-result.json",
    )
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        return {"status": "failed", "passed": False, "errors": errors}
    if (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != "COMPLETE":
        errors.append("invalid_complete_marker")

    sums: dict[str, str] = {}
    for line in (run_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append("invalid_sha256sums_line")
            continue
        name = parts[1].lstrip("*./")
        sums[name] = parts[0]
    for name in required:
        if name == "SHA256SUMS":
            continue
        if name not in sums:
            errors.append(f"unhashed:{name}")
        elif _sha256(run_dir / name) != sums[name]:
            errors.append(f"sha256_mismatch:{name}")

    seal = _load(run_dir / "PRE_GT_SEAL.json", errors)
    if seal.get("schema_version") != "sceneconfig.pre-gt-seal.v1" or seal.get("stage") != "predictions_complete_gt_unread":
        errors.append("invalid_pre_gt_seal_contract")
    if seal.get("holdout_opened") is not False:
        errors.append("pre_gt_seal_holdout_violation")
    for name, expected in seal.get("files", {}).items():
        path = run_dir / name
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"pre_gt_seal_hash_mismatch:{name}")

    frame_manifest = _load(run_dir / "frame-split-manifest.json", errors)
    query = _load(run_dir / "top6-query-trace.json", errors)
    fingerprints = _load(run_dir / "operator-fingerprints.json", errors)
    result = _load(run_dir / "top6-calibrated-result.json", errors)
    provenance = _load(run_dir / "provenance.json", errors)
    expected = set(EXPECTED_CAMERAS)
    if set(frame_manifest.get("cameras", {})) != expected:
        errors.append("frame_manifest_camera_set_mismatch")
    else:
        for camera, row in frame_manifest["cameras"].items():
            calibration = tuple(row.get("calibration_frames", ()))
            evaluation = tuple(row.get("evaluation_frames", ()))
            if len(calibration) != 10 or len(evaluation) != 22 or set(calibration) & set(evaluation) or len(set(calibration) | set(evaluation)) != 32:
                errors.append(f"invalid_10_22_split:{camera}")
    if query.get("gt_used") is not False or query.get("top_k") != 6 or set(query.get("cameras", {})) != expected:
        errors.append("invalid_top6_query_contract")
    else:
        for camera, rows in query["cameras"].items():
            if len(rows) != 6 or [row.get("proxy_rank") for row in rows] != list(range(1, 7)):
                errors.append(f"invalid_top6_trace:{camera}")

    prediction_rows = []
    try:
        prediction_rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_predictions_jsonl:{exc}")
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        by_camera.setdefault(row.get("camera_id", ""), []).append(row)
    if set(by_camera) != expected:
        errors.append("prediction_camera_set_mismatch")
    else:
        for camera, rows in by_camera.items():
            if len(rows) != 7 or sum(row.get("config_id") == "B0" for row in rows) != 1:
                errors.append(f"invalid_prediction_condition_count:{camera}")
    operators = fingerprints.get("operators", {})
    if len(operators) != 28:
        errors.append("operator_fingerprint_count_not_28")
    else:
        for key, operator in operators.items():
            claimed = operator.get("fingerprint_sha256")
            payload = {name: value for name, value in operator.items() if name != "fingerprint_sha256"}
            if claimed != _canonical_sha(payload):
                errors.append(f"operator_fingerprint_mismatch:{key}")

    if result.get("schema_version") != "sceneconfig.top6-calibrated-replay.v1":
        errors.append("invalid_result_schema")
    if result.get("evidence_status") != "DEVELOPMENT_ONLY" or result.get("holdout_opened") is not False or result.get("sealed_camera_used") is not False or result.get("qwen_called") is not False:
        errors.append("result_evidence_boundary_violation")
    if result.get("gt_access_order") != "after_pre_gt_seal" or result.get("calibration_is_mechanism_only_not_statistical_guarantee") is not True:
        errors.append("result_calibration_boundary_missing")
    if set(result.get("per_camera", {})) != expected or result.get("gate", {}).get("all_four_cameras_retained") is not True:
        errors.append("result_camera_retention_failure")
    else:
        for camera, row in result["per_camera"].items():
            if row.get("c10", {}).get("metrics", {}).get("frame_count") != 22.0:
                errors.append(f"c10_not_evaluated_on_22_frames:{camera}")
            if len(row.get("calibration_selection_diagnostics", ())) != 6:
                errors.append(f"missing_six_calibration_diagnostics:{camera}")
    if provenance.get("evidence_status") != "DEVELOPMENT_ONLY" or provenance.get("holdout_opened") is not False or provenance.get("gt_access_after_pre_gt_seal") is not True:
        errors.append("invalid_provenance_boundary")

    method_passed = result.get("gate", {}).get("passed") is True
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "validation_outcome": ("VALIDATED_METHOD_PASS" if method_passed else "VALIDATED_REJECTION_OR_FALLBACK") if not errors else "INVALID_EVIDENCE_PACK",
        "method_gate_passed": method_passed,
        "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False,
        "errors": errors,
        "summary": "Evidence pack is structurally valid; method success is reported separately from validator success." if not errors else "Evidence pack failed structural or boundary validation.",
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

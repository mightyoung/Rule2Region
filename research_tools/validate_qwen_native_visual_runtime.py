from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_CASES = {
    *(f"image_r{repeat}" for repeat in range(3)),
    *(f"video4_r{repeat}" for repeat in range(3)),
    *(f"video8_r{repeat}" for repeat in range(3)),
}
HF_REFERENCE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODELSCOPE_SOURCE = "ModelScope official Qwen organization"
WARNING_PATTERNS = {
    "token_position": re.compile(r"non-consecutive token position|token.position|position.*token", re.I),
    "truncation": re.compile(r"truncat|context[^\n]*(?:exceed|full)|too many tokens", re.I),
    "oom": re.compile(r"out of memory|cuda.*alloc|failed to allocate", re.I),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "case-manifest.json"
    provenance_path = run_dir / "provenance.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"status": "failed", "passed": False, "errors": ["CASE_MANIFEST_MISSING"]}
    if not provenance_path.is_file():
        return {"status": "failed", "passed": False, "errors": ["PROVENANCE_MISSING"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    server_log_path = run_dir / "server.log"
    if not server_log_path.is_file():
        errors.append("SERVER_LOG_MISSING")
        server_log = ""
    else:
        server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
        for warning_name, pattern in WARNING_PATTERNS.items():
            if pattern.search(server_log):
                errors.append(f"SERVER_WARNING:{warning_name}")
    cases: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for row in manifest.get("cases", []):
        case_id = row.get("case_id")
        if not isinstance(case_id, str):
            errors.append("CASE_ID_INVALID")
            continue
        observed_ids.add(case_id)
        response_path = run_dir / "cases" / f"{case_id}.response.json"
        meta_path = run_dir / "cases" / f"{case_id}.meta.json"
        log_path = run_dir / "cases" / f"{case_id}.log"
        if not all(path.is_file() for path in (response_path, meta_path, log_path)):
            errors.append(f"CASE_FILES_MISSING:{case_id}")
            continue
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            errors.append(f"RESPONSE_NOT_JSON:{case_id}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        expected_evidence_id = row.get("evidence_id")
        schema_valid = (
            isinstance(response, dict)
            and set(response) == {"role", "evidence_ids", "seen_frame_ids", "observations", "summary"}
            and response.get("role") == row.get("role")
            and response.get("evidence_ids") == [expected_evidence_id]
            and isinstance(response.get("observations"), list)
            and bool(response.get("observations"))
            and all(
                isinstance(observation, str) and bool(observation.strip())
                for observation in response.get("observations", [])
            )
            and isinstance(response.get("summary"), str)
            and bool(response.get("summary", "").strip())
        )
        warnings = {name: bool(pattern.search(log)) for name, pattern in WARNING_PATTERNS.items()}
        media_kind = row.get("media_kind")
        requested_frames = row.get("requested_frames")
        prepared_frames = meta.get("prepared_frames")
        expected_frame_ids = row.get("expected_frame_ids")
        seen_frame_ids = response.get("seen_frame_ids")
        frame_count_valid = (
            isinstance(requested_frames, int)
            and requested_frames > 0
            and prepared_frames == requested_frames
            and row.get("prepared_frames") == requested_frames
            and seen_frame_ids == expected_frame_ids
            and len(seen_frame_ids) == requested_frames
        )
        success = (
            meta.get("exit_code") == 0
            and schema_valid
            and frame_count_valid
            and not any(warnings.values())
            and isinstance(meta.get("prompt_tokens"), int) and meta["prompt_tokens"] > 0
            and isinstance(meta.get("completion_tokens"), int) and meta["completion_tokens"] > 0
            and isinstance(meta.get("wall_ms"), (int, float)) and meta["wall_ms"] > 0
            and isinstance(meta.get("peak_gpu_memory_mib"), int) and meta["peak_gpu_memory_mib"] >= 0
            and bool(re.fullmatch(r"[0-9a-f]{64}", row.get("media_sha256", "")))
        )
        if not success:
            errors.append(f"CASE_GATE_FAILED:{case_id}")
        cases.append(
            {
                "case_id": case_id,
                "media_kind": media_kind,
                "requested_frames": requested_frames,
                "prepared_frames": prepared_frames,
                "seen_frame_ids": seen_frame_ids,
                "schema_valid": schema_valid,
                "frame_count_valid": frame_count_valid,
                "warnings": warnings,
                "success": success,
                "wall_ms": meta.get("wall_ms"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "peak_gpu_memory_mib": meta.get("peak_gpu_memory_mib"),
            }
        )

    if observed_ids != EXPECTED_CASES:
        errors.append("CASE_MATRIX_INCOMPLETE")
    required_provenance = {
        "model_id",
        "model_source",
        "model_revision",
        "model_identity_mode",
        "model_content_manifest",
        "model_content_manifest_sha256",
        "runtime",
        "runtime_version",
        "transformers_version",
        "torch_version",
        "gpu",
        "holdout_opened",
        "deployment_authority",
    }
    if not required_provenance <= set(provenance):
        errors.append("PROVENANCE_FIELDS_MISSING")
    if provenance.get("model_id") != "Qwen/Qwen3.5-9B":
        errors.append("MODEL_ID_MISMATCH")
    model_source = provenance.get("model_source")
    model_revision = provenance.get("model_revision")
    if model_source == "Hugging Face official Qwen organization":
        if model_revision != HF_REFERENCE_REVISION:
            errors.append("MODEL_REVISION_MISMATCH")
    elif model_source == MODELSCOPE_SOURCE:
        if model_revision != "master":
            errors.append("MODELSCOPE_REVISION_MISMATCH")
        if provenance.get("hf_reference_revision") != HF_REFERENCE_REVISION:
            errors.append("HF_REFERENCE_REVISION_MISSING")
    else:
        errors.append("MODEL_SOURCE_INVALID")
    model_content_manifest = provenance.get("model_content_manifest")
    if provenance.get("model_identity_mode") != "content-addressed-full-recursive-manifest":
        errors.append("MODEL_IDENTITY_MODE_INVALID")
    if not isinstance(model_content_manifest, list) or not model_content_manifest:
        errors.append("MODEL_CONTENT_MANIFEST_MISSING")
    else:
        paths = {row.get("path") for row in model_content_manifest if isinstance(row, dict)}
        required_config = {"config.json", "preprocessor_config.json"}
        if not required_config <= paths:
            errors.append("MODEL_CONFIG_HASHES_MISSING")
        if not any(isinstance(name, str) and name.endswith(".safetensors") for name in paths):
            errors.append("MODEL_WEIGHT_HASHES_MISSING")
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
            for row in model_content_manifest
        ):
            errors.append("MODEL_FILE_HASH_INVALID")
        canonical = json.dumps(
            model_content_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected_manifest_hash = hashlib.sha256(canonical).hexdigest()
        if provenance.get("model_content_manifest_sha256") != expected_manifest_hash:
            errors.append("MODEL_CONTENT_MANIFEST_HASH_MISMATCH")
    if provenance.get("holdout_opened") is not False:
        errors.append("HOLDOUT_BOUNDARY_VIOLATED")
    if provenance.get("deployment_authority") != "DETERMINISTIC_COMPILER_AND_HUMAN":
        errors.append("DEPLOYMENT_AUTHORITY_INVALID")

    passed = not errors and len(cases) == len(EXPECTED_CASES) and all(row["success"] for row in cases)
    return {
        "schema_version": "sceneconfig.qwen-native-visual-runtime-result.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "evidence_status": "DEVELOPMENT_ONLY",
        "run_dir": str(run_dir),
        "gate": {
            "expected_cases": len(EXPECTED_CASES),
            "observed_cases": len(cases),
            "successful_cases": sum(row["success"] for row in cases),
            "token_position_warning_cases": sum(row["warnings"]["token_position"] for row in cases),
            "truncation_warning_cases": sum(row["warnings"]["truncation"] for row in cases),
            "oom_warning_cases": sum(row["warnings"]["oom"] for row in cases),
        },
        "provenance": provenance,
        "cases": cases,
        "errors": errors,
        "artifact_sha256": {
            "case_manifest": _sha256(manifest_path),
            "provenance": _sha256(provenance_path),
            **({"server_log": _sha256(server_log_path)} if server_log_path.is_file() else {}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

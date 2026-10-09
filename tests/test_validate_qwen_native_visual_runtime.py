from __future__ import annotations

import hashlib
import json
from pathlib import Path

from research_tools.validate_qwen_native_visual_runtime import EXPECTED_CASES, validate_run


def _make_run(root: Path, *, warning_case: str | None = None) -> Path:
    run = root / "run"
    cases_dir = run / "cases"
    cases_dir.mkdir(parents=True)
    cases = []
    for case_id in sorted(EXPECTED_CASES):
        if case_id.startswith("image"):
            kind, frames = "image", 1
        elif case_id.startswith("video4"):
            kind, frames = "video", 4
        else:
            kind, frames = "video", 8
        evidence_id = f"evidence:{case_id}"
        expected_frame_ids = ["I0"] if kind == "image" else [f"F{index}" for index in range(frames)]
        role = "detection_overlay_visual_analyst" if case_id.startswith("video4") else "clean_visual_scene_analyst"
        cases.append(
            {
                "case_id": case_id,
                "media_kind": kind,
                "requested_frames": frames,
                "prepared_frames": frames,
                "expected_frame_ids": expected_frame_ids,
                "evidence_id": evidence_id,
                "role": role,
                "media_sha256": "d" * 64,
            }
        )
        response = {
            "role": role,
            "evidence_ids": [evidence_id],
            "seen_frame_ids": expected_frame_ids,
            "observations": ["pixel-grounded transport probe"],
            "summary": "bounded observation",
        }
        meta = {
            "exit_code": 0,
            "prepared_frames": frames,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "wall_ms": 25,
            "peak_gpu_memory_mib": 100,
        }
        (cases_dir / f"{case_id}.response.json").write_text(json.dumps(response), encoding="utf-8")
        (cases_dir / f"{case_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        log = "non-consecutive token position" if case_id == warning_case else "clean"
        (cases_dir / f"{case_id}.log").write_text(log, encoding="utf-8")
    (run / "case-manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    (run / "server.log").write_text("clean", encoding="utf-8")
    model_content_manifest = [
        {"path": "config.json", "size": 1, "sha256": "a" * 64},
        {"path": "preprocessor_config.json", "size": 2, "sha256": "b" * 64},
        {"path": "model-00001-of-00001.safetensors", "size": 3, "sha256": "c" * 64},
    ]
    model_content_manifest_sha256 = hashlib.sha256(
        json.dumps(model_content_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.5-9B",
                "model_source": "ModelScope official Qwen organization",
                "model_revision": "master",
                "hf_reference_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "model_identity_mode": "content-addressed-full-recursive-manifest",
                "model_content_manifest": model_content_manifest,
                "model_content_manifest_sha256": model_content_manifest_sha256,
                "runtime": "vllm",
                "runtime_version": "test",
                "transformers_version": "test",
                "torch_version": "test",
                "gpu": "test",
                "holdout_opened": False,
                "deployment_authority": "DETERMINISTIC_COMPILER_AND_HUMAN",
            }
        ),
        encoding="utf-8",
    )
    return run


def test_validator_accepts_complete_native_visual_matrix(tmp_path: Path) -> None:
    result = validate_run(_make_run(tmp_path))
    assert result["passed"] is True
    assert result["gate"]["successful_cases"] == 9


def test_validator_rejects_token_position_warning(tmp_path: Path) -> None:
    result = validate_run(_make_run(tmp_path, warning_case="video8_r1"))
    assert result["passed"] is False
    assert result["gate"]["token_position_warning_cases"] == 1


def test_validator_rejects_missing_case(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    manifest_path = run / "case-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_run(run)
    assert result["passed"] is False
    assert "CASE_MATRIX_INCOMPLETE" in result["errors"]


def test_validator_rejects_modelscope_without_weight_hashes(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["model_content_manifest"] = provenance["model_content_manifest"][:2]
    provenance["model_content_manifest_sha256"] = hashlib.sha256(
        json.dumps(
            provenance["model_content_manifest"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    result = validate_run(run)
    assert result["passed"] is False
    assert "MODEL_WEIGHT_HASHES_MISSING" in result["errors"]


def test_validator_rejects_unseen_video_frame(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    response_path = run / "cases" / "video8_r1.response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["seen_frame_ids"].pop()
    response_path.write_text(json.dumps(response), encoding="utf-8")
    result = validate_run(run)
    assert result["passed"] is False
    assert "CASE_GATE_FAILED:video8_r1" in result["errors"]

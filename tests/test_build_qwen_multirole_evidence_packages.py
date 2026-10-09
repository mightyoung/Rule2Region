from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_tools.build_qwen_multirole_evidence_packages import (
    DEPLOYMENT_AUTHORITY,
    build_packages,
    validate_proposal_package,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / ".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2"
ARTIFACT = ROOT / "artifacts/sceneconfig/sceneconfig-top6-calibrated-20260814-v1"


def _make_valid_visual_run(root: Path) -> Path:
    run = root / "visual-run"
    cases_dir = run / "cases"
    cases_dir.mkdir(parents=True)
    cases = []
    for repeat in range(3):
        for prefix, role, frames, markers in (
            ("image", "clean_visual_scene_analyst", 1, ["K7Q"]),
            ("video4", "detection_overlay_visual_analyst", 4, ["M3Z", "R8C", "T2N", "V5H"]),
            ("video8", "clean_visual_scene_analyst", 8, ["P4D", "L9X", "B6K", "W2S", "G7M", "C3V", "N8F", "H5R"]),
        ):
            case_id = f"{prefix}_r{repeat}"
            evidence_id = f"visual:{case_id}"
            cases.append(
                {
                    "case_id": case_id,
                    "media_kind": "image" if prefix == "image" else "video",
                    "requested_frames": frames,
                    "prepared_frames": frames,
                    "expected_frame_ids": markers,
                    "evidence_id": evidence_id,
                    "role": role,
                    "media_sha256": "d" * 64,
                }
            )
            (cases_dir / f"{case_id}.response.json").write_text(
                json.dumps(
                    {
                        "role": role,
                        "evidence_ids": [evidence_id],
                        "seen_frame_ids": markers,
                        "observations": ["pixel-grounded transport probe"],
                        "summary": "bounded",
                    }
                ),
                encoding="utf-8",
            )
            (cases_dir / f"{case_id}.meta.json").write_text(
                json.dumps(
                    {
                        "exit_code": 0,
                        "prepared_frames": frames,
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "wall_ms": 25,
                        "peak_gpu_memory_mib": 100,
                    }
                ),
                encoding="utf-8",
            )
            (cases_dir / f"{case_id}.log").write_text("clean", encoding="utf-8")
    model_manifest = [
        {"path": "config.json", "size": 1, "sha256": "a" * 64},
        {"path": "preprocessor_config.json", "size": 2, "sha256": "b" * 64},
        {"path": "model.safetensors", "size": 3, "sha256": "c" * 64},
    ]
    model_manifest_hash = hashlib.sha256(
        json.dumps(model_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (run / "case-manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    (run / "server.log").write_text("clean", encoding="utf-8")
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.5-9B",
                "model_source": "ModelScope official Qwen organization",
                "model_revision": "master",
                "hf_reference_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                "model_identity_mode": "content-addressed-full-recursive-manifest",
                "model_content_manifest": model_manifest,
                "model_content_manifest_sha256": model_manifest_hash,
                "runtime": "vllm-openai-api-server",
                "runtime_version": "0.27.1",
                "transformers_version": "test",
                "torch_version": "test",
                "gpu": "test",
                "holdout_opened": False,
                "deployment_authority": DEPLOYMENT_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    return run


def _build(tmp_path: Path, *, visual_runtime_run_dir: Path | None = None) -> tuple[dict, dict, dict]:
    paths = build_packages(
        camera_id="VIRAT_S_010204",
        decision_catalog_path=SPEC / "camera-catalog-010204-v1/decision-catalog.json",
        evaluation_audit_path=SPEC / "camera-catalog-010204-v1/evaluation-audit.json",
        predictions_path=ARTIFACT / "predictions.jsonl",
        frame_split_path=ARTIFACT / "frame-split-manifest.json",
        fingerprints_path=ARTIFACT / "operator-fingerprints.json",
        output_dir=tmp_path / "package",
        visual_manifest_path=SPEC / "visual-manifest-010204-v1.json",
        business_profile_registry_path=SPEC / "preference-profile-v1/profile-registry.json",
        visual_runtime_run_dir=visual_runtime_run_dir,
    )
    return tuple(json.loads(path.read_text(encoding="utf-8")) for path in paths.values())  # type: ignore[return-value]


def test_proposal_is_calibration_only_and_holdout_is_physically_separate(tmp_path: Path) -> None:
    proposal, evaluation, receipt = _build(tmp_path)
    serialized = json.dumps(proposal, ensure_ascii=False).lower()
    assert "evaluation_metrics" not in serialized
    assert "evaluation_frames" not in serialized
    assert "evaluation_audit" not in serialized
    assert proposal["proposal_scope"] == "CALIBRATION_ONLY"
    assert evaluation["stage"] == "POST_PROPOSAL_EVALUATION"
    assert evaluation["accessible_by"] == ["holdout_evaluator"]
    assert evaluation["proposal_sha256"] == proposal["proposal_sha256"]
    assert receipt["holdout_isolation"] == "LOGICALLY_SEPARATE_POST_FREEZE_FILE_SAME_OUTPUT_ROOT"


def test_role_assignment_enforces_abstention_and_authority(tmp_path: Path) -> None:
    proposal, _, _ = _build(tmp_path)
    roles = proposal["role_packages"]
    assert roles["clean_visual_scene_analyst"]["required_response"] == "ABSTAIN_VISUAL_RUNTIME_REJECTED"
    assert (
        roles["detection_overlay_visual_analyst"]["required_response"]
        == "ABSTAIN_VISUAL_RUNTIME_REJECTED"
    )
    assert "detector_overlays" in roles["clean_visual_scene_analyst"]["forbidden_inputs"]
    assert roles["tracking_temporal_analyst"]["required_response"] == "ABSTAIN_TRACKER_NOT_EXECUTED"
    assert roles["deployment_resource_expert"]["qwen_may_deploy"] is False
    assert roles["deployment_resource_expert"]["deployment_authority"] == DEPLOYMENT_AUTHORITY
    assert roles["unified_synthesizer"]["raw_evidence_access"] is False
    assets = roles["clean_visual_scene_analyst"]["registered_scene_assets"]
    assert len(assets) == 14
    assert sum(asset["split"] == "CALIBRATION" for asset in assets) == 4
    assert all(len(asset["sha256"]) == 64 for asset in assets)


def test_detection_ledger_contains_only_ten_calibration_frames(tmp_path: Path) -> None:
    proposal, _, _ = _build(tmp_path)
    role = proposal["role_packages"]["algorithm_diagnostician"]
    allowed = {186, 189, 193, 196, 199, 202, 205, 209, 212, 215}
    assert {item["config_id"] for item in role["calibration_detection_ledger"]} == {
        "B0",
        "C00",
        "C01",
        "C02",
        "C03",
        "C04",
    }
    for ledger in role["calibration_detection_ledger"]:
        assert {frame["frame"] for frame in ledger["frames"]} == allowed
        assert all("merged_boxes" in frame for frame in ledger["frames"])


def test_eligible_ids_and_tracker_receipt_are_exact(tmp_path: Path) -> None:
    proposal, _, _ = _build(tmp_path)
    roles = proposal["role_packages"]
    assert roles["unified_synthesizer"]["eligible_config_ids"] == ["C00", "C02", "C03", "C04"]
    assert roles["tracking_temporal_analyst"]["track_ids"] is None
    fingerprints = roles["deployment_resource_expert"]["operator_fingerprints"]
    assert fingerprints
    assert {value["tracker"] for value in fingerprints.values()} == {"none"}


def test_validator_rejects_holdout_leakage() -> None:
    bad = {
        "stage": "PROPOSAL_CONFIGURATION",
        "deployment_authority": DEPLOYMENT_AUTHORITY,
        "qwen_deployment_authority": False,
        "evaluation_metrics": {"mae": 1.0},
        "role_packages": {},
    }
    with pytest.raises(ValueError, match="forbidden proposal field"):
        validate_proposal_package(bad)


def test_output_directory_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "package"
    _build(tmp_path)
    assert output.exists()
    with pytest.raises(FileExistsError):
        _build(tmp_path)


def test_validated_run_only_makes_visual_roles_execution_eligible(tmp_path: Path) -> None:
    visual_run = _make_valid_visual_run(tmp_path)
    proposal, _, receipt = _build(tmp_path, visual_runtime_run_dir=visual_run)
    roles = proposal["role_packages"]
    for role_name in ("clean_visual_scene_analyst", "detection_overlay_visual_analyst"):
        assert roles[role_name]["availability"] == "EXECUTION_ELIGIBLE_NATIVE_VISUAL_RUNTIME"
        assert roles[role_name]["role_execution_status"] == "NOT_RUN"
        assert roles[role_name]["role_report_status"] == "ABSENT"
    assert roles["tracking_temporal_analyst"]["availability"] == "UNAVAILABLE"
    assert proposal["qwen_deployment_authority"] is False
    assert (
        receipt["visual_runtime_status"]
        == "PASSED_NATIVE_QWEN35_MULTIMODAL_TRANSPORT_CAPABILITY"
    )


def test_tampered_visual_case_cannot_enable_roles(tmp_path: Path) -> None:
    visual_run = _make_valid_visual_run(tmp_path)
    response_path = visual_run / "cases/video8_r1.response.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["seen_frame_ids"] = ["P4D"]
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="did not satisfy"):
        _build(tmp_path, visual_runtime_run_dir=visual_run)

import hashlib
import json
from pathlib import Path

from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS
from research_tools.validate_sceneconfig_top6_calibrated import validate


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_pack(root: Path) -> None:
    root.mkdir()
    (root / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    (root / "gpu-admission.txt").write_text("free_mib=20000\n", encoding="utf-8")
    (root / "gpu-final.csv").write_text("1, gpu\n", encoding="utf-8")
    _write(root / "provenance.json", {"evidence_status": "DEVELOPMENT_ONLY", "holdout_opened": False, "gt_access_after_pre_gt_seal": True})
    cameras = {
        camera: {"calibration_frames": list(range(1, 11)), "evaluation_frames": list(range(11, 33))}
        for camera in EXPECTED_CAMERAS
    }
    _write(root / "frame-split-manifest.json", {"cameras": cameras})
    query_cameras = {
        camera: [{"config_id": f"C{i:02d}", "proxy_rank": i + 1} for i in range(6)]
        for camera in EXPECTED_CAMERAS
    }
    _write(root / "top6-query-trace.json", {"gt_used": False, "top_k": 6, "cameras": query_cameras})
    prediction_rows = []
    for camera in EXPECTED_CAMERAS:
        prediction_rows.append({"camera_id": camera, "config_id": "B0", "frames": []})
        prediction_rows.extend({"camera_id": camera, "config_id": f"C{i:02d}", "frames": []} for i in range(6))
    (root / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8")
    operators = {}
    for row in prediction_rows:
        key = f"{row['camera_id']}:{row['config_id']}"
        payload = {"camera_id": row["camera_id"], "config": {"config_id": row["config_id"]}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        operators[key] = {**payload, "fingerprint_sha256": digest}
    _write(root / "operator-fingerprints.json", {"operators": operators})
    _write(root / "pre-gt-execution-summary.json", {"cameras": {camera: {} for camera in EXPECTED_CAMERAS}})
    seal_files = ["frame-split-manifest.json", "top6-query-trace.json", "predictions.jsonl", "operator-fingerprints.json", "pre-gt-execution-summary.json"]
    _write(root / "PRE_GT_SEAL.json", {"schema_version": "sceneconfig.pre-gt-seal.v1", "stage": "predictions_complete_gt_unread", "holdout_opened": False, "files": {name: _sha(root / name) for name in seal_files}})
    per_camera = {
        camera: {"c10": {"metrics": {"frame_count": 22.0}}, "calibration_selection_diagnostics": [{}] * 6}
        for camera in EXPECTED_CAMERAS
    }
    _write(root / "top6-calibrated-result.json", {
        "schema_version": "sceneconfig.top6-calibrated-replay.v1", "evidence_status": "DEVELOPMENT_ONLY",
        "holdout_opened": False, "sealed_camera_used": False, "qwen_called": False,
        "gt_access_order": "after_pre_gt_seal", "calibration_is_mechanism_only_not_statistical_guarantee": True,
        "per_camera": per_camera, "gate": {"all_four_cameras_retained": True, "passed": False},
    })
    names = [path for path in root.iterdir() if path.name != "SHA256SUMS"]
    (root / "SHA256SUMS").write_text("".join(f"{_sha(path)}  ./{path.name}\n" for path in sorted(names)), encoding="utf-8")


def test_validator_accepts_validated_rejection(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _valid_pack(run_dir)
    report = validate(run_dir)
    assert report["passed"] is True
    assert report["method_gate_passed"] is False
    assert report["validation_outcome"] == "VALIDATED_REJECTION_OR_FALLBACK"


def test_validator_rejects_tampered_pre_gt_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _valid_pack(run_dir)
    with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    report = validate(run_dir)
    assert report["passed"] is False
    assert any("predictions.jsonl" in error for error in report["errors"])

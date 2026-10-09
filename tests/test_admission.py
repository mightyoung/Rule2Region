import json
from pathlib import Path

from codecref_lite.admission import validate_admission_receipt
from codecref_lite.io import sha256_file


def _receipt() -> dict[str, object]:
    return {
        "source_video_id": "video-1",
        "agreement_version": "1.0",
        "agreement_signed_at": "2026-07-31",
        "release_id": "release-1",
        "archive_path": "archive.zip",
        "archive_sha256": "a" * 64,
        "video_path": "video.mp4",
        "video_sha256": "b" * 64,
        "annotation_revision": "rev-1",
        "annotation_path": "tracks.json",
        "annotation_sha256": "c" * 64,
        "split": "validate",
        "scene_id": "scene-1",
        "timeline_mapping_version": "v1",
        "timeline_mapping_path": "timeline.json",
        "timeline_mapping_sha256": "d" * 64,
        "video_time_base_seconds": 1 / 30,
        "annotation_time_base_seconds": 1 / 30,
        "timeline_offset_seconds": 0.0,
        "video_duration_seconds": 10.0,
        "annotation_duration_seconds": 10.0,
        "source_fps": 30.0,
        "timeline_max_error_seconds": 0.01,
        "cloud_processing_allowed": True,
        "derivative_processing_allowed": True,
        "redistribution_prohibited_acknowledged": True,
    }


def test_admission_passes_complete_receipt_without_file_check() -> None:
    decision = validate_admission_receipt(_receipt(), verify_files=False)
    assert decision.admitted
    assert decision.errors == ()


def test_admission_fails_closed_on_permission() -> None:
    receipt = _receipt()
    receipt["cloud_processing_allowed"] = False
    decision = validate_admission_receipt(receipt, verify_files=False)
    assert not decision.admitted
    assert "permission_not_confirmed:cloud_processing_allowed" in decision.errors


def test_admission_resolves_relative_paths_and_binds_timeline_mapping(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    for name in ("archive.zip", "video.mp4", "tracks.json"):
        (tmp_path / name).write_bytes(name.encode("ascii"))
    mapping_fields = (
        "source_video_id",
        "timeline_mapping_version",
        "video_time_base_seconds",
        "annotation_time_base_seconds",
        "timeline_offset_seconds",
        "video_duration_seconds",
        "annotation_duration_seconds",
        "source_fps",
        "timeline_max_error_seconds",
    )
    mapping = {field: receipt[field] for field in mapping_fields}
    (tmp_path / "timeline.json").write_text(json.dumps(mapping), encoding="utf-8")
    receipt["archive_sha256"] = sha256_file(tmp_path / "archive.zip")
    receipt["video_sha256"] = sha256_file(tmp_path / "video.mp4")
    receipt["annotation_sha256"] = sha256_file(tmp_path / "tracks.json")
    receipt["timeline_mapping_sha256"] = sha256_file(tmp_path / "timeline.json")

    decision = validate_admission_receipt(receipt, base_dir=tmp_path)
    assert decision.admitted

    mapping["source_fps"] = 25.0
    (tmp_path / "timeline.json").write_text(json.dumps(mapping), encoding="utf-8")
    receipt["timeline_mapping_sha256"] = sha256_file(tmp_path / "timeline.json")
    decision = validate_admission_receipt(receipt, base_dir=tmp_path)
    assert not decision.admitted
    assert "timeline_mapping_mismatch:source_fps" in decision.errors

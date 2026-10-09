from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tools.freeze_industrial_g0 import freeze_g0_pilot


def _admission(state: str) -> dict:
    return {
        "schema_version": "fieldpack.dataset-admission.v1",
        "datasets": [
            {
                "role": "primary",
                "dataset_id": "virat-ground-release2",
                "official_url": "https://viratdata.org/",
                "agreement_execution": state,
                "processing_permission": state,
                "video_inventory": "verified",
                "annotation_mapping": "verified",
                "source_hashes": "verified",
            }
        ],
    }


def _inventory() -> dict:
    return {
        "schema_version": "fieldpack.video-inventory.v1",
        "frames_per_video": 4,
        "minimum_gap_seconds": 5,
        "videos": [
            {
                "dataset_id": "virat-ground-release2",
                "camera_id": "cam-01",
                "source_video_id": "video-a",
                "total_frames": 1_000,
                "fps": 10.0,
                "video_sha256": "a" * 64,
                "annotation_sha256": "b" * 64,
            }
        ],
    }


def test_freeze_refuses_blocked_dataset(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="dataset_admission_blocked"):
        freeze_g0_pilot(_admission("unsealed"), _inventory(), tmp_path / "pilot.json")


def test_freeze_writes_immutable_manifest_and_allows_identical_rerun(tmp_path: Path) -> None:
    output = tmp_path / "pilot.json"
    first = freeze_g0_pilot(_admission("sealed"), _inventory(), output)
    second = freeze_g0_pilot(_admission("sealed"), _inventory(), output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert first == second
    assert first["status"] == "FROZEN"
    assert first["sample_count"] == 4
    assert payload["admissions"][0]["status"] == "READY"
    assert len(payload["manifest_sha256"]) == 64

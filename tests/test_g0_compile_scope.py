import json
from pathlib import Path

from research_tools.g0_compile_scope import build_scope, load_sealed


def test_scope_binds_real_ids_and_deployment_subsets() -> None:
    video = {
        "samples": [
                {
                    "sample_id": f"video-{index}",
                    "sequence_id": f"MOT17-{index % 7:02d}",
                "camera_motion_stratum": (
                    "STATIC_PRIMARY" if index < 17 else "MOVING_CAMERA_STRESS"
                ),
                "deployment_member": index < 24,
            }
            for index in range(50)
        ],
        "manifest_sha256": "a" * 64,
    }
    image = {
        "samples": [
            {
                "image": f"image-{index}.jpg",
                "tier": "formal",
                "deployment_member": index < 12,
            }
            for index in range(48)
        ]
        + [
            {
                "image": f"development-{index}.jpg",
                "tier": "development",
                "deployment_member": False,
            }
            for index in range(12)
        ],
        "manifest_sha256": "b" * 64,
    }
    scope = build_scope(video, image)
    assert len(scope.video_ids) == 50
    assert len(scope.image_ids) == 48
    assert len(scope.video_deployment_ids) == 24
    assert len(scope.image_deployment_ids) == 12


def test_load_sealed_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"manifest_sha256": "0" * 64, "value": 1}),
        encoding="utf-8",
    )
    try:
        load_sealed(path)
    except ValueError as error:
        assert "manifest_hash_mismatch" in str(error)
    else:
        raise AssertionError("hash mismatch was not rejected")

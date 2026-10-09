import json
from pathlib import Path

import pytest

from research_tools.prepare_virat_microclips import (
    CONFIRMATORY_FRAMES,
    CONFIRMATORY_VAL_COUNT,
    freeze_time_split,
    write_confirmatory_split,
)
from research_tools.route_a_clip_commissioning import VAL_COUNT, load_clips
from research_tools.run_dcrq_development_gate import SEALED_CAMERA


def test_confirmatory_pack_constants_exceed_development_val() -> None:
    assert CONFIRMATORY_VAL_COUNT > VAL_COUNT
    assert CONFIRMATORY_VAL_COUNT == 32
    assert CONFIRMATORY_FRAMES == 64
    assert CONFIRMATORY_FRAMES - CONFIRMATORY_VAL_COUNT == 32


def test_freeze_time_split_is_tail_validation_and_one_based() -> None:
    sampled = tuple(range(10, 74))
    split = freeze_time_split(sampled, val_count=32)
    assert split["frames"][0] == 11
    assert split["validation_frames"] == list(range(43, 75))
    assert split["history_frames"] == list(range(11, 43))
    assert len(split["history_frames"]) == 32
    assert set(split["history_frames"]).isdisjoint(split["validation_frames"])


def test_freeze_time_split_rejects_development_val_count() -> None:
    with pytest.raises(ValueError, match="confirmatory_val_count_must_exceed_development_16"):
        freeze_time_split(tuple(range(64)), val_count=16)


def test_write_confirmatory_split_hashes_and_seals_camera(tmp_path: Path) -> None:
    sources = [
        {
            "camera_id": "VIRAT_S_010004",
            "source_video_id": "VIRAT_S_010004_01_000163_000188",
            "sampled_frame_indices": list(range(64)),
        },
        {
            "camera_id": "VIRAT_S_010113",
            "source_video_id": "VIRAT_S_010113_00_000000_000100",
            "sampled_frame_indices": list(range(64)),
        },
    ]
    with pytest.raises(ValueError, match="sealed_camera"):
        write_confirmatory_split(tmp_path, sources, val_count=32, sealed_camera=SEALED_CAMERA)

    sources[1]["camera_id"] = "VIRAT_S_010204"
    sources[1]["source_video_id"] = "VIRAT_S_010204_03_000606_000632"
    payload = write_confirmatory_split(tmp_path, sources, val_count=32, sealed_camera=SEALED_CAMERA)
    split_path = tmp_path / "split-manifest.json"
    digest_path = tmp_path / "split-manifest.sha256"
    saved = json.loads(split_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "fieldpack.virat-h1-confirmatory-split.v1"
    assert saved["validation_frame_count"] == 32
    assert saved["frames_per_video"] == 64
    assert saved["same_view"] == {"VIRAT_S_010112": "VIRAT_S_010111"}
    assert payload["sha256"] == digest_path.read_text(encoding="utf-8").strip()
    assert SEALED_CAMERA not in json.dumps(saved["sequences"])
    assert saved["sealed_camera_id"] == SEALED_CAMERA


def test_load_clips_prefers_sealed_split(tmp_path: Path) -> None:
    sequence = "VIRAT_S_010004_01_000163_000188"
    seq_dir = tmp_path / "sequences" / sequence
    seq_dir.mkdir(parents=True)
    (seq_dir / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")
    cameras = [
        ("VIRAT_S_010004", "VIRAT_S_010004_01_000163_000188"),
        ("VIRAT_S_010004", "VIRAT_S_010004_07_000729_000809"),
        ("VIRAT_S_010111", "VIRAT_S_010111_02_000198_000310"),
        ("VIRAT_S_010111", "VIRAT_S_010111_07_000872_000909"),
        ("VIRAT_S_010112", "VIRAT_S_010112_00_000015_000095"),
        ("VIRAT_S_010112", "VIRAT_S_010112_01_000128_000187"),
        ("VIRAT_S_010204", "VIRAT_S_010204_03_000606_000632"),
        ("VIRAT_S_010204", "VIRAT_S_010204_09_001285_001336"),
    ]
    sequences = {}
    for camera, name in cameras:
        path = tmp_path / "sequences" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "gt.txt").write_text("", encoding="utf-8")
        frames = list(range(1, 65))
        sequences[name] = {
            "camera_id": camera,
            "history_frames": frames[:32],
            "validation_frames": frames[32:],
        }
    (tmp_path / "split-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "fieldpack.virat-h1-confirmatory-split.v1",
                "validation_frame_count": 32,
                "frames_per_video": 64,
                "sealed_camera_id": SEALED_CAMERA,
                "sequences": sequences,
            }
        ),
        encoding="utf-8",
    )
    clips = load_clips(tmp_path / "missing-cache", tmp_path)
    assert len(clips) == 8
    row = clips["VIRAT_S_010004_01_000163_000188"]
    assert len(row["validation_frames"]) == 32
    assert len(row["history_frames"]) == 32
    assert row["validation_frames"][0] == 33

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencegate.runner import AttemptArtifactStore
from evidencegate.terminals import TerminalState
from research_tools.g3_formal import (
    enumerate_video_source_paths,
    expected_label,
    load_quality_attempts,
    quality_gate_summary,
    validate_quality_contract,
)


ROOT = Path(__file__).resolve().parents[1]
GOAL = (
    ROOT
    / ".omx"
    / "goals"
    / "autoresearch"
    / "execute-evidencegate-engineering-paper-plan-v2-f"
    / "artifacts"
)


def _quality_unit(index: int) -> dict[str, object]:
    return {"unit_id": f"quality:{index:04d}"}


def _quality_record(
    index: int, state: TerminalState, *, correct: bool | None = None
) -> dict[str, object]:
    result = None if correct is None else {"correct": correct}
    return {
        "unit_id": f"quality:{index:04d}",
        "attempt_id": f"g3q-quality-{index:04d}-r0",
        "retry_index": 0,
        "terminal_state": state.value,
        "result": result,
    }


def test_enumerate_video_source_paths_uses_inclusive_exclusive_bounds(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "MOT17-02-DPM" / "img1"
    gt_path = tmp_path / "MOT17-02-DPM" / "gt" / "gt.txt"
    image_root.mkdir(parents=True)
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text("", encoding="utf-8")
    for index in range(1, 121):
        (image_root / f"{index:06d}.jpg").write_bytes(b"frame")
    sample = {
        "sample_id": "MOT17-02:4s:00",
        "start_frame_inclusive": 1,
        "end_frame_exclusive": 121,
        "frame_count": 120,
        "first_frame": "000001.jpg",
        "last_frame": "000120.jpg",
    }
    paths = enumerate_video_source_paths(sample, {"gt_path": str(gt_path)})
    assert len(paths) == 120
    assert paths[0].name == "000001.jpg"
    assert paths[-1].name == "000120.jpg"
    assert len(set(paths)) == 120


def test_enumerate_video_source_paths_rejects_missing_frame(tmp_path: Path) -> None:
    gt_path = tmp_path / "seq" / "gt" / "gt.txt"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text("", encoding="utf-8")
    (tmp_path / "seq" / "img1").mkdir()
    sample = {
        "sample_id": "window",
        "start_frame_inclusive": 1,
        "end_frame_exclusive": 2,
        "frame_count": 1,
        "first_frame": "000001.jpg",
        "last_frame": "000001.jpg",
    }
    with pytest.raises(FileNotFoundError, match="video_source_frames_missing"):
        enumerate_video_source_paths(sample, {"gt_path": str(gt_path)})


def test_expected_label_preserves_image_and_peak_video_count() -> None:
    assert (
        expected_label({"label": "yes", "text": "Is there a person?"}, "image") is True
    )
    assert expected_label({"label": "no", "text": "Is there a car?"}, "image") is False
    video = {
        "label": {"count": 23, "peak_frame": 117, "peak_track_ids": [1, 2]},
        "label_contract": "frozen",
    }
    assert expected_label(video, "video") == 23
    assert video["label"]["peak_frame"] == 117
    with pytest.raises(ValueError, match="formal_video_label_contract_missing"):
        expected_label({"label": {"count": 1}}, "video")


def test_quality_gate_exact_boundary_and_coverage() -> None:
    units = [_quality_unit(index) for index in range(986)]
    records = {
        f"quality:{index:04d}": [
            _quality_record(
                index,
                TerminalState.SUCCESS if index < 937 else TerminalState.INVALID,
                correct=index < 900,
            )
        ]
        for index in range(986)
    }
    passed = quality_gate_summary(units, records)
    assert passed["verdict"] == "PASS"
    assert passed["quality_successes"] == 937
    records["quality:0936"] = [_quality_record(936, TerminalState.INVALID)]
    assert quality_gate_summary(units, records)["verdict"] == "FAIL"
    records["quality:0936"] = [
        _quality_record(936, TerminalState.SUCCESS, correct=True)
    ]
    records.pop("quality:0985")
    missing = quality_gate_summary(units, records)
    assert missing["quality_successes"] == 937
    assert missing["quality_terminal_coverage"] == 985
    assert missing["verdict"] == "FAIL"


def test_load_quality_attempts_preserves_retry_and_resume(tmp_path: Path) -> None:
    store = AttemptArtifactStore(tmp_path)
    common = {
        "config_file_sha256": "a" * 64,
        "formal_manifest_file_sha256": "b" * 64,
        "unit_id": "quality:one",
    }
    first = {
        **common,
        "attempt_id": "g3q-quality-one-r0",
        "retry_index": 0,
        "terminal_state": "timeout",
    }
    second = {
        **common,
        "attempt_id": "g3q-quality-one-r1",
        "retry_index": 1,
        "terminal_state": "success",
        "result": {"correct": True},
    }
    store.write(first, attempt_id=first["attempt_id"])
    store.write(second, attempt_id=second["attempt_id"])
    grouped = load_quality_attempts(store, {"quality:one"}, "a" * 64, "b" * 64)
    assert [item["terminal_state"] for item in grouped["quality:one"]] == [
        "timeout",
        "success",
    ]
    assert (
        quality_gate_summary([{"unit_id": "quality:one"}], grouped, success_minimum=1)[
            "verdict"
        ]
        == "PASS"
    )
    with pytest.raises(ValueError, match="quality_attempt_frozen_identity_mismatch"):
        load_quality_attempts(store, {"quality:one"}, "c" * 64, "b" * 64)


def test_real_frozen_g3_contract_validates() -> None:
    g0 = GOAL / "g0"
    g1 = GOAL / "g1"
    g2 = GOAL / "g2"
    contract = validate_quality_contract(
        config_path=g2 / "g2-formal-execution-config-v8.json",
        formal_manifest_path=g2 / "g3-formal-quality-manifest-v1.json",
        scope_path=g0 / "g0-execution-scope-v2.json",
        image_scope_path=g0 / "g0-image-scope-v2.json",
        video_scope_path=g0 / "g0-video-scope.json",
        codec_manifest_path=g1 / "codec-derivative-manifest.json",
        model_index_path=g1 / "model-cache-index.json",
        g2_output_index_path=g2 / "g2-smoke-v1" / "g2-output-index.json",
        prompt_adjustment_path=g2 / "g2-prompt-adjustment-v1.json",
        verify_model_files=False,
        verify_source_files=False,
    )
    assert len(contract["formal"]["units"]) == 986
    assert len(contract["image_samples"]) == 48
    assert len(contract["video_samples"]) == 50
    aliases = contract["config"]["runtime"]["formal_condition_resolution"]
    assert aliases["image"]["query_aware_crop50"] == "center_crop50"
    assert aliases["video"]["reduced"] == "codec_guided50"


def test_real_frozen_contract_rejects_formal_sample_adjustment(tmp_path: Path) -> None:
    g0 = GOAL / "g0"
    g1 = GOAL / "g1"
    g2 = GOAL / "g2"
    receipt = json.loads(
        (g2 / "g2-prompt-adjustment-v1.json").read_text(encoding="utf-8")
    )
    receipt["source_evidence"]["formal_samples_used"] = True
    tampered = tmp_path / "prompt.json"
    tampered.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt_adjustment_receipt_hash_mismatch"):
        validate_quality_contract(
            config_path=g2 / "g2-formal-execution-config-v8.json",
            formal_manifest_path=g2 / "g3-formal-quality-manifest-v1.json",
            scope_path=g0 / "g0-execution-scope-v2.json",
            image_scope_path=g0 / "g0-image-scope-v2.json",
            video_scope_path=g0 / "g0-video-scope.json",
            codec_manifest_path=g1 / "codec-derivative-manifest.json",
            model_index_path=g1 / "model-cache-index.json",
            g2_output_index_path=g2 / "g2-smoke-v1" / "g2-output-index.json",
            prompt_adjustment_path=tampered,
            verify_model_files=False,
            verify_source_files=False,
        )

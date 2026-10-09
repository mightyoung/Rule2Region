from pathlib import Path

import pytest

from evidencegate import (
    ManifestConflictError,
    IOReceipt,
    ResourceLedger,
    ScopeVerdict,
    TerminalState,
    VisualEvidenceIR,
    ScopeManifest,
    VideoSample,
    aggregate_terminals,
    compile_matrix,
    decide_scope,
    manifest_sha256,
    write_immutable_manifest,
)
from evidencegate.terminals import AttemptRecord


def test_ir_validates_hashes_and_separates_resources():
    value = VisualEvidenceIR("image", "x", "a" * 64, "b" * 64, encoded_bytes=3, decoded_pixels=4, model_pixels=2, visual_tokens=1, prefill_tokens=5)
    assert value.source_hash == "a" * 64
    assert value.to_dict()["decoded_pixels"] == 4
    with pytest.raises(ValueError, match="nonnegative"):
        VisualEvidenceIR("image", "x", "a" * 64, "b" * 64, vram_bytes=-1)


@pytest.mark.parametrize(("video", "image", "expected"), [(True, True, ScopeVerdict.BROAD), (False, True, ScopeVerdict.IMAGE_ONLY), (True, False, ScopeVerdict.VIDEO_ONLY), (False, False, ScopeVerdict.STOP)])
def test_scope_is_deterministic_and_fail_closed(video, image, expected):
    assert decide_scope(video_admitted=video, image_admitted=image, video_preregistered=video and not image) is expected


@pytest.mark.parametrize(("scope", "counts"), [("BROAD", (986, 896, 36, 189, 2107)), ("IMAGE_ONLY", (348, 392, 12, 74, 826))])
def test_matrix_contract_is_exact_and_stable(scope, counts):
    manifest = _manifest()
    first, second = compile_matrix(scope, manifest), compile_matrix(scope, manifest)
    assert (first.quality_count, first.latency_count, first.smoke_count, first.retry_count, first.total_calls) == counts
    assert [u.unit_id for u in first.quality_units] == [u.unit_id for u in second.quality_units]
    assert first.retry_budget == counts[3]


def test_terminals_preserve_retries_and_have_one_aggregate():
    records = aggregate_terminals((AttemptRecord("u", "a1", TerminalState.TIMEOUT), AttemptRecord("u", "a2", TerminalState.SUCCESS, retry_index=1)))
    assert len(records) == 1 and records[0].state is TerminalState.SUCCESS and records[0].attempt_ids == ("a1", "a2")
    with pytest.raises(ValueError, match="duplicate_attempt"):
        aggregate_terminals((AttemptRecord("u", "a1", TerminalState.CRASH), AttemptRecord("v", "a1", TerminalState.CRASH)))


def test_resource_ledger_keeps_stage_dimensions_and_combined_stage():
    ledger = ResourceLedger()
    ledger.record("processor", encoded_bytes=10, decoded_pixels=20)
    ledger.record("vision+prefill", visual_tokens=3, prefill_tokens=4, combined=True)
    assert ledger.totals()["encoded_bytes"] == 10
    assert ledger.records[1].combined is True


def test_io_receipt_records_nfs_staging_and_peak_vram():
    ledger = ResourceLedger()
    ledger.record("vision", vram_bytes=10)
    ledger.record("generation", vram_bytes=20)
    ledger.record_io(
        IOReceipt(
            stage="input_read",
            source_path="/share/data4/input.jpg",
            source_sha256="a" * 64,
            storage_class="nfs",
            cache_state="staged",
            bytes_read=123,
            wall_clock_ms=4.5,
            included_in_latency=False,
            staging_path="/tmp/input.jpg",
            staging_sha256="a" * 64,
        )
    )
    assert ledger.totals()["vram_bytes"] == 20
    assert ledger.io_receipts[0].storage_class == "nfs"


def test_manifest_is_canonical_and_immutable(tmp_path: Path):
    path = tmp_path / "manifest.json"
    artifact = write_immutable_manifest(path, {"b": 2, "a": 1})
    assert artifact.sha256 == manifest_sha256({"a": 1, "b": 2})
    assert write_immutable_manifest(path, {"a": 1, "b": 2}) == artifact
    with pytest.raises(ManifestConflictError):
        write_immutable_manifest(path, {"a": 9})


def _manifest():
    return ScopeManifest(
        tuple(
            VideoSample(
                f"video-{i:02d}",
                f"sequence-{i % 7}",
                "STATIC_PRIMARY" if i < 17 else "MOVING_CAMERA_STRESS",
            )
            for i in range(50)
        ),
        tuple(f"image-{i:02d}" for i in range(48)),
        tuple(f"video-{i:02d}" for i in range(24)),
        tuple(f"image-{i:02d}" for i in range(12)),
        tuple(f"development-{i:02d}" for i in range(12)),
        "/share/data4/project/development.mp4",
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )


def test_scope_manifest_exact_strata_and_deployment_subsets():
    manifest = _manifest()
    assert len(manifest.video_samples) == 50
    assert sum(v.stratum == "STATIC_PRIMARY" for v in manifest.video_samples) == 17
    assert sum(v.stratum == "MOVING_CAMERA_STRESS" for v in manifest.video_samples) == 33
    with pytest.raises(ValueError, match="subset"):
        ScopeManifest(
            manifest.video_samples,
            manifest.formal_image_ids,
            ("not-a-video",) * 24,
            manifest.image_deployment_ids,
            manifest.development_image_ids,
            manifest.development_video_fixture_path,
            manifest.development_video_fixture_sha256,
            manifest.video_source_manifest_sha256,
            manifest.image_source_manifest_sha256,
        )


def test_video_only_scope_fails_closed_without_future_preregistration():
    assert decide_scope(video_admitted=True, image_admitted=False) is ScopeVerdict.STOP
    assert decide_scope(video_admitted=True, image_admitted=False, video_preregistered=True) is ScopeVerdict.VIDEO_ONLY


def test_broad_retry_is_budget_not_new_logical_units():
    plan = compile_matrix("BROAD", _manifest())
    assert plan.retry_budget == 189
    assert not hasattr(plan, "retry_units")
    assert len({u.unit_id for u in plan.latency_attempts}) == 64
    video_units = [unit for unit in plan.quality_units if unit.modality == "video"]
    assert all(unit.sequence_id and unit.camera_motion_stratum for unit in video_units)
    image_conditions = {unit.condition for unit in plan.quality_units if unit.modality == "image" and unit.model == "0.5b-bf16"}
    assert "center_crop50" in image_conditions
    assert "query_aware_crop50" not in image_conditions


def test_video_only_matrix_cannot_bypass_fail_closed_scope():
    with pytest.raises(ValueError, match="precise_preregistration"):
        compile_matrix("VIDEO_ONLY", _manifest())

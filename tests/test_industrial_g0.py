from __future__ import annotations

import pytest

from codecref_lite.datasets.industrial_g0 import (
    DatasetAdmission,
    IndustrialFrameRecord,
    PersonBox,
    PilotVideo,
    build_pilot_manifest,
    select_evenly_spaced_frames,
)


def test_dataset_admission_fails_closed_until_every_required_gate_is_sealed() -> None:
    admission = DatasetAdmission(
        dataset_id="virat-ground-release2",
        official_url="https://viratdata.org/",
        agreement_execution="unsealed",
        processing_permission="unsealed",
        video_inventory="unverified",
        annotation_mapping="verified",
        source_hashes="unverified",
    )

    assert admission.status == "BLOCKED"
    assert admission.blockers == (
        "agreement_execution:unsealed",
        "processing_permission:unsealed",
        "video_inventory:unverified",
        "source_hashes:unverified",
    )


def test_dataset_admission_allows_not_required_legal_gates_for_public_data() -> None:
    admission = DatasetAdmission(
        dataset_id="wildtrack",
        official_url="https://www.epfl.ch/labs/cvlab/data/data-wildtrack/",
        agreement_execution="not_required",
        processing_permission="not_required",
        video_inventory="verified",
        annotation_mapping="verified",
        source_hashes="verified",
    )

    assert admission.status == "READY"
    assert admission.blockers == ()


def test_dataset_admission_treats_verified_absence_as_a_blocker() -> None:
    admission = DatasetAdmission(
        dataset_id="meva",
        official_url="https://mevadata.org/",
        agreement_execution="not_required",
        processing_permission="not_required",
        video_inventory="absent",
        annotation_mapping="unverified",
        source_hashes="unverified",
    )

    assert admission.status == "BLOCKED"
    assert "video_inventory:absent" in admission.blockers


def test_even_frame_selection_includes_zero_person_frames_by_using_video_extent() -> None:
    frames = select_evenly_spaced_frames(
        total_frames=1_000,
        count=5,
        minimum_gap_frames=100,
    )

    assert frames == (99, 299, 499, 699, 899)


def test_even_frame_selection_rejects_impossible_spacing() -> None:
    with pytest.raises(ValueError, match="insufficient_frames_for_gap"):
        select_evenly_spaced_frames(
            total_frames=100,
            count=5,
            minimum_gap_frames=30,
        )


def test_frame_record_accepts_zero_people_and_clips_boxes() -> None:
    empty = IndustrialFrameRecord(
        dataset_id="virat-ground-release2",
        camera_id="cam-01",
        source_video_id="video-a",
        frame_index=10,
        timestamp_seconds=1.0,
        image_width=100,
        image_height=80,
        persons=(),
    )
    occupied = IndustrialFrameRecord(
        dataset_id="virat-ground-release2",
        camera_id="cam-01",
        source_video_id="video-a",
        frame_index=11,
        timestamp_seconds=1.1,
        image_width=100,
        image_height=80,
        persons=(PersonBox(track_id="7", x=-2, y=10, width=12, height=90),),
    )

    assert empty.person_count == 0
    assert occupied.person_count == 1
    assert occupied.to_dict()["persons"][0]["bbox_xyxy"] == [0.0, 10.0, 10.0, 80.0]


def test_pilot_manifest_is_deterministic_and_keeps_camera_video_identity() -> None:
    videos = (
        PilotVideo(
            dataset_id="virat-ground-release2",
            camera_id="cam-01",
            source_video_id="video-a",
            total_frames=1_000,
            fps=10.0,
            video_sha256="a" * 64,
            annotation_sha256="b" * 64,
        ),
        PilotVideo(
            dataset_id="virat-ground-release2",
            camera_id="cam-02",
            source_video_id="video-b",
            total_frames=800,
            fps=20.0,
            video_sha256="c" * 64,
            annotation_sha256="d" * 64,
        ),
    )

    first = build_pilot_manifest(videos, frames_per_video=4, minimum_gap_seconds=5)
    second = build_pilot_manifest(tuple(reversed(videos)), frames_per_video=4, minimum_gap_seconds=5)

    assert first == second
    assert first["schema_version"] == "fieldpack.g0-pilot.v1"
    assert first["independent_camera_count"] == 2
    assert first["source_video_count"] == 2
    assert first["sample_count"] == 8
    assert len(first["manifest_sha256"]) == 64
    assert first["samples"][0]["camera_id"] == "cam-01"
    assert first["samples"][0]["source_video_id"] == "video-a"


def test_pilot_manifest_rejects_duplicate_source_video_identity() -> None:
    video = PilotVideo(
        dataset_id="virat-ground-release2",
        camera_id="cam-01",
        source_video_id="video-a",
        total_frames=1_000,
        fps=10.0,
        video_sha256="a" * 64,
        annotation_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="duplicate_source_video_id"):
        build_pilot_manifest((video, video), frames_per_video=4, minimum_gap_seconds=5)

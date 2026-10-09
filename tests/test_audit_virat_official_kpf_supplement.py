from pathlib import Path

from research_tools.data_audit.audit_virat_official_kpf_supplement import audit


ROOT = Path(__file__).resolve().parents[1]


def test_downloaded_official_kpf_supplement_is_complete_and_bounded() -> None:
    supplement = ROOT / "artifacts/data/virat-official-diva-kpf-missing-20260820-v1"
    result = audit(
        supplement,
        ROOT / "artifacts/data/virat-videos00-media-audit-development-20260820-v1/result.json",
        supplement / "compatibility/VIRAT_S_000200_03_000657_000899.viratdata.objects.txt",
    )

    assert result["official_kpf_group_found_count"] == 6
    assert result["official_kpf_group_absent_count"] == 2
    assert len(result["files"]) == 24
    assert result["frame_aligned_group_count"] == 5
    assert result["frame_mismatched_source_video_ids"] == ["VIRAT_S_000205_03_000860_000922"]
    assert result["all_downloaded_frames_in_video_range"] is False
    assert result["integration_status"] == "SEPARATE_SUPPLEMENT_DO_NOT_MIX_WITH_RELEASE2_OBJECTS"

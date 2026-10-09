from pathlib import Path

from research_tools.data_audit.analyze_current_dataset import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_current_dataset_summary_matches_audited_artifacts() -> None:
    result = analyze(
        ROOT / "artifacts/data/dataset-expansion-audit-20260820-v4/inventory.json",
        ROOT / "artifacts/data/virat-videos00-media-audit-development-20260820-v1/result.json",
        ROOT / "artifacts/data/virat-videos00-labeled-development-20260820-v1/manifest.json",
    )

    assert result["virat"]["archive_video_count"] == 226
    assert result["virat"]["videos00_labeled_video_count"] == 38
    assert result["virat"]["all_videos_corrected_labeled_count"] == 218
    assert result["virat"]["stale_inventory_overcount"] == 8
    assert [row["labeled_video_count"] for row in result["virat"]["scenes"]] == [5, 2, 31]
    assert result["readiness"]["remaining_independent_scene_gap"] == 3
    assert result["readiness"]["development_gpu_ready"] is False

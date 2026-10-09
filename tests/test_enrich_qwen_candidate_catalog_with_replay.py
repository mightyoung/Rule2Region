import json
from pathlib import Path

from research_tools.enrich_qwen_candidate_catalog_with_replay import build_camera_views


ROOT = Path("artifacts/sceneconfig/sceneconfig-top6-calibrated-20260814-v1")
BASE = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/candidate-config-catalog-v1/catalog.json")


def test_camera_decision_view_withholds_evaluation_and_marks_eligible_configs() -> None:
    decision, audit = build_camera_views(
        base_catalog_path=BASE,
        replay_result_path=ROOT / "top6-calibrated-result.json",
        fingerprint_path=ROOT / "operator-fingerprints.json",
        pre_gt_seal_path=ROOT / "PRE_GT_SEAL.json",
        camera_id="VIRAT_S_010204",
    )
    assert decision["eligible_config_ids"] == ["C00", "C02", "C03", "C04"]
    assert "evaluation_metrics" not in json.dumps(decision)
    rows = {row["config_id"]: row for row in decision["configs"]}
    assert rows["C01"]["camera_replay"]["eligible_for_ranking"] is False
    assert rows["C05"]["camera_replay"]["status"] == "NOT_IN_REGISTERED_TOP6_EXACT_REPLAY"
    assert {row["config_id"] for row in audit["rows"]} == {"C00", "C01", "C02", "C03", "C04"}


def test_zero_precision_cameras_are_not_promoted_as_safe() -> None:
    decision, _ = build_camera_views(
        base_catalog_path=BASE,
        replay_result_path=ROOT / "top6-calibrated-result.json",
        fingerprint_path=ROOT / "operator-fingerprints.json",
        pre_gt_seal_path=ROOT / "PRE_GT_SEAL.json",
        camera_id="VIRAT_S_010111",
    )
    assert decision["eligible_config_ids"] == []
    assert all(
        row["camera_replay"].get("statistical_warning") == "ZERO_PRECISION_BASE_RATE_MAKES_NONINFERIORITY_UNINFORMATIVE"
        for row in decision["configs"] if row["camera_replay"]["status"] == "EXACT_END_TO_END_REPLAY_AVAILABLE"
    )

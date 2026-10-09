import json
from pathlib import Path

from research_tools.evaluate_qwen_camera_ranking import evaluate_camera_ranking


CATALOG = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2/camera-catalog-010204-v1/decision-catalog.json")


def test_qwen_camera_ranking_is_pareto_valid_but_not_incremental(tmp_path: Path) -> None:
    response = tmp_path / "response.json"
    response.write_text(json.dumps({
        "ranked_config_ids": ["C00", "C02", "C03", "C04"],
        "abstain": False,
    }), encoding="utf-8")
    result = evaluate_camera_ranking(CATALOG, response)
    assert result["pareto_frontier_config_ids"] == ["C00", "C04"]
    assert result["ranked_dominated_config_ids"] == ["C02", "C03"]
    assert result["qwen_top1_is_pareto_valid"] is True
    assert result["qwen_top1_matches_latency_minimizer"] is True
    assert result["incremental_value_established"] is False
    assert result["holdout_opened"] is False

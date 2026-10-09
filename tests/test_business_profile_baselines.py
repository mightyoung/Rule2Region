from pathlib import Path

from research_tools.business_profile_baselines import evaluate_baselines


ROOT = Path(".omx/specs/autoresearch-qwen-cross-domain-scene-expert-v2")


def test_postblind_baselines_are_train_only_and_report_diagnostic_status() -> None:
    result = evaluate_baselines(
        development_cards_path=ROOT / "preference-profile-v1/business-cards.json",
        blind_cards_path=ROOT / "preference-profile-blind-v1/blind-cards.json",
        gold_path=ROOT / "preference-profile-blind-v1/annotator-gold.json",
        qwen_result_path=ROOT / "business-profile-blind-v1/qwen-business-profile-blind-20260814-v1/result.json",
    )
    assert result["training_card_count"] == 4
    assert result["blind_card_count"] == 16
    assert result["evidence_status"] == "DEVELOPMENT_ONLY_POST_BLIND_DIAGNOSTIC"
    assert result["holdout_opened"] is False
    assert set(result["metrics"]) == {
        "qwen",
        "keyword_rule",
        "char_ngram_logistic_regression",
        "char_ngram_nearest_development_card",
    }

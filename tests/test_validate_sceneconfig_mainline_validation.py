import json
from pathlib import Path

from research_tools.validate_sceneconfig_mainline_validation import validate


def test_current_e2_e3_form_a_validated_rejection() -> None:
    result = validate(
        Path("artifacts/sceneconfig/e2-20260813-v2/e2-result.json"),
        Path("artifacts/sceneconfig/e3-20260813-v1/e3-result.json"),
    )
    assert result["passed"] is True
    assert result["validation_outcome"] == "VALIDATED_REJECTION_OF_QWEN_INCREMENTAL_CLAIM"

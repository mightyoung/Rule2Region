from research_tools.validate_detector_conditioned_residual_query import validate


def _metrics() -> dict:
    return {
        "static_raw_head_joint_proposal_recall": .125,
        "mean_joint_proposal_recall": .15,
        "worst_camera_joint_proposal_recall": .08,
        "mean_roi_area": .10,
        "max_roi_area": .15,
        "camera_split_leakage": False,
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
    }


def test_validator_passes_only_complete_gate() -> None:
    assert validate(_metrics())["passed"] is True


def test_validator_rejects_static_gain_shortfall() -> None:
    metrics = _metrics()
    metrics["static_raw_head_joint_proposal_recall"] = .13
    result = validate(metrics)
    assert result["passed"] is False
    assert result["checks"]["relative_to_static_at_least_1_20"] is False


def test_validator_rejects_sealed_camera_use() -> None:
    metrics = _metrics()
    metrics["sealed_camera_used"] = True
    assert validate(metrics)["passed"] is False

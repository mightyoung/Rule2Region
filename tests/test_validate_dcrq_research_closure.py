from research_tools.validate_dcrq_research_closure import validate_closure


def _metrics() -> dict:
    return {
        "evidence_status": "BOUNDED_FOUR_CAMERA_ENGINEERING_DEVELOPMENT_GATE_FAILED",
        "mean_joint_proposal_recall": .03,
        "worst_camera_joint_proposal_recall": 0.0,
        "static_raw_head_joint_proposal_recall": .06,
        "mean_roi_area": .095,
        "max_roi_area": .096,
        "camera_split_leakage": False,
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
    }


def _evidence(metrics: dict) -> dict:
    return {**metrics, "evidence_status": "BOUNDED_FOUR_CAMERA_ENGINEERING_DEVELOPMENT_GATE", "folds": [{}, {}, {}, {}]}


def _provenance() -> dict:
    return {"configuration_search": False, "gt_used_online": False, "strong_model_called": False, "sealed_camera_used": False}


def test_complete_negative_result_passes_research_closure_but_rejects_algorithm() -> None:
    metrics = _metrics()
    result = validate_closure(metrics, _evidence(metrics), _provenance(), complete=True)
    assert result["passed"] is True
    assert result["research_outcome"] == "rejected"
    assert result["algorithm_promoted"] is False
    assert not all(result["performance_checks"].values())


def test_closure_rejects_missing_complete_marker() -> None:
    metrics = _metrics()
    assert validate_closure(metrics, _evidence(metrics), _provenance(), complete=False)["passed"] is False


def test_closure_rejects_scope_violation_even_for_negative_result() -> None:
    metrics = _metrics()
    metrics["sealed_camera_used"] = True
    assert validate_closure(metrics, _evidence(metrics), _provenance(), complete=True)["passed"] is False


def test_closure_rejects_mismatched_evidence() -> None:
    metrics = _metrics()
    evidence = _evidence(metrics)
    evidence["mean_joint_proposal_recall"] = .99
    assert validate_closure(metrics, evidence, _provenance(), complete=True)["passed"] is False

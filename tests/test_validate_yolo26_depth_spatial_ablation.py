from research_tools.validate_yolo26_depth_spatial_ablation import validate


def _complete() -> dict:
    return {
        "evidence_status": "BOUNDED_256_FRAME_YOLO26_DETECT_DEPTH_ABLATION",
        "comparison_complete": True,
        "development_camera_count": 4,
        "frame_count": 256,
        "configuration_search": False,
        "camera_split_leakage": False,
        "gt_used_online": False,
        "strong_model_called": False,
        "sealed_camera_used": False,
        "checkpoint_hashes_recorded": True,
        "latency_reported": True,
        "yolo26_depth_has_boxes": False,
        "numpy_input_color": "BGR_CONVERTED_FROM_PIL_RGB",
        "detect_replacement_recommended": False,
        "depth_aux_recommended": False,
    }


def test_complete_negative_ablation_closes_as_both_rejected() -> None:
    result = validate(_complete())
    assert result["passed"] is True
    assert result["research_outcome"] == "both_rejected"


def test_promotion_flags_are_preserved_without_changing_closure_meaning() -> None:
    metrics = _complete()
    metrics["detect_replacement_recommended"] = True
    result = validate(metrics)
    assert result["passed"] is True
    assert result["research_outcome"] == "detect_promoted_depth_rejected"


def test_sealed_camera_use_blocks_closure() -> None:
    metrics = _complete()
    metrics["sealed_camera_used"] = True
    assert validate(metrics)["passed"] is False


def test_depth_boxes_must_be_false_not_unknown() -> None:
    metrics = _complete()
    metrics["yolo26_depth_has_boxes"] = None
    assert validate(metrics)["passed"] is False


def test_wrong_numpy_color_contract_blocks_closure() -> None:
    metrics = _complete()
    metrics["numpy_input_color"] = "RGB_ARRAY"
    assert validate(metrics)["passed"] is False

from research_tools.sceneconfig_m0_robustness_audit import audit


def test_registered_pass_can_still_fail_operational_robustness() -> None:
    e4 = {
        "passed": True,
        "per_camera": {
            "a": {"precision_change": -.2, "relative_count_mae_improvement": .5},
            "b": {"precision_change": .1, "relative_count_mae_improvement": -.5},
        },
    }
    batch = {"batch_invariant": False, "count_disagreement_frame_rate": .4}
    result = audit(e4, batch)
    assert result["registered_e4_passed"] is True
    assert result["operational_robustness_passed"] is False
    assert result["decision"].startswith("BLOCK_CONFIRMATION")

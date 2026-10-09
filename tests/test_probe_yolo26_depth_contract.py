from pathlib import Path


def test_contract_probe_is_pinned_fail_closed_and_reports_both_heads() -> None:
    text = Path("research_tools/probe_yolo26_depth_contract.py").read_text(encoding="utf-8")
    assert '"end2end"] = end2end' in text
    assert '"one_to_one_boxes"' in text
    assert '"one_to_many_boxes"' in text
    assert '"depth_has_boxes"' in text
    assert 'len(payload["depth_shape"]) == 2' in text
    assert '"8.4.118"' in text
    assert "refuse_existing_output" in text
    assert "checkpoint_sha256" in text
    assert '"one_to_one_head_state"' in text
    assert '"one_to_many_head_state"' in text
    assert '"head_weight_mean_absolute_difference"' in text

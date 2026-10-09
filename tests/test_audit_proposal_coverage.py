import pytest

from research_tools.audit_proposal_coverage import summarize


def test_proposal_audit_summary_keeps_temporal_and_spatial_recall_separate():
    result = summarize([
        {"route": "heuristic", "temporal_capture_recall": .2, "spatial_recall_given_selected": .5, "joint_proposal_recall": .1},
        {"route": "heuristic", "temporal_capture_recall": .4, "spatial_recall_given_selected": .5, "joint_proposal_recall": .2},
    ])
    assert result["mean_temporal_capture_recall"] == pytest.approx(.3)
    assert result["mean_spatial_recall_given_selected"] == .5
    assert result["mean_joint_proposal_recall"] == pytest.approx(.15)


def test_proposal_audit_summary_rejects_mixed_routes():
    with pytest.raises(ValueError, match="rows_invalid"):
        summarize([
            {"route": "heuristic", "temporal_capture_recall": 0, "spatial_recall_given_selected": 0, "joint_proposal_recall": 0},
            {"route": "motion", "temporal_capture_recall": 0, "spatial_recall_given_selected": 0, "joint_proposal_recall": 0},
        ])

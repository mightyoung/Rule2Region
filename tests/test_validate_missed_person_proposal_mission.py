from research_tools.validate_missed_person_proposal_mission import validate


def test_validator_requires_every_mission_gate():
    result = validate({
        "joint_proposal_recall": .15,
        "mean_roi_area": .10,
        "max_roi_area": .15,
        "sequence_split_leakage": False,
        "gt_used_online": False,
    })
    assert result["passed"] is True


def test_validator_rejects_gt_online_even_when_metrics_are_high():
    result = validate({
        "joint_proposal_recall": .8,
        "mean_roi_area": .01,
        "max_roi_area": .02,
        "sequence_split_leakage": False,
        "gt_used_online": True,
    })
    assert result["passed"] is False
    assert result["checks"]["gt_not_used_online"] is False

import numpy as np

from research_tools.cascade_experiment import Box
from research_tools.missed_person_proposal import (
    audit_sequence_split,
    build_residual_targets,
    fit_pointwise_ridge,
    missed_ground_truth,
    predict_pointwise_ridge,
)


def test_missed_ground_truth_uses_frozen_iou_gate():
    truth = [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]
    predictions = [Box(0, 0, 10, 10)]
    assert missed_ground_truth(truth, predictions) == (truth[1],)


def test_residual_targets_mark_only_missed_centers_and_size():
    truth = [Box(0, 0, 10, 10), Box(40, 20, 60, 40)]
    result = build_residual_targets(
        truth, [Box(0, 0, 10, 10)], image_width=100, image_height=50,
        output_width=20, output_height=10,
    )
    assert result.heatmap.max() > 0.9
    assert result.scale_mask.sum() == 1
    iy, ix = np.argwhere(result.scale_mask)[0]
    assert np.allclose(result.scale[:, iy, ix], [0.2, 0.4])
    assert result.missed_boxes == (truth[1],)


def test_sequence_split_audit_fails_closed_on_overlap_and_empty_split():
    leaked = audit_sequence_split(["a"], ["b"], ["a"])
    assert leaked["sequence_split_leakage"] is True
    assert leaked["valid"] is False
    assert audit_sequence_split(["a"], ["b"], ["c"])["valid"] is True
    assert audit_sequence_split(["a"], [], ["c"])["valid"] is False


def test_pointwise_ridge_is_a_runnable_pipeline_sentinel():
    features = np.zeros((2, 2, 3, 3), dtype=np.float32)
    features[:, 0, 1, 1] = 1
    targets = np.zeros((2, 3, 3), dtype=np.float32)
    targets[:, 1, 1] = 1
    weights = fit_pointwise_ridge(features, targets)
    predicted = predict_pointwise_ridge(features, weights)
    assert predicted[:, 1, 1].mean() > predicted[:, 0, 0].mean()

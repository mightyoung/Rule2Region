import numpy as np
import pytest

from research_tools.yolo_dense_person_score import split_dense_scores


def test_split_dense_scores_uses_feature_geometry() -> None:
    scores = np.arange(20, dtype=np.float32).reshape(2, 10)
    scales = split_dense_scores(scores, ((2, 4), (1, 2)))
    assert scales[0].shape == (2, 2, 4)
    assert scales[1].shape == (2, 1, 2)
    assert scales[1][1, 0, 1] == 19


def test_split_dense_scores_rejects_geometry_mismatch() -> None:
    with pytest.raises(ValueError, match="geometry"):
        split_dense_scores(np.zeros((1, 4), dtype=np.float32), ((3, 2),))

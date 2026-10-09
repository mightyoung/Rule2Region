import numpy as np

from research_tools.diagnose_dcrq_factorization import factorized_scores


def test_factorized_scores_preserve_frame_utility_and_static_argmax() -> None:
    frame = np.asarray([[[0.1, 0.9], [0.2, 0.3]], [[0.8, 0.1], [0.0, 0.2]]], dtype=np.float32)
    spatial = np.asarray([[[4.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [5.0, 0.0]]], dtype=np.float32)
    result = factorized_scores(frame, spatial)
    np.testing.assert_allclose(result.max(axis=(1, 2)), frame.max(axis=(1, 2)))
    np.testing.assert_array_equal(result.reshape(2, -1).argmax(axis=1), spatial.reshape(2, -1).argmax(axis=1))


def test_factorized_scores_reject_shape_mismatch() -> None:
    try:
        factorized_scores(np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))
    except ValueError as exc:
        assert str(exc) == "factorized_score_shape_mismatch"
    else:
        raise AssertionError("shape mismatch was accepted")

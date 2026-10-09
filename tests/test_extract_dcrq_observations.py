import numpy as np
import pytest

from PIL import Image

from research_tools.extract_dcrq_observations import (
    load_ultralytics_bgr,
    normalized_categorical_entropy,
    robust_quantile_normalize,
)


def test_robust_quantile_normalization_is_per_map_and_bounded() -> None:
    values = np.stack((np.arange(20).reshape(4, 5), np.arange(20, 40).reshape(4, 5))).astype(np.float32)
    normalized = robust_quantile_normalize(values)
    assert normalized.shape == values.shape
    assert normalized.min() >= -3
    assert normalized.max() <= 3
    assert np.allclose(normalized[0], normalized[1])


def test_loader_converts_pil_rgb_to_ultralytics_bgr(tmp_path) -> None:
    path = tmp_path / "pixel.png"
    Image.fromarray(np.asarray([[[1, 2, 3]]], dtype=np.uint8), mode="RGB").save(path)
    bgr = load_ultralytics_bgr(path)
    np.testing.assert_array_equal(bgr, np.asarray([[[3, 2, 1]]], dtype=np.uint8))
    assert bgr.flags.c_contiguous


def test_normalized_entropy_has_known_extremes() -> None:
    uniform = np.full((2, 4), .25, dtype=np.float32)
    certain = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    assert np.allclose(normalized_categorical_entropy(uniform, axis=1), 1.0)
    assert np.all(normalized_categorical_entropy(certain, axis=1) < 1e-5)


def test_normalized_entropy_rejects_single_bin() -> None:
    with pytest.raises(ValueError, match="two_bins"):
        normalized_categorical_entropy(np.ones((2, 1), dtype=np.float32), axis=1)

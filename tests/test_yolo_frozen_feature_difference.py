import pytest

from research_tools.yolo_frozen_feature_difference import _actual_batch, _feature_tensor


class _TensorLike:
    ndim = 4


def test_feature_tensor_accepts_direct_or_nested_4d_output():
    value = _TensorLike()
    assert _feature_tensor(value) is value
    assert _feature_tensor(("ignored", value)) is value


def test_feature_tensor_rejects_non_tensor_outputs():
    with pytest.raises(ValueError, match="no_4d_tensor"):
        _feature_tensor(("ignored",))


def test_actual_batch_discards_different_shape_warmup():
    import torch
    warmup = torch.zeros((1, 2, 80, 80))
    actual = torch.zeros((4, 2, 48, 80))
    result = _actual_batch([warmup, actual], 4)
    assert result.shape == (4, 2, 48, 80)

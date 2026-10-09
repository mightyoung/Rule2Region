import numpy as np

from research_tools.run_dcrq_development_gate import causal_channels, fixed_depthwise_head_features


def test_causal_channels_do_not_use_future_frames() -> None:
    base = np.arange(5 * 2 * 3 * 4, dtype=np.float32).reshape(5, 2, 3, 4)
    changed = base.copy()
    changed[3:] += 10000
    first = causal_channels(base)
    second = causal_channels(changed)
    np.testing.assert_allclose(first[:3], second[:3])


def test_fixed_depthwise_head_preserves_geometry_and_expands_channels() -> None:
    causal = np.ones((4, 21, 5, 7), dtype=np.float32)
    result = fixed_depthwise_head_features(causal)
    assert result.shape == (4, 63, 5, 7)
    np.testing.assert_allclose(result[:, :21], 1.0)
    np.testing.assert_allclose(result[:, 21:42], 1.0)
    np.testing.assert_allclose(result[:, 42:], 0.0, atol=1e-6)

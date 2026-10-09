from __future__ import annotations

from research_tools.sceneconfig_e0_smoke import _owned
from research_tools.cascade_experiment import Box


def test_primary_ownership_uses_box_center() -> None:
    core = Box(0, 0, 100, 100)
    assert _owned(Box(89, 20, 109, 40), core)
    assert not _owned(Box(100, 20, 120, 40), core)


def test_primary_ownership_has_half_open_right_and_bottom_edges() -> None:
    core = Box(0, 0, 100, 100)
    assert _owned(Box(98, 98, 100, 100), core)
    assert not _owned(Box(99, 99, 101, 101), core)

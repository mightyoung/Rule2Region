from research_tools.cascade_experiment import Box, Prediction
from research_tools.sceneconfig_e2_proxy_validation import ConfigCard, _ranks, frozen_panel, spearman


def test_frozen_panel_has_24_distinct_configs_and_baseline() -> None:
    panel = frozen_panel()
    assert len(panel) == 24
    assert len({(card.tiles, card.tile_confidence) for card in panel}) == 24
    assert [card.config_id for card in panel] == [f"C{index:02d}" for index in range(24)]
    assert panel[0] == ConfigCard("C00", (), .25)
    assert all(set(card.tiles) <= {"t0", "t1", "t2", "t3"} for card in panel)


def test_average_ranks_and_spearman_handle_ties() -> None:
    assert _ranks([1, 1, 3]).tolist() == [.5, .5, 2.0]
    assert spearman([1, 2, 3], [10, 20, 30]) == 1.0
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0

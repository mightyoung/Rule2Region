import json

from research_tools.build_supervision_demo import condition_table, write_index
from research_tools.supervision_dashboard import asset_choices, load_demo_index, select_asset


def test_condition_table_is_stable_and_includes_budget_metrics():
    receipt = {"conditions": {"b": {"count_mae": 2, "budget_fraction": .1}, "a": {"count_mae": 1}}}
    rows = condition_table(receipt)
    assert [row[0] for row in rows] == ["a", "b"]
    assert rows[1][1] == 2.0 and rows[1][5] == .1


def test_read_only_demo_index_round_trip(tmp_path):
    video = tmp_path / "demo.mp4"; video.write_bytes(b"video")
    asset = {"sequence_id": "seq", "condition": "oracle_budget_10", "video": str(video), "video_sha256": "a" * 64, "frames": 20, "summary": {}, "condition_table": [["x", 1]]}
    index_path = tmp_path / "index.json"
    write_index(index_path, (asset,), pilot_manifest_sha256="b" * 64)
    index = load_demo_index(index_path)
    assert asset_choices(index) == ["seq | oracle_budget_10"]
    selected = select_asset(index, asset_choices(index)[0])
    assert selected[0] == str(video) and selected[1] == [["x", 1]]

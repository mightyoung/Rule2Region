from datetime import datetime, timezone

from research_tools.parallel_agent_review import aggregate_receipts, parse_receipt_name, receipt_name


def test_receipt_name_is_agent_frame_timestamp() -> None:
    when = datetime(2026, 8, 17, 18, 45, 12, tzinfo=timezone.utc)
    name = receipt_name("A01", "f249", when)
    assert name == "A01_f249_20260817T184512Z.json"
    parsed = parse_receipt_name(name)
    assert parsed == {"agent_id": "A01", "frame_id": "f249", "timestamp": "20260817T184512Z"}


def test_aggregate_rejects_mismatched_filename_identity(tmp_path) -> None:
    path = tmp_path / "A01_f249_20260817T184512Z.json"
    path.write_text('{"agent_id":"A02","frame_id":"f249","box_id":"f249b01","preconfig":{"cell":"t2","source":"x"},"camera_id":"VIRAT_S_010004","receipt_name":"x"}', encoding="utf-8")
    try:
        aggregate_receipts([path], tmp_path / "agg.json")
    except ValueError as exc:
        assert "receipt_identity_mismatch" in str(exc)
    else:
        raise AssertionError("expected identity mismatch")

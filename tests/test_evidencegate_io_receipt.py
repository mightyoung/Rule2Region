import pytest

from evidencegate import IOReceipt


def output_receipt(**overrides):
    values = {
        "stage": "output",
        "source_path": "source",
        "source_sha256": "a" * 64,
        "storage_class": "nfs",
        "cache_state": "staged",
        "bytes_read": 10,
        "wall_clock_ms": 2.0,
        "included_in_latency": False,
        "target_path": "result.json",
        "target_sha256": "b" * 64,
        "bytes_written": 8,
        "started_at": "2026-08-10T00:00:00Z",
        "ended_at": "2026-08-10T00:00:01Z",
        "atomic_seal_state": "sealed",
        "attempt_artifact_sha256": "c" * 64,
    }
    values.update(overrides)
    return IOReceipt(**values)


def test_output_receipt_requires_hash_timestamps_and_atomic_terminal_state():
    output_receipt().validate_output_receipt()
    with pytest.raises(ValueError, match="timestamps"):
        output_receipt(started_at=None, ended_at=None).validate_output_receipt()
    with pytest.raises(ValueError, match="atomic_terminal"):
        output_receipt(atomic_seal_state="partial").validate_output_receipt()
    with pytest.raises(ValueError, match="cannot_have_error"):
        output_receipt(error="write failed").validate_output_receipt()

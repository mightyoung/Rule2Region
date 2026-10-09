from pathlib import Path


def test_confirmatory_h1_protocol_freezes_gate_claim_and_pass_set() -> None:
    text = Path("docs/plans/2026-08-19-h1-rehearsal-gate-confirmatory.md").read_text(encoding="utf-8")
    assert "PROTOCOL_ONLY" in text
    assert "decide_hard_path" in text
    assert "mae_gate" in text
    assert "VIRAT_S_010204_09_001285_001336" in text
    assert "VIRAT_S_010112" in text
    assert "VIRAT_S_010113" in text
    assert "primary_both_pass" in text
    assert "official_h1" in text
    assert "VAL_COUNT=16" in text
    assert "64" in text
    assert "32" in text
    assert "--close-hard-views" in text
    mainline = Path("docs/research/2026-08-19-camera-execution-stack-mainline.md").read_text(encoding="utf-8")
    assert "排练门" in mainline
    assert "v0 裁块" in mainline
    assert "010204" in mainline

from pathlib import Path


def test_v6_prereg_does_not_reopen_v5_or_change_h1() -> None:
    text = Path("docs/plans/2026-08-20-future-qualification-v6-preregistration.md").read_text(encoding="utf-8")
    assert "PRE-REGISTERED" in text
    assert "NOT EXECUTED" in text
    assert "−0.05" in text or "-0.05" in text
    assert "VIRAT_S_010113" in text
    assert "错误放行" in text
    assert "始终关闭" in text
    assert "只看 MAE" in text
    assert "decide_hard_path" in text
    assert "Learn then Test" in text
    assert "不得写有限样本保证" in text
    assert "v5 确认包可作" in text
    confirmatory = Path("docs/plans/2026-08-19-h1-rehearsal-gate-confirmatory.md").read_text(encoding="utf-8")
    assert "official_h1=false" in confirmatory
    assert "010004_01" in confirmatory
    assert "010204_09" in confirmatory


def test_claims_ledger_forbids_promoting_failed_h1() -> None:
    text = Path("docs/handoff/paper-eval-dossier-20260819/16-results-and-claims-ledger.md").read_text(encoding="utf-8")
    assert "official_h1=false" in text
    assert "不得" in text
    assert "010204_09" in text or "ΔMAE −0.089" in text

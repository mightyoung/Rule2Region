from pathlib import Path

from research_tools.validate_virat_kpf_supplement import validate


ROOT = Path(__file__).resolve().parents[1]


def test_downloaded_kpf_supplement_validates() -> None:
    result = validate(ROOT / "artifacts/data/virat-official-diva-kpf-missing-20260820-v1/package")

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["failures"] == []

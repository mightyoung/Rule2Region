from pathlib import Path


def test_bootstrap_is_isolated_pinned_and_auditable() -> None:
    text = Path("research_tools/remote/bootstrap_yolo26_depth_runtime.sh").read_text(encoding="utf-8")
    assert "--system-site-packages" in text
    assert "ULTRALYTICS_VERSION=8.4.118" in text
    assert '"ultralytics==$ULTRALYTICS_VERSION"' in text
    assert "RELEASE_TAG=v8.4.0" in text
    assert "DETECT_SEED=" in text
    assert "DEPTH_SEED=" in text
    assert "MISSING_VERIFIED_SEED" in text
    assert "seed-provenance.txt" in text
    assert "yolo26s.pt yolo26s-depth.pt" in text
    assert 'row["digest"] == f"sha256:{digest}"' in text
    assert "REFUSE_EXISTING_RUNTIME" in text
    assert "FAILED_INCOMPLETE_YOLO26_DEPTH_RUNTIME" in text
    assert "SHA256SUMS" in text

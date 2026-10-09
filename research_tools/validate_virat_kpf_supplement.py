from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe_path:{relative}")
            continue
        target = root / relative
        if not target.is_file():
            failures.append(f"missing:{relative.as_posix()}")
        elif sha256_file(target) != row["sha256"]:
            failures.append(f"hash_mismatch:{relative.as_posix()}")
    legacy_files = list(root.rglob("*.viratdata.objects.txt"))
    checks = {
        "schema": manifest.get("schema_version") == "fieldpack.virat-official-diva-kpf-supplement.v1",
        "official_commit_pinned": manifest.get("commit") == "f1eff0d7db0dc0b25e1c99951ac0907521f3a216",
        "six_groups_found": manifest.get("official_kpf_group_found_count") == 6,
        "two_groups_absent": manifest.get("official_kpf_group_absent_count") == 2,
        "five_groups_frame_aligned": manifest.get("frame_aligned_group_count") == 5,
        "twenty_four_files": len(manifest.get("files", [])) == 24,
        "all_file_hashes": not failures,
        "no_legacy_objects_injected": not legacy_files,
        "separate_supplement": manifest.get("integration_status")
        == "SEPARATE_SUPPLEMENT_DO_NOT_MIX_WITH_RELEASE2_OBJECTS",
        "gpu_not_allowed": manifest.get("gpu_experiment_allowed") is False,
    }
    return {
        "schema_version": "fieldpack.virat-official-diva-kpf-supplement-validation.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

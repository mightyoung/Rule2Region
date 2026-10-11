"""Hash public learning deliverables, excluding this manifest and Python caches."""
import argparse
import hashlib
import json
from pathlib import Path
from .contract import require

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "SOURCE_MANIFEST.json"


def entries():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path == DEST:
            continue
        require(path.suffix in (".py", ".md", ".json"), "non-public deliverable type: " + path.name)
        data = path.read_bytes()
        files[path.relative_to(ROOT).as_posix()] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="explicitly update manifest after authorized edits")
    args = parser.parse_args()
    actual = {"version": "learning-source-v1", "files": entries()}
    if args.write:
        DEST.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    else:
        require(json.loads(DEST.read_text()) == actual, "learning source manifest mismatch")
    print(json.dumps({"verified": not args.write, "files": len(actual["files"]), "manifest_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

"""Strictly finalize raw llama-mtmd scene routes into a sealed profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_tools.qwen_scene_route_teacher import seal_profile
from research_tools.run_dcrq_development_gate import EXPECTED_CAMERAS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse_existing_output:{args.output}")
    responses = {camera: (args.responses_dir / f"{camera}.raw.json").read_text(encoding="utf-8") for camera in EXPECTED_CAMERAS}
    contact = json.loads(args.contact_manifest.read_text(encoding="utf-8"))
    receipts = {camera: contact["cameras"][camera] for camera in EXPECTED_CAMERAS}
    offline_ms = float((args.responses_dir / "offline-ms.txt").read_text(encoding="utf-8").strip())
    payload = seal_profile(responses, receipts, model=args.model, mmproj=args.mmproj, offline_ms=offline_ms)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "sealed", "camera_count": len(payload["cameras"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

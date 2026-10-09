"""Per-camera agent context: history boxes minus observer, plus run-status files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import GroundTruth, Prediction
from research_tools.execution_policy.card import FRAME_HEIGHT, FRAME_WIDTH, parse_policy_card
from research_tools.execution_policy.rules import COUNTING_RULE_CARD, DOOR_RULE_CARD
from research_tools.execution_policy.score import compare_camera
from research_tools.execution_policy.skills.registry import install_skills


def _box_row(pred: Prediction) -> dict[str, Any]:
    return {
        "xyxy": [round(pred.box.x1, 1), round(pred.box.y1, 1), round(pred.box.x2, 1), round(pred.box.y2, 1)],
        "confidence": pred.confidence,
        "h": round(pred.box.y2 - pred.box.y1, 1),
    }


def _boxes_payload(frame: int, source: str, preds: Sequence[Prediction], *, sequence: str) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "frame": frame,
        "source": source,
        "boxes": [_box_row(pred) for pred in preds],
    }


def _box_name(sequence: str, frame: int, stream: str) -> str:
    return f"{sequence}__{frame:06d}.{stream}.json"


MATCH_IOU = 0.30
FOCUS_PAD = 32.0
MAX_CAMERA_REGIONS = 4


def _matched(box, others: Sequence, *, iou: float = MATCH_IOU) -> bool:
    return any(box.iou(other.box) >= iou for other in others)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [
        max(0.0, min(FRAME_WIDTH, x1)),
        max(0.0, min(FRAME_HEIGHT, y1)),
        max(0.0, min(FRAME_WIDTH, x2)),
        max(0.0, min(FRAME_HEIGHT, y2)),
    ]


def _intersect_area(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _union_xyxy(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])]


def _camera_regions(focus: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for item in focus:
        x1, y1, x2, y2 = (float(value) for value in item["xyxy"])
        padded = _clip_xyxy(x1 - FOCUS_PAD, y1 - FOCUS_PAD, x2 + FOCUS_PAD, y2 + FOCUS_PAD)
        frame_key = f"{item.get('sequence', '')}:{int(item['frame'])}"
        hit = next((row for row in clusters if _intersect_area(padded, row["xyxy"]) > 0), None)
        if hit is None:
            clusters.append({"xyxy": padded, "support": 1, "frames": {frame_key}})
            continue
        hit["xyxy"] = _union_xyxy(hit["xyxy"], padded)
        hit["support"] += 1
        hit["frames"].add(frame_key)
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, Any]] = []
        for cluster in clusters:
            hit = next((row for row in merged if _intersect_area(cluster["xyxy"], row["xyxy"]) > 0), None)
            if hit is None:
                merged.append(cluster)
                continue
            hit["xyxy"] = _union_xyxy(hit["xyxy"], cluster["xyxy"])
            hit["support"] += cluster["support"]
            hit["frames"].update(cluster["frames"])
            changed = True
        clusters = merged
    clusters.sort(key=lambda row: (-int(row["support"]), -((row["xyxy"][2] - row["xyxy"][0]) * (row["xyxy"][3] - row["xyxy"][1]))))
    regions: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters[:MAX_CAMERA_REGIONS], start=1):
        regions.append(
            {
                "region_id": f"cam26_{index}",
                "model": "yolo26",
                "conf": 0.25,
                "role": "count",
                "xyxy": [round(float(value), 1) for value in cluster["xyxy"]],
                "support_boxes": int(cluster["support"]),
                "support_frames": len(cluster["frames"]),
            }
        )
    return regions


def _history_focus(
    sources: Sequence[Mapping[str, Any]],
    *,
    observer_sequence: str,
    observer: int,
) -> dict[str, Any]:
    focus: list[dict[str, Any]] = []
    for source in sources:
        sequence = str(source["sequence"])
        blocked = set(source["validation"])
        if sequence == observer_sequence:
            blocked.add(observer)
        gt_by_frame: dict[int, list[GroundTruth]] = {}
        for row in source["truth"]:
            if row.frame in blocked:
                continue
            gt_by_frame.setdefault(row.frame, []).append(row)
        ledgers = source["ledgers"]
        for frame in source["history"]:
            if frame in blocked or frame not in ledgers:
                continue
            ledger = ledgers[frame]
            gts = gt_by_frame.get(frame, [])
            for pred in ledger.yolo26:
                if _matched(pred.box, ledger.yolo11s):
                    continue
                if not gts or not _matched(pred.box, gts):
                    continue
                focus.append(
                    {
                        "sequence": sequence,
                        "frame": frame,
                        **_box_row(pred),
                        "source": "yolo26_not_11s_and_gt",
                    }
                )
    proposed = _camera_regions(focus)
    return {
        "scope": "camera",
        "camera_id": sources[0]["camera_id"] if sources else "",
        "rule": "yolo26_has_box_and_yolo11s_does_not_and_human_gt_matches",
        "iou": MATCH_IOU,
        "regions_must_not_overlap": True,
        "overlap_policy": "merge",
        "observer_sequence": observer_sequence,
        "observer_frame_withheld": observer,
        "validation_frames_withheld": True,
        "streams_mixed": False,
        "evidence_box_count": len(focus),
        "evidence_sequences": sorted({item["sequence"] for item in focus}),
        "evidence_frames": sorted({f"{item['sequence']}:{item['frame']}" for item in focus}),
        "proposed_regions": proposed,
    }


def _history_digest(
    sources: Sequence[Mapping[str, Any]],
    *,
    observer_sequence: str,
    observer: int,
    focus_count: int,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for source in sources:
        sequence = str(source["sequence"])
        ledgers = source["ledgers"]
        for frame in source["history"]:
            if sequence == observer_sequence and frame == observer:
                continue
            if frame not in ledgers:
                continue
            ledger = ledgers[frame]
            frames.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "n11": len(ledger.yolo11s),
                    "n26": len(ledger.yolo26),
                }
            )
    return {
        "observer_sequence": observer_sequence,
        "observer_frame_withheld": observer,
        "streams_mixed": False,
        "focus_file": "history_boxes/_focus.json",
        "focus_count": focus_count,
        "frames": frames,
    }


def write_agent_context(
    root: Path,
    *,
    camera_id: str,
    observer_sequence: str,
    observer: int,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not sources:
        raise ValueError("camera_sources_empty")
    sequences = [str(source["sequence"]) for source in sources]
    if observer_sequence not in set(sequences):
        raise ValueError("observer_sequence_missing")
    camera_ids = sorted({str(source.get("camera_id") or camera_id) for source in sources})
    observer_source = next(source for source in sources if str(source["sequence"]) == observer_sequence)
    if observer in set(observer_source["validation"]):
        raise ValueError("validation_frame_selected_for_context")
    root.mkdir(parents=True)
    rules = root / "rules"
    boxes = root / "history_boxes"
    status = root / "run_status"
    rules.mkdir()
    boxes.mkdir()
    status.mkdir()
    (rules / "counting_rule_card.txt").write_text(COUNTING_RULE_CARD, encoding="utf-8")
    (rules / "door_rule_card_appendix.txt").write_text(
        "APPENDIX_NOT_H1. Do not use this card to fill the H1 schedule.\n" + DOOR_RULE_CARD,
        encoding="utf-8",
    )
    (rules / "menu.txt").write_text(
        "H1 menu: model in {yolo11s, yolo26}; conf in {0.15, 0.25, 0.35}; "
        "1 to 4 non-overlapping axis-aligned count regions; "
        "if two regions overlap, merge them into one union box; "
        "one card per physical camera view, shared by every clip of that view; "
        "VIRAT_S_010111 and VIRAT_S_010112 are the same parking-lot camera; "
        f"xyxy pixels in [0, {int(FRAME_WIDTH)}] x [0, {int(FRAME_HEIGHT)}].\n",
        encoding="utf-8",
    )
    skill_rows = install_skills(root / "skills")
    written: list[str] = []
    for source in sources:
        sequence = str(source["sequence"])
        ledgers = source["ledgers"]
        for frame in source["history"]:
            if sequence == observer_sequence and frame == observer:
                continue
            ledger = ledgers[frame]
            name11 = _box_name(sequence, frame, "yolo11s")
            name26 = _box_name(sequence, frame, "yolo26")
            (boxes / name11).write_text(
                json.dumps(_boxes_payload(frame, "yolo11s", ledger.yolo11s, sequence=sequence), indent=2) + "\n",
                encoding="utf-8",
            )
            (boxes / name26).write_text(
                json.dumps(_boxes_payload(frame, "yolo26", ledger.yolo26, sequence=sequence), indent=2) + "\n",
                encoding="utf-8",
            )
            written.extend([name11, name26])
    focus = _history_focus(sources, observer_sequence=observer_sequence, observer=observer)
    focus["camera_id"] = camera_id
    focus["camera_ids"] = camera_ids
    digest = _history_digest(
        sources,
        observer_sequence=observer_sequence,
        observer=observer,
        focus_count=int(focus["evidence_box_count"]),
    )
    (boxes / "_focus.json").write_text(json.dumps(focus, separators=(",", ":")) + "\n", encoding="utf-8")
    (boxes / "_digest.json").write_text(json.dumps(digest, separators=(",", ":")) + "\n", encoding="utf-8")
    written.insert(0, "_focus.json")
    written.insert(0, "_digest.json")
    observer_key = f"{observer_sequence}:{observer}"
    if any(item["sequence"] == observer_sequence and item["frame"] == observer for item in digest["frames"]):
        raise ValueError("observer_boxes_leaked")
    if observer_key in set(focus.get("evidence_frames") or ()):
        raise ValueError("observer_boxes_leaked")
    if (boxes / _box_name(observer_sequence, observer, "yolo11s")).exists() or (
        boxes / _box_name(observer_sequence, observer, "yolo26")
    ).exists():
        raise ValueError("observer_boxes_leaked")
    for source in sources:
        sequence = str(source["sequence"])
        for frame in source["validation"]:
            if (boxes / _box_name(sequence, frame, "yolo11s")).exists():
                raise ValueError("validation_boxes_leaked")
    baseline = parse_policy_card(
        {
            "regions": [
                {
                    "region_id": "full",
                    "xyxy": [0, 0, int(FRAME_WIDTH), int(FRAME_HEIGHT)],
                    "model": "yolo11s",
                    "conf": 0.25,
                }
            ]
        }
    )
    scored = compare_camera(baseline, sources, compiled_time_ms=0.4, b26_time_ms=1.0)
    b11 = {**scored["b11"], "scope": "camera", "sequences": scored["sequences"]}
    b26 = {**scored["b26"], "scope": "camera", "sequences": scored["sequences"]}
    index = {
        "scope": "camera",
        "view_id": camera_id,
        "camera_id": camera_id,
        "camera_ids": camera_ids,
        "sequences": sequences,
        "observer_sequence": observer_sequence,
        "observer_frame": observer,
        "observer_boxes_withheld": True,
        "validation_frames_withheld": True,
        "one_card_per_camera": True,
        "regions_must_not_overlap": True,
        "overlap_policy": "merge",
        "history_frames": {str(source["sequence"]): list(source["history"]) for source in sources},
        "history_box_files": written,
        "history_digest": "history_boxes/_digest.json",
        "history_focus": "history_boxes/_focus.json",
        "streams_mixed": False,
    }
    (status / "history_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (status / "b11_history.json").write_text(json.dumps(b11, indent=2) + "\n", encoding="utf-8")
    (status / "b26_history.json").write_text(json.dumps(b26, indent=2) + "\n", encoding="utf-8")
    listing = [
        "rules/counting_rule_card.txt",
        "rules/door_rule_card_appendix.txt",
        "rules/menu.txt",
        "skills/INDEX.json",
        *[row["path"] for row in skill_rows],
        "run_status/history_index.json",
        "run_status/b11_history.json",
        "run_status/b26_history.json",
        *[f"history_boxes/{name}" for name in written],
    ]
    (root / "INDEX.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    return {
        "root": str(root),
        "camera_id": camera_id,
        "camera_ids": camera_ids,
        "sequences": sequences,
        "observer_boxes_withheld": True,
        "validation_frames_withheld": True,
        "history_box_files": written,
        "skills": [row["name"] for row in skill_rows],
        "file_count": 1 + len(listing),
    }

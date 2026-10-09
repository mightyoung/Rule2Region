"""Parse and reject illegal execution-policy cards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research_tools.cascade_experiment import Box


MODELS = frozenset({"yolo11s", "yolo26"})
CONFS = frozenset({0.15, 0.25, 0.35})
ROLES = frozenset({"count", "exclude"})
MAX_REGIONS = 4
FRAME_WIDTH = 1280.0
FRAME_HEIGHT = 720.0


class CardInvalid(ValueError):
    pass


@dataclass(frozen=True)
class RegionCard:
    region_id: str
    box: Box
    model: str
    conf: float
    role: str = "count"

    @property
    def area(self) -> float:
        return self.box.area


@dataclass(frozen=True)
class PolicyCard:
    regions: tuple[RegionCard, ...]

    @property
    def yolo26_area(self) -> float:
        return sum(row.box.area for row in self.regions if row.role == "count" and row.model == "yolo26")


def _box(raw: Mapping[str, Any]) -> Box:
    coords = raw.get("xyxy") or raw.get("box")
    if isinstance(coords, (list, tuple)) and len(coords) == 4:
        x1, y1, x2, y2 = (float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))
    else:
        x1, y1, x2, y2 = float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"])
    if x1 < 0 or y1 < 0 or x2 > FRAME_WIDTH or y2 > FRAME_HEIGHT:
        raise CardInvalid("region_outside_frame")
    try:
        return Box(x1, y1, x2, y2)
    except ValueError as exc:
        raise CardInvalid("region_box_unordered") from exc


def _intersection_area(left: Box, right: Box) -> float:
    ix1, iy1 = max(left.x1, right.x1), max(left.y1, right.y1)
    ix2, iy2 = min(left.x2, right.x2), min(left.y2, right.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _xyxy_intersect(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    ix2, iy2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _xyxy_union(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    ]


def _row_xyxy(row: Mapping[str, Any]) -> list[float] | None:
    coords = row.get("xyxy") or row.get("box")
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return None
    return [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]


def _merge_region_row(hit: dict[str, Any], item: Mapping[str, Any]) -> None:
    left = _row_xyxy(hit)
    right = _row_xyxy(item)
    if left is None or right is None:
        return
    hit["xyxy"] = _xyxy_union(left, right)
    models = {str(hit.get("model") or ""), str(item.get("model") or "")}
    if "yolo26" in models:
        if str(hit.get("model") or "") != "yolo26":
            if item.get("conf") is not None:
                hit["conf"] = item.get("conf")
        hit["model"] = "yolo26"
    roles = {str(hit.get("role") or "count"), str(item.get("role") or "count")}
    if "count" in roles:
        hit["role"] = "count"


def merge_overlapping_regions(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("regions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return dict(payload)
    items = [dict(row) for row in raw if isinstance(row, Mapping)]
    changed = True
    while changed:
        changed = False
        merged: list[dict[str, Any]] = []
        for item in items:
            item_box = _row_xyxy(item)
            hit = next(
                (
                    other
                    for other in merged
                    if item_box is not None
                    and _row_xyxy(other) is not None
                    and _xyxy_intersect(item_box, _row_xyxy(other) or ()) > 0
                ),
                None,
            )
            if hit is None:
                merged.append(item)
                continue
            _merge_region_row(hit, item)
            changed = True
        items = merged
    if len(items) > MAX_REGIONS:
        items = sorted(
            items,
            key=lambda row: -max(0.0, float((row.get("xyxy") or [0, 0, 0, 0])[2]) - float((row.get("xyxy") or [0, 0, 0, 0])[0]))
            * max(0.0, float((row.get("xyxy") or [0, 0, 0, 0])[3]) - float((row.get("xyxy") or [0, 0, 0, 0])[1])),
        )[:MAX_REGIONS]
    out = dict(payload)
    out["regions"] = items
    return out


def parse_policy_card(payload: Mapping[str, Any]) -> PolicyCard:
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
        raise CardInvalid("regions_must_be_list")
    if len(raw_regions) > MAX_REGIONS:
        raise CardInvalid("too_many_regions")
    regions: list[RegionCard] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, Mapping):
            raise CardInvalid("region_must_be_object")
        region_id = str(raw.get("region_id") or f"r{index}")
        if region_id in seen:
            raise CardInvalid("duplicate_region_id")
        seen.add(region_id)
        model = str(raw.get("model") or "")
        if model not in MODELS:
            raise CardInvalid("illegal_model")
        conf = float(raw.get("conf") if raw.get("conf") is not None else 0.25)
        if conf not in CONFS:
            raise CardInvalid("illegal_conf")
        role = str(raw.get("role") or "count")
        if role not in ROLES:
            raise CardInvalid("illegal_role")
        regions.append(RegionCard(region_id, _box(raw), model, conf, role))
    for i, left in enumerate(regions):
        for right in regions[i + 1 :]:
            if _intersection_area(left.box, right.box) > 0:
                raise CardInvalid("regions_overlap")
    if not any(row.role == "count" for row in regions):
        raise CardInvalid("need_at_least_one_count_region")
    return PolicyCard(tuple(regions))


def card_to_dict(card: PolicyCard) -> dict[str, Any]:
    return {
        "regions": [
            {
                "region_id": row.region_id,
                "xyxy": list(row.box.coords),
                "model": row.model,
                "conf": row.conf,
                "role": row.role,
            }
            for row in card.regions
        ]
    }


def apply_patch(card: PolicyCard, patch: Mapping[str, Any]) -> PolicyCard:
    action = str(patch.get("action") or "")
    if action == "stop":
        return card
    payload = card_to_dict(card)
    region_id = str(patch.get("region_id") or "")
    if action == "drop_region":
        payload["regions"] = [row for row in payload["regions"] if row["region_id"] != region_id]
        return parse_policy_card(payload)
    found = False
    for row in payload["regions"]:
        if row["region_id"] != region_id:
            continue
        found = True
        if action == "set_conf":
            row["conf"] = float(patch["conf"])
        elif action == "set_model":
            row["model"] = str(patch["model"])
        else:
            raise CardInvalid("illegal_patch_action")
        break
    if not found:
        raise CardInvalid("patch_region_missing")
    return parse_policy_card(payload)



def latency_proxy(card: PolicyCard, *, frame_area: float = FRAME_WIDTH * FRAME_HEIGHT) -> float:
    if frame_area <= 0:
        raise CardInvalid("frame_area_must_be_positive")
    cost = 0.0
    for region in card.regions:
        if region.role != "count":
            continue
        weight = 1.0 if region.model == "yolo26" else 0.25
        cost += weight * region.box.area / frame_area
    return cost

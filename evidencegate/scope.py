from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ScopeVerdict(str, Enum):
    BROAD = "BROAD"
    IMAGE_ONLY = "IMAGE_ONLY"
    VIDEO_ONLY = "VIDEO_ONLY"
    STOP = "STOP"


def decide_scope(*, video_admitted: bool, image_admitted: bool, video_preregistered: bool = False) -> ScopeVerdict:
    """Select exactly one scope; a missing lane never silently broadens scope."""
    if video_admitted and image_admitted:
        return ScopeVerdict.BROAD
    if image_admitted:
        return ScopeVerdict.IMAGE_ONLY
    if video_admitted and video_preregistered:
        return ScopeVerdict.VIDEO_ONLY
    return ScopeVerdict.STOP


def scope_verdict(video: bool | Mapping[str, Any], image: bool | None = None, *, video_preregistered: bool = False) -> str:
    """Compatibility wrapper returning the stable string verdict."""
    if isinstance(video, Mapping):
        image = bool(video.get("image_admitted", video.get("image", False)))
        video = bool(video.get("video_admitted", video.get("video", False)))
        video_preregistered = bool(video.get("video_preregistered", video_preregistered))
    return decide_scope(video_admitted=bool(video), image_admitted=bool(image), video_preregistered=video_preregistered).value

"""Derive plate-crossing targets from bracket clicks and target quality metadata."""

from __future__ import annotations

import math
from typing import Any, Optional

TARGET_SOURCE_POST_PITCH_MARKER = "post_pitch_marker"
TARGET_SOURCE_BRACKET_INTERPOLATION = "bracket_interpolation"
TARGET_SOURCE_BRACKET_MIDPOINT = "bracket_midpoint"
TARGET_SOURCE_NEAREST_FRAME = "nearest_frame"
TARGET_SOURCE_NEAREST_FRAME_OLD = "nearest_frame_old"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
CONFIDENCE_UNKNOWN = "unknown"

CONFIDENCE_RANK = {
    CONFIDENCE_HIGH: 3,
    CONFIDENCE_MEDIUM: 2,
    CONFIDENCE_LOW: 1,
    CONFIDENCE_UNKNOWN: 0,
}

CONFIDENCE_WEIGHTS = {
    CONFIDENCE_HIGH: 1.0,
    CONFIDENCE_MEDIUM: 0.5,
    CONFIDENCE_LOW: 0.2,
    CONFIDENCE_UNKNOWN: 0.0,
}

DEFAULT_UNCERTAINTY_PX = {
    TARGET_SOURCE_POST_PITCH_MARKER: 10.0,
    TARGET_SOURCE_BRACKET_MIDPOINT: 40.0,
    TARGET_SOURCE_BRACKET_INTERPOLATION: 30.0,
    TARGET_SOURCE_NEAREST_FRAME: 60.0,
    TARGET_SOURCE_NEAREST_FRAME_OLD: 80.0,
}


def _bracket_point_complete(point: Optional[dict[str, Any]]) -> bool:
    if not isinstance(point, dict):
        return False
    return point.get("frame") is not None and point.get("x") is not None and point.get("y") is not None


def bracket_span_px(before: dict[str, Any], after: dict[str, Any]) -> float:
    return float(
        math.hypot(
            float(after["x"]) - float(before["x"]),
            float(after["y"]) - float(before["y"]),
        )
    )


def bracket_uncertainty_px(before: dict[str, Any], after: dict[str, Any], alpha: float) -> float:
    """Worst-case half-span at midpoint; scales down near endpoints."""
    span = bracket_span_px(before, after)
    return float(0.5 * span * max(0.0, 1.0 - abs(float(alpha) - 0.5) * 2.0))


def interpolate_bracket(
    before: dict[str, Any],
    after: dict[str, Any],
    alpha: float,
) -> tuple[float, float, float]:
    alpha = max(0.0, min(1.0, float(alpha)))
    bx, by = float(before["x"]), float(before["y"])
    ax, ay = float(after["x"]), float(after["y"])
    b_frame = float(before["frame"])
    a_frame = float(after["frame"])
    cross_x = bx + alpha * (ax - bx)
    cross_y = by + alpha * (ay - by)
    cross_frame = b_frame + alpha * (a_frame - b_frame)
    return cross_x, cross_y, cross_frame


def confidence_for_source(target_source: str, *, alpha: Optional[float] = None) -> str:
    if target_source == TARGET_SOURCE_POST_PITCH_MARKER:
        return CONFIDENCE_HIGH
    if target_source == TARGET_SOURCE_BRACKET_MIDPOINT:
        return CONFIDENCE_MEDIUM
    if target_source == TARGET_SOURCE_BRACKET_INTERPOLATION:
        if alpha is not None and abs(float(alpha) - 0.5) < 0.15:
            return CONFIDENCE_MEDIUM
        return CONFIDENCE_MEDIUM
    if target_source == TARGET_SOURCE_NEAREST_FRAME:
        return CONFIDENCE_LOW
    if target_source == TARGET_SOURCE_NEAREST_FRAME_OLD:
        return CONFIDENCE_UNKNOWN
    return CONFIDENCE_UNKNOWN


def is_estimated_source(target_source: str) -> bool:
    return target_source not in {
        TARGET_SOURCE_POST_PITCH_MARKER,
        TARGET_SOURCE_NEAREST_FRAME,
    }


def target_weight_from_confidence(confidence: str) -> float:
    return float(CONFIDENCE_WEIGHTS.get(confidence, 0.0))


def default_target_quality(
    target_source: str,
    *,
    alpha: Optional[float] = None,
    uncertainty_px: Optional[float] = None,
) -> dict[str, Any]:
    confidence = confidence_for_source(target_source, alpha=alpha)
    if uncertainty_px is None:
        uncertainty_px = DEFAULT_UNCERTAINTY_PX.get(target_source, 80.0)
    return {
        "target_source": target_source,
        "confidence": confidence,
        "estimated": is_estimated_source(target_source),
        "uncertainty_px": round(float(uncertainty_px), 3),
        "weight": target_weight_from_confidence(confidence),
    }


def derive_target_from_bracket(
    target_bracket: dict[str, Any],
    *,
    alpha: Optional[float] = None,
    force_midpoint: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = target_bracket.get("before")
    after = target_bracket.get("after")
    if not _bracket_point_complete(before) or not _bracket_point_complete(after):
        raise ValueError("target_bracket requires complete before and after points")

    chosen_alpha = 0.5 if force_midpoint else float(alpha if alpha is not None else target_bracket.get("alpha", 0.5))
    target_bracket["alpha"] = round(chosen_alpha, 4)
    cross_x, cross_y, cross_frame = interpolate_bracket(before, after, chosen_alpha)

    if force_midpoint or abs(chosen_alpha - 0.5) < 1e-6:
        source = TARGET_SOURCE_BRACKET_MIDPOINT
    else:
        source = TARGET_SOURCE_BRACKET_INTERPOLATION

    uncertainty = bracket_uncertainty_px(before, after, chosen_alpha)
    target = {
        "cross_x": round(cross_x, 3),
        "cross_y": round(cross_y, 3),
        "cross_frame": round(cross_frame, 3),
        "target_source": source,
    }
    quality = default_target_quality(source, alpha=chosen_alpha, uncertainty_px=uncertainty)
    return target, quality


def sync_legacy_quality(data: dict[str, Any], target_quality: dict[str, Any]) -> None:
    quality = data.setdefault("quality", {})
    if not isinstance(quality, dict):
        quality = {}
        data["quality"] = quality
    quality["crossing_estimated"] = bool(target_quality.get("estimated"))
    conf = str(target_quality.get("confidence") or CONFIDENCE_UNKNOWN)
    if conf in {CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW}:
        quality["label_confidence"] = conf


def finalize_label_target(data: dict[str, Any]) -> None:
    """Ensure target, target_bracket, and target_quality are consistent before save."""
    target = data.setdefault("target", {})
    target_bracket = data.get("target_bracket")
    click_mode = str(data.get("_target_click_mode") or "")

    if isinstance(target_bracket, dict):
        before_ok = _bracket_point_complete(target_bracket.get("before"))
        after_ok = _bracket_point_complete(target_bracket.get("after"))
        if before_ok and after_ok:
            derived_target, target_quality = derive_target_from_bracket(target_bracket)
            target.update(derived_target)
            data["target_quality"] = target_quality
            sync_legacy_quality(data, target_quality)
            data.pop("_target_click_mode", None)
            return

    if target.get("cross_x") is not None and target.get("cross_y") is not None:
        source = str(target.get("target_source") or "")
        if not source:
            if click_mode == "post_pitch_marker":
                source = TARGET_SOURCE_POST_PITCH_MARKER
            elif data.get("target_quality"):
                source = str(data["target_quality"].get("target_source") or TARGET_SOURCE_NEAREST_FRAME)
            else:
                source = TARGET_SOURCE_NEAREST_FRAME
            target["target_source"] = source

        if not data.get("target_quality"):
            uncertainty = DEFAULT_UNCERTAINTY_PX.get(source, 60.0)
            data["target_quality"] = default_target_quality(source, uncertainty_px=uncertainty)
        sync_legacy_quality(data, data["target_quality"])
        data.pop("_target_click_mode", None)
        return

    data.pop("_target_click_mode", None)


def normalize_label_target_fields(raw: dict[str, Any]) -> None:
    """Backfill target_quality for legacy labels (in-memory, before load/predict)."""
    if raw.get("target_quality"):
        tq = raw["target_quality"]
        if "weight" not in tq and tq.get("confidence"):
            tq["weight"] = target_weight_from_confidence(str(tq["confidence"]))
        target = raw.get("target") or {}
        if tq.get("target_source") and not target.get("target_source"):
            target["target_source"] = tq["target_source"]
        sync_legacy_quality(raw, tq)
        return

    target = raw.get("target") or {}
    if target.get("cross_x") is None or target.get("cross_y") is None:
        return

    source = str(target.get("target_source") or TARGET_SOURCE_NEAREST_FRAME_OLD)
    if not target.get("target_source"):
        target["target_source"] = source

    raw["target_quality"] = default_target_quality(
        source,
        uncertainty_px=DEFAULT_UNCERTAINTY_PX.get(source, 80.0),
    )
    sync_legacy_quality(raw, raw["target_quality"])


def meets_min_confidence(confidence: str, min_confidence: str) -> bool:
    return CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK.get(min_confidence, 0)

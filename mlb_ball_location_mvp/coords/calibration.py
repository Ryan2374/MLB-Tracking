"""Coordinate calibration for full-frame pixel space.

All ball labels and predictions use the video frame as a 2D grid:

    top-left = (0, 0)
    x increases right
    y increases down
    bottom-right = (frame_width - 1, frame_height - 1) for the last pixel index

Store coordinates in full-frame pixels even when detection runs on a crop.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

COORDINATE_SPACE = "full_frame_pixels"
ORIGIN = "top_left"
X_DIRECTION = "right"
Y_DIRECTION = "down"

SMALL_GRID_PX = 25
LARGE_GRID_PX = 100


def full_frame_metadata(frame_width: int, frame_height: int) -> dict[str, Any]:
    return {
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "coordinate_space": COORDINATE_SPACE,
        "origin": ORIGIN,
        "x_direction": X_DIRECTION,
        "y_direction": Y_DIRECTION,
    }


def roi_to_full_frame(crop_x: float, crop_y: float, local_x: float, local_y: float) -> tuple[float, float]:
    """Convert a point inside a crop back to full-frame pixel coordinates."""
    return crop_x + local_x, crop_y + local_y


def normalize_to_zone(
    x: float,
    y: float,
    zone: dict[str, float],
) -> tuple[float, float]:
    """Map a full-frame point into normalized strike-zone coordinates."""
    left = float(zone["left"])
    right = float(zone["right"])
    top = float(zone["top"])
    bottom = float(zone["bottom"])
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("zone must have positive width and height")
    zone_x = (x - left) / width
    zone_y = (y - top) / height
    return zone_x, zone_y


def draw_grid(
    img: np.ndarray,
    *,
    small_spacing: int = SMALL_GRID_PX,
    large_spacing: int = LARGE_GRID_PX,
    fast: bool = False,
) -> None:
    """Draw optional debug grid lines over the image (in-place)."""
    height, width = img.shape[:2]
    small_color = (48, 48, 48)
    large_color = (72, 72, 72)
    line_type = cv2.LINE_8 if fast else cv2.LINE_AA

    for x in range(0, width, small_spacing):
        color = large_color if x % large_spacing == 0 else small_color
        cv2.line(img, (x, 0), (x, height - 1), color, 1, line_type)
    for y in range(0, height, small_spacing):
        color = large_color if y % large_spacing == 0 else small_color
        cv2.line(img, (0, y), (width - 1, y), color, 1, line_type)


def draw_zone(img: np.ndarray, zone: dict[str, float], *, color: tuple[int, int, int] = (255, 128, 0)) -> None:
    """Draw the strike-zone rectangle if present in the label."""
    left = int(round(float(zone["left"])))
    top = int(round(float(zone["top"])))
    right = int(round(float(zone["right"])))
    bottom = int(round(float(zone["bottom"])))
    cv2.rectangle(img, (left, top), (right, bottom), color, 2, cv2.LINE_AA)


def zone_dict_or_none(raw: Optional[dict[str, Any]]) -> Optional[dict[str, float]]:
    if not raw:
        return None
    required = ("left", "top", "right", "bottom")
    if any(raw.get(key) is None for key in required):
        return None
    return {key: float(raw[key]) for key in required}

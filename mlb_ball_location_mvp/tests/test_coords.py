from __future__ import annotations

from coords.calibration import (
    COORDINATE_SPACE,
    full_frame_metadata,
    normalize_to_zone,
    roi_to_full_frame,
)


def test_full_frame_metadata() -> None:
    meta = full_frame_metadata(1920, 1080)
    assert meta["frame_width"] == 1920
    assert meta["frame_height"] == 1080
    assert meta["coordinate_space"] == COORDINATE_SPACE
    assert meta["origin"] == "top_left"
    assert meta["x_direction"] == "right"
    assert meta["y_direction"] == "down"


def test_roi_to_full_frame() -> None:
    full_x, full_y = roi_to_full_frame(300, 100, 166, 138)
    assert full_x == 466
    assert full_y == 238


def test_normalize_to_zone() -> None:
    zone = {"left": 485.0, "top": 285.0, "right": 690.0, "bottom": 525.0}
    zone_x, zone_y = normalize_to_zone(560.0, 412.0, zone)
    assert abs(zone_x - 0.3658536585365854) < 1e-9
    assert abs(zone_y - 0.5291666666666666) < 1e-9


def test_normalize_to_zone_edges() -> None:
    zone = {"left": 0.0, "top": 0.0, "right": 100.0, "bottom": 100.0}
    assert normalize_to_zone(0.0, 0.0, zone) == (0.0, 0.0)
    assert normalize_to_zone(100.0, 100.0, zone) == (1.0, 1.0)
    assert normalize_to_zone(50.0, 50.0, zone) == (0.5, 0.5)

#!/usr/bin/env python3
"""Render a debug image with observed points, prediction, and ground truth."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from coords.calibration import draw_grid, draw_zone, zone_dict_or_none


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_prediction(pred_payload: dict[str, Any], pitch_id: str) -> dict[str, Any]:
    predictions = pred_payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Prediction JSON must contain a non-empty predictions list")
    for pred in predictions:
        if pred.get("pitch_id") == pitch_id:
            return pred
    if len(predictions) == 1:
        return predictions[0]
    raise ValueError(f"No prediction found for pitch_id={pitch_id}")


def resolve_video_path(label_path: Path, label: dict[str, Any], override: Optional[str]) -> Optional[Path]:
    raw = override or label.get("video")
    if not raw:
        return None
    p = Path(raw)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(label_path.parent / p)
        candidates.append(label_path.parent.parent / p)
        candidates.append(Path.cwd() / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return p


def make_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    return canvas


def load_frame(video_path: Optional[Path], frame_idx: Optional[int], width: int, height: int) -> np.ndarray:
    if video_path is None or not video_path.exists():
        return make_canvas(width, height)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return make_canvas(width, height)
    if frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return make_canvas(width, height)
    return frame


def draw_text(img, text: str, x: int, y: int, scale: float = 0.65, thickness: int = 2) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_cross(img, x: float, y: float, color, label: str) -> None:  # noqa: ANN001
    xi, yi = int(round(x)), int(round(y))
    cv2.drawMarker(img, (xi, yi), color, markerType=cv2.MARKER_CROSS, markerSize=28, thickness=3)
    draw_text(img, label, xi + 12, yi - 12, scale=0.55, thickness=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Pitch label JSON")
    parser.add_argument("--prediction", required=True, help="Prediction JSON from prediction/predict_location.py")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--video", help="Optional override video path")
    parser.add_argument("--width", type=int, default=1920, help="Fallback blank canvas width")
    parser.add_argument("--height", type=int, default=1080, help="Fallback blank canvas height")
    parser.add_argument("--grid", action="store_true", help="Draw 25px/100px debug grid")
    parser.add_argument("--zone", action="store_true", help="Draw strike-zone box if present in label")
    args = parser.parse_args()

    label_path = Path(args.label)
    label = load_json(label_path)
    pred_payload = load_json(Path(args.prediction))
    pitch_id = str(label.get("pitch_id") or label_path.stem)
    pred = find_prediction(pred_payload, pitch_id)

    width = int(label.get("frame_width") or args.width)
    height = int(label.get("frame_height") or args.height)

    target = label["target"]
    frame_idx = target.get("cross_frame")
    if frame_idx is None and label.get("early_points"):
        frame_idx = int(label["early_points"][-1]["frame"])

    video_path = resolve_video_path(label_path, label, args.video)
    img = load_frame(video_path, frame_idx, width, height)

    if args.grid:
        draw_grid(img)

    zone = zone_dict_or_none(label.get("zone"))
    if args.zone and zone is not None:
        draw_zone(img, zone)

    pts = []
    for item in label.get("early_points", []):
        x, y = float(item["x"]), float(item["y"])
        pts.append((int(round(x)), int(round(y))))
        cv2.circle(img, pts[-1], 6, (0, 255, 255), 2)
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)

    predicted = pred["predicted_cross"]
    actual = pred["actual_cross"]
    pred_x, pred_y = float(predicted["x"]), float(predicted["y"])
    actual_x, actual_y = float(actual["x"]), float(actual["y"])
    error = math.hypot(pred_x - actual_x, pred_y - actual_y)

    draw_cross(img, actual_x, actual_y, (0, 0, 255), "actual")
    draw_cross(img, pred_x, pred_y, (0, 255, 0), "pred")
    cv2.line(img, (int(round(actual_x)), int(round(actual_y))), (int(round(pred_x)), int(round(pred_y))), (255, 255, 255), 1)

    lines = [
        f"pitch_id={pitch_id}",
        f"method={pred.get('method')} n_points={pred.get('n_points')}",
        f"error_px={error:.2f}",
        f"frame={frame_idx}",
        f"actual: x={actual_x:.1f}, y={actual_y:.1f}",
        f"pred:   x={pred_x:.1f}, y={pred_y:.1f}",
        f"space={label.get('coordinate_space', 'full_frame_pixels')}",
    ]
    if label.get("release_frame") is not None:
        lines.append(f"release_frame={label['release_frame']}")
    if target.get("cross_frame") is not None:
        lines.append(f"cross_frame={target['cross_frame']}")

    for i, line in enumerate(lines):
        draw_text(img, line, 20, 35 + i * 28, scale=0.55, thickness=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise SystemExit(f"Failed to write {out_path}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

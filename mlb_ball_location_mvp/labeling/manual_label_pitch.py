#!/usr/bin/env python3
"""Manual pitch labeling tool.

Use this to create clean MVP labels before automating ball detection.
All coordinates are stored in full-frame pixel space (top-left origin).
The output label contains:
- coordinate metadata (frame size, coordinate_space)
- release_frame
- first 5-10 early ball center points
- target crossing point: cross_frame, cross_x, cross_y
- optional strike-zone rectangle for later normalization
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from coords.calibration import draw_grid, draw_zone, full_frame_metadata, zone_dict_or_none


HELP_LINES = [
    "Controls:",
    "n/p: next/prev frame",
    "j/k: +/- 10 frames",
    "r: set release_frame",
    "e: early-point mode",
    "t: target-crossing mode",
    "left click: add point for current mode",
    "u: undo early point or clear target",
    "g: toggle grid overlay",
    "z: toggle strike-zone box",
    "c: toggle coordinate readout",
    "s: save label",
    "h: toggle this help",
    "q: quit",
]


class LabelState:
    def __init__(
        self,
        video_path: Path,
        out_path: Path,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.video_path = video_path
        self.out_path = out_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mode = "early"
        self.frame_idx = 0
        self.show_help = True
        self.show_grid = True
        self.show_zone = True
        self.show_coords = True
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None
        self.data: dict[str, Any] = {
            **full_frame_metadata(frame_width, frame_height),
            "pitch_id": video_path.stem,
            "video": str(video_path),
            "fps": fps if fps > 0 else None,
            "release_frame": None,
            "early_points": [],
            "target": {
                "cross_frame": None,
                "cross_x": None,
                "cross_y": None,
            },
        }

    def add_click(self, x: int, y: int) -> None:
        if self.mode == "early":
            point = {"frame": int(self.frame_idx), "x": float(x), "y": float(y)}
            points = self.data["early_points"]
            replaced = False
            for i, existing in enumerate(points):
                if int(existing["frame"]) == self.frame_idx:
                    points[i] = point
                    replaced = True
                    break
            if not replaced:
                points.append(point)
            points.sort(key=lambda p: int(p["frame"]))
        else:
            self.data["target"] = {
                "cross_frame": int(self.frame_idx),
                "cross_x": float(x),
                "cross_y": float(y),
            }

    def undo(self) -> None:
        if self.mode == "early":
            if self.data["early_points"]:
                self.data["early_points"].pop()
        else:
            self.data["target"] = {"cross_frame": None, "cross_x": None, "cross_y": None}

    def target_set(self) -> bool:
        target = self.data["target"]
        return target.get("cross_x") is not None and target.get("cross_y") is not None

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
        print(f"Saved {self.out_path}")


def draw_text(img, text: str, x: int, y: int, scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_overlay(frame, state: LabelState):
    img = frame.copy()
    h, w = img.shape[:2]

    if state.show_grid:
        draw_grid(img)

    zone = zone_dict_or_none(state.data.get("zone"))
    if state.show_zone and zone is not None:
        draw_zone(img, zone)

    for idx, p in enumerate(state.data["early_points"], start=1):
        x, y = int(round(p["x"])), int(round(p["y"]))
        same_frame = int(p["frame"]) == state.frame_idx
        radius = 7 if same_frame else 4
        cv2.circle(img, (x, y), radius, (0, 255, 255), 2)
        draw_text(img, str(idx), x + 8, y - 8, scale=0.45)

    target = state.data["target"]
    if target.get("cross_x") is not None and target.get("cross_y") is not None:
        tx, ty = int(round(target["cross_x"])), int(round(target["cross_y"]))
        cv2.drawMarker(img, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        draw_text(img, "target", tx + 10, ty - 10, scale=0.5)

    release = state.data.get("release_frame")
    target_frame = target.get("cross_frame")
    mode_label = "early-point" if state.mode == "early" else "target-crossing"
    target_status = "set" if state.target_set() else "not set"

    top = [
        f"Frame: {state.frame_idx}",
        f"Mode: {mode_label}",
        f"Early points: {len(state.data['early_points'])}",
        f"Target: {target_status}",
        f"release_frame: {release}",
        f"cross_frame: {target_frame}",
    ]
    if state.show_coords and state.mouse_x is not None and state.mouse_y is not None:
        top.append(f"Mouse: x={state.mouse_x}, y={state.mouse_y}")
    top.append(f"space: {state.data.get('coordinate_space')} ({w}x{h})")

    for i, line in enumerate(top):
        draw_text(img, line, 16, 28 + i * 24)

    if state.show_help:
        y0 = max(220, h - 280)
        for i, line in enumerate(HELP_LINES):
            draw_text(img, line, 16, y0 + i * 22, scale=0.48)

    return img


def seek_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Pitch video path")
    parser.add_argument("--out", required=True, help="Output label JSON path")
    parser.add_argument("--zone", help="Optional JSON file with strike-zone rectangle")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    state = LabelState(
        video_path=video_path,
        out_path=out_path,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if args.zone:
        with Path(args.zone).open("r", encoding="utf-8") as f:
            zone_payload = json.load(f)
        zone = zone_dict_or_none(zone_payload.get("zone", zone_payload))
        if zone is not None:
            state.data["zone"] = zone

    window = "manual_pitch_label"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):  # noqa: ANN001, ARG001
        state.mouse_x = int(x)
        state.mouse_y = int(y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_click(x, y)

    cv2.setMouseCallback(window, on_mouse)

    while True:
        frame = seek_frame(cap, state.frame_idx)
        if frame is None:
            state.frame_idx = max(0, min(state.frame_idx, frame_count - 1))
            frame = seek_frame(cap, state.frame_idx)
            if frame is None:
                break

        img = render_overlay(frame, state)
        cv2.imshow(window, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("n"):
            state.frame_idx = min(frame_count - 1 if frame_count else state.frame_idx + 1, state.frame_idx + 1)
        elif key == ord("p"):
            state.frame_idx = max(0, state.frame_idx - 1)
        elif key == ord("j"):
            state.frame_idx = min(frame_count - 1 if frame_count else state.frame_idx + 10, state.frame_idx + 10)
        elif key == ord("k"):
            state.frame_idx = max(0, state.frame_idx - 10)
        elif key == ord("r"):
            state.data["release_frame"] = int(state.frame_idx)
        elif key == ord("e"):
            state.mode = "early"
        elif key == ord("t"):
            state.mode = "target"
        elif key == ord("u"):
            state.undo()
        elif key == ord("g"):
            state.show_grid = not state.show_grid
        elif key == ord("z"):
            state.show_zone = not state.show_zone
        elif key == ord("c"):
            state.show_coords = not state.show_coords
        elif key == ord("s"):
            state.save()
        elif key == ord("h"):
            state.show_help = not state.show_help

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

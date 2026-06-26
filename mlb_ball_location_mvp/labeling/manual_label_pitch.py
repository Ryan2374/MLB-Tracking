#!/usr/bin/env python3
"""Manual pitch labeling tool.

Label the full observed ball flight after release, plus the plate-crossing target.
All coordinates are stored in full-frame pixel space (top-left origin).

The `early_points` field stores every visible ball position after release.
Prediction later uses only the first N of those points via --n-points.
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
import numpy as np

from coords.calibration import draw_grid, draw_zone, full_frame_metadata, zone_dict_or_none

PITCH_TYPES = [
    "fastball",
    "slider",
    "curveball",
    "changeup",
    "cutter",
    "sinker",
    "splitter",
    "unknown",
]
ZONE_RESULTS = ["strike", "borderline", "ball", "unknown"]
LOCATION_BUCKETS = [
    "middle",
    "high",
    "low",
    "inside",
    "outside",
    "high_inside",
    "high_outside",
    "low_inside",
    "low_outside",
    "above_zone",
    "below_zone",
    "way_inside",
    "way_outside",
    "unknown",
]
LABEL_CONFIDENCE = ["high", "medium", "low"]

HELP_LINES = [
    "Workflow: r=release | e=ball points | click every visible frame | t=target | s=save",
    "n/p or arrows: next/prev frame | j/k: +/-10 | space: play/pause | f: next gap",
    "Ball points: click center of ball each frame (skip hidden frames)",
    "Target: plate/strike-zone crossing, not catcher glove",
    "Metadata: [/]=pitch type | ;/'=zone | ,/.=location | m=confidence | b=est. cross | i=notes",
    "v: all-frame markers | g: grid | z: zone box | u: undo | h: help | q: quit",
]

ARROW_LEFT = {2424832, 81, 2}
ARROW_RIGHT = {2555904, 83, 3}


def default_quality() -> dict[str, Any]:
    return {
        "ball_visible": True,
        "crossing_estimated": False,
        "label_confidence": "high",
    }


def cycle_option(options: list[str], current: str, direction: int) -> str:
    if current not in options:
        return options[0]
    idx = (options.index(current) + direction) % len(options)
    return options[idx]


class LabelState:
    def __init__(
        self,
        video_path: Path,
        out_path: Path,
        fps: float,
        frame_width: int,
        frame_height: int,
        *,
        auto_step: int = 1,
        difficulty: str = "GOAT",
    ) -> None:
        self.video_path = video_path
        self.out_path = out_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.auto_step = max(0, auto_step)
        self.mode = "early"
        self.frame_idx = 0
        self.show_help = True
        self.show_grid = True
        self.show_zone = True
        self.show_all_points = False
        self.show_coords = True
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None
        self.data: dict[str, Any] = {
            **full_frame_metadata(frame_width, frame_height),
            "pitch_id": video_path.stem,
            "video": str(video_path),
            "fps": fps if fps > 0 else None,
            "difficulty": difficulty,
            "pitch_type": "unknown",
            "zone_result": "unknown",
            "location_bucket": "unknown",
            "release_frame": None,
            "early_points": [],
            "target": {
                "cross_frame": None,
                "cross_x": None,
                "cross_y": None,
            },
            "quality": default_quality(),
            "notes": "",
        }

    def labeled_frames(self) -> set[int]:
        return {int(p["frame"]) for p in self.data["early_points"]}

    def ball_point_range_end(self) -> Optional[int]:
        target = self.data["target"]
        cross_frame = target.get("cross_frame")
        if cross_frame is not None:
            return int(cross_frame) - 1
        return None

    def first_unlabeled_frame(self) -> Optional[int]:
        release = self.data.get("release_frame")
        if release is None:
            return None
        end = self.ball_point_range_end()
        if end is None:
            if self.data["early_points"]:
                end = int(self.data["early_points"][-1]["frame"]) + 5
            else:
                end = int(release) + 20
        labeled = self.labeled_frames()
        for frame in range(int(release) + 1, int(end) + 1):
            if frame not in labeled:
                return frame
        return None

    def missing_frame_count(self) -> int:
        release = self.data.get("release_frame")
        if release is None:
            return 0
        end = self.ball_point_range_end()
        if end is None:
            return 0
        labeled = self.labeled_frames()
        return sum(
            1 for frame in range(int(release) + 1, int(end) + 1)
            if frame not in labeled
        )

    def validation_warnings(self) -> list[str]:
        warnings: list[str] = []
        release = self.data.get("release_frame")
        points = self.data["early_points"]
        target = self.data["target"]

        if release is None:
            warnings.append("release_frame not set")
        if len(points) < 5:
            warnings.append(f"only {len(points)} ball points (aim for 5-10 on GOAT)")
        if release is not None and points:
            first_frame = int(points[0]["frame"])
            if first_frame <= int(release):
                warnings.append("first ball point should be after release_frame")
        if target.get("cross_x") is None or target.get("cross_y") is None:
            warnings.append("target crossing not set")
        missing = self.missing_frame_count()
        if missing > 0:
            warnings.append(f"{missing} visible frames still unlabeled before crossing")
        return warnings

    def load_existing(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.data.update(payload)
        self.data["pitch_id"] = self.video_path.stem
        self.data["video"] = str(self.video_path)
        quality = self.data.get("quality")
        if not isinstance(quality, dict):
            self.data["quality"] = default_quality()
        else:
            merged = default_quality()
            merged.update(quality)
            self.data["quality"] = merged
        if "notes" not in self.data:
            self.data["notes"] = ""
        release = self.data.get("release_frame")
        if release is not None:
            self.frame_idx = int(release)
        print(f"Loaded existing label from {path}")

    def add_click(self, x: int, y: int, frame_count: int) -> None:
        if self.mode == "early":
            release = self.data.get("release_frame")
            if release is not None and self.frame_idx <= int(release):
                print("Warning: ball point should be after release_frame")
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
            if not replaced and self.auto_step > 0 and frame_count > 0:
                self.frame_idx = min(frame_count - 1, self.frame_idx + self.auto_step)
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
        for warning in self.validation_warnings():
            print(f"Warning: {warning}")
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
        point_frame = int(p["frame"])
        if not state.show_all_points and point_frame != state.frame_idx:
            continue
        x, y = int(round(p["x"])), int(round(p["y"]))
        cv2.circle(img, (x, y), 7, (0, 255, 255), 2)
        draw_text(img, str(idx), x + 8, y - 8, scale=0.45)

    pts = state.data["early_points"]
    if state.show_all_points and len(pts) >= 2:
        poly = [(int(round(p["x"])), int(round(p["y"]))) for p in pts]
        cv2.polylines(img, [np.array(poly, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=1)

    release = state.data.get("release_frame")
    if release is not None and int(release) == state.frame_idx:
        draw_text(img, "RELEASE", 16, h - 24, scale=0.7, thickness=2)

    labeled = state.labeled_frames()
    if release is not None and state.frame_idx > int(release) and state.frame_idx not in labeled:
        target_frame = state.data["target"].get("cross_frame")
        if target_frame is None or state.frame_idx < int(target_frame):
            draw_text(img, "unlabeled frame", w - 220, 28, scale=0.55, thickness=1)

    target = state.data["target"]
    target_frame = target.get("cross_frame")
    show_target = (
        target.get("cross_x") is not None
        and target.get("cross_y") is not None
        and (state.show_all_points or target_frame == state.frame_idx)
    )
    if show_target:
        tx, ty = int(round(target["cross_x"])), int(round(target["cross_y"]))
        cv2.drawMarker(img, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        draw_text(img, "target", tx + 10, ty - 10, scale=0.5)

    mode_label = "ball-point" if state.mode == "early" else "target-crossing"
    target_status = "set" if state.target_set() else "not set"
    missing = state.missing_frame_count()
    quality = state.data.get("quality", {})

    top = [
        f"Frame: {state.frame_idx}",
        f"Mode: {mode_label}",
        f"Ball points: {len(state.data['early_points'])} (label every visible frame)",
        f"Markers: {'all frames' if state.show_all_points else 'current frame only'}",
        f"Target: {target_status}",
        f"release_frame: {release}",
        f"cross_frame: {target_frame}",
        f"type={state.data.get('pitch_type')} zone={state.data.get('zone_result')} loc={state.data.get('location_bucket')}",
        f"confidence={quality.get('label_confidence')} est_cross={quality.get('crossing_estimated')}",
    ]
    if missing > 0:
        top.append(f"Missing ball frames before crossing: {missing}")
    notes = str(state.data.get("notes") or "")
    if notes:
        top.append(f"notes: {notes[:60]}{'...' if len(notes) > 60 else ''}")
    if state.show_coords and state.mouse_x is not None and state.mouse_y is not None:
        top.append(f"Mouse: x={state.mouse_x}, y={state.mouse_y}")

    for i, line in enumerate(top):
        draw_text(img, line, 16, 28 + i * 24)

    if state.show_help:
        y0 = max(220, h - 160)
        for i, line in enumerate(HELP_LINES):
            draw_text(img, line, 16, y0 + i * 22, scale=0.48)

    return img


def seek_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def clamp_frame(frame_idx: int, frame_count: int) -> int:
    if frame_count <= 0:
        return max(0, frame_idx)
    return max(0, min(frame_count - 1, frame_idx))


def handle_key(
    key_raw: int,
    state: LabelState,
    frame_count: int,
    *,
    playing: bool,
) -> bool:
    """Return updated playing flag."""
    key = key_raw & 0xFF
    if key == ord("q"):
        return playing
    if key_raw in ARROW_RIGHT or key == ord("n"):
        state.frame_idx = clamp_frame(state.frame_idx + 1, frame_count)
    elif key_raw in ARROW_LEFT or key == ord("p"):
        state.frame_idx = clamp_frame(state.frame_idx - 1, frame_count)
    elif key == ord("j"):
        state.frame_idx = clamp_frame(state.frame_idx + 10, frame_count)
    elif key == ord("k"):
        state.frame_idx = clamp_frame(state.frame_idx - 10, frame_count)
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
    elif key == ord("v"):
        state.show_all_points = not state.show_all_points
    elif key == ord("c"):
        state.show_coords = not state.show_coords
    elif key == ord("s"):
        state.save()
    elif key == ord("h"):
        state.show_help = not state.show_help
    elif key == ord(" "):
        playing = not playing
    elif key == ord("a"):
        release = state.data.get("release_frame")
        if release is not None:
            state.frame_idx = clamp_frame(int(release), frame_count)
    elif key == ord("f"):
        nxt = state.first_unlabeled_frame()
        if nxt is not None:
            state.frame_idx = clamp_frame(nxt, frame_count)
    elif key == ord("["):
        state.data["pitch_type"] = cycle_option(PITCH_TYPES, str(state.data.get("pitch_type", "unknown")), -1)
    elif key == ord("]"):
        state.data["pitch_type"] = cycle_option(PITCH_TYPES, str(state.data.get("pitch_type", "unknown")), 1)
    elif key == ord(";"):
        state.data["zone_result"] = cycle_option(ZONE_RESULTS, str(state.data.get("zone_result", "unknown")), -1)
    elif key == ord("'"):
        state.data["zone_result"] = cycle_option(ZONE_RESULTS, str(state.data.get("zone_result", "unknown")), 1)
    elif key == ord(","):
        state.data["location_bucket"] = cycle_option(
            LOCATION_BUCKETS, str(state.data.get("location_bucket", "unknown")), -1
        )
    elif key == ord("."):
        state.data["location_bucket"] = cycle_option(
            LOCATION_BUCKETS, str(state.data.get("location_bucket", "unknown")), 1
        )
    elif key == ord("m"):
        quality = state.data.setdefault("quality", default_quality())
        quality["label_confidence"] = cycle_option(
            LABEL_CONFIDENCE, str(quality.get("label_confidence", "high")), 1
        )
    elif key == ord("b"):
        quality = state.data.setdefault("quality", default_quality())
        quality["crossing_estimated"] = not bool(quality.get("crossing_estimated"))
    elif key == ord("i"):
        print("Enter notes (blank to clear). Check this terminal:")
        try:
            state.data["notes"] = input("notes> ").strip()
        except EOFError:
            pass
    return playing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Pitch video path")
    parser.add_argument("--out", required=True, help="Output label JSON path")
    parser.add_argument("--zone", help="Optional JSON file with strike-zone rectangle")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Frames to advance after each new ball point. Default: 1",
    )
    parser.add_argument(
        "--difficulty",
        default="GOAT",
        help="Capture difficulty tag saved in the label. Default: GOAT",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load existing label from --out if the file exists",
    )
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
        auto_step=args.frame_step,
        difficulty=args.difficulty,
    )

    if args.load and out_path.exists():
        state.load_existing(out_path)
    elif out_path.exists():
        state.load_existing(out_path)
        print("Tip: resuming existing label. Use a new --out path to start fresh.")

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
            state.add_click(x, y, frame_count)

    cv2.setMouseCallback(window, on_mouse)

    playing = False
    quit_requested = False
    play_delay_ms = max(1, int(1000 / fps)) if fps > 0 else 33

    while True:
        frame = seek_frame(cap, state.frame_idx)
        if frame is None:
            state.frame_idx = clamp_frame(state.frame_idx, frame_count)
            frame = seek_frame(cap, state.frame_idx)
            if frame is None:
                break

        img = render_overlay(frame, state)
        cv2.imshow(window, img)
        key_raw = cv2.waitKeyEx(play_delay_ms if playing else 0)
        if key_raw == -1 and playing:
            state.frame_idx = clamp_frame(state.frame_idx + 1, frame_count)
            if state.frame_idx >= frame_count - 1:
                playing = False
            continue
        if key_raw == -1:
            continue

        if (key_raw & 0xFF) == ord("q"):
            quit_requested = True
            break

        playing = handle_key(key_raw, state, frame_count, playing=playing)

    cap.release()
    cv2.destroyAllWindows()
    if quit_requested:
        return


if __name__ == "__main__":
    main()

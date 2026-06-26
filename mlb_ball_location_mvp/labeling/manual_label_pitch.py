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

CROSSHAIR_HALF = 14
MOUSE_HUD_X = 16
MOUSE_HUD_Y = 28 + 7 * 24
MOUSE_HUD_W = 300
MOUSE_HUD_H = 26


HELP_LINES = [
    "Workflow: r=release | e=ball points | t=target | s=save | q=quit",
    "n/p or arrows: next/prev | j/k: +/-10 | space: play/pause | f: next gap",
    "Metadata: [/]=pitch type | ;/'=zone | ,/.=location | m=confidence | b=est. cross | i=notes",
    "v: all-frame markers | g: grid | z: zone box | u: undo | c: coords | h: help",
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


def meta_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(".meta.json")


def resolve_meta_sidecar(video_path: Path) -> Optional[dict[str, Any]]:
    candidates = [video_path, Path.cwd() / video_path]
    seen: set[Path] = set()
    for candidate in candidates:
        meta_path = meta_sidecar_path(candidate.resolve() if candidate.exists() else candidate)
        if meta_path in seen:
            continue
        seen.add(meta_path)
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def apply_recording_fps_fields(
    data: dict[str, Any],
    video_path: Path,
    *,
    container_fps: float,
) -> Optional[float]:
    """Prefer measured capture FPS from .meta.json over MP4 container metadata."""
    meta = resolve_meta_sidecar(video_path)
    if meta is not None:
        actual = meta.get("actual_fps")
        if actual is not None and float(actual) > 0:
            data["fps"] = float(actual)
            requested = meta.get("requested_fps")
            if requested is not None and float(requested) > 0:
                data["requested_fps"] = float(requested)
            if container_fps > 0:
                data["container_fps"] = float(container_fps)
            else:
                data.pop("container_fps", None)
            return float(actual)

    data.pop("requested_fps", None)
    data.pop("container_fps", None)
    if container_fps > 0:
        data["fps"] = float(container_fps)
        return float(container_fps)
    data["fps"] = None
    return None


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
        self.min_points = 5
        self.mode = "early"
        self.frame_idx = 0
        self.show_help = True
        self.show_grid = False
        self.show_zone = True
        self.show_all_points = False
        self.show_coords = True
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None
        self.needs_redraw = True
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
        apply_recording_fps_fields(self.data, video_path, container_fps=fps)

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

    def validation_warnings(self, min_points: int = 5) -> list[str]:
        warnings: list[str] = []
        release = self.data.get("release_frame")
        points = self.data["early_points"]
        target = self.data["target"]
        w = int(self.data.get("frame_width") or self.frame_width or 0)
        h = int(self.data.get("frame_height") or self.frame_height or 0)

        if release is None:
            warnings.append("release_frame not set")
        if len(points) < min_points:
            warnings.append(f"only {len(points)} early points (aim for {min_points}+ on GOAT)")
        if release is not None and points:
            first_frame = int(points[0]["frame"])
            if first_frame <= int(release):
                warnings.append("first ball point should be after release_frame")
        if target.get("cross_x") is None or target.get("cross_y") is None:
            warnings.append("target crossing not set")
        if target.get("cross_frame") is None:
            warnings.append("target cross_frame is not set")
        missing = self.missing_frame_count()
        if missing > 0:
            warnings.append(f"{missing} visible frames still unlabeled before crossing")
        if release is not None:
            for idx, point in enumerate(points, start=1):
                frame = int(point["frame"])
                if frame <= int(release):
                    warnings.append(f"early point {idx} frame {frame} is not after release_frame {release}")
        cross_frame = target.get("cross_frame")
        if release is not None and cross_frame is not None and int(cross_frame) <= int(release):
            warnings.append(f"cross_frame {cross_frame} is not after release_frame {release}")
        if w > 0 and h > 0:
            for idx, point in enumerate(points, start=1):
                x, y = float(point["x"]), float(point["y"])
                if x < 0 or y < 0 or x >= w or y >= h:
                    warnings.append(f"early point {idx} ({x:.1f}, {y:.1f}) is outside frame bounds ({w}x{h})")
            if target.get("cross_x") is not None and target.get("cross_y") is not None:
                cx, cy = float(target["cross_x"]), float(target["cross_y"])
                if cx < 0 or cy < 0 or cx >= w or cy >= h:
                    warnings.append(f"target ({cx:.1f}, {cy:.1f}) is outside frame bounds ({w}x{h})")
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
        apply_recording_fps_fields(self.data, self.video_path, container_fps=float(self.data.get("fps") or 0))
        release = self.data.get("release_frame")
        if release is not None:
            self.frame_idx = int(release)
        print(f"Loaded existing label from {path}")

    @classmethod
    def from_existing(
        cls,
        video_path: Path,
        out_path: Path,
        fps: float,
        frame_width: int,
        frame_height: int,
        existing: dict[str, Any],
    ) -> LabelState:
        state = cls(video_path, out_path, fps, frame_width, frame_height)
        state.data.update(existing)
        state.data.setdefault("early_points", [])
        state.data.setdefault(
            "target",
            {"cross_frame": None, "cross_x": None, "cross_y": None},
        )
        state.data.setdefault("notes", "")
        quality = state.data.get("quality")
        if not isinstance(quality, dict):
            state.data["quality"] = default_quality()
        state.data["video"] = str(video_path)
        state.data["frame_width"] = frame_width
        state.data["frame_height"] = frame_height
        apply_recording_fps_fields(state.data, video_path, container_fps=fps)
        release = state.data.get("release_frame")
        if release is not None:
            state.frame_idx = int(release)
        elif state.data["early_points"]:
            state.frame_idx = int(state.data["early_points"][0]["frame"])
        return state

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

    def apply_metadata(
        self,
        pitch_type: Optional[str],
        zone_result: Optional[str],
        location_bucket: Optional[str],
        notes: str,
    ) -> None:
        if pitch_type is not None:
            self.data["pitch_type"] = pitch_type
        if zone_result is not None:
            self.data["zone_result"] = zone_result
        if location_bucket is not None:
            self.data["location_bucket"] = location_bucket
        if notes:
            self.data["notes"] = notes

    def save(self, min_points: int = 5) -> None:
        warnings = self.validation_warnings(min_points=min_points)
        if warnings:
            print("WARNING: label may be incomplete:")
            for warning in warnings:
                print(f" - {warning}")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
        print(f"Saved {self.out_path}")


def draw_text(img, text: str, x: int, y: int, scale: float = 0.55, thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def base_overlay_key(state: LabelState) -> tuple[Any, ...]:
    target = state.data["target"]
    zone = state.data.get("zone") or {}
    points = state.data["early_points"]
    return (
        state.frame_idx,
        state.show_grid,
        state.show_zone,
        state.show_help,
        state.mode,
        state.data.get("release_frame"),
        tuple((int(p["frame"]), float(p["x"]), float(p["y"])) for p in points),
        target.get("cross_frame"),
        target.get("cross_x"),
        target.get("cross_y"),
        tuple(sorted(zone.items())),
    )


def render_base_overlay(frame, state: LabelState):
    """Draw static overlay elements; cache this layer between mouse moves."""
    img = frame.copy()
    h, w = img.shape[:2]

    if state.show_grid:
        draw_grid(img, fast=True)

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
        f"fps: {state.data.get('fps')} (container {state.data.get('container_fps', 'n/a')})",
        f"type={state.data.get('pitch_type')} zone={state.data.get('zone_result')} loc={state.data.get('location_bucket')}",
        f"confidence={quality.get('label_confidence')} est_cross={quality.get('crossing_estimated')}",
    ]
    if missing > 0:
        top.append(f"Missing ball frames before crossing: {missing}")
    notes = str(state.data.get("notes") or "")
    if notes:
        top.append(f"notes: {notes[:60]}{'...' if len(notes) > 60 else ''}")
    for i, line in enumerate(top):
        if line:
            draw_text(img, line, 16, 28 + i * 24)

    if state.show_help:
        y0 = max(220, h - 160)
        for i, line in enumerate(HELP_LINES):
            draw_text(img, line, 16, y0 + i * 22, scale=0.48)

    return img


def restore_mouse_hud(display, base) -> None:
    y0 = MOUSE_HUD_Y - 18
    y1 = y0 + MOUSE_HUD_H
    x1 = MOUSE_HUD_X + MOUSE_HUD_W
    display[y0:y1, MOUSE_HUD_X:x1] = base[y0:y1, MOUSE_HUD_X:x1]


def erase_crosshair(display, base, mx: int, my: int) -> None:
    h, w = display.shape[:2]
    half = CROSSHAIR_HALF
    x0, x1 = max(0, mx - half), min(w, mx + half + 1)
    display[max(0, my - 1) : min(h, my + 2), x0:x1] = base[max(0, my - 1) : min(h, my + 2), x0:x1]
    y0, y1 = max(0, my - half), min(h, my + half + 1)
    display[y0:y1, max(0, mx - 1) : min(w, mx + 2)] = base[y0:y1, max(0, mx - 1) : min(w, mx + 2)]


def draw_crosshair(display, mx: int, my: int) -> None:
    half = CROSSHAIR_HALF
    color = (0, 255, 255)
    cv2.line(display, (mx - half, my), (mx + half, my), color, 1, cv2.LINE_8)
    cv2.line(display, (mx, my - half), (mx, my + half), color, 1, cv2.LINE_8)


def draw_mouse_hud(display, mx: int, my: int) -> None:
    cv2.putText(
        display,
        f"Mouse: x={mx}, y={my}",
        (MOUSE_HUD_X, MOUSE_HUD_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_8,
    )


def reset_display_from_base(base, state: LabelState):
    """One full copy when the static overlay changes."""
    display = base.copy()
    if state.show_coords and state.mouse_x is not None and state.mouse_y is not None:
        draw_crosshair(display, state.mouse_x, state.mouse_y)
        draw_mouse_hud(display, state.mouse_x, state.mouse_y)
    return display


def update_mouse_on_display(display, base, state: LabelState, last_mx: Optional[int], last_my: Optional[int]) -> None:
    """Patch only crosshair + HUD regions instead of copying the full frame."""
    if last_mx is not None and last_my is not None:
        erase_crosshair(display, base, last_mx, last_my)
    restore_mouse_hud(display, base)

    if state.show_coords and state.mouse_x is not None and state.mouse_y is not None:
        draw_crosshair(display, state.mouse_x, state.mouse_y)
        draw_mouse_hud(display, state.mouse_x, state.mouse_y)


def render_mouse_overlay(base, state: LabelState):
    """Fallback full redraw (tests / one-off renders)."""
    display = base.copy()
    if state.show_coords and state.mouse_x is not None and state.mouse_y is not None:
        draw_crosshair(display, state.mouse_x, state.mouse_y)
        draw_mouse_hud(display, state.mouse_x, state.mouse_y)
    return display


def render_overlay(frame, state: LabelState):
    return render_mouse_overlay(render_base_overlay(frame, state), state)


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
        state.save(min_points=state.min_points)
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
    parser.add_argument("--grid", action="store_true", help="Show grid overlay on launch (default: hidden)")
    parser.add_argument("--min-points", type=int, default=5, help="Minimum early points for save warnings")
    parser.add_argument("--pitch-type", help="Pitch metadata, e.g. fastball")
    parser.add_argument("--zone-result", help="Pitch metadata, e.g. strike")
    parser.add_argument("--location-bucket", help="Pitch metadata, e.g. high_inside")
    parser.add_argument("--notes", default="", help="Freeform notes for this pitch")
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

    container_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    state = LabelState(
        video_path=video_path,
        out_path=out_path,
        fps=container_fps,
        frame_width=frame_width,
        frame_height=frame_height,
        auto_step=args.frame_step,
        difficulty=args.difficulty,
    )
    state.min_points = max(1, args.min_points)
    state.apply_metadata(args.pitch_type, args.zone_result, args.location_bucket, args.notes)
    if state.data.get("fps") != container_fps and state.data.get("requested_fps") is not None:
        print(
            f"Using measured fps={state.data['fps']} from sidecar "
            f"(container={container_fps}, requested={state.data['requested_fps']})"
        )
    if args.grid:
        state.show_grid = True

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
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            state.needs_redraw = True
        if event == cv2.EVENT_LBUTTONDOWN:
            state.add_click(x, y, frame_count)

    cv2.setMouseCallback(window, on_mouse)

    playing = False
    quit_requested = False
    label_fps = float(state.data.get("fps") or container_fps or 0.0)
    play_delay_ms = max(1, int(1000 / label_fps)) if label_fps > 0 else 33

    cached_frame_idx = -1
    cached_frame = None
    cached_base_key: Optional[tuple[Any, ...]] = None
    cached_base = None
    display_img = None
    last_drawn_mouse: tuple[Optional[int], Optional[int]] = (None, None)

    while True:
        if playing:
            state.frame_idx = clamp_frame(state.frame_idx + 1, frame_count)
            if state.frame_idx >= max(0, frame_count - 1):
                playing = False
            state.needs_redraw = True

        if state.frame_idx != cached_frame_idx:
            cached_frame = seek_frame(cap, state.frame_idx)
            if cached_frame is None:
                state.frame_idx = clamp_frame(state.frame_idx, frame_count)
                cached_frame = seek_frame(cap, state.frame_idx)
                if cached_frame is None:
                    break
            cached_frame_idx = state.frame_idx
            cached_base_key = None
            state.needs_redraw = True

        base_rebuilt = False
        base_key = base_overlay_key(state)
        if base_key != cached_base_key:
            cached_base = render_base_overlay(cached_frame, state)
            cached_base_key = base_key
            display_img = reset_display_from_base(cached_base, state)
            last_drawn_mouse = (
                (state.mouse_x, state.mouse_y)
                if state.show_coords and state.mouse_x is not None
                else (None, None)
            )
            base_rebuilt = True
            state.needs_redraw = True

        if state.needs_redraw and display_img is not None and cached_base is not None:
            if not base_rebuilt:
                update_mouse_on_display(
                    display_img,
                    cached_base,
                    state,
                    last_drawn_mouse[0],
                    last_drawn_mouse[1],
                )
            last_drawn_mouse = (state.mouse_x, state.mouse_y)
            cv2.imshow(window, display_img)
            state.needs_redraw = False

        key_raw = cv2.waitKeyEx(play_delay_ms if playing else 1)
        if key_raw == -1:
            continue

        if (key_raw & 0xFF) == ord("q"):
            quit_requested = True
            break

        state.needs_redraw = True
        playing = handle_key(key_raw, state, frame_count, playing=playing)

    cap.release()
    cv2.destroyAllWindows()
    if quit_requested:
        return


if __name__ == "__main__":
    main()

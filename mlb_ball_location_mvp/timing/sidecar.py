"""Capture timing sidecars and label timing enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2

TIMING_SOURCE_CAPTURE = "capture_sidecar"
TIMING_SOURCE_BACKFILL = "backfill_uniform_fps"
TIMING_SOURCE_MISSING = "missing"


def frames_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(".frames.jsonl")


def meta_sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(".meta.json")


def resolve_video_path(video_path: Path) -> Path:
    candidates = [video_path.resolve(), (Path.cwd() / video_path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return video_path


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def load_frame_timestamps(video_path: Path) -> dict[int, float]:
    sidecar = frames_sidecar_path(resolve_video_path(video_path))
    if not sidecar.exists():
        return {}
    timestamps: dict[int, float] = {}
    with sidecar.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamps[int(record["frame"])] = float(record["timestamp"])
    return timestamps


def load_meta_sidecar(video_path: Path) -> Optional[dict[str, Any]]:
    meta_path = meta_sidecar_path(resolve_video_path(video_path))
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_ms(frame_idx: int, timestamps: dict[int, float]) -> Optional[float]:
    ts = timestamps.get(int(frame_idx))
    if ts is None:
        return None
    return round(ts * 1000.0, 3)


def write_uniform_sidecars(
    video_path: Path,
    *,
    fps: float,
    frame_count: Optional[int] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create approximate sidecars from frame index / fps (not wall-clock exact)."""
    video_path = resolve_video_path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames_path = frames_sidecar_path(video_path)
    meta_path = meta_sidecar_path(video_path)
    if not overwrite and frames_path.exists() and meta_path.exists():
        meta = load_meta_sidecar(video_path) or {}
        if meta.get("timing_source") == TIMING_SOURCE_CAPTURE:
            return {"status": "skipped", "reason": "capture sidecar already exists"}

    if frame_count is None:
        frame_count = count_video_frames(video_path)
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"invalid frame_count={frame_count} or fps={fps} for {video_path}")

    timestamps: list[dict[str, float | int]] = []
    with frames_path.open("w", encoding="utf-8") as f:
        for frame in range(frame_count):
            ts = round(frame / fps, 6)
            record = {"frame": frame, "timestamp": ts}
            timestamps.append(record)
            f.write(json.dumps(record) + "\n")

    duration_s = round((frame_count - 1) / fps, 6) if frame_count > 1 else 0.0
    meta = {
        "video": str(video_path),
        "requested_fps": float(fps),
        "writer_fps": float(fps),
        "capture_fps_measured": float(fps),
        "actual_fps": float(fps),
        "frame_count": frame_count,
        "encoded_frame_count": frame_count,
        "sidecar_frame_count": frame_count,
        "frames_in_sync": True,
        "timing_source": TIMING_SOURCE_BACKFILL,
        "duration_s": duration_s,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    return {
        "status": "written",
        "video": str(video_path),
        "frames_path": str(frames_path),
        "meta_path": str(meta_path),
        "frame_count": frame_count,
        "fps": fps,
        "timing_source": TIMING_SOURCE_BACKFILL,
    }


def verify_video_sidecars(video_path: Path) -> dict[str, Any]:
    video_path = resolve_video_path(video_path)
    frames_path = frames_sidecar_path(video_path)
    meta_path = meta_sidecar_path(video_path)
    encoded = count_video_frames(video_path) if video_path.exists() else 0
    sidecar = load_frame_timestamps(video_path) if frames_path.exists() else {}
    meta = load_meta_sidecar(video_path)

    report: dict[str, Any] = {
        "video": str(video_path),
        "video_exists": video_path.exists(),
        "frames_sidecar_exists": frames_path.exists(),
        "meta_sidecar_exists": meta_path.exists(),
        "encoded_frame_count": encoded,
        "sidecar_frame_count": len(sidecar),
        "frames_in_sync": encoded > 0 and len(sidecar) == encoded,
        "timing_source": (meta or {}).get("timing_source"),
    }
    if not frames_path.exists():
        report["status"] = "missing_sidecar"
    elif encoded != len(sidecar):
        report["status"] = "frame_count_mismatch"
    elif (meta or {}).get("timing_source") == TIMING_SOURCE_BACKFILL:
        report["status"] = "approximate_timing"
    else:
        report["status"] = "ok"
    return report


def enrich_label_timing(label: dict[str, Any], video_path: Path) -> dict[str, Any]:
    """Add timing block and per-point timestamp_ms from sidecar when available."""
    timestamps = load_frame_timestamps(video_path)
    meta = load_meta_sidecar(video_path)

    if timestamps:
        timing_source = (meta or {}).get("timing_source", TIMING_SOURCE_CAPTURE)
    else:
        timing_source = TIMING_SOURCE_MISSING

    release_frame = label.get("release_frame")
    target = label.get("target") or {}
    cross_frame = target.get("cross_frame")

    release_ms = timestamp_ms(int(release_frame), timestamps) if release_frame is not None else None
    cross_ms = timestamp_ms(int(cross_frame), timestamps) if cross_frame is not None else None
    release_to_cross_ms = None
    if release_ms is not None and cross_ms is not None:
        release_to_cross_ms = round(cross_ms - release_ms, 3)

    for point in label.get("early_points", []):
        frame = point.get("frame")
        if frame is None:
            continue
        ms = timestamp_ms(int(frame), timestamps)
        if ms is not None:
            point["timestamp_ms"] = ms
        elif "timestamp_ms" in point:
            point.pop("timestamp_ms", None)

    if cross_ms is not None:
        target["cross_timestamp_ms"] = cross_ms
    elif "cross_timestamp_ms" in target:
        target.pop("cross_timestamp_ms", None)

    label["target"] = target
    label["timing"] = {
        "source": timing_source,
        "release_frame": release_frame,
        "cross_frame": cross_frame,
        "release_timestamp_ms": release_ms,
        "cross_timestamp_ms": cross_ms,
        "release_to_cross_ms": release_to_cross_ms,
        "sidecar_frame_count": len(timestamps),
    }
    if meta:
        label["timing"]["capture_fps_measured"] = meta.get("capture_fps_measured") or meta.get("actual_fps")
        label["timing"]["frames_in_sync"] = meta.get("frames_in_sync")
    return label

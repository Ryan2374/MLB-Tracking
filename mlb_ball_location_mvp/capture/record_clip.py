#!/usr/bin/env python3
"""Record a short pitch clip from a capture device.

This is a convenience tool. OBS/capture-card software is also fine as long as
clips are saved into data/raw/ and can be opened by OpenCV.

Each recording also writes timing sidecars next to the video:
- ``pitch_001.frames.jsonl`` per-frame timestamps (seconds from recording start)
- ``pitch_001.meta.json`` capture metadata (requested/actual FPS, frame count, size)

The writer FPS is set from a short warmup measurement so encoded MP4 frame counts
match the sidecar entries. Use frame numbers from the labeled video for now;
treat sidecar timestamps as diagnostic only until capture_fps is stable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int) and sys.platform == "win32":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, requested_fps: float) -> None:
    """Apply capture settings that improve live FPS on Windows USB devices."""
    if sys.platform == "win32":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, requested_fps)


def draw_text(img, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def sidecar_paths(video_path: Path) -> tuple[Path, Path]:
    stem = video_path.with_suffix("")
    return stem.with_suffix(".frames.jsonl"), stem.with_suffix(".meta.json")


def count_encoded_frames(video_path: Path) -> int:
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


def measure_capture_fps(cap: cv2.VideoCapture, warmup_frames: int = 45) -> float:
    """Read frames without writing to estimate the device's sustained capture rate."""
    start = time.perf_counter()
    count = 0
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    elapsed = time.perf_counter() - start
    if count == 0 or elapsed <= 0:
        return 30.0
    return count / elapsed


def sync_sidecars_to_encoded(
    frame_timestamps: list[dict[str, float | int]],
    encoded_frame_count: int,
) -> tuple[list[dict[str, float | int]], bool]:
    sidecar_count = len(frame_timestamps)
    if encoded_frame_count == sidecar_count:
        return frame_timestamps, True
    if encoded_frame_count < sidecar_count:
        print(
            f"WARNING: encoded MP4 has {encoded_frame_count} frames but sidecar has {sidecar_count}; "
            "trimming sidecar to match MP4."
        )
        return frame_timestamps[:encoded_frame_count], False
    print(
        f"WARNING: encoded MP4 has {encoded_frame_count} frames but sidecar has {sidecar_count}; "
        "keeping sidecar length (MP4 read may be inconsistent)."
    )
    return frame_timestamps, False


def write_sidecars(
    out_path: Path,
    *,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    frame_count: int,
    encoded_frame_count: int,
    width: int,
    height: int,
    actual_duration: float,
    frame_timestamps: list[dict[str, float | int]],
    frames_in_sync: bool,
) -> None:
    frames_path, meta_path = sidecar_paths(out_path)
    with frames_path.open("w", encoding="utf-8") as frames_file:
        for record in frame_timestamps:
            frames_file.write(json.dumps(record) + "\n")

    actual_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
    meta = {
        "video": str(out_path),
        "requested_fps": float(requested_fps),
        "writer_fps": round(writer_fps, 2),
        "capture_fps_measured": round(capture_fps_measured, 2),
        "actual_fps": round(actual_fps, 2),
        "frame_count": frame_count,
        "encoded_frame_count": encoded_frame_count,
        "sidecar_frame_count": len(frame_timestamps),
        "frames_in_sync": frames_in_sync,
        "width": width,
        "height": height,
        "duration_s": round(actual_duration, 3),
        "timing_source": "capture_sidecar",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved {out_path}")
    print(f"Saved {frames_path}")
    print(f"Saved {meta_path}")
    print(
        f"frames={frame_count} encoded={encoded_frame_count} "
        f"duration={actual_duration:.3f}s capture_fps={capture_fps_measured:.2f} "
        f"writer_fps={writer_fps:.2f} in_sync={frames_in_sync}"
    )
    if capture_fps_measured < requested_fps * 0.75:
        print(
            f"WARNING: capture measured {capture_fps_measured:.1f} fps vs requested {requested_fps:.1f}. "
            "Expect fewer ball positions per pitch; labels still use video frame numbers."
        )


def finalize_recording(
    out_path: Path,
    frame_timestamps: list[dict[str, float | int]],
    *,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    width: int,
    height: int,
    rec_start: float,
) -> None:
    encoded_frame_count = count_encoded_frames(out_path)
    frame_timestamps, frames_in_sync = sync_sidecars_to_encoded(frame_timestamps, encoded_frame_count)
    frame_count = len(frame_timestamps)
    actual_duration = time.perf_counter() - rec_start
    write_sidecars(
        out_path,
        requested_fps=requested_fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        frame_count=frame_count,
        encoded_frame_count=encoded_frame_count,
        width=width,
        height=height,
        actual_duration=actual_duration,
        frame_timestamps=frame_timestamps,
        frames_in_sync=frames_in_sync,
    )


def create_writer(out_path: Path, writer_fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, writer_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")
    return writer


def append_recorded_frame(
    writer: cv2.VideoWriter,
    frame,
    frame_timestamps: list[dict[str, float | int]],
    rec_start: float,
) -> None:
    writer.write(frame)
    elapsed = time.perf_counter() - rec_start
    frame_timestamps.append({"frame": len(frame_timestamps), "timestamp": round(elapsed, 6)})


def next_clip_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for path in out_dir.glob("pitch_*.mp4"):
        suffix = path.stem.removeprefix("pitch_")
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return out_dir / f"pitch_{max_num + 1:03d}.mp4"


def run_interactive_preview(
    cap: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    clip_seconds: float,
    clip_dir: Path,
) -> None:
    frame_count = 0
    preview_start = time.perf_counter()

    recording = False
    writer: cv2.VideoWriter | None = None
    out_path: Path | None = None
    rec_start = 0.0
    frame_timestamps: list[dict[str, float | int]] = []

    print(f"Interactive preview ready. Press r to record a {clip_seconds:.0f}s clip, q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        preview = frame.copy()
        elapsed = time.perf_counter() - preview_start
        fps_estimate = frame_count / elapsed if elapsed > 0 else 0.0

        if recording and writer is not None and out_path is not None:
            append_recorded_frame(writer, frame, frame_timestamps, rec_start)
            rec_elapsed = time.perf_counter() - rec_start
            draw_text(
                preview,
                f"REC {rec_elapsed:.1f}s / {clip_seconds:.1f}s -> {out_path.name}",
                20,
                40,
            )
            if rec_elapsed >= clip_seconds:
                writer.release()
                writer = None
                finalize_recording(
                    out_path,
                    frame_timestamps,
                    requested_fps=requested_fps,
                    writer_fps=writer_fps,
                    capture_fps_measured=capture_fps_measured,
                    width=width,
                    height=height,
                    rec_start=rec_start,
                )
                recording = False
                out_path = None
                frame_timestamps = []
        else:
            draw_text(
                preview,
                f"preview fps={fps_estimate:.1f}  |  r=record {clip_seconds:.0f}s  |  q=quit",
                20,
                40,
            )

        cv2.imshow("record_clip", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if recording and writer is not None and out_path is not None:
                writer.release()
                finalize_recording(
                    out_path,
                    frame_timestamps,
                    requested_fps=requested_fps,
                    writer_fps=writer_fps,
                    capture_fps_measured=capture_fps_measured,
                    width=width,
                    height=height,
                    rec_start=rec_start,
                )
            break
        if key == ord("r") and not recording:
            out_path = next_clip_path(clip_dir)
            writer = create_writer(out_path, writer_fps, width, height)
            recording = True
            rec_start = time.perf_counter()
            frame_timestamps = []
            append_recorded_frame(writer, frame, frame_timestamps, rec_start)
            print(f"Recording {clip_seconds:.1f}s to {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Preview ended. frames={frame_count} avg_fps={fps_estimate:.2f}")


def record_clip(
    cap: cv2.VideoCapture,
    out_path: Path,
    *,
    seconds: float,
    requested_fps: float,
    writer_fps: float,
    capture_fps_measured: float,
    width: int,
    height: int,
    first_frame,
    preview: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_writer(out_path, writer_fps, width, height)

    rec_start = time.perf_counter()
    frame_timestamps: list[dict[str, float | int]] = []
    append_recorded_frame(writer, first_frame, frame_timestamps, rec_start)

    print(f"Recording {seconds:.1f}s to {out_path} (writer_fps={writer_fps:.2f})")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed = time.perf_counter() - rec_start
        if elapsed >= seconds:
            break

        append_recorded_frame(writer, frame, frame_timestamps, rec_start)

        if preview:
            preview_img = frame.copy()
            fps_estimate = len(frame_timestamps) / elapsed if elapsed > 0 else 0.0
            draw_text(
                preview_img,
                f"REC frame={len(frame_timestamps) - 1} elapsed={elapsed:.2f}s fps={fps_estimate:.1f}",
                20,
                40,
            )
            cv2.imshow("record_clip", preview_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    writer.release()
    if preview:
        cv2.destroyAllWindows()

    finalize_recording(
        out_path,
        frame_timestamps,
        requested_fps=requested_fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        width=width,
        height=height,
        rec_start=rec_start,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Capture device index or video source path. Default: 0")
    parser.add_argument("--out", help="Output video path, e.g. data/raw/pitch_001.mp4")
    parser.add_argument("--seconds", type=float, default=8.0, help="Recording duration")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--warmup-frames", type=int, default=45, help="Frames to measure capture FPS before recording")
    parser.add_argument("--preview", action="store_true", help="Show a preview window while recording")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Show live preview. Press r to record a clip, q to quit.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=6.0,
        help="Clip length when pressing r in preview mode. Default: 6",
    )
    parser.add_argument(
        "--clip-dir",
        default="data/raw",
        help="Output folder for clips recorded with r in preview mode. Default: data/raw",
    )
    args = parser.parse_args()

    if not args.preview_only and not args.out:
        parser.error("--out is required unless --preview-only is set")

    source = int(args.device) if str(args.device).isdigit() else args.device
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open capture source: {args.device}")

    configure_capture(cap, args.width, args.height, args.fps)

    capture_fps_measured = measure_capture_fps(cap, warmup_frames=args.warmup_frames)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Capture source opened but returned no frame after warmup.")

    height, width = frame.shape[:2]
    writer_fps = max(1.0, round(capture_fps_measured, 2))
    print(
        f"Capture: {width}x{height} device={args.device} "
        f"requested_fps={args.fps:.1f} measured_fps={capture_fps_measured:.2f} writer_fps={writer_fps:.2f}"
    )

    if args.preview_only:
        run_interactive_preview(
            cap,
            width=width,
            height=height,
            requested_fps=args.fps,
            writer_fps=writer_fps,
            capture_fps_measured=capture_fps_measured,
            clip_seconds=args.clip_seconds,
            clip_dir=Path(args.clip_dir),
        )
        return

    record_clip(
        cap,
        Path(args.out),
        seconds=args.seconds,
        requested_fps=args.fps,
        writer_fps=writer_fps,
        capture_fps_measured=capture_fps_measured,
        width=width,
        height=height,
        first_frame=frame,
        preview=args.preview,
    )
    cap.release()


if __name__ == "__main__":
    main()

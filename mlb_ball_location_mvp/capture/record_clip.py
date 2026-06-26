#!/usr/bin/env python3
"""Record a short pitch clip from a capture device.

This is a convenience tool. OBS/capture-card software is also fine as long as
clips are saved into data/raw/ and can be opened by OpenCV.

Each recording also writes timing sidecars next to the video:
- ``pitch_001.frames.jsonl`` per-frame timestamps (seconds from recording start)
- ``pitch_001.meta.json`` capture metadata (requested/actual FPS, frame count, size)
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


def draw_text(img, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def sidecar_paths(video_path: Path) -> tuple[Path, Path]:
    stem = video_path.with_suffix("")
    return stem.with_suffix(".frames.jsonl"), stem.with_suffix(".meta.json")


def next_clip_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for path in out_dir.glob("pitch_*.mp4"):
        suffix = path.stem.removeprefix("pitch_")
        if suffix.isdigit():
            max_num = max(max_num, int(suffix))
    return out_dir / f"pitch_{max_num + 1:03d}.mp4"


def write_sidecars(
    out_path: Path,
    *,
    requested_fps: float,
    frame_count: int,
    width: int,
    height: int,
    actual_duration: float,
    frame_timestamps: list[dict[str, float | int]],
) -> None:
    frames_path, meta_path = sidecar_paths(out_path)
    with frames_path.open("w", encoding="utf-8") as frames_file:
        for record in frame_timestamps:
            frames_file.write(json.dumps(record) + "\n")

    actual_fps = frame_count / actual_duration if actual_duration > 0 else 0.0
    meta = {
        "video": str(out_path),
        "requested_fps": float(requested_fps),
        "actual_fps": round(actual_fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": round(actual_duration, 3),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Saved {out_path}")
    print(f"Saved {frames_path}")
    print(f"Saved {meta_path}")
    print(f"frames={frame_count} duration={actual_duration:.3f}s actual_fps={actual_fps:.2f}")


def finish_recording(
    writer: cv2.VideoWriter | None,
    out_path: Path | None,
    rec_start: float,
    frame_timestamps: list[dict[str, float | int]],
    *,
    requested_fps: float,
    width: int,
    height: int,
) -> None:
    if writer is None or out_path is None:
        return
    writer.release()
    frame_count = len(frame_timestamps)
    actual_duration = time.perf_counter() - rec_start
    write_sidecars(
        out_path,
        requested_fps=requested_fps,
        frame_count=frame_count,
        width=width,
        height=height,
        actual_duration=actual_duration,
        frame_timestamps=frame_timestamps,
    )


def run_interactive_preview(
    cap: cv2.VideoCapture,
    *,
    width: int,
    height: int,
    fps: float,
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

        if recording and writer is not None:
            writer.write(frame)
            rec_elapsed = time.perf_counter() - rec_start
            frame_timestamps.append({"frame": len(frame_timestamps), "timestamp": round(rec_elapsed, 6)})
            draw_text(
                preview,
                f"REC {rec_elapsed:.1f}s / {clip_seconds:.1f}s -> {out_path.name}",
                20,
                40,
            )
            if rec_elapsed >= clip_seconds:
                finish_recording(
                    writer,
                    out_path,
                    rec_start,
                    frame_timestamps,
                    requested_fps=fps,
                    width=width,
                    height=height,
                )
                writer = None
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
            if recording:
                finish_recording(
                    writer,
                    out_path,
                    rec_start,
                    frame_timestamps,
                    requested_fps=fps,
                    width=width,
                    height=height,
                )
            break
        if key == ord("r") and not recording:
            out_path = next_clip_path(clip_dir)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"Could not create output video: {out_path}")
                writer = None
                out_path = None
                continue
            recording = True
            rec_start = time.perf_counter()
            frame_timestamps = []
            writer.write(frame)
            frame_timestamps.append({"frame": 0, "timestamp": 0.0})
            print(f"Recording {clip_seconds:.1f}s to {out_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Preview ended. frames={frame_count} avg_fps={fps_estimate:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Capture device index or video source path. Default: 0")
    parser.add_argument("--out", help="Output video path, e.g. data/raw/pitch_001.mp4")
    parser.add_argument("--seconds", type=float, default=8.0, help="Recording duration")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Capture source opened but returned no frame.")

    height, width = frame.shape[:2]
    print(f"Capture: {width}x{height} device={args.device}")

    if args.preview_only:
        run_interactive_preview(
            cap,
            width=width,
            height=height,
            fps=args.fps,
            clip_seconds=args.clip_seconds,
            clip_dir=Path(args.clip_dir),
        )
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")

    start = time.perf_counter()
    frame_count = 0
    fps_estimate = 0.0
    frame_timestamps: list[dict[str, float | int]] = []

    print(f"Recording {args.seconds:.1f}s to {out_path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed = time.perf_counter() - start
        frame_count += 1
        if elapsed > 0:
            fps_estimate = frame_count / elapsed

        writer.write(frame)
        frame_timestamps.append({"frame": frame_count - 1, "timestamp": round(elapsed, 6)})

        if args.preview:
            preview = frame.copy()
            draw_text(
                preview,
                f"REC frame={frame_count - 1} elapsed={elapsed:.2f}s fps={fps_estimate:.1f}",
                20,
                40,
            )
            cv2.imshow("record_clip", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if elapsed >= args.seconds:
            break

    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    actual_duration = time.perf_counter() - start
    write_sidecars(
        out_path,
        requested_fps=args.fps,
        frame_count=frame_count,
        width=width,
        height=height,
        actual_duration=actual_duration,
        frame_timestamps=frame_timestamps,
    )


if __name__ == "__main__":
    main()

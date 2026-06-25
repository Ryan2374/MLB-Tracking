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
import time
from pathlib import Path

import cv2


def draw_text(img, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def sidecar_paths(video_path: Path) -> tuple[Path, Path]:
    stem = video_path.with_suffix("")
    return stem.with_suffix(".frames.jsonl"), stem.with_suffix(".meta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Capture device index or video source path. Default: 0")
    parser.add_argument("--out", required=True, help="Output video path, e.g. data/raw/pitch_001.mp4")
    parser.add_argument("--seconds", type=float, default=8.0, help="Recording duration")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--preview", action="store_true", help="Show a preview window while recording")
    args = parser.parse_args()

    source = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open capture source: {args.device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Capture source opened but returned no frame.")

    height, width = frame.shape[:2]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_path, meta_path = sidecar_paths(out_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")

    start = time.perf_counter()
    frame_count = 0
    fps_estimate = 0.0
    frame_timestamps: list[dict[str, float | int]] = []

    print(f"Recording {args.seconds:.1f}s to {out_path}")
    with frames_path.open("w", encoding="utf-8") as frames_file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            elapsed = time.perf_counter() - start
            frame_count += 1
            if elapsed > 0:
                fps_estimate = frame_count / elapsed

            writer.write(frame)
            record = {"frame": frame_count - 1, "timestamp": round(elapsed, 6)}
            frame_timestamps.append(record)
            frames_file.write(json.dumps(record) + "\n")

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
    actual_fps = frame_count / actual_duration if actual_duration > 0 else 0.0

    meta = {
        "video": str(out_path),
        "requested_fps": float(args.fps),
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


if __name__ == "__main__":
    main()

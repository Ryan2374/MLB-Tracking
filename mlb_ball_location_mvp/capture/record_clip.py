#!/usr/bin/env python3
"""Record a short pitch clip from a capture device.

This is a convenience tool. OBS/capture-card software is also fine as long as
clips are saved into data/raw/ and can be opened by OpenCV.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def draw_text(img, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {out_path}")

    start = time.perf_counter()
    frame_count = 0
    fps_estimate = 0.0

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

        if args.preview:
            preview = frame.copy()
            draw_text(preview, f"REC frame={frame_count} elapsed={elapsed:.2f}s fps={fps_estimate:.1f}", 20, 40)
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
    print(f"Saved {out_path}")
    print(f"frames={frame_count} duration={actual_duration:.3f}s actual_fps={actual_fps:.2f}")


if __name__ == "__main__":
    main()

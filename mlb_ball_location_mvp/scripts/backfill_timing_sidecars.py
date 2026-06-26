#!/usr/bin/env python3
"""Backfill .frames.jsonl and .meta.json for existing MP4 clips.

Capture-time sidecars (from record_clip.py) are exact wall-clock timestamps.
This script creates uniform frame/fps estimates when sidecars are missing.
Those backfilled timestamps are useful but marked timing_source=backfill_uniform_fps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from timing.sidecar import TIMING_SOURCE_BACKFILL, write_uniform_sidecars


def infer_fps(video_path: Path, label_path: Path | None) -> float:
    if label_path and label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            label = json.load(f)
        fps = label.get("fps")
        if fps is not None and float(fps) > 0:
            return float(fps)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps > 0:
        return fps
    raise ValueError(f"Could not infer fps for {video_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with pitch_XXX.mp4 files")
    parser.add_argument("--labels-dir", default="data/labels", help="Optional labels dir for fps lookup")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing backfilled sidecars")
    parser.add_argument("--video", help="Backfill one video instead of all in --raw-dir")
    args = parser.parse_args()

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(Path(args.raw_dir).glob("pitch_*.mp4"))

    if not videos:
        raise SystemExit(f"No videos found in {args.raw_dir}")

    labels_dir = Path(args.labels_dir)
    for video in videos:
        label_path = labels_dir / f"{video.stem}.json"
        try:
            fps = infer_fps(video, label_path if label_path.exists() else None)
            result = write_uniform_sidecars(video, fps=fps, overwrite=args.overwrite)
            print(f"{video.name}: {result['status']} ({result.get('frame_count', '?')} frames @ {fps:.2f} fps)")
            if result["status"] == "written":
                print(f"  timing_source={TIMING_SOURCE_BACKFILL} (uniform estimate, not capture wall-clock)")
        except Exception as exc:  # noqa: BLE001
            print(f"{video.name}: ERROR {exc}")


if __name__ == "__main__":
    main()

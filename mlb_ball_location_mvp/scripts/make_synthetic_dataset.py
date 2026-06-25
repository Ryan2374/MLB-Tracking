#!/usr/bin/env python3
"""Create synthetic labels to smoke-test the MVP pipeline.

This does not generate video. It creates label JSON files with plausible
screen-space trajectories so prediction/evaluation can be tested immediately.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from coords.calibration import full_frame_metadata


def point_at(frame: int, start_frame: int, x0: float, y0: float, vx: float, vy: float, ax: float, ay: float):
    t = frame - start_frame
    x = x0 + vx * t + 0.5 * ax * t * t
    y = y0 + vy * t + 0.5 * ay * t * t
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output labels directory")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early-points", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.05, help="Uniform label noise in pixels")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--frame-height", type=int, default=1080)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.count + 1):
        pitch_id = f"synthetic_{i:03d}"
        release_frame = 100 + rng.randint(-3, 3)
        first_visible = release_frame + 1
        cross_frame = release_frame + rng.randint(23, 29)

        x0 = rng.uniform(390, 470)
        y0 = rng.uniform(180, 250)
        vx = rng.uniform(7.5, 12.5)
        vy = rng.uniform(5.0, 10.5)
        ax = rng.uniform(-0.08, 0.08)
        ay = rng.uniform(0.04, 0.20)

        early = []
        for j in range(args.early_points):
            frame = first_visible + j
            x, y = point_at(frame, first_visible, x0, y0, vx, vy, ax, ay)
            # Add small label noise.
            x += rng.uniform(-args.noise, args.noise)
            y += rng.uniform(-args.noise, args.noise)
            early.append({"frame": frame, "x": round(x, 3), "y": round(y, 3)})

        cross_x, cross_y = point_at(cross_frame, first_visible, x0, y0, vx, vy, ax, ay)
        label = {
            **full_frame_metadata(args.frame_width, args.frame_height),
            "pitch_id": pitch_id,
            "video": None,
            "fps": 60.0,
            "release_frame": release_frame,
            "early_points": early,
            "target": {
                "cross_frame": cross_frame,
                "cross_x": round(cross_x, 3),
                "cross_y": round(cross_y, 3),
            },
        }

        path = out_dir / f"{pitch_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(label, f, indent=2)
            f.write("\n")

    print(f"Wrote {args.count} synthetic labels to {out_dir}")


if __name__ == "__main__":
    main()

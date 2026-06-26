#!/usr/bin/env python3
"""Verify capture timing sidecars for pitch videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from timing.sidecar import enrich_label_timing, verify_video_sidecars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--enrich-labels", action="store_true", help="Write timing fields into label JSON files")
    parser.add_argument("--out", help="Optional JSON report path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    labels_dir = Path(args.labels_dir)
    videos = sorted(raw_dir.glob("pitch_*.mp4"))
    reports = []

    for video in videos:
        report = verify_video_sidecars(video)
        label_path = labels_dir / f"{video.stem}.json"
        report["label_exists"] = label_path.exists()
        if args.enrich_labels and label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                label = json.load(f)
            enrich_label_timing(label, video)
            with label_path.open("w", encoding="utf-8") as f:
                json.dump(label, f, indent=2)
                f.write("\n")
            report["label_enriched"] = True
            timing = label.get("timing", {})
            report["release_to_cross_ms"] = timing.get("release_to_cross_ms")
        reports.append(report)
        print(
            f"{video.stem}: {report['status']} "
            f"(mp4={report['encoded_frame_count']} sidecar={report['sidecar_frame_count']} "
            f"source={report.get('timing_source')})"
        )

    payload = {"videos": reports, "count": len(reports)}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

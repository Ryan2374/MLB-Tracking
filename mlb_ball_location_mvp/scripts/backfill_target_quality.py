#!/usr/bin/env python3
"""Backfill target_quality fields on legacy pitch labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labeling.target_derivation import finalize_label_target, normalize_label_target_fields


def backfill_label(path: Path, *, write: bool) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    had_quality = bool(data.get("target_quality"))
    normalize_label_target_fields(data)
    finalize_label_target(data)
    tq = data.get("target_quality") or {}
    if write:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    return {
        "pitch_id": data.get("pitch_id") or path.stem,
        "had_target_quality": had_quality,
        "target_source": tq.get("target_source"),
        "confidence": tq.get("confidence"),
        "uncertainty_px": tq.get("uncertainty_px"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write files")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    paths = sorted(labels_dir.glob("pitch_*.json"))
    if not paths:
        raise SystemExit(f"No pitch_*.json files in {labels_dir}")

    for path in paths:
        report = backfill_label(path, write=not args.dry_run)
        action = "would update" if args.dry_run else "updated"
        print(
            f"{report['pitch_id']}: {action} "
            f"source={report['target_source']} confidence={report['confidence']} "
            f"uncertainty={report['uncertainty_px']}px"
        )


if __name__ == "__main__":
    main()

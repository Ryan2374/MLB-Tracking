#!/usr/bin/env python3
"""Auto-bracket relabel for high-error pitches using adjacent early_points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labeling.target_derivation import derive_target_from_bracket, finalize_label_target


def load_predictions(pred_path: Path) -> list[dict]:
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    return list(payload.get("predictions") or [])


def bracket_from_early_points(
    early_points: list[dict],
    cross_frame: float,
    target: dict,
) -> dict | None:
    points = sorted(
        (p for p in early_points if p.get("frame") is not None and p.get("x") is not None),
        key=lambda p: int(p["frame"]),
    )
    if not points:
        return None

    cf = float(cross_frame)
    before = None
    after = None
    for point in points:
        frame = int(point["frame"])
        if frame < cf:
            before = point
        elif frame > cf and after is None:
            after = point
        elif frame == cf and after is None:
            after = point

    if before is None:
        before = points[0]
    if after is None:
        if target.get("cross_x") is None or target.get("cross_y") is None:
            return None
        after = {
            "frame": int(round(cf)),
            "x": float(target["cross_x"]),
            "y": float(target["cross_y"]),
        }
        if target.get("cross_timestamp_ms") is not None:
            after["timestamp_ms"] = float(target["cross_timestamp_ms"])

    if int(before["frame"]) >= int(after["frame"]):
        return None

    span = float(int(after["frame"]) - int(before["frame"]))
    alpha = max(0.0, min(1.0, (cf - int(before["frame"])) / span))
    return {
        "before": {
            "frame": int(before["frame"]),
            "x": float(before["x"]),
            "y": float(before["y"]),
            **({"timestamp_ms": before["timestamp_ms"]} if before.get("timestamp_ms") is not None else {}),
        },
        "after": {
            "frame": int(after["frame"]),
            "x": float(after["x"]),
            "y": float(after["y"]),
            **({"timestamp_ms": after["timestamp_ms"]} if after.get("timestamp_ms") is not None else {}),
        },
        "alpha": round(alpha, 4),
    }


def relabel_label(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    target = data.get("target") or {}
    cross_frame = target.get("cross_frame")
    if cross_frame is None:
        return {"pitch_id": path.stem, "status": "skipped", "reason": "no cross_frame"}

    bracket = bracket_from_early_points(data.get("early_points") or [], float(cross_frame), target)
    if bracket is None:
        return {"pitch_id": path.stem, "status": "skipped", "reason": "no adjacent bracket frames"}

    data["target_bracket"] = bracket
    target, target_quality = derive_target_from_bracket(bracket, alpha=bracket["alpha"])
    data["target"].update(target)
    data["target_quality"] = target_quality
    finalize_label_target(data)

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "pitch_id": path.stem,
        "status": "updated",
        "target_source": target_quality.get("target_source"),
        "confidence": target_quality.get("confidence"),
        "uncertainty_px": target_quality.get("uncertainty_px"),
        "alpha": bracket["alpha"],
        "before_frame": bracket["before"]["frame"],
        "after_frame": bracket["after"]["frame"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="data/predictions/fastball_compact_ridge_n7.json",
        help="Prediction JSON used to rank outliers.",
    )
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--top", type=int, default=15, help="Number of highest-error pitches to relabel.")
    parser.add_argument(
        "--pitch-ids",
        nargs="*",
        help="Optional explicit pitch IDs instead of top-N by error.",
    )
    parser.add_argument(
        "--include-untrusted",
        action="store_true",
        help="Include missing_video_unverified / unreliable_unverified pitches.",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if args.pitch_ids:
        pitch_ids = list(args.pitch_ids)
    else:
        preds = load_predictions(Path(args.predictions))
        if not args.include_untrusted:
            preds = [p for p in preds if p.get("review_status", "verified") == "verified"]
        preds.sort(key=lambda p: float(p.get("error_px", 0.0)), reverse=True)
        pitch_ids = [str(p["pitch_id"]) for p in preds[: args.top]]

    print(f"Bracket relabel for {len(pitch_ids)} pitches:")
    updated = 0
    skipped = 0
    for pitch_id in pitch_ids:
        path = labels_dir / f"{pitch_id}.json"
        if not path.is_file():
            print(f"  {pitch_id}: missing label file")
            skipped += 1
            continue
        report = relabel_label(path)
        if report["status"] == "updated":
            updated += 1
            print(
                f"  {pitch_id}: {report['target_source']} "
                f"frames {report['before_frame']}-{report['after_frame']} alpha={report['alpha']:.3f} "
                f"confidence={report['confidence']} uncertainty~{report['uncertainty_px']:.1f}px"
            )
        else:
            skipped += 1
            print(f"  {pitch_id}: skipped ({report.get('reason')})")

    print(f"Done: {updated} updated, {skipped} skipped")


if __name__ == "__main__":
    main()

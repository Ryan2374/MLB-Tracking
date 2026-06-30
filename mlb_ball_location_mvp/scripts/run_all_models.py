#!/usr/bin/env python3
"""Run every evaluation model spec and compile JSON + HTML reports.

Designed to run from an external Windows terminal (or packaged .exe) while
Cursor stays open for other work. Add future models in prediction/model_specs.py.

Example:
  python scripts/run_all_models.py
  python scripts/run_all_models.py --project-dir .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from prediction.model_specs import EVAL_RUN_SPECS
from prediction.predict_location import run_batch, write_json
from scripts.export_fastball_predictions import build_export
from scripts.render_eval_report import write_html_report


def resolve_project_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("MLB_EVAL_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    if getattr(sys, "frozen", False):
        cfg = Path(sys.executable).parent / "eval_config.json"
        if cfg.is_file():
            raw = json.loads(cfg.read_text(encoding="utf-8"))
            project = raw.get("project_dir")
            if project:
                return Path(project).resolve()
    return REPO


def validate_label_set(labels_dir: Path) -> int:
    paths = sorted(labels_dir.glob("pitch_*.json"))
    if not paths:
        raise SystemExit(f"No pitch_*.json labels found in {labels_dir}")
    synthetic = list(labels_dir.glob("synthetic_*.json"))
    if synthetic:
        raise SystemExit(
            f"Found {len(synthetic)} synthetic_*.json in {labels_dir}; "
            "move them out before full eval."
        )
    return len(paths)


def build_run_args(
    spec: dict,
    labels_dir: Path,
    out_path: Path,
) -> Namespace:
    calibration = spec.get("calibration")
    return Namespace(
        labels=str(labels_dir),
        label=None,
        n_points=int(spec["n_points"]),
        method=str(spec["method"]),
        calibration=calibration if calibration else "linear",
        alpha=10.0,
        ridge_cv=True,
        no_ridge_outlier_guard=False,
        use_actual_cross_frame=False,
        poly_degree=int(spec.get("poly_degree", 1)),
        out=str(out_path),
    )


def run_spec(spec: dict, labels_dir: Path, predictions_dir: Path, log_path: Path) -> dict:
    out_path = predictions_dir / spec["source_file"]
    started = time.time()
    line = f"[{datetime.now().isoformat(timespec='seconds')}] START {spec['run_id']} -> {out_path.name}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(line)
        log.flush()
        print(line.strip())

        args = build_run_args(spec, labels_dir, out_path)
        payload = run_batch(args)
        write_json(out_path, payload)

        summary = payload.get("summary") or {}
        count = summary.get("count")
        median = summary.get("median_error_px")
        elapsed = time.time() - started
        done = (
            f"DONE {spec['run_id']}: count={count} median_error_px={median} "
            f"elapsed_s={elapsed:.1f}\n"
        )
        log.write(done)
        log.flush()
        print(done.strip())

        if count and any(str(p.get("pitch_id", "")).startswith("synthetic_") for p in payload.get("predictions", [])):
            raise SystemExit(f"{spec['run_id']} included synthetic pitches — check labels_dir")

    return {
        "run_id": spec["run_id"],
        "source_file": spec["source_file"],
        "count": count,
        "median_error_px": median,
        "elapsed_s": round(elapsed, 1),
    }


def compile_reports(
    project_dir: Path,
    predictions_dir: Path,
    labels_dir: Path,
    json_out: Path,
    html_out: Path,
) -> None:
    payload = build_export(predictions_dir, labels_dir)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    write_html_report(payload, html_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {html_out}")
    if payload.get("leaderboard_trusted", {}).get("overall"):
        top = payload["leaderboard_trusted"]["overall"][0]
        print(f"Trusted leaderboard #1: {top['run_id']} (avg rank {top['average_rank']:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        type=Path,
        help="Path to mlb_ball_location_mvp inner project root (default: repo or MLB_EVAL_PROJECT_DIR).",
    )
    parser.add_argument("--labels-dir", type=Path, help="Default: <project-dir>/data/labels")
    parser.add_argument("--predictions-dir", type=Path, help="Default: <project-dir>/data/predictions")
    parser.add_argument(
        "--json-out",
        type=Path,
        help='Default: <predictions-dir>/Fastball predictions.json',
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        help='Default: <predictions-dir>/Fastball predictions.html',
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Append progress log (default: <predictions-dir>/full_eval_<timestamp>.log)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Run models only; do not bundle JSON/HTML.",
    )
    args = parser.parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    os.chdir(project_dir)

    labels_dir = (args.labels_dir or project_dir / "data" / "labels").resolve()
    predictions_dir = (args.predictions_dir or project_dir / "data" / "predictions").resolve()
    json_out = (args.json_out or predictions_dir / "Fastball predictions.json").resolve()
    html_out = (args.html_out or predictions_dir / "Fastball predictions.html").resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (args.log or predictions_dir / f"full_eval_{stamp}.log").resolve()

    pitch_count = validate_label_set(labels_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project: {project_dir}")
    print(f"Labels:  {labels_dir} ({pitch_count} pitch files)")
    print(f"Output:  {predictions_dir}")
    print(f"Models:  {len(EVAL_RUN_SPECS)} runs from prediction/model_specs.py")
    print(f"Log:     {log_path}")

    results: list[dict] = []
    total_start = time.time()
    for spec in EVAL_RUN_SPECS:
        results.append(run_spec(spec, labels_dir, predictions_dir, log_path))

    if not args.skip_compile:
        print("\n=== Compiling reports ===")
        compile_reports(project_dir, predictions_dir, labels_dir, json_out, html_out)

    total_elapsed = time.time() - total_start
    print(f"\nAll {len(results)} models finished in {total_elapsed / 60:.1f} min")
    for row in results:
        print(f"  {row['run_id']}: median={row.get('median_error_px')} px ({row['elapsed_s']}s)")


if __name__ == "__main__":
    main()

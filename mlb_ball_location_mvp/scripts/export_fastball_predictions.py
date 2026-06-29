#!/usr/bin/env python3
"""Bundle fastball prediction JSON outputs into one standalone review file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prediction.predict_location import (
    ACCURACY_CHAMPION_METHOD,
    ACCURACY_CHAMPION_N_POINTS,
    EARLY_CHAMPION_METHOD,
    EARLY_CHAMPION_N_POINTS,
    STABILITY_CHAMPION_CALIBRATION,
    STABILITY_CHAMPION_METHOD,
    STABILITY_CHAMPION_N_POINTS,
    UNTRUSTED_REVIEW_STATUSES,
    build_leaderboard,
    summarize,
    summarize_by_location,
)
from prediction.predict_location import Prediction, filter_trusted_predictions

EXPORT_SCHEMA_VERSION = 5

STABILITY_CHAMPION_RUN_ID = "calibrated_velocity_linear_n7"
ACCURACY_CHAMPION_RUN_ID = "compact_ridge_calibrated_n7"
EARLY_CHAMPION_RUN_ID = "compact_ridge_calibrated_n5"

# Backward-compatible alias used to mark the stability run in console output.
CHAMPION_RUN_ID = STABILITY_CHAMPION_RUN_ID

CHAMPION_ROLES: dict[str, dict] = {
    "accuracy_champion": {
        "run_id": ACCURACY_CHAMPION_RUN_ID,
        "method": ACCURACY_CHAMPION_METHOD,
        "n_points": ACCURACY_CHAMPION_N_POINTS,
        "calibration": None,
        "purpose": "Best trusted median error; use for offline eval and model development.",
    },
    "stability_champion": {
        "run_id": STABILITY_CHAMPION_RUN_ID,
        "method": STABILITY_CHAMPION_METHOD,
        "n_points": STABILITY_CHAMPION_N_POINTS,
        "calibration": STABILITY_CHAMPION_CALIBRATION,
        "purpose": "Safest fallback with fewer extreme misses; default live prediction path.",
    },
    "early_champion": {
        "run_id": EARLY_CHAMPION_RUN_ID,
        "method": EARLY_CHAMPION_METHOD,
        "n_points": EARLY_CHAMPION_N_POINTS,
        "calibration": None,
        "purpose": "Best balance when fewer early ball points are available.",
    },
}

RUN_SPECS: list[dict] = [
    {
        "run_id": "velocity_n3",
        "source_file": "eval_n3.json",
        "method": "velocity",
        "n_points": 3,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "velocity_n5",
        "source_file": "fastball_velocity_n5.json",
        "method": "velocity",
        "n_points": 5,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "velocity_n7",
        "source_file": "fastball_velocity_n7.json",
        "method": "velocity",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "ridge_n5",
        "source_file": "fastball_ridge_n5.json",
        "method": "ridge",
        "n_points": 5,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "ridge_n7",
        "source_file": "fastball_ridge_n7.json",
        "method": "ridge",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "calibrated_velocity_median_n7",
        "source_file": "fastball_calib_median_n7.json",
        "method": "calibrated_velocity",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": "median",
    },
    {
        "run_id": CHAMPION_RUN_ID,
        "source_file": "fastball_calib_linear_n7.json",
        "method": "calibrated_velocity",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": "linear",
    },
    {
        "run_id": "calibrated_velocity_linear_n5",
        "source_file": "fastball_calib_linear_n5.json",
        "method": "calibrated_velocity",
        "n_points": 5,
        "poly_degree": 1,
        "calibration": "linear",
        "optional": True,
    },
    {
        "run_id": "calibrated_velocity_linear_n3",
        "source_file": "fastball_calib_linear_n3.json",
        "method": "calibrated_velocity",
        "n_points": 3,
        "poly_degree": 1,
        "calibration": "linear",
        "optional": True,
    },
    {
        "run_id": "compact_ridge_calibrated_n5",
        "source_file": "fastball_compact_ridge_n5.json",
        "method": "compact_ridge_calibrated",
        "n_points": 5,
        "poly_degree": 1,
        "calibration": None,
        "optional": True,
    },
    {
        "run_id": "compact_ridge_calibrated_n7",
        "source_file": "fastball_compact_ridge_n7.json",
        "method": "compact_ridge_calibrated",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": None,
        "optional": True,
    },
]


def load_run(pred_dir: Path, spec: dict) -> dict:
    path = pred_dir / spec["source_file"]
    if not path.is_file():
        if spec.get("optional"):
            raise FileNotFoundError(f"optional:{spec['run_id']}")
        raise FileNotFoundError(f"Missing prediction file: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "run_id": spec["run_id"],
        "source_file": spec["source_file"],
        "method": spec["method"],
        "n_points": spec["n_points"],
        "poly_degree": spec.get("poly_degree", payload.get("poly_degree")),
        "calibration": spec.get("calibration", payload.get("calibration")),
        "mode": payload.get("mode"),
        "summary": payload.get("summary"),
        "summary_by_location": payload.get("summary_by_location"),
        "predictions": payload.get("predictions", []),
    }


def load_review_status_map(labels_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not labels_dir.is_dir():
        return mapping
    for path in sorted(labels_dir.glob("pitch_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pitch_id = str(raw.get("pitch_id") or path.stem)
        mapping[pitch_id] = str(raw.get("review_status") or "verified")
    return mapping


def predictions_from_run(run: dict, review_status_map: dict[str, str]) -> list[Prediction]:
    preds: list[Prediction] = []
    for row in run.get("predictions") or []:
        actual = row["actual_cross"]
        pitch_id = row["pitch_id"]
        review_status = str(
            row.get("review_status") or review_status_map.get(pitch_id) or "verified"
        )
        preds.append(
            Prediction(
                pitch_id=pitch_id,
                method=row.get("method", run.get("method", "")),
                pred_x=float(row["predicted_cross"]["x"]),
                pred_y=float(row["predicted_cross"]["y"]),
                actual_x=float(actual["x"]),
                actual_y=float(actual["y"]),
                error_px=float(row["error_px"]),
                x_error_px=float(row["x_error_px"]),
                y_error_px=float(row["y_error_px"]),
                n_points=int(row.get("n_points", run.get("n_points", 0))),
                details=row.get("details") or {},
                review_status=review_status,
            )
        )
    return preds


def trusted_summary_for_run(run: dict, review_status_map: dict[str, str]) -> dict:
    trusted = filter_trusted_predictions(predictions_from_run(run, review_status_map))
    return summarize(trusted)


def trusted_location_summary_for_run(run: dict, review_status_map: dict[str, str]) -> dict:
    trusted = filter_trusted_predictions(predictions_from_run(run, review_status_map))
    return summarize_by_location(trusted)


def comparison_row(run: dict, review_status_map: dict[str, str], *, trusted: bool = False) -> dict:
    if trusted:
        summary = trusted_summary_for_run(run, review_status_map)
        run_id = f"{run['run_id']}_trusted"
    else:
        summary = run.get("summary") or {}
        run_id = run["run_id"]
    row = {
        "run_id": run_id,
        "source_file": run["source_file"],
        "method": run["method"],
        "n_points": run["n_points"],
        "poly_degree": run.get("poly_degree"),
        "calibration": run.get("calibration"),
        "trusted_only": trusted,
        "count": summary.get("count"),
        "mean_error_px": summary.get("mean_error_px"),
        "median_error_px": summary.get("median_error_px"),
        "p90_error_px": summary.get("p90_error_px"),
        "max_error_px": summary.get("max_error_px"),
        "mean_abs_x_error_px": summary.get("mean_abs_x_error_px"),
        "mean_abs_y_error_px": summary.get("mean_abs_y_error_px"),
        "mean_x_error_px": summary.get("mean_x_error_px"),
        "mean_y_error_px": summary.get("mean_y_error_px"),
    }
    return row


def pitch_ids_from_runs(runs: list[dict]) -> list[str]:
    for run in runs:
        preds = run.get("predictions") or []
        if preds:
            return [p["pitch_id"] for p in preds]
    return []


def load_runs(pred_dir: Path) -> tuple[list[dict], list[str]]:
    runs: list[dict] = []
    skipped: list[str] = []
    for spec in RUN_SPECS:
        try:
            runs.append(load_run(pred_dir, spec))
        except FileNotFoundError as exc:
            message = str(exc)
            if message.startswith("optional:"):
                skipped.append(message.split(":", 1)[1])
            else:
                raise
    return runs, skipped


def champion_payload(
    role_key: str,
    spec: dict,
    comparison_summary: list[dict],
    comparison_summary_trusted: list[dict],
) -> dict:
    run_id = spec["run_id"]
    summary = next((row for row in comparison_summary if row["run_id"] == run_id), None)
    summary_trusted = next(
        (row for row in comparison_summary_trusted if row["run_id"] == f"{run_id}_trusted"),
        None,
    )
    return {
        "role": role_key,
        "run_id": run_id,
        "method": spec["method"],
        "calibration": spec.get("calibration"),
        "n_points": spec["n_points"],
        "purpose": spec["purpose"],
        "summary": summary,
        "summary_trusted": summary_trusted,
    }


def build_export(pred_dir: Path, labels_dir: Path | None = None) -> dict:
    runs, skipped_optional = load_runs(pred_dir)
    labels_dir = labels_dir or (REPO / "data" / "labels")
    review_status_map = load_review_status_map(labels_dir)
    pitch_ids = pitch_ids_from_runs(runs)
    comparison_summary = [
        comparison_row(run, review_status_map, trusted=False) for run in runs
    ]
    comparison_summary_trusted = [
        comparison_row(run, review_status_map, trusted=True) for run in runs
    ]
    leaderboard = build_leaderboard(comparison_summary)
    leaderboard_trusted = build_leaderboard(comparison_summary_trusted)
    champions = {
        role_key: champion_payload(role_key, spec, comparison_summary, comparison_summary_trusted)
        for role_key, spec in CHAMPION_ROLES.items()
    }
    stability = champions["stability_champion"]

    enriched_runs = []
    for run in runs:
        enriched = dict(run)
        enriched["summary_trusted"] = trusted_summary_for_run(run, review_status_map)
        enriched["summary_by_location_trusted"] = trusted_location_summary_for_run(
            run, review_status_map
        )
        enriched_runs.append(enriched)

    return {
        "title": "Fastball predictions",
        "schema_version": EXPORT_SCHEMA_VERSION,
        "description": (
            "Standalone export: leave-one-out fastball crossing predictions across "
            "velocity, calibrated_velocity, ridge, and compact_ridge_calibrated models. "
            "Use comparison_summary / comparison_summary_trusted, leaderboard, and runs. "
            "Champion roles: accuracy_champion (best median), stability_champion (safest), "
            "early_champion (fewer early points). Trusted metrics exclude "
            "missing_video_unverified and unreliable_unverified. Prediction JSON may also "
            "include summary_by_target_confidence* and summary_trusted_high_confidence."
        ),
        "exported_at": date.today().isoformat(),
        "pitch_count": len(pitch_ids),
        "pitch_ids": pitch_ids,
        "trusted_excludes_review_status": sorted(UNTRUSTED_REVIEW_STATUSES),
        "champions": champions,
        "champion": {
            "alias_of": "stability_champion",
            "note": "Backward-compatible alias; prefer champions.stability_champion.",
            "run_id": stability["run_id"],
            "method": stability["method"],
            "calibration": stability["calibration"],
            "n_points": stability["n_points"],
            "summary": stability["summary"],
            "summary_trusted": stability["summary_trusted"],
        },
        "comparison_summary": comparison_summary,
        "comparison_summary_trusted": comparison_summary_trusted,
        "leaderboard": leaderboard,
        "leaderboard_trusted": leaderboard_trusted,
        "runs": enriched_runs,
        "skipped_optional_runs": skipped_optional,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=REPO / "data" / "predictions",
        help="Directory containing individual prediction JSON files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "predictions" / "Fastball predictions.json",
        help="Combined export path.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=REPO / "data" / "labels",
        help="Label directory for review_status lookup.",
    )
    args = parser.parse_args()

    payload = build_export(args.predictions_dir, args.labels_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(f"  schema_version: {payload['schema_version']}")
    print(f"  pitch_count: {payload['pitch_count']}")
    print(f"  runs bundled: {len(payload['runs'])}")
    if payload["skipped_optional_runs"]:
        print(f"  skipped optional: {', '.join(payload['skipped_optional_runs'])}")
    if payload["leaderboard"]["overall"]:
        top = payload["leaderboard"]["overall"][0]
        print(f"  leaderboard #1: {top['run_id']} (avg rank {top['average_rank']:.2f})")
    if payload["leaderboard_trusted"]["overall"]:
        top_t = payload["leaderboard_trusted"]["overall"][0]
        print(f"  trusted leaderboard #1: {top_t['run_id']} (avg rank {top_t['average_rank']:.2f})")
    acc_t = payload["champions"]["accuracy_champion"].get("summary_trusted") or {}
    stab_t = payload["champions"]["stability_champion"].get("summary_trusted") or {}
    if acc_t:
        print(
            f"  accuracy champion trusted: median={acc_t.get('median_error_px', 0):.1f} "
            f"p90={acc_t.get('p90_error_px', 0):.1f} max={acc_t.get('max_error_px', 0):.1f}"
        )
    if stab_t:
        print(
            f"  stability champion trusted: median={stab_t.get('median_error_px', 0):.1f} "
            f"p90={stab_t.get('p90_error_px', 0):.1f} max={stab_t.get('max_error_px', 0):.1f}"
        )
    for row in payload["comparison_summary"]:
        median = row.get("median_error_px")
        median_s = f"{median:.1f}" if isinstance(median, (int, float)) else "n/a"
        marker = " *" if row["run_id"] == STABILITY_CHAMPION_RUN_ID else ""
        print(f"  {row['run_id']}: median_error_px={median_s}{marker}")


if __name__ == "__main__":
    main()

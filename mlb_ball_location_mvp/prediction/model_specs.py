"""Single source of truth for evaluation model runs.

Add new models here; run_all_models.py and export_fastball_predictions.py
both read from EVAL_RUN_SPECS.
"""

from __future__ import annotations

STABILITY_CHAMPION_RUN_ID = "calibrated_velocity_linear_n7"
ACCURACY_CHAMPION_RUN_ID = "compact_ridge_calibrated_n7"
EARLY_CHAMPION_RUN_ID = "compact_ridge_calibrated_n5"

# Every entry is executed by run_all_models.py (full eval sweep).
EVAL_RUN_SPECS: list[dict] = [
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
        "run_id": STABILITY_CHAMPION_RUN_ID,
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
    },
    {
        "run_id": "calibrated_velocity_linear_n3",
        "source_file": "fastball_calib_linear_n3.json",
        "method": "calibrated_velocity",
        "n_points": 3,
        "poly_degree": 1,
        "calibration": "linear",
    },
    {
        "run_id": "compact_ridge_calibrated_n5",
        "source_file": "fastball_compact_ridge_n5.json",
        "method": "compact_ridge_calibrated",
        "n_points": 5,
        "poly_degree": 1,
        "calibration": None,
    },
    {
        "run_id": "compact_ridge_calibrated_n7",
        "source_file": "fastball_compact_ridge_n7.json",
        "method": "compact_ridge_calibrated",
        "n_points": 7,
        "poly_degree": 1,
        "calibration": None,
    },
]

# Backward-compatible alias used by export script.
RUN_SPECS: list[dict] = EVAL_RUN_SPECS

# Runs that export tolerates if missing (legacy partial bundles only).
EXPORT_OPTIONAL_RUN_IDS = frozenset(
    {
        "calibrated_velocity_linear_n5",
        "calibrated_velocity_linear_n3",
        "compact_ridge_calibrated_n5",
        "compact_ridge_calibrated_n7",
    }
)


def spec_for_run_id(run_id: str) -> dict | None:
    for spec in EVAL_RUN_SPECS:
        if spec["run_id"] == run_id:
            return spec
    return None


def export_specs() -> list[dict]:
    """Specs for bundling; marks legacy optional runs for skip-if-missing."""
    out: list[dict] = []
    for spec in EVAL_RUN_SPECS:
        row = dict(spec)
        if row["run_id"] in EXPORT_OPTIONAL_RUN_IDS:
            row["optional"] = True
        out.append(row)
    return out

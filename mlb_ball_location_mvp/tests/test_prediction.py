from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from prediction.predict_location import load_label, predict_velocity, summarize, Prediction


def test_load_label_and_velocity_prediction(tmp_path: Path) -> None:
    label_path = tmp_path / "pitch_001.json"
    label = {
        "pitch_id": "pitch_001",
        "video": None,
        "fps": 60.0,
        "release_frame": 100,
        "early_points": [
            {"frame": 101, "x": 400.0, "y": 200.0},
            {"frame": 102, "x": 410.0, "y": 206.0},
            {"frame": 103, "x": 420.0, "y": 212.0},
            {"frame": 104, "x": 430.0, "y": 218.0},
            {"frame": 105, "x": 440.0, "y": 224.0},
        ],
        "target": {"cross_frame": 111, "cross_x": 500.0, "cross_y": 260.0},
    }
    label_path.write_text(json.dumps(label), encoding="utf-8")
    loaded = load_label(label_path)
    pred_x, pred_y, details = predict_velocity(
        loaded,
        n_points=5,
        use_actual_cross_frame=True,
        poly_degree=1,
    )
    assert abs(pred_x - 500.0) < 1e-6
    assert abs(pred_y - 260.0) < 1e-6
    assert details["timing_source"] == "actual_cross_frame"
    assert details["degree"] == 1


def test_linear_default_is_degree_one_on_noisy_synthetic(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    labels_dir = tmp_path / "labels"
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "make_synthetic_dataset.py"),
            "--out",
            str(labels_dir),
            "--count",
            "20",
            "--noise",
            "2.0",
        ],
        check=True,
        cwd=repo,
    )
    out_linear = tmp_path / "linear.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "prediction" / "predict_location.py"),
            "--labels",
            str(labels_dir),
            "--n-points",
            "5",
            "--method",
            "velocity",
            "--use-actual-cross-frame",
            "--poly-degree",
            "1",
            "--out",
            str(out_linear),
        ],
        check=True,
        cwd=repo,
    )
    out_quad = tmp_path / "quad.json"
    subprocess.run(
        [
            sys.executable,
            str(repo / "prediction" / "predict_location.py"),
            "--labels",
            str(labels_dir),
            "--n-points",
            "5",
            "--method",
            "velocity",
            "--use-actual-cross-frame",
            "--poly-degree",
            "2",
            "--out",
            str(out_quad),
        ],
        check=True,
        cwd=repo,
    )
    linear = json.loads(out_linear.read_text(encoding="utf-8"))["summary"]["median_error_px"]
    quad = json.loads(out_quad.read_text(encoding="utf-8"))["summary"]["median_error_px"]
    assert linear < quad


def test_single_label_velocity_requires_actual_cross_frame(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    label_path = tmp_path / "pitch_001.json"
    label_path.write_text(
        json.dumps(
            {
                "pitch_id": "pitch_001",
                "early_points": [
                    {"frame": 101, "x": 400.0, "y": 200.0},
                    {"frame": 102, "x": 410.0, "y": 206.0},
                    {"frame": 103, "x": 420.0, "y": 212.0},
                    {"frame": 104, "x": 430.0, "y": 218.0},
                    {"frame": 105, "x": 440.0, "y": 224.0},
                ],
                "target": {"cross_frame": 111, "cross_x": 500.0, "cross_y": 260.0},
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "prediction" / "predict_location.py"),
            "--label",
            str(label_path),
            "--n-points",
            "5",
            "--method",
            "velocity",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "use-actual-cross-frame" in result.stderr.lower() or "use-actual-cross-frame" in result.stdout.lower()


def test_summarize() -> None:
    preds = [
        Prediction("a", "velocity", 0, 0, 3, 4, 5, -3, -4, 5, {}),
        Prediction("b", "velocity", 0, 0, 6, 8, 10, -6, -8, 5, {}),
    ]
    summary = summarize(preds)
    assert summary["count"] == 2
    assert summary["mean_error_px"] == 7.5
    assert summary["median_error_px"] == 7.5


def test_build_leaderboard() -> None:
    from prediction.predict_location import build_leaderboard

    rows = [
        {"run_id": "slow", "median_error_px": 200.0, "p90_error_px": 300.0, "max_error_px": 400.0,
         "mean_abs_x_error_px": 80.0, "mean_abs_y_error_px": 150.0},
        {"run_id": "fast", "median_error_px": 70.0, "p90_error_px": 120.0, "max_error_px": 200.0,
         "mean_abs_x_error_px": 40.0, "mean_abs_y_error_px": 45.0},
    ]
    board = build_leaderboard(rows)
    assert board["overall"][0]["run_id"] == "fast"
    assert board["by_metric"]["median_error_px"][0]["run_id"] == "fast"


def test_ridge_model_robust_scaling(tmp_path: Path) -> None:
    from prediction.predict_location import (
        PitchLabel,
        Point,
        Target,
        default_target_quality_obj,
        fit_ridge_model,
        predict_ridge_model,
    )

    labels = []
    tq = default_target_quality_obj()
    for i in range(4):
        early = tuple(
            Point(frame=100 + j, x=400.0 + 10 * j, y=200.0 + 8 * j)
            for j in range(5)
        )
        labels.append(
            PitchLabel(
                path=tmp_path / f"pitch_{i:03d}.json",
                pitch_id=f"pitch_{i:03d}",
                video=None,
                fps=60.0,
                release_frame=100,
                early_points=early,
                target=Target(cross_x=500.0 + i, cross_y=260.0 + i, cross_frame=111),
                target_quality=tq,
            )
        )
    model = fit_ridge_model(labels, n_points=5, alpha=10.0)
    pred_x, pred_y, details = predict_ridge_model(labels[0], n_points=5, model=model)
    assert details["feature_scaling"] == "robust"
    assert math.isfinite(pred_x)
    assert math.isfinite(pred_y)


def test_trusted_review_filter() -> None:
    from prediction.predict_location import (
        REVIEW_STATUS_MISSING_VIDEO,
        REVIEW_STATUS_UNRELIABLE,
        Prediction,
        filter_trusted_predictions,
        summarize,
    )

    preds = [
        Prediction("a", "velocity", 0, 0, 0, 0, 10, 0, 0, 5, {}, REVIEW_STATUS_MISSING_VIDEO),
        Prediction("b", "velocity", 0, 0, 0, 0, 15, 0, 0, 5, {}, REVIEW_STATUS_UNRELIABLE),
        Prediction("c", "velocity", 0, 0, 0, 0, 20, 0, 0, 5, {}),
    ]
    trusted = filter_trusted_predictions(preds)
    assert len(trusted) == 1
    assert summarize(trusted)["median_error_px"] == 20.0


def test_export_champion_roles(tmp_path: Path) -> None:
    from scripts.export_fastball_predictions import CHAMPION_ROLES, build_export

    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    stub_prediction = {
        "mode": "batch_leave_one_out",
        "method": "velocity",
        "n_points": 7,
        "summary": {
            "count": 1,
            "median_error_px": 10.0,
            "p90_error_px": 12.0,
            "max_error_px": 15.0,
            "mean_error_px": 10.0,
            "mean_abs_x_error_px": 5.0,
            "mean_abs_y_error_px": 5.0,
            "mean_x_error_px": 0.0,
            "mean_y_error_px": 0.0,
        },
        "predictions": [
            {
                "pitch_id": "pitch_001",
                "method": "velocity",
                "predicted_cross": {"x": 1.0, "y": 2.0},
                "actual_cross": {"x": 0.0, "y": 0.0},
                "error_px": 10.0,
                "x_error_px": 1.0,
                "y_error_px": 2.0,
                "n_points": 7,
                "details": {},
            }
        ],
    }
    for spec in (
        {"source_file": "eval_n3.json"},
        {"source_file": "fastball_velocity_n5.json"},
        {"source_file": "fastball_velocity_n7.json"},
        {"source_file": "fastball_ridge_n5.json"},
        {"source_file": "fastball_ridge_n7.json"},
        {"source_file": "fastball_calib_median_n7.json"},
        {"source_file": "fastball_calib_linear_n7.json"},
        {"source_file": "fastball_calib_linear_n5.json"},
        {"source_file": "fastball_calib_linear_n3.json"},
        {"source_file": "fastball_compact_ridge_n5.json"},
        {"source_file": "fastball_compact_ridge_n7.json"},
    ):
        (pred_dir / spec["source_file"]).write_text(json.dumps(stub_prediction), encoding="utf-8")

    payload = build_export(pred_dir, labels_dir)
    assert payload["schema_version"] == 5
    assert set(payload["champions"]) == set(CHAMPION_ROLES)
    assert payload["champions"]["accuracy_champion"]["run_id"] == "compact_ridge_calibrated_n7"
    assert payload["champions"]["stability_champion"]["run_id"] == "calibrated_velocity_linear_n7"
    assert payload["champion"]["alias_of"] == "stability_champion"


def test_cli_synthetic_pipeline(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    labels_dir = tmp_path / "labels"
    out_path = tmp_path / "predictions.json"
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "make_synthetic_dataset.py"), "--out", str(labels_dir), "--count", "8"],
        check=True,
        cwd=repo,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "prediction" / "predict_location.py"),
            "--labels",
            str(labels_dir),
            "--n-points",
            "5",
            "--method",
            "velocity",
            "--use-actual-cross-frame",
            "--poly-degree",
            "2",
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=repo,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["summary"]["count"] == 8
    assert payload["summary"]["median_error_px"] < 20

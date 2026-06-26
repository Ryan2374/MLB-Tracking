from __future__ import annotations

import json
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

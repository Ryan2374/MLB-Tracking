#!/usr/bin/env python3
"""Predict final pitch crossing location from early ball positions.

The MVP target is screen-space final crossing location: (cross_x, cross_y).
This module supports two baselines:

1. velocity: fit x(frame) and y(frame) from the first N points, then extrapolate.
2. ridge: train a direct ridge-regression map from early points to crossing point.

Both are intentionally simple. Use them to create a measurable baseline before
adding ball detection, filters, or pitch-type-specific models.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Point:
    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class Target:
    cross_x: float
    cross_y: float
    cross_frame: Optional[int] = None


@dataclass(frozen=True)
class PitchLabel:
    path: Path
    pitch_id: str
    video: Optional[str]
    fps: Optional[float]
    release_frame: Optional[int]
    early_points: tuple[Point, ...]
    target: Target


@dataclass(frozen=True)
class Prediction:
    pitch_id: str
    method: str
    pred_x: float
    pred_y: float
    actual_x: float
    actual_y: float
    error_px: float
    x_error_px: float
    y_error_px: float
    n_points: int
    details: dict


def load_label(path: Path) -> PitchLabel:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "early_points" not in raw or "target" not in raw:
        raise ValueError(f"{path} is missing early_points or target")

    target_raw = raw["target"]
    if target_raw.get("cross_x") is None or target_raw.get("cross_y") is None:
        raise ValueError(f"{path} target must include cross_x and cross_y")

    early = []
    for item in raw["early_points"]:
        if item.get("frame") is None or item.get("x") is None or item.get("y") is None:
            raise ValueError(f"{path} has an incomplete early point: {item}")
        early.append(Point(frame=int(item["frame"]), x=float(item["x"]), y=float(item["y"])))
    early.sort(key=lambda p: p.frame)

    target = Target(
        cross_x=float(target_raw["cross_x"]),
        cross_y=float(target_raw["cross_y"]),
        cross_frame=int(target_raw["cross_frame"]) if target_raw.get("cross_frame") is not None else None,
    )

    return PitchLabel(
        path=path,
        pitch_id=str(raw.get("pitch_id") or path.stem),
        video=raw.get("video"),
        fps=float(raw["fps"]) if raw.get("fps") is not None else None,
        release_frame=int(raw["release_frame"]) if raw.get("release_frame") is not None else None,
        early_points=tuple(early),
        target=target,
    )


def iter_label_paths(labels_dir: Path) -> list[Path]:
    paths = sorted(p for p in labels_dir.rglob("*.json") if p.is_file())
    # Ignore prediction/evaluation JSON files accidentally placed nearby.
    return [p for p in paths if not p.name.endswith("_prediction.json") and not p.name.endswith("_eval.json")]


def load_labels(labels_dir: Path) -> list[PitchLabel]:
    labels = []
    errors = []
    for path in iter_label_paths(labels_dir):
        try:
            labels.append(load_label(path))
        except Exception as exc:  # noqa: BLE001 - collect all validation issues for the user.
            errors.append(f"{path}: {exc}")
    if errors:
        joined = "\n".join(errors[:10])
        raise ValueError(f"Failed to load some labels:\n{joined}")
    return labels


def require_points(label: PitchLabel, n_points: int) -> tuple[Point, ...]:
    if len(label.early_points) < n_points:
        raise ValueError(
            f"{label.pitch_id} has {len(label.early_points)} early points; need at least {n_points}"
        )
    return tuple(label.early_points[:n_points])


def time_to_cross_from_release(label: PitchLabel) -> Optional[int]:
    if label.release_frame is None or label.target.cross_frame is None:
        return None
    return int(label.target.cross_frame - label.release_frame)


def median_time_to_cross(labels: Iterable[PitchLabel]) -> Optional[float]:
    values = [time_to_cross_from_release(label) for label in labels]
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    return float(statistics.median(values))


def fit_poly_predict(points: tuple[Point, ...], future_frame: float) -> tuple[float, float, dict]:
    """Fit x(frame) and y(frame), then predict location at future_frame."""
    frames = np.array([p.frame for p in points], dtype=float)
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    # Center frames for numerical stability.
    frame0 = frames[0]
    t = frames - frame0
    future_t = float(future_frame) - frame0

    # Quadratic needs at least 3 points. Linear is safer for 2 points.
    degree = min(2, len(points) - 1)
    if degree < 1:
        raise ValueError("Need at least 2 points for trajectory extrapolation")

    x_coef = np.polyfit(t, xs, degree)
    y_coef = np.polyfit(t, ys, degree)
    pred_x = float(np.polyval(x_coef, future_t))
    pred_y = float(np.polyval(y_coef, future_t))

    details = {
        "degree": degree,
        "future_frame": float(future_frame),
        "frame0": float(frame0),
        "x_coef": [float(v) for v in x_coef],
        "y_coef": [float(v) for v in y_coef],
    }
    return pred_x, pred_y, details


def predict_velocity(
    label: PitchLabel,
    n_points: int,
    train_labels_for_timing: Iterable[PitchLabel] = (),
    use_actual_cross_frame: bool = False,
) -> tuple[float, float, dict]:
    points = require_points(label, n_points)

    future_frame: Optional[float] = None
    timing_source = "unknown"

    if use_actual_cross_frame and label.target.cross_frame is not None:
        future_frame = float(label.target.cross_frame)
        timing_source = "actual_cross_frame"
    else:
        median_dt = median_time_to_cross(train_labels_for_timing)
        if median_dt is not None and label.release_frame is not None:
            future_frame = float(label.release_frame + median_dt)
            timing_source = "median_train_time_to_cross"
        elif label.target.cross_frame is not None:
            # Fallback makes single-label smoke testing possible, but do not treat it as a fair live test.
            future_frame = float(label.target.cross_frame)
            timing_source = "fallback_actual_cross_frame"
        else:
            # Last-resort rough estimate: extend by the same observed duration four more times.
            observed_span = max(1, points[-1].frame - points[0].frame)
            future_frame = float(points[-1].frame + 4 * observed_span)
            timing_source = "rough_observed_span_extrapolation"

    pred_x, pred_y, details = fit_poly_predict(points, future_frame)
    details["timing_source"] = timing_source
    return pred_x, pred_y, details


def feature_vector(label: PitchLabel, n_points: int) -> np.ndarray:
    """Create a fixed-width feature vector from the first N points.

    Features are mostly relative to the first visible point so the model learns
    motion shape rather than only absolute screen location.
    """
    points = require_points(label, n_points)
    p0 = points[0]
    values: list[float] = [1.0, p0.x, p0.y]
    for p in points[1:]:
        values.extend([float(p.frame - p0.frame), p.x - p0.x, p.y - p0.y])
    return np.array(values, dtype=float)


def target_vector(label: PitchLabel) -> np.ndarray:
    return np.array([label.target.cross_x, label.target.cross_y], dtype=float)


def fit_ridge(labels: list[PitchLabel], n_points: int, alpha: float = 1.0) -> np.ndarray:
    if not labels:
        raise ValueError("Need at least one training label")
    X = np.vstack([feature_vector(label, n_points) for label in labels])
    Y = np.vstack([target_vector(label) for label in labels])

    penalty = np.eye(X.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0  # Do not penalize bias.
    XtX = X.T @ X + penalty
    XtY = X.T @ Y
    W = np.linalg.pinv(XtX) @ XtY
    return W


def predict_ridge(label: PitchLabel, n_points: int, W: np.ndarray) -> tuple[float, float, dict]:
    x = feature_vector(label, n_points)
    pred = x @ W
    details = {"feature_count": int(x.shape[0])}
    return float(pred[0]), float(pred[1]), details


def make_prediction(
    label: PitchLabel,
    method: str,
    n_points: int,
    train_labels: list[PitchLabel],
    alpha: float,
    use_actual_cross_frame: bool,
) -> Prediction:
    if method == "velocity":
        pred_x, pred_y, details = predict_velocity(
            label=label,
            n_points=n_points,
            train_labels_for_timing=train_labels,
            use_actual_cross_frame=use_actual_cross_frame,
        )
    elif method == "ridge":
        W = fit_ridge(train_labels, n_points=n_points, alpha=alpha)
        pred_x, pred_y, details = predict_ridge(label, n_points=n_points, W=W)
        details["train_count"] = len(train_labels)
        details["alpha"] = float(alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    actual_x = label.target.cross_x
    actual_y = label.target.cross_y
    x_error = pred_x - actual_x
    y_error = pred_y - actual_y
    error = math.hypot(x_error, y_error)

    return Prediction(
        pitch_id=label.pitch_id,
        method=method,
        pred_x=pred_x,
        pred_y=pred_y,
        actual_x=actual_x,
        actual_y=actual_y,
        error_px=error,
        x_error_px=x_error,
        y_error_px=y_error,
        n_points=n_points,
        details=details,
    )


def prediction_to_dict(pred: Prediction) -> dict:
    return {
        "pitch_id": pred.pitch_id,
        "method": pred.method,
        "n_points": pred.n_points,
        "predicted_cross": {"x": pred.pred_x, "y": pred.pred_y},
        "actual_cross": {"x": pred.actual_x, "y": pred.actual_y},
        "error_px": pred.error_px,
        "x_error_px": pred.x_error_px,
        "y_error_px": pred.y_error_px,
        "details": pred.details,
    }


def summarize(predictions: list[Prediction]) -> dict:
    if not predictions:
        return {}
    errors = np.array([p.error_px for p in predictions], dtype=float)
    x_errors = np.array([abs(p.x_error_px) for p in predictions], dtype=float)
    y_errors = np.array([abs(p.y_error_px) for p in predictions], dtype=float)
    return {
        "count": int(len(predictions)),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "max_error_px": float(np.max(errors)),
        "mean_abs_x_error_px": float(np.mean(x_errors)),
        "mean_abs_y_error_px": float(np.mean(y_errors)),
    }


def print_predictions(predictions: list[Prediction], summary: dict) -> None:
    if not predictions:
        print("No predictions produced.")
        return

    print("\nPer-pitch results")
    print("pitch_id                 method     pred_x   pred_y   actual_x actual_y error_px")
    print("-" * 82)
    for p in predictions:
        print(
            f"{p.pitch_id[:24]:24s} {p.method:9s} "
            f"{p.pred_x:8.2f} {p.pred_y:8.2f} {p.actual_x:8.2f} {p.actual_y:8.2f} {p.error_px:8.2f}"
        )

    print("\nSummary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def run_single(args: argparse.Namespace) -> dict:
    label = load_label(Path(args.label))
    train_labels: list[PitchLabel] = []
    method = args.method

    if method == "ridge":
        raise SystemExit("Single-label ridge prediction needs --labels so it has training data. Use --method velocity for one label.")

    pred = make_prediction(
        label=label,
        method=method,
        n_points=args.n_points,
        train_labels=train_labels,
        alpha=args.alpha,
        use_actual_cross_frame=args.use_actual_cross_frame,
    )
    summary = summarize([pred])
    print_predictions([pred], summary)
    return {"mode": "single", "predictions": [prediction_to_dict(pred)], "summary": summary}


def run_batch(args: argparse.Namespace) -> dict:
    labels = load_labels(Path(args.labels))
    labels = [label for label in labels if len(label.early_points) >= args.n_points]
    if not labels:
        raise SystemExit(f"No labels with at least {args.n_points} early points found.")

    predictions: list[Prediction] = []

    if args.method == "ridge" and len(labels) < 2:
        raise SystemExit("Ridge evaluation needs at least 2 labels for leave-one-out evaluation.")

    for i, label in enumerate(labels):
        train = [other for j, other in enumerate(labels) if j != i]
        if args.method == "velocity" and not train:
            train = []
        pred = make_prediction(
            label=label,
            method=args.method,
            n_points=args.n_points,
            train_labels=train,
            alpha=args.alpha,
            use_actual_cross_frame=args.use_actual_cross_frame,
        )
        predictions.append(pred)

    summary = summarize(predictions)
    print_predictions(predictions, summary)
    return {
        "mode": "batch_leave_one_out",
        "method": args.method,
        "n_points": args.n_points,
        "predictions": [prediction_to_dict(p) for p in predictions],
        "summary": summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", help="Path to one pitch label JSON.")
    group.add_argument("--labels", help="Directory containing pitch label JSON files.")
    parser.add_argument("--n-points", type=int, default=5, help="Number of early ball points to use.")
    parser.add_argument("--method", choices=["velocity", "ridge"], default="velocity")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge penalty for --method ridge.")
    parser.add_argument(
        "--use-actual-cross-frame",
        action="store_true",
        help="Use labeled cross_frame for trajectory extrapolation. Good for smoke tests, not a fair live-style evaluation.",
    )
    parser.add_argument("--out", help="Optional output JSON path for predictions.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_points < 2:
        raise SystemExit("--n-points must be at least 2")

    if args.label:
        payload = run_single(args)
    else:
        payload = run_batch(args)

    if args.out:
        write_json(Path(args.out), payload)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

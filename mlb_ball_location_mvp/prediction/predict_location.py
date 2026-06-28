#!/usr/bin/env python3
"""Predict final pitch crossing location from early ball positions.

The MVP target is screen-space final crossing location: (cross_x, cross_y).
This module supports two baselines:

1. velocity: fit x(frame) and y(frame) from the first N points, then extrapolate.
2. ridge: train a direct ridge-regression map from early points to crossing point.

Both are intentionally simple. Use them to create a measurable baseline before
adding ball detection, filters, or pitch-type-specific models.

3. calibrated_velocity: velocity extrapolation plus a leave-one-out residual
   correction learned from training pitches (median/mean bias or linear map).
   Stability champion (live fallback): calibrated_velocity + linear + 7 points.

4. compact_ridge_calibrated: robust-scaled ridge on compact trajectory features
   Accuracy champion at 100 fastballs: compact_ridge_calibrated + 7 points.
   Falls back to calibrated_velocity linear when off-screen or disagreeing by
   COMPACT_STABILITY_DISAGREEMENT_PX (when ridge_outlier_guard is on).
   (raw velocity prediction, last point, velocity, span, estimated frames to cross).
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

# Role-based fastball defaults (see export_fastball_predictions champion roles).
ACCURACY_CHAMPION_METHOD = "compact_ridge_calibrated"
ACCURACY_CHAMPION_N_POINTS = 7

STABILITY_CHAMPION_METHOD = "calibrated_velocity"
STABILITY_CHAMPION_CALIBRATION = "linear"
STABILITY_CHAMPION_N_POINTS = 7

EARLY_CHAMPION_METHOD = "compact_ridge_calibrated"
EARLY_CHAMPION_N_POINTS = 5

# Backward-compatible aliases for CLI defaults and stability fallback.
CHAMPION_METHOD = STABILITY_CHAMPION_METHOD
CHAMPION_CALIBRATION = STABILITY_CHAMPION_CALIBRATION
CHAMPION_N_POINTS = STABILITY_CHAMPION_N_POINTS

# Compact ridge uses calibrated_velocity linear when predictions diverge this far.
COMPACT_STABILITY_DISAGREEMENT_PX = 200.0

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
OFFSCREEN_MARGIN = 100

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

REVIEW_STATUS_VERIFIED = "verified"
REVIEW_STATUS_MISSING_VIDEO = "missing_video_unverified"
REVIEW_STATUS_UNRELIABLE = "unreliable_unverified"

UNTRUSTED_REVIEW_STATUSES = frozenset(
    {REVIEW_STATUS_MISSING_VIDEO, REVIEW_STATUS_UNRELIABLE}
)

LEADERBOARD_METRICS = (
    "median_error_px",
    "p90_error_px",
    "max_error_px",
    "mean_abs_x_error_px",
    "mean_abs_y_error_px",
)


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
    review_status: str = REVIEW_STATUS_VERIFIED


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
    review_status: str = REVIEW_STATUS_VERIFIED


def is_trusted_review(review_status: str) -> bool:
    return review_status not in UNTRUSTED_REVIEW_STATUSES


def filter_trusted_predictions(predictions: list[Prediction]) -> list[Prediction]:
    return [p for p in predictions if is_trusted_review(p.review_status)]


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
        review_status=str(raw.get("review_status") or REVIEW_STATUS_VERIFIED),
    )


def iter_label_paths(labels_dir: Path) -> list[Path]:
    paths = sorted(p for p in labels_dir.rglob("*.json") if p.is_file())
    # Ignore prediction/evaluation JSON and schema examples accidentally placed nearby.
    return [
        p
        for p in paths
        if not p.name.endswith("_prediction.json")
        and not p.name.endswith("_eval.json")
        and not p.name.startswith("example_")
        and not p.name.startswith("eval_")
    ]


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


def fit_poly_predict(
    points: tuple[Point, ...],
    future_frame: float,
    degree: int = 1,
) -> tuple[float, float, dict]:
    """Fit x(frame) and y(frame), then predict location at future_frame."""
    frames = np.array([p.frame for p in points], dtype=float)
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)

    # Center frames for numerical stability.
    frame0 = frames[0]
    t = frames - frame0
    future_t = float(future_frame) - frame0

    degree = min(int(degree), len(points) - 1)
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
    poly_degree: int = 1,
) -> tuple[float, float, dict]:
    points = require_points(label, n_points)

    future_frame: Optional[float] = None
    timing_source = "unknown"

    if use_actual_cross_frame:
        if label.target.cross_frame is None:
            raise ValueError(
                f"{label.pitch_id}: --use-actual-cross-frame requires target.cross_frame in the label"
            )
        future_frame = float(label.target.cross_frame)
        timing_source = "actual_cross_frame"
    else:
        median_dt = median_time_to_cross(train_labels_for_timing)
        if median_dt is not None and label.release_frame is not None:
            future_frame = float(label.release_frame + median_dt)
            timing_source = "median_train_time_to_cross"
        else:
            observed_span = max(1, points[-1].frame - points[0].frame)
            future_frame = float(points[-1].frame + 4 * observed_span)
            timing_source = "rough_observed_span_extrapolation"

    pred_x, pred_y, details = fit_poly_predict(points, future_frame, degree=poly_degree)
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


@dataclass(frozen=True)
class RidgeModel:
    W: np.ndarray
    feature_median: np.ndarray
    feature_scale: np.ndarray
    alpha: float


def robust_feature_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust scale for ridge features; column 0 (bias) is left untouched."""
    if X.shape[1] <= 1:
        return np.zeros(0, dtype=float), np.ones(0, dtype=float)
    values = X[:, 1:]
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.where(mad > 1e-6, mad * 1.4826, 1.0)
    return median, scale


def scale_feature_vector(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = x.astype(float).copy()
    if median.size:
        out[1:] = (out[1:] - median) / scale
    return out


def scale_feature_matrix(X: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = X.astype(float).copy()
    if median.size:
        out[:, 1:] = (out[:, 1:] - median) / scale
    return out


def fit_ridge_weights(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    XtX = X.T @ X + penalty
    XtY = X.T @ Y
    return np.linalg.pinv(XtX) @ XtY


def fit_ridge_model(labels: list[PitchLabel], n_points: int, alpha: float) -> RidgeModel:
    if not labels:
        raise ValueError("Need at least one training label")
    X_raw = np.vstack([feature_vector(label, n_points) for label in labels])
    median, scale = robust_feature_scale(X_raw)
    X = scale_feature_matrix(X_raw, median, scale)
    Y = np.vstack([target_vector(label) for label in labels])
    W = fit_ridge_weights(X, Y, alpha)
    return RidgeModel(W=W, feature_median=median, feature_scale=scale, alpha=float(alpha))


def select_ridge_alpha(labels: list[PitchLabel], n_points: int, alphas: Iterable[float] = RIDGE_ALPHAS) -> float:
    if len(labels) < 3:
        return float(next(iter(alphas), 10.0))

    best_alpha = float(next(iter(alphas), 10.0))
    best_median = float("inf")
    for alpha in alphas:
        fold_errors: list[float] = []
        for i, label in enumerate(labels):
            fold_train = [other for j, other in enumerate(labels) if j != i]
            model = fit_ridge_model(fold_train, n_points, float(alpha))
            pred_x, pred_y, _ = predict_ridge_model(label, n_points, model)
            fold_errors.append(
                math.hypot(pred_x - label.target.cross_x, pred_y - label.target.cross_y)
            )
        median_error = float(np.median(fold_errors))
        if median_error < best_median:
            best_median = median_error
            best_alpha = float(alpha)
    return best_alpha


def predict_ridge_model(label: PitchLabel, n_points: int, model: RidgeModel) -> tuple[float, float, dict]:
    x = scale_feature_vector(
        feature_vector(label, n_points),
        model.feature_median,
        model.feature_scale,
    )
    pred = x @ model.W
    details = {
        "feature_count": int(x.shape[0]),
        "alpha": model.alpha,
        "feature_scaling": "robust",
    }
    return float(pred[0]), float(pred[1]), details


def compact_feature_vector(
    label: PitchLabel,
    n_points: int,
    raw_pred_x: float,
    raw_pred_y: float,
    train_labels_for_timing: Iterable[PitchLabel],
) -> np.ndarray:
    points = require_points(label, n_points)
    p0 = points[0]
    pn = points[-1]
    frame_span = float(max(1, pn.frame - p0.frame))
    dx = float(pn.x - p0.x)
    dy = float(pn.y - p0.y)
    vel_x = dx / frame_span
    vel_y = dy / frame_span
    median_dt = median_time_to_cross(train_labels_for_timing)
    if median_dt is not None and label.release_frame is not None:
        est_frames = float(median_dt)
    else:
        est_frames = float(4 * frame_span)
    return np.array(
        [
            1.0,
            float(raw_pred_x),
            float(raw_pred_y),
            float(pn.x),
            float(pn.y),
            vel_x,
            vel_y,
            dx,
            dy,
            frame_span,
            est_frames,
        ],
        dtype=float,
    )


def build_compact_training_matrix(
    labels: list[PitchLabel],
    n_points: int,
    *,
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for label in labels:
        raw_x, raw_y, _ = predict_velocity(
            label=label,
            n_points=n_points,
            train_labels_for_timing=labels,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
        rows.append(
            compact_feature_vector(label, n_points, raw_x, raw_y, labels)
        )
        targets.append(target_vector(label))
    return np.vstack(rows), np.vstack(targets)


def fit_ridge_model_from_matrix(X: np.ndarray, Y: np.ndarray, alpha: float) -> RidgeModel:
    median, scale = robust_feature_scale(X)
    X_scaled = scale_feature_matrix(X, median, scale)
    W = fit_ridge_weights(X_scaled, Y, alpha)
    return RidgeModel(W=W, feature_median=median, feature_scale=scale, alpha=float(alpha))


def fit_compact_ridge_model(
    labels: list[PitchLabel],
    n_points: int,
    alpha: float,
    *,
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> RidgeModel:
    X, Y = build_compact_training_matrix(
        labels,
        n_points,
        use_actual_cross_frame=use_actual_cross_frame,
        poly_degree=poly_degree,
    )
    return fit_ridge_model_from_matrix(X, Y, alpha)


def select_compact_ridge_alpha(
    labels: list[PitchLabel],
    n_points: int,
    *,
    use_actual_cross_frame: bool,
    poly_degree: int,
    alphas: Iterable[float] = RIDGE_ALPHAS,
) -> float:
    if len(labels) < 3:
        return float(next(iter(alphas), 10.0))

    best_alpha = float(next(iter(alphas), 10.0))
    best_median = float("inf")
    for alpha in alphas:
        fold_errors: list[float] = []
        for i, label in enumerate(labels):
            fold_train = [other for j, other in enumerate(labels) if j != i]
            model = fit_compact_ridge_model(
                fold_train,
                n_points,
                float(alpha),
                use_actual_cross_frame=use_actual_cross_frame,
                poly_degree=poly_degree,
            )
            pred_x, pred_y, _ = predict_compact_ridge_model(
                label,
                n_points,
                model,
                fold_train,
                use_actual_cross_frame=use_actual_cross_frame,
                poly_degree=poly_degree,
            )
            fold_errors.append(
                math.hypot(pred_x - label.target.cross_x, pred_y - label.target.cross_y)
            )
        median_error = float(np.median(fold_errors))
        if median_error < best_median:
            best_median = median_error
            best_alpha = float(alpha)
    return best_alpha


def predict_compact_ridge_model(
    label: PitchLabel,
    n_points: int,
    model: RidgeModel,
    train_labels_for_timing: list[PitchLabel],
    *,
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> tuple[float, float, dict]:
    raw_x, raw_y, vel_details = predict_velocity(
        label=label,
        n_points=n_points,
        train_labels_for_timing=train_labels_for_timing,
        use_actual_cross_frame=use_actual_cross_frame,
        poly_degree=poly_degree,
    )
    x = scale_feature_vector(
        compact_feature_vector(label, n_points, raw_x, raw_y, train_labels_for_timing),
        model.feature_median,
        model.feature_scale,
    )
    pred = x @ model.W
    details = {
        "feature_count": int(x.shape[0]),
        "alpha": model.alpha,
        "feature_scaling": "robust",
        "model_type": "compact_ridge_calibrated",
        "raw_pred_x": raw_x,
        "raw_pred_y": raw_y,
        "velocity_timing_source": vel_details.get("timing_source"),
    }
    return float(pred[0]), float(pred[1]), details


def prediction_on_screen(
    x: float,
    y: float,
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    margin: int = OFFSCREEN_MARGIN,
) -> bool:
    return (-margin <= x <= width + margin) and (-margin <= y <= height + margin)


def fit_ridge(labels: list[PitchLabel], n_points: int, alpha: float = 1.0) -> np.ndarray:
    """Legacy unscaled ridge weights (kept for compatibility)."""
    if not labels:
        raise ValueError("Need at least one training label")
    X = np.vstack([feature_vector(label, n_points) for label in labels])
    Y = np.vstack([target_vector(label) for label in labels])
    return fit_ridge_weights(X, Y, alpha)


def predict_ridge(label: PitchLabel, n_points: int, W: np.ndarray) -> tuple[float, float, dict]:
    x = feature_vector(label, n_points)
    pred = x @ W
    details = {"feature_count": int(x.shape[0])}
    return float(pred[0]), float(pred[1]), details


def raw_velocity_on_labels(
    labels: list[PitchLabel],
    n_points: int,
    train_labels_for_timing: Iterable[PitchLabel],
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_x: list[float] = []
    raw_y: list[float] = []
    actual_x: list[float] = []
    actual_y: list[float] = []
    for label in labels:
        px, py, _ = predict_velocity(
            label=label,
            n_points=n_points,
            train_labels_for_timing=train_labels_for_timing,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
        raw_x.append(px)
        raw_y.append(py)
        actual_x.append(label.target.cross_x)
        actual_y.append(label.target.cross_y)
    return (
        np.array(raw_x, dtype=float),
        np.array(raw_y, dtype=float),
        np.array(actual_x, dtype=float),
        np.array(actual_y, dtype=float),
    )


def fit_velocity_calibration(
    train_labels: list[PitchLabel],
    n_points: int,
    *,
    calibration: str,
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> dict:
    if not train_labels:
        raise ValueError("calibrated_velocity needs at least one training label")

    raw_x, raw_y, actual_x, actual_y = raw_velocity_on_labels(
        train_labels,
        n_points,
        train_labels,
        use_actual_cross_frame,
        poly_degree,
    )
    x_res = actual_x - raw_x
    y_res = actual_y - raw_y

    if calibration == "median":
        return {
            "calibration": calibration,
            "x_offset": float(np.median(x_res)),
            "y_offset": float(np.median(y_res)),
            "train_count": len(train_labels),
        }
    if calibration == "mean":
        return {
            "calibration": calibration,
            "x_offset": float(np.mean(x_res)),
            "y_offset": float(np.mean(y_res)),
            "train_count": len(train_labels),
        }
    if calibration == "linear":
        if len(train_labels) < 3:
            raise ValueError("linear calibration needs at least 3 training labels")
        X = np.column_stack([raw_x, raw_y, np.ones(len(train_labels))])
        x_coef = np.linalg.lstsq(X, actual_x, rcond=None)[0]
        y_coef = np.linalg.lstsq(X, actual_y, rcond=None)[0]
        return {
            "calibration": calibration,
            "x_coef": [float(v) for v in x_coef],
            "y_coef": [float(v) for v in y_coef],
            "train_count": len(train_labels),
        }
    raise ValueError(f"Unknown calibration: {calibration}")


def apply_velocity_calibration(
    raw_x: float,
    raw_y: float,
    calibration_params: dict,
) -> tuple[float, float]:
    mode = calibration_params["calibration"]
    if mode in ("median", "mean"):
        return (
            raw_x + float(calibration_params["x_offset"]),
            raw_y + float(calibration_params["y_offset"]),
        )
    if mode == "linear":
        vec = np.array([raw_x, raw_y, 1.0], dtype=float)
        x_coef = np.array(calibration_params["x_coef"], dtype=float)
        y_coef = np.array(calibration_params["y_coef"], dtype=float)
        return float(vec @ x_coef), float(vec @ y_coef)
    raise ValueError(f"Unknown calibration: {mode}")


def predict_calibrated_velocity(
    label: PitchLabel,
    n_points: int,
    train_labels: list[PitchLabel],
    *,
    calibration: str,
    use_actual_cross_frame: bool,
    poly_degree: int,
) -> tuple[float, float, dict]:
    raw_x, raw_y, vel_details = predict_velocity(
        label=label,
        n_points=n_points,
        train_labels_for_timing=train_labels,
        use_actual_cross_frame=use_actual_cross_frame,
        poly_degree=poly_degree,
    )
    calib = fit_velocity_calibration(
        train_labels,
        n_points,
        calibration=calibration,
        use_actual_cross_frame=use_actual_cross_frame,
        poly_degree=poly_degree,
    )
    pred_x, pred_y = apply_velocity_calibration(raw_x, raw_y, calib)
    details = {
        **vel_details,
        "raw_pred_x": raw_x,
        "raw_pred_y": raw_y,
        **calib,
    }
    return pred_x, pred_y, details


def crossing_y_bucket(actual_y: float) -> str:
    """Bucket final crossing y for error reporting (y increases downward)."""
    if actual_y < 480:
        return "high"
    if actual_y < 580:
        return "middle"
    if actual_y < 720:
        return "low"
    return "very_low"


def summarize_by_location(predictions: list[Prediction]) -> dict:
    buckets: dict[str, list[Prediction]] = {}
    for pred in predictions:
        bucket = crossing_y_bucket(pred.actual_y)
        buckets.setdefault(bucket, []).append(pred)

    out: dict[str, dict] = {}
    for name in ("high", "middle", "low", "very_low"):
        if name in buckets:
            out[name] = summarize(buckets[name])
            out[name]["count"] = len(buckets[name])
    return out


def make_prediction(
    label: PitchLabel,
    method: str,
    n_points: int,
    train_labels: list[PitchLabel],
    alpha: float,
    use_actual_cross_frame: bool,
    poly_degree: int = 1,
    calibration: str = CHAMPION_CALIBRATION,
    ridge_cv: bool = True,
    ridge_outlier_guard: bool = True,
) -> Prediction:
    if method == "velocity":
        pred_x, pred_y, details = predict_velocity(
            label=label,
            n_points=n_points,
            train_labels_for_timing=train_labels,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
    elif method == "calibrated_velocity":
        pred_x, pred_y, details = predict_calibrated_velocity(
            label=label,
            n_points=n_points,
            train_labels=train_labels,
            calibration=calibration,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
    elif method == "ridge":
        chosen_alpha = select_ridge_alpha(train_labels, n_points) if ridge_cv else float(alpha)
        model = fit_ridge_model(train_labels, n_points=n_points, alpha=chosen_alpha)
        ridge_x, ridge_y, details = predict_ridge_model(label, n_points=n_points, model=model)
        details["train_count"] = len(train_labels)
        details["ridge_cv"] = bool(ridge_cv)
        pred_x, pred_y = ridge_x, ridge_y
        if ridge_outlier_guard and not prediction_on_screen(ridge_x, ridge_y):
            fallback_x, fallback_y, fallback_details = predict_calibrated_velocity(
                label=label,
                n_points=n_points,
                train_labels=train_labels,
                calibration="linear",
                use_actual_cross_frame=use_actual_cross_frame,
                poly_degree=poly_degree,
            )
            details["outlier_guard"] = True
            details["ridge_raw_pred_x"] = ridge_x
            details["ridge_raw_pred_y"] = ridge_y
            details["fallback_method"] = "calibrated_velocity_linear"
            details["fallback_details"] = fallback_details
            pred_x, pred_y = fallback_x, fallback_y
    elif method == "compact_ridge_calibrated":
        chosen_alpha = (
            select_compact_ridge_alpha(
                train_labels,
                n_points,
                use_actual_cross_frame=use_actual_cross_frame,
                poly_degree=poly_degree,
            )
            if ridge_cv
            else float(alpha)
        )
        model = fit_compact_ridge_model(
            train_labels,
            n_points,
            chosen_alpha,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
        compact_x, compact_y, details = predict_compact_ridge_model(
            label,
            n_points,
            model,
            train_labels,
            use_actual_cross_frame=use_actual_cross_frame,
            poly_degree=poly_degree,
        )
        details["train_count"] = len(train_labels)
        details["ridge_cv"] = bool(ridge_cv)
        pred_x, pred_y = compact_x, compact_y
        if ridge_outlier_guard:
            stab_x, stab_y, stab_details = predict_calibrated_velocity(
                label=label,
                n_points=n_points,
                train_labels=train_labels,
                calibration="linear",
                use_actual_cross_frame=use_actual_cross_frame,
                poly_degree=poly_degree,
            )
            disagreement = math.hypot(compact_x - stab_x, compact_y - stab_y)
            off_screen = not prediction_on_screen(compact_x, compact_y)
            if off_screen or disagreement > COMPACT_STABILITY_DISAGREEMENT_PX:
                details["outlier_guard"] = True
                details["compact_raw_pred_x"] = compact_x
                details["compact_raw_pred_y"] = compact_y
                details["fallback_method"] = "calibrated_velocity_linear"
                details["fallback_details"] = stab_details
                details["fallback_reason"] = (
                    "off_screen" if off_screen else "stability_disagreement"
                )
                if not off_screen:
                    details["stability_disagreement_px"] = disagreement
                pred_x, pred_y = stab_x, stab_y
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
        review_status=label.review_status,
    )


def prediction_to_dict(pred: Prediction) -> dict:
    return {
        "pitch_id": pred.pitch_id,
        "method": pred.method,
        "n_points": pred.n_points,
        "review_status": pred.review_status,
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
    signed_x = np.array([p.x_error_px for p in predictions], dtype=float)
    signed_y = np.array([p.y_error_px for p in predictions], dtype=float)
    return {
        "count": int(len(predictions)),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "max_error_px": float(np.max(errors)),
        "mean_abs_x_error_px": float(np.mean(x_errors)),
        "mean_abs_y_error_px": float(np.mean(y_errors)),
        "mean_x_error_px": float(np.mean(signed_x)),
        "mean_y_error_px": float(np.mean(signed_y)),
    }


def build_leaderboard(rows: list[dict], metrics: tuple[str, ...] = LEADERBOARD_METRICS) -> dict:
    """Rank runs by error metrics (lower is better)."""
    if not rows:
        return {"by_metric": {}, "overall": []}

    rank_scores: dict[str, float] = {row["run_id"]: 0.0 for row in rows if row.get("run_id")}
    by_metric: dict[str, list[dict]] = {}

    for metric in metrics:
        ranked = sorted(
            rows,
            key=lambda row: float(row.get(metric, float("inf"))),
        )
        metric_rows: list[dict] = []
        for rank, row in enumerate(ranked, start=1):
            run_id = row.get("run_id")
            if not run_id:
                continue
            rank_scores[run_id] = rank_scores.get(run_id, 0.0) + rank
            metric_rows.append(
                {
                    "rank": rank,
                    "run_id": run_id,
                    "value": row.get(metric),
                }
            )
        by_metric[metric] = metric_rows

    overall = sorted(rank_scores.items(), key=lambda item: item[1])
    return {
        "by_metric": by_metric,
        "overall": [
            {"rank": rank, "run_id": run_id, "average_rank": avg_rank / len(metrics)}
            for rank, (run_id, avg_rank) in enumerate(overall, start=1)
        ],
    }


def batch_summary_payload(predictions: list[Prediction]) -> dict:
    trusted = filter_trusted_predictions(predictions)
    return {
        "summary": summarize(predictions),
        "summary_trusted": summarize(trusted),
        "summary_by_location": summarize_by_location(predictions),
        "summary_by_location_trusted": summarize_by_location(trusted),
        "trusted_excludes_review_status": sorted(UNTRUSTED_REVIEW_STATUSES),
        "trusted_count": len(trusted),
        "excluded_count": len(predictions) - len(trusted),
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

    if method in ("ridge", "calibrated_velocity", "compact_ridge_calibrated"):
        raise SystemExit(
            f"Single-label {method} prediction needs --labels so it has training data. "
            "Use --method velocity for one label."
        )

    if method == "velocity" and not args.use_actual_cross_frame:
        raise SystemExit(
            "Single-label velocity prediction requires --use-actual-cross-frame for smoke tests, "
            "or use --labels so timing can be estimated from training history."
        )

    pred = make_prediction(
        label=label,
        method=method,
        n_points=args.n_points,
        train_labels=train_labels,
        alpha=args.alpha,
        use_actual_cross_frame=args.use_actual_cross_frame,
        poly_degree=args.poly_degree,
        calibration=args.calibration,
        ridge_cv=args.ridge_cv,
        ridge_outlier_guard=not args.no_ridge_outlier_guard,
    )
    summary = summarize([pred])
    print_predictions([pred], summary)
    return {
        "mode": "single",
        "predictions": [prediction_to_dict(pred)],
        "summary": summary,
        "summary_by_location": summarize_by_location([pred]),
    }


def run_batch(args: argparse.Namespace) -> dict:
    labels = load_labels(Path(args.labels))
    labels = [label for label in labels if len(label.early_points) >= args.n_points]
    if not labels:
        raise SystemExit(f"No labels with at least {args.n_points} early points found.")

    predictions: list[Prediction] = []

    if args.method == "ridge" and len(labels) < 2:
        raise SystemExit("Ridge evaluation needs at least 2 labels for leave-one-out evaluation.")
    if args.method == "compact_ridge_calibrated" and len(labels) < 4:
        raise SystemExit("compact_ridge_calibrated needs at least 4 labels for leave-one-out evaluation.")
    if args.method == "calibrated_velocity" and len(labels) < 2:
        raise SystemExit("calibrated_velocity needs at least 2 labels for leave-one-out evaluation.")
    if args.method == "calibrated_velocity" and args.calibration == "linear" and len(labels) < 4:
        raise SystemExit("calibrated_velocity with linear calibration needs at least 4 labels.")

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
            poly_degree=args.poly_degree,
            calibration=args.calibration,
            ridge_cv=args.ridge_cv,
            ridge_outlier_guard=not args.no_ridge_outlier_guard,
        )
        predictions.append(pred)

    summaries = batch_summary_payload(predictions)
    print_predictions(predictions, summaries["summary"])
    if summaries["summary_by_location"]:
        print("\nSummary by crossing height (actual_y)")
        for bucket, bucket_summary in summaries["summary_by_location"].items():
            print(f"  {bucket}: median_error_px={bucket_summary['median_error_px']:.1f} n={bucket_summary['count']}")
    if summaries["excluded_count"]:
        trusted = summaries["summary_trusted"]
        print(
            f"\nTrusted summary (excludes {summaries['excluded_count']} untrusted): "
            f"median_error_px={trusted.get('median_error_px', 0):.1f}"
        )
    return {
        "mode": "batch_leave_one_out",
        "method": args.method,
        "n_points": args.n_points,
        "poly_degree": args.poly_degree,
        "calibration": args.calibration if args.method == "calibrated_velocity" else None,
        "predictions": [prediction_to_dict(p) for p in predictions],
        **summaries,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", help="Path to one pitch label JSON.")
    group.add_argument("--labels", help="Directory containing pitch label JSON files.")
    parser.add_argument("--n-points", type=int, default=CHAMPION_N_POINTS, help="Number of early ball points to use.")
    parser.add_argument(
        "--method",
        choices=["velocity", "calibrated_velocity", "ridge", "compact_ridge_calibrated"],
        default=CHAMPION_METHOD,
    )
    parser.add_argument(
        "--calibration",
        choices=["median", "mean", "linear"],
        default=CHAMPION_CALIBRATION,
        help="Residual correction for --method calibrated_velocity.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge penalty when --no-ridge-cv is set.",
    )
    parser.add_argument(
        "--ridge-cv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pick ridge alpha by leave-one-out CV on training pitches (default: on).",
    )
    parser.add_argument(
        "--no-ridge-outlier-guard",
        action="store_true",
        help="Disable ridge fallback to calibrated_velocity linear when prediction is off-screen.",
    )
    parser.add_argument(
        "--use-actual-cross-frame",
        action="store_true",
        help="Use labeled cross_frame for trajectory extrapolation. Good for smoke tests, not a fair live-style evaluation.",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        choices=[1, 2],
        default=1,
        help="Polynomial degree for --method velocity (default: 1 linear; 2 quadratic).",
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

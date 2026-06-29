from __future__ import annotations

import json
from pathlib import Path

from labeling.target_derivation import (
    TARGET_SOURCE_BRACKET_MIDPOINT,
    TARGET_SOURCE_NEAREST_FRAME,
    TARGET_SOURCE_NEAREST_FRAME_OLD,
    derive_target_from_bracket,
    finalize_label_target,
    interpolate_bracket,
    normalize_label_target_fields,
    target_weight_from_confidence,
)


def test_interpolate_bracket_midpoint() -> None:
    before = {"frame": 10, "x": 100.0, "y": 200.0}
    after = {"frame": 12, "x": 200.0, "y": 400.0}
    cross_x, cross_y, cross_frame = interpolate_bracket(before, after, 0.5)
    assert cross_x == 150.0
    assert cross_y == 300.0
    assert cross_frame == 11.0


def test_derive_target_from_bracket() -> None:
    bracket = {
        "before": {"frame": 123, "x": 1010.0, "y": 455.0},
        "after": {"frame": 124, "x": 1065.0, "y": 525.0},
        "alpha": 0.5,
    }
    target, quality = derive_target_from_bracket(bracket, force_midpoint=True)
    assert target["target_source"] == TARGET_SOURCE_BRACKET_MIDPOINT
    assert target["cross_x"] == 1037.5
    assert target["cross_y"] == 490.0
    assert quality["confidence"] == "medium"
    assert quality["estimated"] is True
    assert quality["uncertainty_px"] > 0


def test_finalize_label_target_from_bracket() -> None:
    data = {
        "target": {},
        "target_bracket": {
            "before": {"frame": 1, "x": 0.0, "y": 0.0},
            "after": {"frame": 3, "x": 10.0, "y": 10.0},
            "alpha": 0.5,
        },
    }
    finalize_label_target(data)
    assert data["target"]["cross_x"] == 5.0
    assert data["target_quality"]["target_source"] == TARGET_SOURCE_BRACKET_MIDPOINT


def test_normalize_legacy_label() -> None:
    raw = {
        "target": {"cross_frame": 50, "cross_x": 500.0, "cross_y": 400.0},
        "quality": {"label_confidence": "high", "crossing_estimated": False},
    }
    normalize_label_target_fields(raw)
    assert raw["target_quality"]["target_source"] == TARGET_SOURCE_NEAREST_FRAME_OLD
    assert raw["target_quality"]["confidence"] == "unknown"
    assert raw["quality"]["crossing_estimated"] is True


def test_finalize_single_frame_target() -> None:
    data = {
        "target": {
            "cross_frame": 50,
            "cross_x": 500.0,
            "cross_y": 400.0,
            "target_source": TARGET_SOURCE_NEAREST_FRAME,
        }
    }
    finalize_label_target(data)
    assert data["target_quality"]["confidence"] == "low"
    assert data["target_quality"]["weight"] == target_weight_from_confidence("low")


def test_label_state_bracket_apply() -> None:
    from labeling.manual_label_pitch import LabelState

    state = LabelState(Path("pitch_001.mp4"), Path("out.json"), 60.0, 1920, 1080)
    state.data["target_bracket"] = {
        "before": {"frame": 10, "x": 0.0, "y": 0.0},
        "after": {"frame": 20, "x": 100.0, "y": 100.0},
        "alpha": 0.5,
    }
    state.apply_bracket_target(force_midpoint=True)
    assert state.data["target"]["cross_x"] == 50.0


def test_load_label_includes_target_quality(tmp_path: Path) -> None:
    from prediction.predict_location import load_label

    label_path = tmp_path / "pitch_001.json"
    label_path.write_text(
        json.dumps(
            {
                "pitch_id": "pitch_001",
                "release_frame": 1,
                "early_points": [
                    {"frame": 2, "x": 1.0, "y": 2.0},
                    {"frame": 3, "x": 2.0, "y": 3.0},
                ],
                "target": {"cross_frame": 10, "cross_x": 100.0, "cross_y": 200.0},
            }
        ),
        encoding="utf-8",
    )
    label = load_label(label_path)
    assert label.target_quality.target_source == TARGET_SOURCE_NEAREST_FRAME_OLD
    assert label.target_quality.confidence == "unknown"


def test_filter_by_target_confidence() -> None:
    from prediction.predict_location import (
        CONFIDENCE_HIGH,
        CONFIDENCE_MEDIUM,
        Prediction,
        filter_by_min_target_confidence,
    )

    preds = [
        Prediction("a", "m", 0, 0, 0, 0, 1, 0, 0, 3, {}, target_confidence=CONFIDENCE_HIGH, target_weight=1.0),
        Prediction("b", "m", 0, 0, 0, 0, 2, 0, 0, 3, {}, target_confidence=CONFIDENCE_MEDIUM, target_weight=0.5),
        Prediction("c", "m", 0, 0, 0, 0, 3, 0, 0, 3, {}, target_confidence="low", target_weight=0.2),
    ]
    medium_plus = filter_by_min_target_confidence(preds, CONFIDENCE_MEDIUM)
    assert len(medium_plus) == 2
    high_only = filter_by_min_target_confidence(preds, CONFIDENCE_HIGH)
    assert len(high_only) == 1

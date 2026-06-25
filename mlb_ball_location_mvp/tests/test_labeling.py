from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labeling.manual_label_pitch import LabelState


def test_validation_warnings_incomplete() -> None:
    state = LabelState(Path("pitch_001.mp4"), Path("out.json"), 60.0, 1920, 1080)
    warnings = state.validation_warnings(min_points=5)
    assert any("release_frame" in w for w in warnings)
    assert any("early points" in w for w in warnings)
    assert any("target crossing" in w for w in warnings)


def test_validation_warnings_detects_bad_frames() -> None:
    state = LabelState(Path("pitch_001.mp4"), Path("out.json"), 60.0, 1920, 1080)
    state.data["release_frame"] = 100
    state.data["early_points"] = [{"frame": 99, "x": 100.0, "y": 100.0}] * 5
    state.data["target"] = {"cross_frame": 100, "cross_x": 500.0, "cross_y": 400.0}
    warnings = state.validation_warnings(min_points=5)
    assert any("not after release_frame" in w for w in warnings)


def test_load_existing_label(tmp_path: Path) -> None:
    existing = {
        "pitch_id": "pitch_001",
        "release_frame": 50,
        "early_points": [{"frame": 51, "x": 10.0, "y": 20.0}],
        "target": {"cross_frame": None, "cross_x": None, "cross_y": None},
        "pitch_type": "fastball",
    }
    state = LabelState.from_existing(
        Path("pitch_001.mp4"),
        tmp_path / "pitch_001.json",
        60.0,
        1920,
        1080,
        existing,
    )
    assert state.frame_idx == 50
    assert state.data["pitch_type"] == "fastball"


def test_save_prints_warnings(capsys, tmp_path: Path) -> None:
    state = LabelState(Path("pitch_001.mp4"), tmp_path / "out.json", 60.0, 1920, 1080)
    state.save(min_points=5)
    captured = capsys.readouterr()
    assert "WARNING: label may be incomplete" in captured.out
    assert (tmp_path / "out.json").exists()

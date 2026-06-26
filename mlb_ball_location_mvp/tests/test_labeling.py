from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from labeling.manual_label_pitch import LabelState, apply_recording_fps_fields, format_timestamp_ms
from timing.sidecar import load_frame_timestamps


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


def test_apply_recording_fps_from_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "pitch_001.mp4"
    video.write_bytes(b"")
    meta = {"actual_fps": 27.1, "requested_fps": 60.0, "frame_count": 163}
    meta_sidecar = tmp_path / "pitch_001.meta.json"
    meta_sidecar.write_text(json.dumps(meta), encoding="utf-8")

    data: dict = {}
    resolved = apply_recording_fps_fields(data, video, container_fps=60.0)
    assert resolved == 27.1
    assert data["fps"] == 27.1
    assert data["requested_fps"] == 60.0
    assert data["container_fps"] == 60.0


def test_apply_recording_fps_falls_back_to_container(tmp_path: Path) -> None:
    video = tmp_path / "pitch_001.mp4"
    video.write_bytes(b"")
    data: dict = {}
    resolved = apply_recording_fps_fields(data, video, container_fps=60.0)
    assert resolved == 60.0
    assert data["fps"] == 60.0
    assert "requested_fps" not in data


def test_early_mode_hides_other_frame_markers() -> None:
    state = LabelState(Path("pitch_001.mp4"), Path("out.json"), 60.0, 1920, 1080)
    state.mode = "early"
    state.frame_idx = 102
    state.data["early_points"] = [
        {"frame": 101, "x": 10.0, "y": 10.0},
        {"frame": 102, "x": 20.0, "y": 20.0},
    ]
    assert state.should_show_ball_marker(101) is False
    assert state.should_show_ball_marker(102) is True


def test_load_frame_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "pitch_001.mp4"
    video.write_bytes(b"")
    sidecar = tmp_path / "pitch_001.frames.jsonl"
    sidecar.write_text(
        '{"frame": 0, "timestamp": 0.0}\n{"frame": 1, "timestamp": 0.016731}\n',
        encoding="utf-8",
    )
    timestamps = load_frame_timestamps(video)
    assert timestamps[1] == 0.016731
    assert format_timestamp_ms(timestamps[1]) == "16.731 ms"


def test_add_click_stores_timestamp_ms(tmp_path: Path) -> None:
    video = tmp_path / "pitch_001.mp4"
    video.write_bytes(b"")
    sidecar = tmp_path / "pitch_001.frames.jsonl"
    sidecar.write_text('{"frame": 5, "timestamp": 0.123456}\n', encoding="utf-8")
    state = LabelState(video, tmp_path / "out.json", 60.0, 1920, 1080)
    state.frame_idx = 5
    state.add_click(100, 200, frame_count=100)
    point = state.data["early_points"][0]
    assert point["timestamp_ms"] == 123.456

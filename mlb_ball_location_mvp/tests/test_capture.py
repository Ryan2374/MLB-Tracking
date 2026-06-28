from __future__ import annotations

import json
from pathlib import Path

from prediction.predict_location import iter_label_paths


def test_iter_label_paths_ignores_example_files(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "pitch_001.json").write_text("{}", encoding="utf-8")
    (labels_dir / "example_pitch.json").write_text("{}", encoding="utf-8")
    (labels_dir / "eval_n5.json").write_text("{}", encoding="utf-8")
    (labels_dir / "pitch_001_prediction.json").write_text("{}", encoding="utf-8")

    paths = iter_label_paths(labels_dir)
    assert [p.name for p in paths] == ["pitch_001.json"]


def test_sync_sidecars_to_encoded_trims_extra() -> None:
    from capture.record_clip import sync_sidecars_to_encoded

    sidecar = [{"frame": i, "timestamp": float(i)} for i in range(5)]
    trimmed, in_sync = sync_sidecars_to_encoded(sidecar, 3)
    assert in_sync is False
    assert len(trimmed) == 3
    assert trimmed[-1]["frame"] == 2


def test_rolling_fps_from_timestamps() -> None:
    from collections import deque

    from capture.record_clip import rolling_fps

    times = deque([0.0, 0.5, 1.0])
    assert rolling_fps(times) == 2.0


def test_rolling_fps_needs_at_least_two_samples() -> None:
    from collections import deque

    from capture.record_clip import rolling_fps

    assert rolling_fps(deque([1.0])) == 0.0
    assert rolling_fps(deque()) == 0.0


class _FakeCapture:
    def __init__(self, frames: list[bool]) -> None:
        self._frames = list(frames)

    def read(self) -> tuple[bool, None]:
        if not self._frames:
            return False, None
        ok = self._frames.pop(0)
        return ok, None


def test_drain_capture_buffer_stops_on_read_failure() -> None:
    from capture.record_clip import drain_capture_buffer

    cap = _FakeCapture([True, True, False, True])
    assert drain_capture_buffer(cap, max_frames=10) == 2


def test_preview_frame_downscales() -> None:
    import numpy as np

    from capture.record_clip import preview_frame

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    preview = preview_frame(frame, 1920, 1080, 0.5)
    assert preview.shape == (540, 960, 3)

from __future__ import annotations

import json
from pathlib import Path

from timing.sidecar import enrich_label_timing, write_uniform_sidecars


def test_enrich_label_timing_with_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "pitch_001.mp4"
    video.write_bytes(b"")
    write_uniform_sidecars(video, fps=30.0, frame_count=140, overwrite=True)

    label = {
        "release_frame": 100,
        "early_points": [{"frame": 101, "x": 1.0, "y": 2.0}],
        "target": {"cross_frame": 120, "cross_x": 3.0, "cross_y": 4.0},
    }
    enrich_label_timing(label, video)

    assert label["early_points"][0]["timestamp_ms"] == 3366.667
    assert label["target"]["cross_timestamp_ms"] == 4000.0
    assert label["timing"]["release_to_cross_ms"] == 666.667
    assert label["timing"]["source"] == "backfill_uniform_fps"


def test_write_uniform_sidecar_skips_capture_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "pitch_002.mp4"
    video.write_bytes(b"")
    meta = tmp_path / "pitch_002.meta.json"
    meta.write_text(json.dumps({"timing_source": "capture_sidecar"}), encoding="utf-8")
    frames = tmp_path / "pitch_002.frames.jsonl"
    frames.write_text('{"frame": 0, "timestamp": 0.0}\n', encoding="utf-8")

    result = write_uniform_sidecars(video, fps=30.0, frame_count=10, overwrite=False)
    assert result["status"] == "skipped"

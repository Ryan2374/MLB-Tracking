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

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from prediction.model_specs import EVAL_RUN_SPECS
from scripts.run_all_models import build_run_args, resolve_project_dir, validate_label_set


def test_validate_label_set_counts_pitch_files(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    for i in range(3):
        (labels / f"pitch_{i:03d}.json").write_text("{}", encoding="utf-8")
    assert validate_label_set(labels) == 3


def test_validate_label_set_rejects_synthetic(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "pitch_001.json").write_text("{}", encoding="utf-8")
    (labels / "synthetic_001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        validate_label_set(labels)


def test_build_run_args_from_spec(tmp_path: Path) -> None:
    spec = EVAL_RUN_SPECS[0]
    out = tmp_path / spec["source_file"]
    args = build_run_args(spec, tmp_path, out)
    assert isinstance(args, Namespace)
    assert args.method == spec["method"]
    assert args.n_points == spec["n_points"]
    assert args.out == str(out)


def test_resolve_project_dir_explicit(tmp_path: Path) -> None:
    assert resolve_project_dir(tmp_path) == tmp_path.resolve()

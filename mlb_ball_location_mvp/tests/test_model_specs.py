from __future__ import annotations

from prediction.model_specs import (
    EVAL_RUN_SPECS,
    EXPORT_OPTIONAL_RUN_IDS,
    export_specs,
    spec_for_run_id,
)


def test_eval_run_specs_count_and_unique_ids() -> None:
    assert len(EVAL_RUN_SPECS) == 11
    run_ids = [spec["run_id"] for spec in EVAL_RUN_SPECS]
    assert len(run_ids) == len(set(run_ids))


def test_export_specs_marks_optional_runs() -> None:
    specs = export_specs()
    assert len(specs) == len(EVAL_RUN_SPECS)
    optional = {spec["run_id"] for spec in specs if spec.get("optional")}
    assert optional == set(EXPORT_OPTIONAL_RUN_IDS)


def test_spec_for_run_id() -> None:
    spec = spec_for_run_id("compact_ridge_calibrated_n7")
    assert spec is not None
    assert spec["method"] == "compact_ridge_calibrated"
    assert spec_for_run_id("not_a_model") is None

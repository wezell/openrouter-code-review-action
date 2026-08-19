"""Tests for the Sub-AC 4.1 overlap threshold gate.

The gate reads a persisted ``eval/overlap_aggregate.json`` rollup and
fails when the configured metric (default: ``micro.recall``) drops
below the configured threshold (default: ``0.80``). These tests pin
the four contracts the sub-AC promises:

1. **Pass path**           — recall ≥ threshold → exit 0.
2. **Fail path**           — recall < threshold → exit 1, reason on
                              stderr.
3. **Unmeasurable**        — ``ready_count == 0`` (fixtures placeholder)
                              fails by default and only soft-passes
                              when ``--allow-unmeasurable`` is set.
4. **Metric switch**       — ``--metric`` selects micro/macro and any
                              of precision/recall/F1; bad metric is
                              rejected by argparse.

Persistence/aggregation math itself is covered in
``tests/test_overlap_aggregate.py`` and ``test_overlap_score_cli.py`` —
this suite exclusively pins the gate semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import check_overlap_threshold as gate

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _aggregate(
    *,
    ready_count: int = 2,
    micro: dict[str, Any] | None = None,
    macro: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal aggregate record shaped like ``overlap_aggregate.py``."""
    return {
        "schema_version": "1",
        "generated_at": "2026-05-07T00:00:00Z",
        "generator": "eval/overlap_aggregate.py",
        "tolerance_lines": 3,
        "sample_size": ready_count,
        "ready_count": ready_count,
        "metrics": {
            "micro": micro
            or {
                "true_positive": 8,
                "false_positive": 1,
                "false_negative": 1,
                "precision": 0.8889,
                "recall": 0.8889,
                "f1": 0.8889,
            },
            "macro": macro
            or {
                "precision": 0.85,
                "recall": 0.85,
                "f1": 0.85,
                "averaged_over_prs": ready_count,
            },
            "severity_match_rate": 0.75,
        },
    }


def _write(path: Path, record: dict[str, Any]) -> Path:
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# (1) Pass path
# ---------------------------------------------------------------------------


def test_evaluate_pass_at_or_above_threshold() -> None:
    record = _aggregate(
        micro={
            "true_positive": 8,
            "false_positive": 0,
            "false_negative": 2,
            "precision": 1.0,
            "recall": 0.80,
            "f1": 0.8889,
        }
    )
    result = gate.evaluate(record)
    assert result.passed is True
    assert result.metric == "micro.recall"
    assert result.value == pytest.approx(0.80)
    assert result.threshold == pytest.approx(0.80)
    assert result.ready_count == 2


def test_main_pass_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    agg = _write(tmp_path / "overlap_aggregate.json", _aggregate())
    rc = gate.main(["--aggregate", str(agg)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "overlap gate PASS" in captured.out
    assert captured.err == ""


# ---------------------------------------------------------------------------
# (2) Fail path
# ---------------------------------------------------------------------------


def test_evaluate_fails_below_threshold() -> None:
    record = _aggregate(
        micro={
            "true_positive": 7,
            "false_positive": 1,
            "false_negative": 3,
            "precision": 0.875,
            "recall": 0.70,
            "f1": 0.7778,
        }
    )
    result = gate.evaluate(record)
    assert result.passed is False
    assert result.value == pytest.approx(0.70)
    assert "0.7000" in result.reason
    assert "<" in result.reason


def test_main_fail_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = _aggregate(
        micro={
            "true_positive": 5,
            "false_positive": 2,
            "false_negative": 5,
            "precision": 0.7143,
            "recall": 0.50,
            "f1": 0.5882,
        }
    )
    agg = _write(tmp_path / "overlap_aggregate.json", bad)
    rc = gate.main(["--aggregate", str(agg)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "overlap gate FAIL" in captured.err
    # Threshold and observed value both surface in the failure message.
    assert "0.5000" in captured.err
    assert "0.8000" in captured.err


# ---------------------------------------------------------------------------
# (3) Unmeasurable: ready_count == 0
# ---------------------------------------------------------------------------


def test_unmeasurable_default_fails() -> None:
    record = _aggregate(
        ready_count=0,
        micro={
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        },
    )
    result = gate.evaluate(record)
    assert result.passed is False
    assert result.value is None
    assert "unmeasurable" in result.reason


def test_unmeasurable_with_allow_passes_with_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _aggregate(
        ready_count=0,
        micro={
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        },
    )
    agg = _write(tmp_path / "overlap_aggregate.json", record)
    rc = gate.main(["--aggregate", str(agg), "--allow-unmeasurable"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "PASS" in captured.out


def test_unmeasurable_default_main_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = _aggregate(
        ready_count=0,
        micro={
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": None,
            "recall": None,
            "f1": None,
        },
    )
    agg = _write(tmp_path / "overlap_aggregate.json", record)
    rc = gate.main(["--aggregate", str(agg)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL" in err
    assert "ready_count=0" in err


# ---------------------------------------------------------------------------
# (4) Metric switch
# ---------------------------------------------------------------------------


def test_metric_switch_to_micro_precision() -> None:
    record = _aggregate(
        micro={
            "true_positive": 8,
            "false_positive": 4,
            "false_negative": 0,
            "precision": 0.6667,  # below threshold
            "recall": 1.0,  # above threshold
            "f1": 0.80,
        }
    )
    # Recall passes...
    assert gate.evaluate(record, metric="micro.recall").passed is True
    # ...but precision should fail.
    failed = gate.evaluate(record, metric="micro.precision")
    assert failed.passed is False
    assert failed.metric == "micro.precision"
    assert failed.value == pytest.approx(0.6667)


def test_metric_switch_to_macro_f1() -> None:
    record = _aggregate(
        macro={
            "precision": 0.95,
            "recall": 0.95,
            "f1": 0.95,
            "averaged_over_prs": 2,
        }
    )
    result = gate.evaluate(record, metric="macro.f1", threshold=0.90)
    assert result.passed is True
    assert result.metric == "macro.f1"
    assert result.value == pytest.approx(0.95)


def test_invalid_metric_rejected_by_argparse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agg = _write(tmp_path / "overlap_aggregate.json", _aggregate())
    with pytest.raises(SystemExit):
        gate.main(["--aggregate", str(agg), "--metric", "weird.thing"])
    err = capsys.readouterr().err
    assert "weird.thing" in err or "invalid choice" in err


# ---------------------------------------------------------------------------
# Threshold validation + missing aggregate
# ---------------------------------------------------------------------------


def test_threshold_out_of_range_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agg = _write(tmp_path / "overlap_aggregate.json", _aggregate())
    with pytest.raises(SystemExit):
        gate.main(["--aggregate", str(agg), "--threshold", "1.5"])


def test_missing_aggregate_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = gate.main(["--aggregate", str(tmp_path / "nope.json")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "missing aggregate" in err


def test_corrupt_aggregate_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    agg = tmp_path / "overlap_aggregate.json"
    agg.write_text("{not json", encoding="utf-8")
    rc = gate.main(["--aggregate", str(agg)])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_custom_threshold_pass_at_lower_bar() -> None:
    record = _aggregate(
        micro={
            "true_positive": 6,
            "false_positive": 2,
            "false_negative": 4,
            "precision": 0.75,
            "recall": 0.60,
            "f1": 0.6667,
        }
    )
    # Default 0.80 fails:
    assert gate.evaluate(record).passed is False
    # 0.50 threshold passes:
    relaxed = gate.evaluate(record, threshold=0.50)
    assert relaxed.passed is True
    assert relaxed.threshold == pytest.approx(0.50)

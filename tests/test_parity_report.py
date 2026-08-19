"""Tests for the consolidated parity report generator (Sub-AC 4.2).

The generator merges three pre-existing artifact tiers into a single
human-readable Markdown report and a JSON sibling. These tests exercise
the contract by constructing tiny synthetic fixtures in a tmp_path tree:

* per-PR overlap.json files under ``runs/<pr-id>/overlap.json``
* an aggregate rollup at ``overlap_aggregate.json``
* a Codex-vs-OpenRouter parity ``_summary.json`` (plus per-PR JSON)

We then drive the public functions ``build_consolidated_report``,
``write_report``, ``check_report``, and the ``main`` CLI to confirm:

* Markdown contains the threshold gate banner, the rollup table, the
  per-PR table, and the source-artifact section.
* JSON payload mirrors the Markdown content one-to-one, including
  unmeasurable rows (status pending, metrics ``None``).
* Drift detection flips on disk content vs. rebuilt content, and the
  CLI returns 0 / 1 accordingly.
* Atomic writes never leave a half-written file behind.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from eval import parity_report

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_runs(runs_root: Path) -> None:
    """Two PRs: one ready with metrics, one pending (unmeasurable)."""
    _write_json(
        runs_root / "dotcms-core-35449" / "overlap.json",
        {
            "pr_id": "dotcms-core-35449",
            "fixture_pr_id": "pr-35449",
            "status": "ready",
            "head_sha_baseline": "abc123",
            "head_sha_candidate": "abc123",
            "counts": {
                "codex_total": 5,
                "openrouter_total": 4,
                "matched_pairs": 4,
                "codex_only": 1,
                "openrouter_only": 0,
            },
            "metrics": {
                "precision": 1.0,
                "recall": 0.8,
                "f1": 0.8888888888888888,
            },
            "models": {
                "codex": "gpt-5-codex",
                "openrouter": "anthropic/claude-opus-4.7",
            },
        },
    )
    _write_json(
        runs_root / "dotcms-core-35458" / "overlap.json",
        {
            "pr_id": "dotcms-core-35458",
            "fixture_pr_id": "pr-35458",
            "status": "pending",
            "head_sha_baseline": None,
            "head_sha_candidate": None,
            "counts": {
                "codex_total": 0,
                "openrouter_total": 0,
                "matched_pairs": 0,
                "codex_only": 0,
                "openrouter_only": 0,
            },
            "metrics": {
                "precision": None,
                "recall": None,
                "f1": None,
            },
            "models": {
                "codex": None,
                "openrouter": None,
            },
        },
    )


def _make_aggregate(path: Path) -> None:
    _write_json(
        path,
        {
            "sample_size": 2,
            "ready_count": 1,
            "status_counts": {"ready": 1, "pending": 1},
            "metrics": {
                "micro": {
                    "precision": 1.0,
                    "recall": 0.8,
                    "f1": 0.8888888888888888,
                },
                "macro": {
                    "precision": 1.0,
                    "recall": 0.8,
                    "f1": 0.8888888888888888,
                },
                "severity_match_rate": 0.75,
            },
            "per_category": {
                "P1": {
                    "codex_total": 3,
                    "openrouter_total": 3,
                    "matched_codex": 3,
                    "matched_openrouter": 3,
                    "precision": 1.0,
                    "recall": 1.0,
                },
                "P2": {
                    "codex_total": 2,
                    "openrouter_total": 1,
                    "matched_codex": 1,
                    "matched_openrouter": 1,
                    "precision": 1.0,
                    "recall": 0.5,
                },
            },
            "prs": [
                {
                    "pr_id": "pr-35449",
                    "status": "ready",
                },
                {
                    "pr_id": "pr-35458",
                    "status": "pending",
                },
            ],
        },
    )


def _make_parity(parity_root: Path) -> None:
    _write_json(
        parity_root / "_summary.json",
        {
            "prs": [
                {
                    "pr_id": "pr-35449",
                    "status": "ready",
                    "severity_match": 3,
                },
                {
                    "pr_id": "pr-35458",
                    "status": "pending",
                    "severity_match": 0,
                },
            ],
        },
    )
    _write_json(
        parity_root / "pr-35449.json",
        {
            "pr_id": "pr-35449",
            "status": "ready",
            "overlap": {
                "overlap_ratio_codex_baseline": 0.8,
                "overlap_ratio_openrouter_candidate": 1.0,
                "severity_match_count": 3,
            },
        },
    )
    _write_json(
        parity_root / "pr-35458.json",
        {
            "pr_id": "pr-35458",
            "status": "pending",
            "overlap": {
                "overlap_ratio_codex_baseline": None,
                "overlap_ratio_openrouter_candidate": None,
                "severity_match_count": 0,
            },
        },
    )


@pytest.fixture()
def fixture_tree(tmp_path: Path) -> dict[str, Path]:
    runs = tmp_path / "runs"
    aggregate = tmp_path / "overlap_aggregate.json"
    parity = tmp_path / "parity_report"
    _make_runs(runs)
    _make_aggregate(aggregate)
    _make_parity(parity)
    return {
        "runs_root": runs,
        "aggregate_path": aggregate,
        "parity_summary_path": parity / "_summary.json",
        "parity_root": parity,
        "md_path": tmp_path / "parity_report.md",
        "json_path": tmp_path / "parity_report.json",
    }


# ---------------------------------------------------------------------------
# build_consolidated_report
# ---------------------------------------------------------------------------


def test_build_consolidated_report_emits_rows_and_markdown(
    fixture_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Per-PR JSONs live under PARITY_REPORT_ROOT (module constant); steer
    # the lookup at the synthetic fixture tree for this test.
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])

    payload, markdown = parity_report.build_consolidated_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
        generated_at="2026-01-01T00:00:00Z",
    )

    # JSON payload structure
    assert payload["schema_version"] == parity_report.SCHEMA_VERSION
    assert payload["generator"] == parity_report.GENERATOR_REL_PATH
    assert payload["threshold_gate"]["metric"] == "micro.recall"
    assert payload["threshold_gate"]["passed"] is True
    assert payload["aggregate"]["sample_size"] == 2
    assert payload["aggregate"]["ready_count"] == 1
    assert payload["aggregate"]["metrics"]["micro"]["recall"] == pytest.approx(0.8)

    pr_ids = sorted(row["pr_id"] for row in payload["prs"])
    assert pr_ids == ["dotcms-core-35449", "dotcms-core-35458"]

    ready_row = next(r for r in payload["prs"] if r["pr_id"] == "dotcms-core-35449")
    assert ready_row["fixture_pr_id"] == "pr-35449"
    assert ready_row["status"] == "ready"
    assert ready_row["matched_pairs"] == 4
    assert ready_row["delta_total"] == -1
    assert ready_row["severity_match"] == 3
    assert ready_row["recall"] == pytest.approx(0.8)
    assert ready_row["overlap_ratio_codex_baseline"] == pytest.approx(0.8)

    pending_row = next(r for r in payload["prs"] if r["pr_id"] == "dotcms-core-35458")
    assert pending_row["status"] == "pending"
    assert pending_row["recall"] is None
    assert pending_row["precision"] is None
    assert pending_row["overlap_ratio_codex_baseline"] is None

    # Markdown surface
    assert "# Codex-vs-OpenRouter Parity Report" in markdown
    assert "**Threshold gate:** PASS" in markdown
    assert "## Sample Rollup" in markdown
    assert "## Per-PR Details" in markdown
    assert "## Source Artifacts" in markdown
    assert "dotcms-core-35449" in markdown
    assert "dotcms-core-35458" in markdown


def test_build_consolidated_report_unmeasurable_aggregate(tmp_path: Path) -> None:
    """An empty pipeline produces n/a metrics and FAIL banner unless allowed."""
    runs = tmp_path / "runs"
    runs.mkdir()
    aggregate = tmp_path / "agg.json"
    _write_json(
        aggregate,
        {
            "sample_size": 0,
            "ready_count": 0,
            "status_counts": {},
            "metrics": {
                "micro": {"precision": None, "recall": None, "f1": None},
                "macro": {"precision": None, "recall": None, "f1": None},
            },
            "per_category": {},
            "prs": [],
        },
    )
    parity = tmp_path / "parity"
    parity.mkdir()
    _write_json(parity / "_summary.json", {"prs": []})

    payload, md = parity_report.build_consolidated_report(
        runs_root=runs,
        aggregate_path=aggregate,
        parity_summary_path=parity / "_summary.json",
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=True,
        generated_at="2026-01-01T00:00:00Z",
    )
    assert payload["threshold_gate"]["passed"] is True  # allow_unmeasurable
    assert payload["aggregate"]["sample_size"] == 0
    assert payload["prs"] == []
    assert "No PRs discovered" in md


# ---------------------------------------------------------------------------
# write_report + check_report (drift)
# ---------------------------------------------------------------------------


def test_write_then_check_is_clean(
    fixture_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])

    md_path, json_path = parity_report.write_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        md_path=fixture_tree["md_path"],
        json_path=fixture_tree["json_path"],
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
        generated_at="2026-01-01T00:00:00Z",
    )
    assert md_path.is_file()
    assert json_path.is_file()
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert md_path.read_text(encoding="utf-8").endswith("\n")

    drift = parity_report.check_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        md_path=md_path,
        json_path=json_path,
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
    )
    assert drift == [], f"unexpected drift: {drift}"


def test_check_report_detects_drift(
    fixture_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])

    parity_report.write_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        md_path=fixture_tree["md_path"],
        json_path=fixture_tree["json_path"],
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
        generated_at="2026-01-01T00:00:00Z",
    )

    # Mutate the JSON on disk (corrupt a count)
    on_disk = json.loads(fixture_tree["json_path"].read_text(encoding="utf-8"))
    on_disk["aggregate"]["sample_size"] = 99
    fixture_tree["json_path"].write_text(json.dumps(on_disk, indent=2, sort_keys=True))

    drift = parity_report.check_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        md_path=fixture_tree["md_path"],
        json_path=fixture_tree["json_path"],
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
    )
    assert any("JSON differs" in d for d in drift)


def test_check_report_missing_files(
    fixture_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])

    drift = parity_report.check_report(
        runs_root=fixture_tree["runs_root"],
        aggregate_path=fixture_tree["aggregate_path"],
        parity_summary_path=fixture_tree["parity_summary_path"],
        md_path=fixture_tree["md_path"],
        json_path=fixture_tree["json_path"],
        threshold=0.80,
        metric="micro.recall",
        allow_unmeasurable=False,
    )
    joined = "\n".join(drift)
    assert "missing JSON report" in joined
    assert "missing Markdown report" in joined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_then_check_returns_zero(
    fixture_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])
    # Steer module-level defaults at the synthetic fixtures.
    monkeypatch.setattr(parity_report, "RUNS_ROOT", fixture_tree["runs_root"])
    monkeypatch.setattr(parity_report, "AGGREGATE_PATH", fixture_tree["aggregate_path"])
    monkeypatch.setattr(parity_report, "PARITY_SUMMARY_PATH", fixture_tree["parity_summary_path"])

    rc = parity_report.main(
        [
            "--md-out",
            str(fixture_tree["md_path"]),
            "--json-out",
            str(fixture_tree["json_path"]),
            "--threshold",
            "0.80",
            "--metric",
            "micro.recall",
        ]
    )
    assert rc == 0
    assert fixture_tree["md_path"].is_file()
    assert fixture_tree["json_path"].is_file()

    rc = parity_report.main(
        [
            "--check",
            "--md-out",
            str(fixture_tree["md_path"]),
            "--json-out",
            str(fixture_tree["json_path"]),
            "--threshold",
            "0.80",
            "--metric",
            "micro.recall",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "in sync" in captured.out


def test_cli_check_returns_one_on_drift(
    fixture_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(parity_report, "PARITY_REPORT_ROOT", fixture_tree["parity_root"])
    monkeypatch.setattr(parity_report, "RUNS_ROOT", fixture_tree["runs_root"])
    monkeypatch.setattr(parity_report, "AGGREGATE_PATH", fixture_tree["aggregate_path"])
    monkeypatch.setattr(parity_report, "PARITY_SUMMARY_PATH", fixture_tree["parity_summary_path"])

    rc = parity_report.main(
        [
            "--check",
            "--md-out",
            str(fixture_tree["md_path"]),
            "--json-out",
            str(fixture_tree["json_path"]),
            "--threshold",
            "0.80",
            "--metric",
            "micro.recall",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "drifted from rebuild" in captured.err

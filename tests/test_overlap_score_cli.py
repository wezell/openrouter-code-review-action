"""CLI tests for ``eval.overlap_score`` (Sub-AC 3.5).

The matching engine itself is exhaustively pinned in
``tests/test_overlap_score.py`` (52 cases against the §6.4 worked
example, the predicate, determinism, and tie-breaks). Sub-AC 3.5 ships
the CLI orchestrator on top — these tests cover the four contracts the
sub-AC calls out:

1. **matching**  — the CLI delegates to the same ``match_findings``
   engine, so a ready fixture lands a real match and writes a non-zero
   ``true_positive`` count to disk.
2. **persistence** — invoking the CLI without ``--check`` writes both
   ``eval/runs/<runs_id>/overlap.json`` and the sample-wide
   ``eval/overlap_aggregate.json`` artifact, atomically and
   idempotently.
3. **--check drift mode** — re-running ``--check`` immediately after a
   write returns 0; mutating either layer flips the exit code to 1 and
   surfaces the drifting layer name on stderr.
4. **aggregate math** — the rolled-up record reflects the per-PR match
   counts (TP/FP/FN summed) and computes pooled precision/recall/F1
   over the staged sample.

Tests use ``tmp_path`` plus ``monkeypatch`` to redirect every on-disk
root (baseline / candidate fixtures, ``eval/runs/``, the sample
registry, the aggregate output path) so the CLI never touches the real
repository state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from eval import compare_findings as cf
from eval import overlap_aggregate as oa
from eval import overlap_metrics as om
from eval import overlap_score as os_mod

# ---------------------------------------------------------------------------
# Synthetic fixtures (mirrors tests/test_overlap_metrics.py shape)
# ---------------------------------------------------------------------------


def _finding(
    *,
    title: str = "🔴 [P1] foo.py:10 thing",
    path: str = "foo.py",
    start: int = 10,
    end: int | None = None,
    priority: int | None = 1,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": "details",
        "confidence_score": 0.9,
        "priority": priority,
        "side": "RIGHT",
        "code_location": {
            "absolute_file_path": path,
            "line_range": {
                "start": start,
                "end": end if end is not None else start,
            },
        },
    }


def _wrapper(
    *,
    pr_id: str,
    head_sha: str,
    findings: list[dict[str, Any]],
    run: str,
    model: str,
) -> dict[str, Any]:
    return {
        "pr_id": pr_id,
        "run": run,
        "head_sha": head_sha,
        "model": model,
        "reasoning_effort": "medium",
        "web_search_mode": "disabled",
        "prior_review_state": None,
        "review_run_result": {
            "overall_correctness": "ok",
            "overall_explanation": "ok",
            "overall_confidence_score": 0.8,
            "findings": findings,
            "carried_forward": [],
        },
        "posted": {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
        "capture": {
            "captured_at": None,
            "command": "synthetic",
            "notes": "synthetic",
            "runner_version": "1",
            "status": "captured" if run == "codex" else "completed",
        },
        "source": {
            "eval_artifact": "eval/runs/synthetic.json",
            "produced_by": "synthetic",
            "mirrored_by": "synthetic",
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stage_pr(
    *,
    baseline_root: Path,
    candidate_root: Path,
    pr_fixture_id: str,
    head_sha: str,
    codex_findings: list[dict[str, Any]],
    openrouter_findings: list[dict[str, Any]],
) -> None:
    _write_json(
        baseline_root / pr_fixture_id / "codex-findings.json",
        _wrapper(
            pr_id=pr_fixture_id,
            head_sha=head_sha,
            findings=codex_findings,
            run="codex",
            model="gpt-5.4",
        ),
    )
    _write_json(
        candidate_root / pr_fixture_id / "openrouter-findings.json",
        _wrapper(
            pr_id=pr_fixture_id,
            head_sha=head_sha,
            findings=openrouter_findings,
            run="openrouter",
            model="anthropic/claude-opus-4.7",
        ),
    )


@pytest.fixture
def staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect every persistence root the CLI touches into ``tmp_path``.

    Lays out two PRs with deterministic match counts:

    * ``dotcms-core-1`` — TP=1, FP=1, FN=1 → p=0.5, r=0.5, f1=0.5
    * ``dotcms-core-2`` — TP=2, FP=0, FN=0 → p=1.0, r=1.0, f1=1.0

    Pooled (micro): TP=3, FP=1, FN=1 → p=0.75, r=0.75, f1=0.75.
    """
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    runs = tmp_path / "runs"
    baseline.mkdir()
    candidate.mkdir()
    runs.mkdir()

    # Sample registry the aggregate layer reads.
    registry = tmp_path / "sample-prs.yaml"
    registry.write_text(
        "samples:\n"
        "  - id: dotcms-core-1\n"
        "    repo: dotcms/core\n"
        "    number: 1\n"
        "  - id: dotcms-core-2\n"
        "    repo: dotcms/core\n"
        "    number: 2\n",
        encoding="utf-8",
    )
    aggregate_path = tmp_path / "overlap_aggregate.json"

    # Stage both PRs.
    sha = "aa" * 20
    _stage_pr(
        baseline_root=baseline,
        candidate_root=candidate,
        pr_fixture_id="pr-1",
        head_sha=sha,
        codex_findings=[
            _finding(path="foo.py", start=10),
            _finding(path="bar.py", start=20),
        ],
        openrouter_findings=[
            _finding(path="foo.py", start=10),  # TP
            _finding(path="other.py", start=99),  # FP
        ],
    )
    _stage_pr(
        baseline_root=baseline,
        candidate_root=candidate,
        pr_fixture_id="pr-2",
        head_sha=sha,
        codex_findings=[
            _finding(path="zap.py", start=5),
            _finding(path="zap.py", start=40),
        ],
        openrouter_findings=[
            _finding(path="zap.py", start=5),
            _finding(path="zap.py", start=40),
        ],
    )

    # Re-route the per-PR + aggregate roots so neither writes into the
    # real repo.
    monkeypatch.setattr(cf, "BASELINE_ROOT", baseline)
    monkeypatch.setattr(cf, "CANDIDATE_ROOT", candidate)
    monkeypatch.setattr(om, "RUNS_ROOT", runs)
    monkeypatch.setattr(oa, "SAMPLE_REGISTRY_PATH", registry)
    monkeypatch.setattr(oa, "AGGREGATE_PATH", aggregate_path)
    # Need ``discover_runs_ids`` to see the staged dirs.
    return {
        "baseline": baseline,
        "candidate": candidate,
        "runs": runs,
        "registry": registry,
        "aggregate_path": aggregate_path,
    }


# ---------------------------------------------------------------------------
# (1) Matching: CLI surfaces a real TP through the engine
# ---------------------------------------------------------------------------


class TestCLIMatching:
    def test_cli_writes_per_pr_record_with_correct_match_count(
        self, staged: dict[str, Path]
    ) -> None:
        # Pre-create the runs directories so ``discover_runs_ids`` finds
        # them — the CLI only persists into directories it discovers,
        # mirroring the production layout under eval/runs/.
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()

        rc = os_mod.main([])
        assert rc == 0

        record_1 = json.loads((staged["runs"] / "dotcms-core-1" / "overlap.json").read_text())
        # Engine semantics: 1 TP, 1 FP, 1 FN — proves the CLI delegates
        # to ``match_findings`` and the predicate fires on path+line.
        assert record_1["metrics"]["true_positive"] == 1
        assert record_1["metrics"]["false_positive"] == 1
        assert record_1["metrics"]["false_negative"] == 1
        assert record_1["metrics"]["precision"] == 0.5
        assert record_1["metrics"]["recall"] == 0.5

        record_2 = json.loads((staged["runs"] / "dotcms-core-2" / "overlap.json").read_text())
        assert record_2["metrics"]["true_positive"] == 2
        assert record_2["metrics"]["false_positive"] == 0
        assert record_2["metrics"]["false_negative"] == 0
        assert record_2["metrics"]["precision"] == 1.0
        assert record_2["metrics"]["recall"] == 1.0

    def test_pr_id_filter_only_writes_filtered_target(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()

        # Only run for PR 2; PR 1 should stay un-persisted at the per-PR
        # layer.
        rc = os_mod.main(["--pr-id", "dotcms-core-2", "--skip-aggregate"])
        assert rc == 0
        assert not (staged["runs"] / "dotcms-core-1" / "overlap.json").exists()
        assert (staged["runs"] / "dotcms-core-2" / "overlap.json").exists()

    def test_unknown_pr_id_returns_nonzero_with_helpful_message(
        self,
        staged: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        rc = os_mod.main(["--pr-id", "dotcms-core-9999"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "unknown pr_id(s)" in err
        assert "dotcms-core-9999" in err


# ---------------------------------------------------------------------------
# (2) Persistence: both per-PR and aggregate artifacts land
# ---------------------------------------------------------------------------


class TestCLIPersistence:
    def test_default_run_writes_both_layers(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()

        rc = os_mod.main([])
        assert rc == 0
        assert (staged["runs"] / "dotcms-core-1" / "overlap.json").is_file()
        assert (staged["runs"] / "dotcms-core-2" / "overlap.json").is_file()
        assert staged["aggregate_path"].is_file()

    def test_skip_aggregate_only_writes_per_pr(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        rc = os_mod.main(["--skip-aggregate"])
        assert rc == 0
        assert (staged["runs"] / "dotcms-core-1" / "overlap.json").is_file()
        assert not staged["aggregate_path"].exists()

    def test_idempotent_writes_produce_identical_content(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()

        os_mod.main([])
        first = (staged["runs"] / "dotcms-core-1" / "overlap.json").read_text()
        first_agg = staged["aggregate_path"].read_text()

        os_mod.main([])
        second = (staged["runs"] / "dotcms-core-1" / "overlap.json").read_text()
        second_agg = staged["aggregate_path"].read_text()

        # ``generated_at`` is the only volatile field; strip it.
        first_obj = json.loads(first)
        second_obj = json.loads(second)
        first_obj.pop("generated_at", None)
        second_obj.pop("generated_at", None)
        assert first_obj == second_obj

        first_agg_obj = json.loads(first_agg)
        second_agg_obj = json.loads(second_agg)
        first_agg_obj.pop("generated_at", None)
        second_agg_obj.pop("generated_at", None)
        assert first_agg_obj == second_agg_obj

    def test_no_runs_directory_skips_per_pr_quietly(
        self,
        staged: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # No ``dotcms-core-*`` subdirs under runs/ → per-PR layer is a
        # no-op but the aggregate still writes (over a sample registry
        # whose targets resolve to ``pending`` placeholders).
        rc = os_mod.main([])
        assert rc == 0
        assert staged["aggregate_path"].is_file()
        err = capsys.readouterr().err
        assert "nothing to do for per-PR layer" in err


# ---------------------------------------------------------------------------
# (3) --check drift mode
# ---------------------------------------------------------------------------


class TestCLICheck:
    def test_check_after_fresh_write_returns_zero(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()

        assert os_mod.main([]) == 0
        assert os_mod.main(["--check"]) == 0

    def test_check_flags_per_pr_drift(
        self,
        staged: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()
        os_mod.main([])

        # Mutate one per-PR record so drift surfaces.
        target = staged["runs"] / "dotcms-core-1" / "overlap.json"
        record = json.loads(target.read_text())
        record["metrics"]["true_positive"] = 999
        target.write_text(json.dumps(record), encoding="utf-8")

        rc = os_mod.main(["--check"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "overlap pipeline drifted" in err
        assert "dotcms-core-1" in err

    def test_check_flags_aggregate_drift(
        self,
        staged: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()
        os_mod.main([])

        agg = json.loads(staged["aggregate_path"].read_text())
        # Mutate any deeply-nested numeric field guaranteed to exist.
        agg["schema_version"] = "tampered"
        staged["aggregate_path"].write_text(json.dumps(agg), encoding="utf-8")

        rc = os_mod.main(["--check"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "aggregate" in err

    def test_check_missing_per_pr_record(
        self,
        staged: dict[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()
        os_mod.main([])
        # Delete one record: the per-PR layer should flag the absence.
        (staged["runs"] / "dotcms-core-1" / "overlap.json").unlink()

        rc = os_mod.main(["--check"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "missing overlap.json" in err

    def test_check_skip_aggregate_only_validates_per_pr(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        os_mod.main(["--skip-aggregate"])
        # Aggregate file does not exist; default --check would fail. With
        # ``--skip-aggregate`` we only validate the per-PR layer, which
        # is in sync.
        assert not staged["aggregate_path"].exists()
        rc = os_mod.main(["--check", "--skip-aggregate"])
        assert rc == 0


# ---------------------------------------------------------------------------
# (4) Aggregate math
# ---------------------------------------------------------------------------


class TestCLIAggregateMath:
    def test_aggregate_pools_per_pr_counts(self, staged: dict[str, Path]) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()
        assert os_mod.main([]) == 0

        agg = json.loads(staged["aggregate_path"].read_text())
        # Per-PR rows for both stage entries should be present.
        per_pr = {row["pr_id"]: row for row in agg["prs"]}
        assert {"dotcms-core-1", "dotcms-core-2"} <= set(per_pr)

        # PR 1: 1/1/1, PR 2: 2/0/0 → micro pool 3/1/1.
        micro = agg["metrics"]["micro"]
        assert micro["true_positive"] == 3
        assert micro["false_positive"] == 1
        assert micro["false_negative"] == 1
        # precision = 3 / (3+1) = 0.75; recall = 3 / (3+1) = 0.75
        assert micro["precision"] == 0.75
        assert micro["recall"] == 0.75
        # F1 = 2 * 0.75 * 0.75 / (0.75 + 0.75) = 0.75
        assert micro["f1"] == 0.75

    def test_aggregate_macro_average_is_mean_of_per_pr_metrics(
        self, staged: dict[str, Path]
    ) -> None:
        (staged["runs"] / "dotcms-core-1").mkdir()
        (staged["runs"] / "dotcms-core-2").mkdir()
        os_mod.main([])

        agg = json.loads(staged["aggregate_path"].read_text())
        macro = agg["metrics"]["macro"]
        # Per-PR precisions: 0.5 (PR1), 1.0 (PR2) → macro mean = 0.75.
        # Same for recall and F1 in this fixture.
        assert macro["precision"] == pytest.approx(0.75, abs=1e-4)
        assert macro["recall"] == pytest.approx(0.75, abs=1e-4)
        assert macro["f1"] == pytest.approx(0.75, abs=1e-4)

    def test_aggregate_excludes_unmeasurable_prs_from_metrics(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # One PR ready, one PR with only the Codex side captured. The
        # aggregate must compute metrics from the ready PR alone — the
        # pending PR contributes counts but not pooled precision/recall.
        baseline = tmp_path / "baseline"
        candidate = tmp_path / "candidate"
        runs = tmp_path / "runs"
        for d in (baseline, candidate, runs):
            d.mkdir()

        registry = tmp_path / "sample-prs.yaml"
        registry.write_text(
            "samples:\n"
            "  - id: dotcms-core-7\n"
            "    repo: dotcms/core\n"
            "    number: 7\n"
            "  - id: dotcms-core-8\n"
            "    repo: dotcms/core\n"
            "    number: 8\n",
            encoding="utf-8",
        )
        agg_path = tmp_path / "overlap_aggregate.json"

        # PR 7 is ready (perfect match).
        sha = "bb" * 20
        _stage_pr(
            baseline_root=baseline,
            candidate_root=candidate,
            pr_fixture_id="pr-7",
            head_sha=sha,
            codex_findings=[_finding(path="foo.py", start=1)],
            openrouter_findings=[_finding(path="foo.py", start=1)],
        )
        # PR 8: only Codex side captured.
        _write_json(
            baseline / "pr-8" / "codex-findings.json",
            _wrapper(
                pr_id="pr-8",
                head_sha=sha,
                findings=[_finding(path="bar.py", start=2)],
                run="codex",
                model="gpt-5.4",
            ),
        )

        monkeypatch.setattr(cf, "BASELINE_ROOT", baseline)
        monkeypatch.setattr(cf, "CANDIDATE_ROOT", candidate)
        monkeypatch.setattr(om, "RUNS_ROOT", runs)
        monkeypatch.setattr(oa, "SAMPLE_REGISTRY_PATH", registry)
        monkeypatch.setattr(oa, "AGGREGATE_PATH", agg_path)

        (runs / "dotcms-core-7").mkdir()

        rc = os_mod.main(["--pr-id", "dotcms-core-7"])
        assert rc == 0

        agg = json.loads(agg_path.read_text())
        # Both PRs surface in per_pr rows (lossless).
        per_pr_ids = {row["pr_id"] for row in agg["prs"]}
        assert per_pr_ids == {"dotcms-core-7", "dotcms-core-8"}
        # Pooled metrics derive only from the ready PR (1 TP, 0 FP, 0 FN).
        micro = agg["metrics"]["micro"]
        assert micro["true_positive"] == 1
        assert micro["false_positive"] == 0
        assert micro["false_negative"] == 0
        assert micro["precision"] == 1.0
        assert micro["recall"] == 1.0


# ---------------------------------------------------------------------------
# Negative tolerance / arg-parser sanity
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_negative_tolerance_rejected(self, staged: dict[str, Path]) -> None:
        with pytest.raises(SystemExit):
            os_mod.main(["--tolerance", "-1"])

    def test_help_flag_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            os_mod.main(["--help"])
        assert exc.value.code == 0

"""Sub-AC 1.4 — pytest contract for the baseline fixture loader.

These tests exercise ``tests/baseline_loader.py`` two ways:

1. **Real-fixture sweep** — load every fixture in
   ``tests/fixtures/baseline/`` and assert the loader's well-formedness
   contract holds. This is the regression gate the later
   parity-comparison ACs depend on: if any fixture goes off-shape, this
   suite fails before bake-off scoring runs.

2. **Synthetic fixture probes** — build deliberately broken fixtures in
   ``tmp_path`` and assert the loader catches each defect with a
   readable error. Without these, a regression in the validator would
   silently weaken the contract for the real fixtures.

Per the methodology (``eval/baseline-methodology.md`` §2.3), fixtures
checked into the repo are *placeholders* until live capture runs with
``GITHUB_TOKEN``. Sub-AC 1.4's "non-empty findings" assertion therefore
only applies to fixtures whose ``capture.status`` is ``"captured"``.
The synthetic-probe tests pin both the captured-status non-emptiness
rule and the placeholder-status null-review rule.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.baseline_loader import (
    ALLOWED_CAPTURE_STATES,
    BaselineFixture,
    BaselineFixtureError,
    iter_baseline_fixtures,
    list_fixture_pr_ids,
    load_all_baseline_fixtures,
    load_baseline_fixture,
)

# ---------------------------------------------------------------------------
# Real fixture sweep — every checked-in fixture must validate.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_fixture_ids() -> list[str]:
    return list_fixture_pr_ids()


def test_sample_registry_yields_at_least_five_fixtures(
    all_fixture_ids: list[str],
) -> None:
    """Sub-AC 1.1 selected 5–10 PRs; the loader must surface them all."""
    assert 5 <= len(all_fixture_ids) <= 10
    # IDs must be unique and follow the pr-<NUMBER> convention.
    assert len(set(all_fixture_ids)) == len(all_fixture_ids)
    for pr_id in all_fixture_ids:
        assert pr_id.startswith("pr-")
        int(pr_id.removeprefix("pr-"))  # raises if non-integer suffix


@pytest.mark.parametrize("pr_id", list_fixture_pr_ids())
def test_each_fixture_loads_and_validates(pr_id: str) -> None:
    """Every PR in the sample registry must have a well-formed fixture."""
    fixture = load_baseline_fixture(pr_id)
    assert isinstance(fixture, BaselineFixture)
    assert fixture.pr_id == pr_id
    assert fixture.fixture_dir.is_dir()
    assert fixture.capture_status in ALLOWED_CAPTURE_STATES
    # Diff must be a real unified diff — guards against accidentally
    # writing an HTML error page or empty stdout.
    assert fixture.diff_text.startswith("diff --git ")
    # The codex baseline pins the merge-commit SHA reviews replay
    # against; that must match metadata.mergeCommit.oid.
    assert fixture.metadata["mergeCommit"]["oid"] == fixture.head_sha
    # Either we have findings (captured) OR we don't (pending/failed).
    if fixture.capture_status == "captured":
        assert fixture.review_run_result is not None
        assert fixture.codex_findings, "captured fixture must have non-empty findings"
    else:
        assert fixture.review_run_result is None
        assert fixture.codex_findings == []


def test_load_all_baseline_fixtures_aggregates_failures() -> None:
    """The eager loader must report every fixture defect at once.

    On a healthy tree this just confirms the dataset is consumable by
    later regression-comparison ACs; on a broken tree it gives the
    fixture-regen author every defect in one error message instead of
    a one-at-a-time game of whack-a-mole.
    """
    fixtures = load_all_baseline_fixtures()
    assert len(fixtures) == len(list_fixture_pr_ids())
    # Order must mirror the sample registry so downstream comparisons
    # are deterministic.
    assert [f.pr_id for f in fixtures] == list_fixture_pr_ids()


def test_iter_baseline_fixtures_streams_in_registry_order() -> None:
    """``iter_baseline_fixtures`` is a streaming variant — same order."""
    streamed = [f.pr_id for f in iter_baseline_fixtures()]
    assert streamed == list_fixture_pr_ids()


# ---------------------------------------------------------------------------
# Synthetic fixture probes — pin the validator's failure surface.
# ---------------------------------------------------------------------------


_CAPTURED_FINDING: dict[str, Any] = {
    "title": "[P1] Possible null deref",
    "body": "The pointer `x` may be null after the early return on L42.",
    "priority": 1,
    "confidence_score": 0.78,
    "code_location": {
        "absolute_file_path": "src/main/java/example/Foo.java",
        "line_range": {"start": 42, "end": 42},
    },
}


def _write_fixture(
    root: Path,
    pr_id: str,
    *,
    diff: str | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    codex_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal-but-valid baseline fixture rooted at ``root``.

    Overrides can patch metadata.json or codex-findings.json before
    write so each probe test mutates a single field.
    """
    pr_dir = root / pr_id
    pr_dir.mkdir(parents=True, exist_ok=True)
    pr_number = int(pr_id.removeprefix("pr-"))
    head_sha = "a" * 40
    base_sha = "b" * 40
    # codex.head_sha must match metadata.mergeCommit.oid so the helper
    # produces a valid fixture by default.
    merge_sha = "c" * 40
    codex_head_sha = merge_sha

    diff_text = (
        diff
        if diff is not None
        else (
            "diff --git a/src/Foo.java b/src/Foo.java\n"
            "--- a/src/Foo.java\n"
            "+++ b/src/Foo.java\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
    )
    (pr_dir / "diff.patch").write_text(diff_text, encoding="utf-8")

    metadata: dict[str, Any] = {
        "number": pr_number,
        "baseRefOid": base_sha,
        "headRefOid": head_sha,
        "mergeCommit": {"oid": merge_sha},
        "files": [{"path": "src/Foo.java", "additions": 1, "deletions": 1}],
        "changedFiles": 1,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    (pr_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    codex: dict[str, Any] = {
        "pr_id": pr_id,
        "run": "codex",
        "head_sha": codex_head_sha,
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "web_search_mode": "cached",
        "capture": {
            "captured_at": None,
            "command": f"python -m eval.run_codex_baseline --pr-id dotcms-core-{pr_number}",
            "notes": "Live capture skipped: GITHUB_TOKEN not set",
            "runner_version": "1",
            "status": "pending",
        },
        "review_run_result": None,
        "posted": {
            "inline_ids": [],
            "summary_id": None,
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
        "prior_review_state": None,
        "source": {
            "produced_by": "eval.run_codex_baseline",
            "mirrored_by": "eval.mirror_codex_baseline",
            "eval_artifact": f"eval/runs/dotcms-core-{pr_number}/codex.json",
        },
    }
    if codex_overrides:
        codex = _deep_merge(codex, codex_overrides)
    (pr_dir / "codex-findings.json").write_text(json.dumps(codex), encoding="utf-8")
    return pr_dir


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def test_synthetic_fixture_baseline_loads(tmp_path: Path) -> None:
    """Sanity check: the helper produces a fixture the loader accepts."""
    _write_fixture(tmp_path, "pr-99001")
    fixture = load_baseline_fixture("pr-99001", fixtures_dir=tmp_path)
    assert fixture.pr_number == 99001
    assert fixture.capture_status == "pending"
    assert fixture.review_run_result is None
    assert fixture.codex_findings == []


def test_loader_rejects_missing_diff(tmp_path: Path) -> None:
    pr_dir = _write_fixture(tmp_path, "pr-99002")
    (pr_dir / "diff.patch").unlink()
    with pytest.raises(BaselineFixtureError, match="missing diff.patch"):
        load_baseline_fixture("pr-99002", fixtures_dir=tmp_path)


def test_loader_rejects_non_diff_content(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "pr-99003", diff="<html>not a diff</html>\n")
    with pytest.raises(BaselineFixtureError, match="not a unified diff"):
        load_baseline_fixture("pr-99003", fixtures_dir=tmp_path)


def test_loader_rejects_missing_metadata_field(tmp_path: Path) -> None:
    pr_dir = _write_fixture(tmp_path, "pr-99004")
    meta = json.loads((pr_dir / "metadata.json").read_text(encoding="utf-8"))
    meta.pop("baseRefOid")
    (pr_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(BaselineFixtureError, match="missing keys.*baseRefOid"):
        load_baseline_fixture("pr-99004", fixtures_dir=tmp_path)


def test_loader_rejects_short_sha(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99005",
        metadata_overrides={"headRefOid": "deadbeef"},
    )
    with pytest.raises(BaselineFixtureError, match="40-char SHA"):
        load_baseline_fixture("pr-99005", fixtures_dir=tmp_path)


def test_loader_rejects_pr_number_mismatch(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99006",
        metadata_overrides={"number": 12345},
    )
    with pytest.raises(BaselineFixtureError, match="fixture pr_id implies"):
        load_baseline_fixture("pr-99006", fixtures_dir=tmp_path)


def test_loader_rejects_unknown_capture_status(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99007",
        codex_overrides={"capture": {"status": "weirdo"}},
    )
    with pytest.raises(BaselineFixtureError, match="capture.status='weirdo'"):
        load_baseline_fixture("pr-99007", fixtures_dir=tmp_path)


def test_loader_rejects_pending_with_synthesized_review(tmp_path: Path) -> None:
    """Placeholder fixtures must not embed fake findings."""
    _write_fixture(
        tmp_path,
        "pr-99008",
        codex_overrides={
            "review_run_result": {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": 1.0,
                "findings": [],
                "carried_forward": [],
            }
        },
    )
    with pytest.raises(BaselineFixtureError, match="review_run_result=null"):
        load_baseline_fixture("pr-99008", fixtures_dir=tmp_path)


def test_loader_requires_capture_notes_for_pending(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99009",
        codex_overrides={"capture": {"notes": ""}},
    )
    with pytest.raises(BaselineFixtureError, match="capture.notes"):
        load_baseline_fixture("pr-99009", fixtures_dir=tmp_path)


def test_loader_rejects_captured_without_review(tmp_path: Path) -> None:
    """``capture.status='captured'`` requires a real review object."""
    _write_fixture(
        tmp_path,
        "pr-99010",
        codex_overrides={
            "capture": {
                "status": "captured",
                "captured_at": "2026-05-07T00:00:00Z",
                "notes": "",
            },
            "review_run_result": None,
        },
    )
    with pytest.raises(BaselineFixtureError, match="review_run_result to be an object"):
        load_baseline_fixture("pr-99010", fixtures_dir=tmp_path)


def test_loader_rejects_captured_with_empty_findings(tmp_path: Path) -> None:
    """Sub-AC 1.4 'non-empty findings' rule for captured fixtures."""
    _write_fixture(
        tmp_path,
        "pr-99011",
        codex_overrides={
            "capture": {
                "status": "captured",
                "captured_at": "2026-05-07T00:00:00Z",
                "notes": "",
            },
            "review_run_result": {
                "overall_correctness": "patch is correct",
                "overall_explanation": "no findings",
                "overall_confidence_score": 1.0,
                "findings": [],
                "carried_forward": [],
            },
        },
    )
    with pytest.raises(BaselineFixtureError, match="findings must be non-empty"):
        load_baseline_fixture("pr-99011", fixtures_dir=tmp_path)


def test_loader_rejects_captured_with_malformed_finding(tmp_path: Path) -> None:
    bad_finding = {
        "title": "missing code_location",
        "body": "x",
        "priority": 1,
        # code_location intentionally omitted
    }
    _write_fixture(
        tmp_path,
        "pr-99012",
        codex_overrides={
            "capture": {
                "status": "captured",
                "captured_at": "2026-05-07T00:00:00Z",
                "notes": "",
            },
            "review_run_result": {
                "overall_correctness": "patch has a bug",
                "overall_explanation": "see finding",
                "overall_confidence_score": 0.9,
                "findings": [bad_finding],
                "carried_forward": [],
            },
        },
    )
    with pytest.raises(BaselineFixtureError, match=r"findings\[0\] missing keys"):
        load_baseline_fixture("pr-99012", fixtures_dir=tmp_path)


def test_loader_accepts_well_formed_captured_fixture(tmp_path: Path) -> None:
    """Positive control: a fixture with a real captured finding loads."""
    _write_fixture(
        tmp_path,
        "pr-99013",
        codex_overrides={
            "capture": {
                "status": "captured",
                "captured_at": "2026-05-07T00:00:00Z",
                "notes": "",
            },
            "review_run_result": {
                "overall_correctness": "patch has a bug",
                "overall_explanation": "see finding",
                "overall_confidence_score": 0.85,
                "findings": [_CAPTURED_FINDING],
                "carried_forward": [],
            },
        },
    )
    fixture = load_baseline_fixture("pr-99013", fixtures_dir=tmp_path)
    assert fixture.capture_status == "captured"
    assert fixture.review_run_result is not None
    assert len(fixture.codex_findings) == 1
    assert fixture.codex_findings[0]["title"].startswith("[P1]")
    assert fixture.has_findings is True


def test_loader_detects_head_sha_disagreement(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99014",
        codex_overrides={"head_sha": "f" * 40},
    )
    with pytest.raises(BaselineFixtureError, match="review-replay SHA disagrees"):
        load_baseline_fixture("pr-99014", fixtures_dir=tmp_path)


def test_loader_detects_pr_id_mismatch_in_codex(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "pr-99015",
        codex_overrides={"pr_id": "pr-00000"},
    )
    with pytest.raises(BaselineFixtureError, match="does not match fixture id"):
        load_baseline_fixture("pr-99015", fixtures_dir=tmp_path)


def test_load_all_aggregates_multiple_failures(tmp_path: Path, monkeypatch) -> None:
    """When several fixtures are broken, the aggregator reports them all."""
    # Build a 2-PR mini sample registry so we can synthesize broken
    # fixtures without polluting the real ones.
    sample_path = tmp_path / "sample.yml"
    sample_path.write_text(
        "prs:\n"
        "  - number: 99100\n"
        "    merge_commit: " + ("a" * 40) + "\n"
        "  - number: 99101\n"
        "    merge_commit: " + ("b" * 40) + "\n",
        encoding="utf-8",
    )
    fixtures_dir = tmp_path / "baseline"
    # First fixture: missing diff.patch
    pr_dir_a = _write_fixture(fixtures_dir, "pr-99100")
    (pr_dir_a / "diff.patch").unlink()
    # Second fixture: short head SHA
    _write_fixture(
        fixtures_dir,
        "pr-99101",
        metadata_overrides={"headRefOid": "deadbeef"},
    )
    with pytest.raises(BaselineFixtureError) as info:
        load_all_baseline_fixtures(fixtures_dir=fixtures_dir, sample_path=sample_path)
    msg = str(info.value)
    assert "pr-99100" in msg
    assert "pr-99101" in msg
    assert "2 baseline fixture(s) failed" in msg

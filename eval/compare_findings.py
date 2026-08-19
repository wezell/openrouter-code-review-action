#!/usr/bin/env python3
"""Diff candidate OpenRouter findings against the Codex baseline (Sub-AC 2.4).

This is the **parity-report** step. Sub-AC 2.1 defined the shared §1
envelope; Sub-AC 2.2 mirrored the Codex baseline into
``tests/fixtures/baseline/<pr-id>/codex-findings.json``; Sub-AC 2.3
mirrored the OpenRouter candidate into
``tests/fixtures/candidate/<pr-id>/openrouter-findings.json``. This
module is the **single consumer** of those two trees that produces
review-ready evidence: per-PR counts, proximity-matched overlap, and
side-by-side deltas, plus a sample-level summary that a reviewer can
open without re-walking either fixture set.

Why a separate report
---------------------

``eval.score_codex_baseline`` scores a single side (Codex) against a
labeled ground-truth dataset. The parity bake-off needs a *second*
view: how many findings does each side emit, how many overlap (same
path + line proximity within ± tolerance lines), and what is the delta
in each severity bucket. That view is independent of the labeled
dataset because it can be generated even before the live captures
land — the pending-placeholder fixtures still produce a valid
zero-finding report.

Inputs (per PR)
---------------

* ``tests/fixtures/baseline/<pr-id>/codex-findings.json``  (Sub-AC 2.2)
* ``tests/fixtures/candidate/<pr-id>/openrouter-findings.json`` (Sub-AC 2.3)

Outputs (committed to git, like every other bake-off artifact)
---------------------------------------------------------------

* ``tests/fixtures/parity_report/<pr-id>.json``    — one per PR
* ``tests/fixtures/parity_report/_summary.json``    — sample-level rollup

The report is **idempotent**: ``--check`` rebuilds in memory and exits
non-zero if the on-disk artifacts drift from the rebuild, so CI can
keep the report in sync with whichever side last got recaptured.

Status semantics
----------------

Each PR's report carries a single ``status``:

* ``pending``  — both sides still placeholders (no live capture).
* ``codex_only``     — only Codex captured.
* ``openrouter_only`` — only OpenRouter captured.
* ``ready``    — both sides captured at the same head_sha. Overlap
                 numbers are meaningful.
* ``head_sha_drift`` — both captured but on different head_sha; counts
                       are still emitted but ``overlap_ratio`` is set
                       to ``null`` because the diffs reviewed differ.

Usage
-----

::

    python -m eval.compare_findings                  # rebuild every PR + summary
    python -m eval.compare_findings --pr-id pr-35449  # one PR
    python -m eval.compare_findings --check          # CI drift check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.overlap_score import (
    LINE_PROXIMITY_TOLERANCE,
    OverlapFinding,
    OverlapMatch,
)
from eval.overlap_score import (
    line_distance as _overlap_line_distance,
)
from eval.overlap_score import (
    match_findings as _overlap_match_findings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = REPO_ROOT / "tests" / "fixtures" / "baseline"
CANDIDATE_ROOT = REPO_ROOT / "tests" / "fixtures" / "candidate"
REPORT_ROOT = REPO_ROOT / "tests" / "fixtures" / "parity_report"
SUMMARY_PATH = REPORT_ROOT / "_summary.json"
SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "eval/compare_findings.py"

# Mirrors the comparison-set / scorer extractors so a single field
# name change in the title prefix propagates to every consumer.
_SEVERITY_TAG_RE = re.compile(r"\[P(?P<level>[0-3])\]")

# Per §4 of findings-schema.md, ``capture.status`` is one of these on
# the codex side, plus ``unfilled`` / ``skipped_dry_run`` / ``error``
# on the OpenRouter side. Only ``captured`` / ``completed`` carry real
# findings; everything else is a placeholder.
_CAPTURED_STATUSES: frozenset[str] = frozenset({"captured", "completed"})


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    """Render ``path`` as repo-relative when possible.

    ``Path.relative_to`` raises when the target is not under
    ``REPO_ROOT`` (e.g. ``tmp_path`` on macOS resolves under
    ``/private/var/...``); fall back to the absolute string so a
    surfaced path message never crashes a test run.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _baseline_path(pr_id: str, *, baseline_root: Path | None = None) -> Path:
    base = baseline_root if baseline_root is not None else BASELINE_ROOT
    return base / pr_id / "codex-findings.json"


def _candidate_path(pr_id: str, *, candidate_root: Path | None = None) -> Path:
    base = candidate_root if candidate_root is not None else CANDIDATE_ROOT
    return base / pr_id / "openrouter-findings.json"


def _report_path(pr_id: str, *, report_root: Path | None = None) -> Path:
    base = report_root if report_root is not None else REPORT_ROOT
    return base / f"{pr_id}.json"


def _load_fixture(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, Mapping):
        return None
    return dict(doc)


# ---------------------------------------------------------------------------
# Finding normalization (fixture wrapper → flat tuples)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Finding:
    """Flat shape used for diff/proximity matching.

    The fixture wrapper carries a §1 envelope under
    ``review_run_result.findings`` with nested ``code_location``. The
    diff layer wants flat ``(path, line_start, line_end, severity)``
    tuples — this dataclass is that shape, plus identity/title fields
    so the report can quote the surfaced finding back to a reviewer.
    """

    finding_id: str
    raw_index: int
    side: str  # "codex" | "openrouter"
    path: str | None
    line_start: int | None
    line_end: int | None
    severity: str | None  # "P0".."P3" | None
    title: str
    confidence_score: float | None


def _extract_severity(title: str, priority: int | None) -> str | None:
    if isinstance(priority, int) and 0 <= priority <= 3:
        return f"P{priority}"
    m = _SEVERITY_TAG_RE.search(title or "")
    if m is None:
        return None
    return f"P{m.group('level')}"


def _normalize_finding(raw: Mapping[str, Any], *, side: str, raw_index: int) -> _Finding:
    code_location = raw.get("code_location") if isinstance(raw, Mapping) else None
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    if isinstance(code_location, Mapping):
        path_raw = code_location.get("absolute_file_path")
        if isinstance(path_raw, str) and path_raw.strip():
            path = path_raw.strip()
        line_range = code_location.get("line_range")
        if isinstance(line_range, Mapping):
            ls = line_range.get("start")
            le = line_range.get("end")
            if isinstance(ls, int) and ls > 0:
                line_start = ls
            if isinstance(le, int) and le > 0:
                line_end = le
    if line_end is None and line_start is not None:
        line_end = line_start

    title_raw = raw.get("title")
    title = title_raw.strip() if isinstance(title_raw, str) else ""

    priority_raw = raw.get("priority") if isinstance(raw, Mapping) else None
    priority = priority_raw if isinstance(priority_raw, int) else None

    confidence_raw = raw.get("confidence_score") if isinstance(raw, Mapping) else None
    if isinstance(confidence_raw, (int, float)):
        confidence_score: float | None = float(confidence_raw)
    else:
        confidence_score = None

    severity = _extract_severity(title, priority)
    return _Finding(
        finding_id=f"{side}.f{raw_index + 1:04d}",
        raw_index=raw_index,
        side=side,
        path=path,
        line_start=line_start,
        line_end=line_end,
        severity=severity,
        title=title,
        confidence_score=confidence_score,
    )


def _extract_findings(fixture: Mapping[str, Any] | None, *, side: str) -> list[_Finding]:
    if fixture is None:
        return []
    rrr = fixture.get("review_run_result")
    if not isinstance(rrr, Mapping):
        return []
    raw_findings = rrr.get("findings")
    if not isinstance(raw_findings, list):
        return []
    out: list[_Finding] = []
    for idx, item in enumerate(raw_findings):
        if isinstance(item, Mapping):
            out.append(_normalize_finding(item, side=side, raw_index=idx))
    return out


# ---------------------------------------------------------------------------
# Proximity matching (methodology §4.5)
# ---------------------------------------------------------------------------
#
# The matcher itself lives in ``eval/overlap_score.py`` (Sub-AC 3.2).
# This module wraps the engine to translate between the local
# ``_Finding`` shape (which carries fixture-display fields like
# ``raw_index``, ``title``, and ``confidence_score``) and the engine's
# ``OverlapFinding`` shape (which carries only the fields the predicate
# reads). The wrapper lets the parity report keep its render-friendly
# fields without leaking them into the matching engine.


# Thin alias kept for any in-module callers that want the proximity
# helper without importing through ``overlap_score`` directly.
_line_distance = _overlap_line_distance


@dataclass(frozen=True)
class _Match:
    codex: _Finding
    openrouter: _Finding
    line_distance: int


def _to_overlap_finding(f: _Finding) -> OverlapFinding:
    return OverlapFinding(
        finding_id=f.finding_id,
        path=f.path,
        line_start=f.line_start,
        line_end=f.line_end,
        severity=f.severity,
    )


def _match_findings(
    codex: Sequence[_Finding],
    openrouter: Sequence[_Finding],
    *,
    tolerance: int,
) -> tuple[list[_Match], list[_Finding], list[_Finding]]:
    """Wrap :func:`eval.overlap_score.match_findings` for fixture records.

    Each finding can participate in at most one match (spec §6).
    Tie-breaks are by Codex finding id then OpenRouter finding id so
    the report is reproducible across runs. Severity is not part of
    the sort key (spec §5).
    """
    codex_overlap = [_to_overlap_finding(c) for c in codex]
    or_overlap = [_to_overlap_finding(o) for o in openrouter]
    by_codex_id = {c.finding_id: c for c in codex}
    by_or_id = {o.finding_id: o for o in openrouter}

    result = _overlap_match_findings(codex_overlap, or_overlap, tolerance=tolerance)

    matches = [
        _Match(
            codex=by_codex_id[m.codex.finding_id],
            openrouter=by_or_id[m.openrouter.finding_id],
            line_distance=m.line_distance,
        )
        for m in result.matches
    ]
    used_codex = {m.codex.finding_id for m in matches}
    used_or = {m.openrouter.finding_id for m in matches}
    codex_only = [c for c in codex if c.finding_id not in used_codex]
    or_only = [o for o in openrouter if o.finding_id not in used_or]
    return matches, codex_only, or_only


# Re-export for callers that want the engine-typed match structure
# without importing from ``eval.overlap_score`` directly. Keeps the
# Sub-AC 3.2 module the single source of truth for the symbol.
_OverlapMatch = OverlapMatch


# ---------------------------------------------------------------------------
# Status / per-side descriptors
# ---------------------------------------------------------------------------


def _capture_status(fixture: Mapping[str, Any] | None) -> str:
    if fixture is None:
        return "missing"
    capture = fixture.get("capture")
    if not isinstance(capture, Mapping):
        return "unknown"
    return str(capture.get("status") or "unknown")


def _capture_notes(fixture: Mapping[str, Any] | None) -> str | None:
    if fixture is None:
        return None
    capture = fixture.get("capture")
    if not isinstance(capture, Mapping):
        return None
    notes = capture.get("notes")
    return notes if isinstance(notes, str) else None


def _head_sha(fixture: Mapping[str, Any] | None) -> str | None:
    if fixture is None:
        return None
    sha = fixture.get("head_sha")
    return sha if isinstance(sha, str) else None


def _model(fixture: Mapping[str, Any] | None) -> str | None:
    if fixture is None:
        return None
    model = fixture.get("model")
    return model if isinstance(model, str) else None


def _by_severity(findings: Iterable[_Finding]) -> dict[str, int]:
    counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unknown": 0}
    for f in findings:
        if f.severity in counts:
            counts[f.severity] += 1
        else:
            counts["unknown"] += 1
    return counts


def _classify_status(
    codex_status: str,
    or_status: str,
    *,
    codex_sha: str | None,
    or_sha: str | None,
) -> str:
    codex_captured = codex_status in _CAPTURED_STATUSES
    or_captured = or_status in _CAPTURED_STATUSES
    if codex_captured and or_captured:
        if codex_sha and or_sha and codex_sha != or_sha:
            return "head_sha_drift"
        return "ready"
    if codex_captured and not or_captured:
        return "codex_only"
    if or_captured and not codex_captured:
        return "openrouter_only"
    return "pending"


# ---------------------------------------------------------------------------
# Per-PR record assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRComparison:
    pr_id: str
    head_sha_baseline: str | None
    head_sha_candidate: str | None
    status: str
    codex_status: str
    openrouter_status: str
    codex_findings: list[_Finding]
    openrouter_findings: list[_Finding]
    matches: list[_Match]
    codex_only: list[_Finding]
    openrouter_only: list[_Finding]
    codex_model: str | None = None
    openrouter_model: str | None = None
    codex_notes: str | None = None
    openrouter_notes: str | None = None


def compare_pr(
    pr_id: str,
    *,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> PRComparison:
    """Build the in-memory comparison for one PR (no I/O on output)."""
    codex_fixture = _load_fixture(_baseline_path(pr_id, baseline_root=baseline_root))
    or_fixture = _load_fixture(_candidate_path(pr_id, candidate_root=candidate_root))

    codex_findings = _extract_findings(codex_fixture, side="codex")
    or_findings = _extract_findings(or_fixture, side="openrouter")

    matches, codex_only, or_only = _match_findings(codex_findings, or_findings, tolerance=tolerance)

    codex_status = _capture_status(codex_fixture)
    or_status = _capture_status(or_fixture)
    codex_sha = _head_sha(codex_fixture)
    or_sha = _head_sha(or_fixture)

    return PRComparison(
        pr_id=pr_id,
        head_sha_baseline=codex_sha,
        head_sha_candidate=or_sha,
        status=_classify_status(codex_status, or_status, codex_sha=codex_sha, or_sha=or_sha),
        codex_status=codex_status,
        openrouter_status=or_status,
        codex_findings=codex_findings,
        openrouter_findings=or_findings,
        matches=matches,
        codex_only=codex_only,
        openrouter_only=or_only,
        codex_model=_model(codex_fixture),
        openrouter_model=_model(or_fixture),
        codex_notes=_capture_notes(codex_fixture),
        openrouter_notes=_capture_notes(or_fixture),
    )


def _finding_payload(f: _Finding) -> dict[str, Any]:
    return {
        "finding_id": f.finding_id,
        "raw_index": f.raw_index,
        "path": f.path,
        "line_start": f.line_start,
        "line_end": f.line_end,
        "severity": f.severity,
        "title": f.title,
        "confidence_score": f.confidence_score,
    }


def _match_payload(m: _Match) -> dict[str, Any]:
    return {
        "codex_finding_id": m.codex.finding_id,
        "openrouter_finding_id": m.openrouter.finding_id,
        "path": m.codex.path,
        "line_distance": m.line_distance,
        "codex_line_range": [m.codex.line_start, m.codex.line_end],
        "openrouter_line_range": [m.openrouter.line_start, m.openrouter.line_end],
        "codex_severity": m.codex.severity,
        "openrouter_severity": m.openrouter.severity,
        "severity_match": (
            m.codex.severity is not None
            and m.openrouter.severity is not None
            and m.codex.severity == m.openrouter.severity
        ),
    }


def _delta_by_severity(codex: Mapping[str, int], openrouter: Mapping[str, int]) -> dict[str, int]:
    keys = sorted(set(codex) | set(openrouter))
    return {k: openrouter.get(k, 0) - codex.get(k, 0) for k in keys}


def _ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _build_pr_report(cmp: PRComparison, *, generated_at: str, tolerance: int) -> dict[str, Any]:
    codex_total = len(cmp.codex_findings)
    or_total = len(cmp.openrouter_findings)
    matched = len(cmp.matches)
    codex_only = len(cmp.codex_only)
    or_only = len(cmp.openrouter_only)
    severity_match_count = sum(1 for m in cmp.matches if _match_payload(m)["severity_match"])
    codex_by_sev = _by_severity(cmp.codex_findings)
    or_by_sev = _by_severity(cmp.openrouter_findings)
    delta_by_sev = _delta_by_severity(codex_by_sev, or_by_sev)

    # Overlap ratio is undefined when either side is missing real
    # findings or when head_sha drifted; surface ``null`` so the
    # downstream reviewer reads "not measurable" rather than "0%".
    if cmp.status == "ready" and codex_total > 0:
        overlap_ratio_codex = _ratio(matched, codex_total)
    else:
        overlap_ratio_codex = None
    if cmp.status == "ready" and or_total > 0:
        overlap_ratio_openrouter = _ratio(matched, or_total)
    else:
        overlap_ratio_openrouter = None

    return {
        "schema_version": SCHEMA_VERSION,
        "pr_id": cmp.pr_id,
        "status": cmp.status,
        "tolerance_lines": tolerance,
        "head_sha_baseline": cmp.head_sha_baseline,
        "head_sha_candidate": cmp.head_sha_candidate,
        "codex": {
            "capture_status": cmp.codex_status,
            "model": cmp.codex_model,
            "findings_count": codex_total,
            "by_severity": codex_by_sev,
            "notes": cmp.codex_notes,
        },
        "openrouter": {
            "capture_status": cmp.openrouter_status,
            "model": cmp.openrouter_model,
            "findings_count": or_total,
            "by_severity": or_by_sev,
            "notes": cmp.openrouter_notes,
        },
        "counts": {
            "codex_total": codex_total,
            "openrouter_total": or_total,
            "matched_pairs": matched,
            "codex_only": codex_only,
            "openrouter_only": or_only,
        },
        "overlap": {
            "overlap_ratio_codex_baseline": overlap_ratio_codex,
            "overlap_ratio_openrouter_candidate": overlap_ratio_openrouter,
            "severity_match_count": severity_match_count,
        },
        "deltas": {
            "delta_total": or_total - codex_total,
            "delta_by_severity": delta_by_sev,
        },
        "matches": [_match_payload(m) for m in cmp.matches],
        "codex_only_findings": [_finding_payload(f) for f in cmp.codex_only],
        "openrouter_only_findings": [_finding_payload(f) for f in cmp.openrouter_only],
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
    }


# ---------------------------------------------------------------------------
# Sample-level summary
# ---------------------------------------------------------------------------


def _build_summary(
    comparisons: Sequence[PRComparison], *, generated_at: str, tolerance: int
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    totals = {
        "codex_findings_total": 0,
        "openrouter_findings_total": 0,
        "matched_pairs_total": 0,
        "codex_only_total": 0,
        "openrouter_only_total": 0,
        "severity_match_total": 0,
    }
    ready_codex = 0
    ready_or = 0
    ready_matched = 0

    for cmp in comparisons:
        status_counts[cmp.status] = status_counts.get(cmp.status, 0) + 1
        codex_total = len(cmp.codex_findings)
        or_total = len(cmp.openrouter_findings)
        matched = len(cmp.matches)
        sev_match = sum(1 for m in cmp.matches if _match_payload(m)["severity_match"])

        totals["codex_findings_total"] += codex_total
        totals["openrouter_findings_total"] += or_total
        totals["matched_pairs_total"] += matched
        totals["codex_only_total"] += len(cmp.codex_only)
        totals["openrouter_only_total"] += len(cmp.openrouter_only)
        totals["severity_match_total"] += sev_match

        if cmp.status == "ready":
            ready_codex += codex_total
            ready_or += or_total
            ready_matched += matched

        rows.append(
            {
                "pr_id": cmp.pr_id,
                "status": cmp.status,
                "codex_findings": codex_total,
                "openrouter_findings": or_total,
                "matched_pairs": matched,
                "codex_only": len(cmp.codex_only),
                "openrouter_only": len(cmp.openrouter_only),
                "delta_total": or_total - codex_total,
                "head_sha_baseline": cmp.head_sha_baseline,
                "head_sha_candidate": cmp.head_sha_candidate,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
        "tolerance_lines": tolerance,
        "sample_size": len(comparisons),
        "status_counts": status_counts,
        "totals": totals,
        "ready_overlap": {
            "codex_findings": ready_codex,
            "openrouter_findings": ready_or,
            "matched_pairs": ready_matched,
            # Weighted overlap ratios across only the PRs whose status
            # is ``ready``; ``null`` when no PR is ready yet so the
            # number is never confused with "0% parity".
            "overlap_ratio_codex_baseline": _ratio(ready_matched, ready_codex),
            "overlap_ratio_openrouter_candidate": _ratio(ready_matched, ready_or),
        },
        "prs": rows,
    }


# ---------------------------------------------------------------------------
# Persistence + drift check
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _strip_volatile(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    if isinstance(out, dict):
        out.pop("generated_at", None)
    return out  # type: ignore[no-any-return]


def write_report(
    comparisons: Sequence[PRComparison],
    *,
    report_root: Path | None = None,
    summary_path: Path | None = None,
    generated_at: str | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> tuple[list[Path], Path]:
    """Persist per-PR reports + the summary. Returns ``(paths, summary)``."""
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    written: list[Path] = []
    for cmp in comparisons:
        path = _report_path(cmp.pr_id, report_root=report_root)
        artifact = _build_pr_report(cmp, generated_at=generated_at, tolerance=tolerance)
        _atomic_write(path, _serialize(artifact))
        written.append(path)
    summary_target = (
        summary_path
        if summary_path is not None
        else ((report_root if report_root is not None else REPORT_ROOT) / "_summary.json")
    )
    summary = _build_summary(comparisons, generated_at=generated_at, tolerance=tolerance)
    _atomic_write(summary_target, _serialize(summary))
    return written, summary_target


def check_report(
    comparisons: Sequence[PRComparison],
    *,
    report_root: Path | None = None,
    summary_path: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[str]:
    """Compare the on-disk report against an in-memory rebuild."""
    drift: list[str] = []
    for cmp in comparisons:
        path = _report_path(cmp.pr_id, report_root=report_root)
        fresh = _build_pr_report(cmp, generated_at="<ignored-for-check>", tolerance=tolerance)
        on_disk = _load_fixture(path)
        if on_disk is None:
            drift.append(f"{cmp.pr_id}: missing report at {_display_path(path)}")
            continue
        if _strip_volatile(on_disk) != _strip_volatile(fresh):
            drift.append(f"{cmp.pr_id}: on-disk report differs from rebuild")
    summary_target = (
        summary_path
        if summary_path is not None
        else ((report_root if report_root is not None else REPORT_ROOT) / "_summary.json")
    )
    fresh_summary = _build_summary(
        comparisons, generated_at="<ignored-for-check>", tolerance=tolerance
    )
    on_disk_summary = _load_fixture(summary_target)
    if on_disk_summary is None:
        drift.append(f"{_display_path(summary_target)}: missing parity summary")
    elif _strip_volatile(on_disk_summary) != _strip_volatile(fresh_summary):
        drift.append(f"{_display_path(summary_target)}: parity summary differs from rebuild")
    return drift


# ---------------------------------------------------------------------------
# Discovery + CLI orchestration
# ---------------------------------------------------------------------------


def discover_pr_ids(
    *,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
) -> list[str]:
    """Return the union of PR ids present in either fixture tree.

    The report is meaningful even when only one side is present (e.g.
    Codex baseline committed but OpenRouter not yet captured), so the
    discovery walks both trees and merges.
    """
    base = baseline_root if baseline_root is not None else BASELINE_ROOT
    cand = candidate_root if candidate_root is not None else CANDIDATE_ROOT
    ids: set[str] = set()
    for root in (base, cand):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("pr-"):
                continue
            ids.add(child.name)
    return sorted(ids)


def compare_all(
    pr_ids: Sequence[str] | None = None,
    *,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[PRComparison]:
    if pr_ids is None:
        pr_ids = discover_pr_ids(baseline_root=baseline_root, candidate_root=candidate_root)
    return [
        compare_pr(
            pid,
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            tolerance=tolerance,
        )
        for pid in pr_ids
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.compare_findings",
        description=(
            "Diff candidate OpenRouter findings against the Codex baseline "
            "(Sub-AC 2.4). Writes tests/fixtures/parity_report/<pr-id>.json "
            "per PR plus tests/fixtures/parity_report/_summary.json for the "
            "sample-level rollup."
        ),
    )
    parser.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help="Compare only this PR id (e.g. ``pr-35449``). May be repeated.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild in memory and exit non-zero if the "
            "on-disk report differs from the rebuild. Use in CI."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=LINE_PROXIMITY_TOLERANCE,
        help=(f"± lines tolerance for proximity matching (default: {LINE_PROXIMITY_TOLERANCE})."),
    )
    args = parser.parse_args(argv)

    discovered = discover_pr_ids()
    if args.pr_ids:
        unknown = [pid for pid in args.pr_ids if pid not in discovered]
        if unknown:
            raise SystemExit(
                f"unknown pr_id(s): {', '.join(unknown)}. "
                f"Known: {', '.join(discovered) or '(none)'}"
            )
        targets = list(args.pr_ids)
    else:
        targets = discovered

    if not targets:
        # No fixtures present yet (fresh checkout, no Sub-AC 2.2/2.3
        # mirror). Treat as a successful no-op so CI doesn't block on
        # an unbuilt sample.
        print(
            "[ok] no baseline/candidate fixtures found; nothing to compare",
            file=sys.stderr,
        )
        return 0

    comparisons = compare_all(targets, tolerance=args.tolerance)

    if args.check:
        drift = check_report(comparisons, tolerance=args.tolerance)
        if drift:
            sys.stderr.write(
                "parity report drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m eval.compare_findings`\n"
            )
            return 1
        print(f"parity report in sync ({len(comparisons)} PRs)")
        return 0

    written, summary_target = write_report(comparisons, tolerance=args.tolerance)
    by_status: dict[str, int] = {}
    for cmp in comparisons:
        by_status[cmp.status] = by_status.get(cmp.status, 0) + 1
    by_status_str = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    print(
        f"wrote {len(written)} parity report(s) and "
        f"{_display_path(summary_target)} ({by_status_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

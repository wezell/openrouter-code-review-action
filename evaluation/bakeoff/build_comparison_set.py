#!/usr/bin/env python3
"""Build the structured comparison reference set for the bake-off.

Sub-AC 1.1.3 deliverable. Codex baseline captures live in
``eval/runs/<pr_id>/codex.json`` (Sub-AC 1.2 / 2.1 schema). OpenRouter
captures live in ``evaluation/bakeoff/runs/<pr_id>/openrouter.json``
(Sub-AC 2.2 schema). The two artifacts are written by independent
runners with subtly different shapes and ID conventions, and the
findings inside them carry only the *raw* model output: a dict per
finding with ``code_location.line_range.{start,end}`` and a free-form
``title`` whose severity is encoded as a ``[P0]``-style prefix.

Downstream Sub-ACs (1.4 labeling; 1.5 overlap + FP-rate scoring)
expect a *normalized* per-PR record they can iterate without
re-deriving severities, re-bridging IDs, or re-computing line-range
proximity for cross-run match candidates. This module produces
exactly that record — one ``evaluation/bakeoff/comparison/<pr_id>.json``
file per sample PR — and a sample-level ``_index.json`` rollup so the
labeling pass can see at a glance which slots are ready and which are
still pending capture.

The schema is documented in ``evaluation/bakeoff/comparison/README.md``
and pinned by ``tests/test_comparison_set.py``.

Design notes
------------

* **Normalization, not interpretation.** This script does not assign
  ``TP``/``FP`` labels — that is a human pass per methodology §2. It
  only re-shapes data already captured into a uniform schema, and
  surfaces *match candidates* (path + line proximity) to make the
  human labeling pass cheaper.

* **Idempotent.** Re-running it overwrites ``comparison/`` from the
  current ``eval/runs/`` + ``evaluation/bakeoff/runs/`` state. That
  keeps the comparison set a *function* of the captured artifacts —
  no hand-edits silently survive a regeneration.

* **Pending-tolerant.** The Codex side may be ``pending`` (placeholder
  awaiting live capture) and the OpenRouter side may be ``unfilled``
  (scaffold). The comparison record records that state explicitly so
  scoring can short-circuit on partially-captured PRs without crashing.

Usage
-----
::

    python -m evaluation.bakeoff.build_comparison_set
    python -m evaluation.bakeoff.build_comparison_set --only pr-35567
    python -m evaluation.bakeoff.build_comparison_set --check

``--check`` rebuilds in memory and fails non-zero if the on-disk
``comparison/`` directory differs from the rebuilt one. Wired into CI
later to keep the reference set honest as captures evolve.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Authoritative sample registry (the registry the methodology binds to).
EVAL_SAMPLE_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"
# Long-form sample with selection metadata (title, surface, change_type, ...).
BAKEOFF_SAMPLE_PATH = REPO_ROOT / "evaluation" / "bakeoff" / "sample-prs.yml"

CODEX_RUNS_DIR = REPO_ROOT / "eval" / "runs"
OPENROUTER_RUNS_DIR = REPO_ROOT / "evaluation" / "bakeoff" / "runs"

OUTPUT_DIR = REPO_ROOT / "evaluation" / "bakeoff" / "comparison"

# Bumped when the comparison-record schema changes in a way that
# downstream consumers (labeling, scoring) must care about.
SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "evaluation/bakeoff/build_comparison_set.py"

# Severity tag pattern — matches the [P0]/[P1]/[P2]/[P3] prefix that
# `prompts/review.md` instructs the model to put on every finding title.
_SEVERITY_TAG_RE = re.compile(r"\[P(?P<level>[0-3])\]")

# ± lines tolerance for the cross-run match-candidate heuristic. Mirrors
# methodology §4.5 ("line ranges overlap or are within ±3 lines"). Set
# at the script level so a single tweak propagates to every PR.
LINE_PROXIMITY_TOLERANCE = 3


# --------------------------------------------------------------------------- #
# Sample loading: bridge the two sample registries.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleEntry:
    """Per-PR registry entry, joined across the two sample files.

    ``eval/sample-prs.yaml`` is the *canonical* registry the methodology
    cites; ``evaluation/bakeoff/sample-prs.yml`` carries richer
    selection metadata. We carry both IDs so artifact paths from either
    side resolve cleanly:

    * ``eval_id``  — e.g. ``"dotcms-core-35567"`` (matches ``eval/runs/<id>/``)
    * ``pr_id``    — e.g. ``"pr-35567"`` (matches ``evaluation/bakeoff/runs/<id>/``)
    """

    pr_id: str
    eval_id: str
    repo: str
    pr_number: int
    head_sha: str
    title: str
    base_ref: str
    head_ref: str
    additions: int
    deletions: int
    changed_files: int
    surface: str
    change_type: str
    security_signal: bool
    languages: list[str]
    slots: list[str]
    loc_changed: int
    why: str

    def codex_artifact_path(self) -> Path:
        # Resolve the runs-dir lazily so tests that monkeypatch
        # ``CODEX_RUNS_DIR`` see their override here, not the value
        # captured at import time.
        return CODEX_RUNS_DIR / self.eval_id / "codex.json"

    def openrouter_artifact_path(self) -> Path:
        return OPENROUTER_RUNS_DIR / self.pr_id / "openrouter.json"

    def comparison_artifact_path(self, base: Path | None = None) -> Path:
        if base is None:
            base = OUTPUT_DIR
        return base / f"{self.pr_id}.json"


def _load_yaml_doc(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SystemExit(f"sample registry not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise SystemExit(f"{path}: top-level document must be a mapping")
    return data


def load_sample_registry(
    *,
    eval_path: Path | None = None,
    bakeoff_path: Path | None = None,
) -> list[SampleEntry]:
    """Return one ``SampleEntry`` per PR, joined across the two registries.

    The eval registry is treated as the authoritative driver: every PR
    in the comparison set must appear there. The bakeoff registry is
    *enriching* — when a PR is missing from it we fall back to safe
    defaults rather than failing the whole build.

    Resolves the registry paths lazily from the module-level constants
    so tests can monkeypatch ``EVAL_SAMPLE_PATH`` / ``BAKEOFF_SAMPLE_PATH``
    and the override is picked up here instead of being baked in at
    function-definition time.
    """
    if eval_path is None:
        eval_path = EVAL_SAMPLE_PATH
    if bakeoff_path is None:
        bakeoff_path = BAKEOFF_SAMPLE_PATH
    eval_doc = _load_yaml_doc(eval_path)
    eval_prs_raw = eval_doc.get("prs") or []
    if not isinstance(eval_prs_raw, list):
        raise SystemExit(f"{eval_path}: 'prs' must be a list")

    bakeoff_doc = _load_yaml_doc(bakeoff_path)
    bakeoff_prs_raw = bakeoff_doc.get("prs") or []
    if not isinstance(bakeoff_prs_raw, list):
        raise SystemExit(f"{bakeoff_path}: 'prs' must be a list")

    bakeoff_by_number: dict[int, Mapping[str, Any]] = {}
    for entry in bakeoff_prs_raw:
        if isinstance(entry, Mapping):
            try:
                bakeoff_by_number[int(entry["number"])] = entry
            except (KeyError, TypeError, ValueError):
                continue

    out: list[SampleEntry] = []
    for entry in eval_prs_raw:
        if not isinstance(entry, Mapping):
            raise SystemExit(f"{eval_path}: each prs[] entry must be a mapping")
        pr_number = int(entry["pr"])
        eval_id = str(entry["id"])
        pr_id = f"pr-{pr_number}"
        bakeoff_entry = bakeoff_by_number.get(pr_number, {})

        languages_raw = entry.get("languages") or []
        languages = [str(x) for x in languages_raw if isinstance(x, str)]
        slots_raw = entry.get("slots") or []
        slots = [str(x) for x in slots_raw if isinstance(x, str)]

        out.append(
            SampleEntry(
                pr_id=pr_id,
                eval_id=eval_id,
                repo=str(entry.get("repo") or bakeoff_entry.get("repo") or ""),
                pr_number=pr_number,
                head_sha=str(entry.get("head_sha") or bakeoff_entry.get("merge_commit") or ""),
                title=str(bakeoff_entry.get("title") or "").strip(),
                base_ref=str(bakeoff_entry.get("base_ref") or "main"),
                head_ref=str(bakeoff_entry.get("head_ref") or ""),
                additions=int(bakeoff_entry.get("additions") or 0),
                deletions=int(bakeoff_entry.get("deletions") or 0),
                changed_files=int(bakeoff_entry.get("changed_files") or 0),
                surface=str(bakeoff_entry.get("surface") or ""),
                change_type=str(bakeoff_entry.get("change_type") or ""),
                security_signal=bool(bakeoff_entry.get("security_signal", False)),
                languages=languages,
                slots=slots,
                loc_changed=int(entry.get("loc_changed") or 0),
                why=str(bakeoff_entry.get("why") or entry.get("notes") or "").strip(),
            )
        )

    return out


# --------------------------------------------------------------------------- #
# Artifact loading and finding normalization.
# --------------------------------------------------------------------------- #


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _extract_severity(title: str, priority: int | None) -> str | None:
    """Return ``"P0"``/``"P1"``/``"P2"``/``"P3"`` from a finding.

    Prefer the explicit ``priority`` field (set by the prompt's rule
    "set priority to 0 for P0, 1 for P1, ..."). Fall back to scanning
    the ``[P<n>]`` tag in the title for findings emitted before the
    numeric field was required, or for hand-written placeholders.
    Returns ``None`` when neither source is available — severity is
    optional in the schema, so an unknown severity is a legitimate
    state, not a parse failure.
    """
    if isinstance(priority, int) and 0 <= priority <= 3:
        return f"P{priority}"
    match = _SEVERITY_TAG_RE.search(title or "")
    if match is None:
        return None
    return f"P{match.group('level')}"


def _normalize_finding(
    raw: Mapping[str, Any],
    *,
    run: str,
    raw_index: int,
) -> dict[str, Any]:
    """Reshape one raw model finding into the comparison-set schema.

    The output is intentionally flat: ``path``, ``line_start``,
    ``line_end``, ``severity``, ``title``, ``message``, ``priority``,
    ``confidence_score``. Downstream labeling and scoring pull from
    these fields directly without re-walking ``code_location``.
    """
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

    title_raw = raw.get("title") if isinstance(raw, Mapping) else None
    title = title_raw.strip() if isinstance(title_raw, str) else ""
    body_raw = raw.get("body") if isinstance(raw, Mapping) else None
    message = body_raw if isinstance(body_raw, str) else ""

    priority_raw = raw.get("priority") if isinstance(raw, Mapping) else None
    priority = priority_raw if isinstance(priority_raw, int) else None

    confidence_raw = raw.get("confidence_score") if isinstance(raw, Mapping) else None
    if isinstance(confidence_raw, (int, float)):
        confidence_score: float | None = float(confidence_raw)
    else:
        confidence_score = None

    severity = _extract_severity(title, priority)

    return {
        "finding_id": f"{run}.f{raw_index + 1:04d}",
        "run": run,
        "raw_index": raw_index,
        "path": path,
        "line_start": line_start,
        "line_end": line_end,
        "severity": severity,
        "priority": priority,
        "title": title,
        "message": message,
        "confidence_score": confidence_score,
        # Labeling fields are populated by the human pass (methodology §2).
        # Pre-creating them here means downstream tools never have to
        # branch on missing-vs-null when reading the schema.
        "label": None,
        "rationale": None,
        "matched_finding_id": None,
        "match_to_other_run": None,
    }


def _extract_run_view(
    artifact: dict[str, Any] | None,
    *,
    run: str,
    expected_head_sha: str,
) -> dict[str, Any]:
    """Pull the comparison-set view out of one raw run artifact.

    Returns a dict with the per-run axes (model, reasoning_effort,
    web_search_mode), the capture status (with both runners' status
    vocabularies normalized to a single key), the top-level review
    summary fields, and the normalized findings array. Findings come
    out of the artifact's ``review_run_result.findings`` when present,
    or as an empty list when the run is still pending/unfilled/error.
    """
    if artifact is None:
        return {
            "present": False,
            "capture_status": "missing",
            "captured_at": None,
            "model": None,
            "reasoning_effort": None,
            "web_search_mode": None,
            "head_sha_at_capture": None,
            "head_sha_matches": None,
            "summary": _empty_summary(),
            "findings": [],
            "carried_forward_count": 0,
            "notes": None,
        }

    if run == "codex":
        capture = artifact.get("capture") if isinstance(artifact, Mapping) else None
        capture = capture if isinstance(capture, Mapping) else {}
        capture_status = str(capture.get("status") or "unknown")
        captured_at = capture.get("captured_at")
        notes = capture.get("notes")
    else:
        capture_status = str(artifact.get("status") or "unknown")
        captured_at = artifact.get("captured_at")
        notes = artifact.get("notes")

    head_sha_at_capture = artifact.get("head_sha")
    head_sha_matches = (
        head_sha_at_capture == expected_head_sha
        if isinstance(head_sha_at_capture, str) and expected_head_sha
        else None
    )

    review = artifact.get("review_run_result") if isinstance(artifact, Mapping) else None
    findings_out: list[dict[str, Any]] = []
    summary = _empty_summary()
    carried_forward_count = 0

    if isinstance(review, Mapping):
        raw_findings = review.get("findings")
        if isinstance(raw_findings, list):
            for idx, item in enumerate(raw_findings):
                if isinstance(item, Mapping):
                    findings_out.append(_normalize_finding(item, run=run, raw_index=idx))
        carried_forward = review.get("carried_forward")
        if isinstance(carried_forward, list):
            carried_forward_count = len(carried_forward)

        oc_raw = review.get("overall_correctness")
        oe_raw = review.get("overall_explanation")
        ocs_raw = review.get("overall_confidence_score")

        summary = {
            "overall_correctness": oc_raw if isinstance(oc_raw, str) else None,
            "overall_explanation": oe_raw if isinstance(oe_raw, str) else None,
            "overall_confidence_score": (
                float(ocs_raw) if isinstance(ocs_raw, (int, float)) else None
            ),
            "findings_count": len(findings_out),
            "carried_forward_count": carried_forward_count,
            "by_severity": _summarize_by_severity(findings_out),
        }

    return {
        "present": True,
        "capture_status": capture_status,
        "captured_at": captured_at if isinstance(captured_at, str) else None,
        "model": artifact.get("model") if isinstance(artifact.get("model"), str) else None,
        "reasoning_effort": (
            artifact.get("reasoning_effort")
            if isinstance(artifact.get("reasoning_effort"), str)
            else None
        ),
        "web_search_mode": (
            artifact.get("web_search_mode")
            if isinstance(artifact.get("web_search_mode"), str)
            else None
        ),
        "head_sha_at_capture": (
            head_sha_at_capture if isinstance(head_sha_at_capture, str) else None
        ),
        "head_sha_matches": head_sha_matches,
        "summary": summary,
        "findings": findings_out,
        "carried_forward_count": carried_forward_count,
        "notes": notes if isinstance(notes, str) else None,
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "overall_correctness": None,
        "overall_explanation": None,
        "overall_confidence_score": None,
        "findings_count": 0,
        "carried_forward_count": 0,
        "by_severity": {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unknown": 0},
    }


def _summarize_by_severity(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unknown": 0}
    for f in findings:
        sev = f.get("severity")
        if isinstance(sev, str) and sev in counts:
            counts[sev] += 1
        else:
            counts["unknown"] += 1
    return counts


# --------------------------------------------------------------------------- #
# Cross-run match candidates.
# --------------------------------------------------------------------------- #


def _line_distance(
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
) -> int | None:
    """Closest-line distance between two ranges. ``None`` if either is unknown.

    Returns 0 when the ranges overlap by ≥ 1 line. Methodology §4.5
    treats overlap or ± 3 lines as matchable; we record the distance
    rather than the boolean so the labeler can re-tune the tolerance
    without re-deriving distances.
    """
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return None
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    if b_end < b_start:
        b_start, b_end = b_end, b_start
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def _build_match_candidates(
    codex_findings: list[dict[str, Any]],
    openrouter_findings: list[dict[str, Any]],
    *,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[dict[str, Any]]:
    """Symmetric all-pairs match-candidate scan.

    Two findings are a *candidate* if they share ``path`` and the
    closest-line distance between them is ``<= tolerance``. Candidacy
    is not the same as a confirmed match — methodology §4.5 still
    requires a human to judge that the two findings describe the same
    underlying bug. We surface candidates so the human pass can start
    from a short list, not the full Cartesian product.

    Each candidate is annotated with:

    * ``line_distance`` — 0 for overlapping ranges, > 0 for adjacent.
    * ``severity_match`` — True when both findings carry the same
      severity tag. Useful for spotting "same path, same line, but
      one says P1 and the other P3" — a class of finding that often
      *is* the same bug despite the severity disagreement.
    """
    out: list[dict[str, Any]] = []
    by_path_or: dict[str, list[dict[str, Any]]] = {}
    for f in openrouter_findings:
        if not isinstance(f.get("path"), str):
            continue
        by_path_or.setdefault(f["path"], []).append(f)

    for cf in codex_findings:
        path = cf.get("path")
        if not isinstance(path, str):
            continue
        for orf in by_path_or.get(path, []):
            distance = _line_distance(
                cf.get("line_start"),
                cf.get("line_end"),
                orf.get("line_start"),
                orf.get("line_end"),
            )
            if distance is None or distance > tolerance:
                continue
            out.append(
                {
                    "codex_finding_id": cf["finding_id"],
                    "openrouter_finding_id": orf["finding_id"],
                    "path": path,
                    "line_distance": distance,
                    "codex_line_range": [cf.get("line_start"), cf.get("line_end")],
                    "openrouter_line_range": [
                        orf.get("line_start"),
                        orf.get("line_end"),
                    ],
                    "severity_match": (
                        cf.get("severity") is not None and cf.get("severity") == orf.get("severity")
                    ),
                }
            )

    # Sort by (line_distance, codex_finding_id, openrouter_finding_id) so the
    # output is stable across runs — important for the --check mode and for
    # diffing the comparison set in code review.
    out.sort(
        key=lambda r: (
            r["line_distance"],
            r["codex_finding_id"],
            r["openrouter_finding_id"],
        )
    )
    return out


# --------------------------------------------------------------------------- #
# Comparison record assembly.
# --------------------------------------------------------------------------- #


_READY_CODEX_STATES = frozenset({"captured"})
_READY_OPENROUTER_STATES = frozenset({"completed"})


def _comparison_status(codex_view: Mapping[str, Any], or_view: Mapping[str, Any]) -> str:
    """Single rollup state used to gate downstream labeling/scoring.

    States, in increasing readiness:

    * ``pending``               — neither side has real findings.
    * ``awaiting_openrouter``   — Codex captured, OpenRouter not yet.
    * ``awaiting_codex``        — OpenRouter completed, Codex not yet.
    * ``ready_for_labeling``    — both sides captured, labeling can begin.
    * ``ready_for_scoring``     — both sides captured AND every finding
                                  carries a non-null ``label``.
    """
    codex_ready = codex_view.get("capture_status") in _READY_CODEX_STATES
    or_ready = or_view.get("capture_status") in _READY_OPENROUTER_STATES
    if codex_ready and or_ready:
        if all(
            isinstance(f.get("label"), str)
            for f in (list(codex_view.get("findings") or []) + list(or_view.get("findings") or []))
        ):
            return "ready_for_scoring"
        return "ready_for_labeling"
    if codex_ready and not or_ready:
        return "awaiting_openrouter"
    if or_ready and not codex_ready:
        return "awaiting_codex"
    return "pending"


def build_comparison_record(entry: SampleEntry, *, generated_at: str) -> dict[str, Any]:
    """Build one fully normalized comparison record for ``entry``."""
    codex_artifact = _load_json_if_present(entry.codex_artifact_path())
    openrouter_artifact = _load_json_if_present(entry.openrouter_artifact_path())

    codex_view = _extract_run_view(codex_artifact, run="codex", expected_head_sha=entry.head_sha)
    or_view = _extract_run_view(
        openrouter_artifact, run="openrouter", expected_head_sha=entry.head_sha
    )

    status = _comparison_status(codex_view, or_view)
    match_candidates = _build_match_candidates(codex_view["findings"], or_view["findings"])

    return {
        "schema_version": SCHEMA_VERSION,
        "pr_id": entry.pr_id,
        "eval_id": entry.eval_id,
        "repo": entry.repo,
        "pr_number": entry.pr_number,
        "head_sha": entry.head_sha,
        "title": entry.title,
        "diff_signal": {
            "additions": entry.additions,
            "deletions": entry.deletions,
            "changed_files": entry.changed_files,
            "loc_changed": entry.loc_changed,
            "surface": entry.surface,
            "change_type": entry.change_type,
            "security_signal": entry.security_signal,
            "languages": list(entry.languages),
            "slots": list(entry.slots),
            "base_ref": entry.base_ref,
            "head_ref": entry.head_ref,
        },
        "selection_rationale": entry.why,
        "runs": {
            "codex": {
                "artifact_path": str(entry.codex_artifact_path().relative_to(REPO_ROOT)),
                **{k: v for k, v in codex_view.items() if k != "findings"},
            },
            "openrouter": {
                "artifact_path": str(entry.openrouter_artifact_path().relative_to(REPO_ROOT)),
                **{k: v for k, v in or_view.items() if k != "findings"},
            },
        },
        "findings": codex_view["findings"] + or_view["findings"],
        "match_candidates": match_candidates,
        "labeling": {
            "status": status,
            "solo_labeled": None,
            "tolerance_lines": LINE_PROXIMITY_TOLERANCE,
        },
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
        "schema_doc": "evaluation/bakeoff/comparison/README.md",
    }


def build_index(records: Sequence[Mapping[str, Any]], *, generated_at: str) -> dict[str, Any]:
    """Sample-level rollup so callers can inspect the whole set in one read."""
    by_status: dict[str, int] = {}
    by_pr: list[dict[str, Any]] = []
    for rec in records:
        status = str(rec.get("labeling", {}).get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        codex_view = rec.get("runs", {}).get("codex", {})
        or_view = rec.get("runs", {}).get("openrouter", {})
        by_pr.append(
            {
                "pr_id": rec.get("pr_id"),
                "eval_id": rec.get("eval_id"),
                "head_sha": rec.get("head_sha"),
                "labeling_status": status,
                "codex_status": codex_view.get("capture_status"),
                "openrouter_status": or_view.get("capture_status"),
                "codex_findings_count": codex_view.get("summary", {}).get("findings_count", 0),
                "openrouter_findings_count": or_view.get("summary", {}).get("findings_count", 0),
                "match_candidate_count": len(rec.get("match_candidates") or []),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
        "sample_size": len(records),
        "status_counts": by_status,
        "prs": by_pr,
    }


# --------------------------------------------------------------------------- #
# Disk I/O.
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_comparison_set(
    entries: Sequence[SampleEntry],
    *,
    output_dir: Path | None = None,
    generated_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build and persist the comparison set. Returns (records, index).

    ``output_dir`` resolves to the module-level ``OUTPUT_DIR`` when
    omitted, so a test that monkeypatches ``OUTPUT_DIR`` is honored.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    records: list[dict[str, Any]] = []
    for entry in entries:
        rec = build_comparison_record(entry, generated_at=generated_at)
        records.append(rec)
        _write_json(entry.comparison_artifact_path(output_dir), rec)
    index = build_index(records, generated_at=generated_at)
    _write_json(output_dir / "_index.json", index)
    return records, index


def check_comparison_set(
    entries: Sequence[SampleEntry],
    *,
    output_dir: Path | None = None,
) -> list[str]:
    """Compare on-disk comparison set against a fresh in-memory rebuild.

    Returns a list of human-readable drift descriptions (one per file
    that differs). Excludes the volatile ``generated_at`` timestamp from
    the comparison so a clean rebuild does not flag itself as drifted.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    drift: list[str] = []
    fresh_records: list[dict[str, Any]] = []
    for entry in entries:
        rec = build_comparison_record(entry, generated_at="<ignored-for-check>")
        fresh_records.append(rec)
        on_disk = _load_json_if_present(entry.comparison_artifact_path(output_dir))
        if on_disk is None:
            drift.append(f"{entry.pr_id}: missing on-disk comparison record")
            continue
        if _strip_volatile(on_disk) != _strip_volatile(rec):
            drift.append(f"{entry.pr_id}: on-disk record drifted from rebuild")

    fresh_index = build_index(fresh_records, generated_at="<ignored-for-check>")
    on_disk_index = _load_json_if_present(output_dir / "_index.json")
    if on_disk_index is None:
        drift.append("_index.json: missing")
    elif _strip_volatile(on_disk_index) != _strip_volatile(fresh_index):
        drift.append("_index.json: drifted from rebuild")
    return drift


def _strip_volatile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with timestamp fields removed.

    The reference set is checked into git, so we don't want a clean
    rebuild on a fresh clone to register as drift purely because the
    ``generated_at`` walltime moved forward.
    """
    out = json.loads(json.dumps(payload))
    if isinstance(out, dict):
        out.pop("generated_at", None)
    return out  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluation.bakeoff.build_comparison_set",
        description=(
            "Build the structured comparison reference set: one normalized "
            "JSON per sample PR plus a sample-level index."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PR_ID",
        help=(
            "Restrict to specific pr_id values (e.g. pr-35567). May be "
            "repeated. Without this flag, every sample PR is rebuilt."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild in memory and exit non-zero if the "
            "on-disk comparison set differs from the rebuild. Use in CI "
            "to keep the reference set in sync with eval/runs/ + "
            "evaluation/bakeoff/runs/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_sample_registry()
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e.pr_id in wanted]
        if not entries:
            raise SystemExit(f"No sample PRs matched --only {args.only!r}")

    if args.check:
        drift = check_comparison_set(entries, output_dir=args.output_dir)
        if drift:
            sys.stderr.write(
                "comparison set drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m evaluation.bakeoff.build_comparison_set`\n"
            )
            return 1
        print(f"comparison set in sync ({len(entries)} PRs)")
        return 0

    records, index = write_comparison_set(entries, output_dir=args.output_dir)
    summary = index["status_counts"]
    summary_str = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
    print(
        "wrote comparison set: "
        f"{len(records)} PRs to {args.output_dir.relative_to(REPO_ROOT)}/ "
        f"({summary_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

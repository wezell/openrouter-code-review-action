"""Core overlap matching engine (Sub-AC 3.2).

This module is the **single implementation** of the overlap matcher
specified in ``eval/overlap_spec.md``. It pairs candidate (OpenRouter)
findings to baseline (Codex) findings by:

* exact ``path`` equality,
* closed-interval line-range proximity within a configurable
  ``± tolerance`` (default 3 lines, per
  ``baseline-methodology.md`` §4.5 rule 2),
* an explicit category-equivalence rule (currently a no-op because the
  schema does not yet carry a ``category`` field — see §4 of the spec),

using a deterministic, reproducible greedy 1:1 assignment:

* sort the full Cartesian set of *path-equal, in-tolerance* candidate
  pairs by ``(line_distance, codex_finding_id, openrouter_finding_id)``,
* walk in order, claiming a pair when neither finding has been used,
* leftover findings on either side become the side-only complements.

Severity is **not** part of the match predicate or the tie-break key
(spec §5). It is carried into the output as an annotation
(``severity_match``) so the parity report can surface miscalibration
without erasing matched bugs.

Why a separate module
---------------------

Sub-AC 2.4 (`compare_findings.py`) is the *consumer* that loads
fixtures, builds reports, and writes JSON. Sub-AC 3.2 is the
*matching engine* it delegates to. Splitting them lets:

* the score-baseline scorer (`score_codex_baseline.py`) reuse the
  exact same predicate without dragging in fixture I/O,
* the test surface for matching semantics (1:1 assignment, tie-breaks,
  tolerance edges) be exercised against this module directly without
  building a parity-report fixture,
* future schema extensions (per-finding ``category`` field) land in
  one place, with the consumers automatically picking up the change.

Public API
----------

* :class:`OverlapFinding`  — input record (one normalized finding).
* :class:`OverlapMatch`    — output record (one matched pair).
* :class:`OverlapResult`   — ``(matches, codex_only, openrouter_only)``.
* :func:`line_distance`    — closed-interval proximity helper.
* :func:`match_findings`   — the engine entry point.
* :data:`LINE_PROXIMITY_TOLERANCE` — default ``± lines`` tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "LINE_PROXIMITY_TOLERANCE",
    "OverlapFinding",
    "OverlapMatch",
    "OverlapResult",
    "line_distance",
    "match_findings",
]


# Methodology §4.5 rule 2: "line ranges overlap or are within ± 3
# lines". The CLI surface in ``eval.compare_findings`` exposes a
# ``--tolerance`` knob that overrides this default for sensitivity
# analysis; the matcher itself is parameterized in ``match_findings``.
LINE_PROXIMITY_TOLERANCE = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapFinding:
    """One normalized finding the matcher consumes.

    The matcher reads only ``finding_id``, ``path``, ``line_start``,
    ``line_end``, and (for the annotation in the output) ``severity``.
    Other fields (``category``, ``title``, etc.) are accepted so callers
    can hand the same record straight through to a report row without
    re-zipping.

    Per spec §2, a finding **without** a usable ``path`` or **without**
    both line endpoints is non-matchable: it never participates in a
    match and is automatically routed to the side-only bucket.
    """

    finding_id: str
    path: str | None
    line_start: int | None
    line_end: int | None
    severity: str | None = None
    # Reserved for the future schema extension described in spec §4.1.
    # Until a ``category`` field exists in ``findings-schema.md`` §2,
    # this stays ``None`` for every input and the matcher's category
    # equivalence rule is a no-op.
    category: str | None = None


@dataclass(frozen=True)
class OverlapMatch:
    """A single matched pair produced by the engine.

    ``line_distance`` is the closed-interval gap (0 for overlap). The
    ``severity_match`` and ``category_match`` flags are pure
    annotations — they never gate the match itself (spec §4, §5).
    """

    codex: OverlapFinding
    openrouter: OverlapFinding
    line_distance: int
    severity_match: bool
    category_match: bool


@dataclass(frozen=True)
class OverlapResult:
    """The triple returned by :func:`match_findings`.

    ``matches`` is ordered by selection order (closest-first, with
    deterministic tie-breaking). ``codex_only`` and ``openrouter_only``
    preserve the *input* order so callers can render side-only buckets
    without re-sorting.
    """

    matches: list[OverlapMatch]
    codex_only: list[OverlapFinding]
    openrouter_only: list[OverlapFinding]


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------


def line_distance(
    a_start: int | None,
    a_end: int | None,
    b_start: int | None,
    b_end: int | None,
) -> int | None:
    """Closed-interval line-range proximity (spec §3.2).

    * Returns ``0`` when the two closed integer intervals overlap by
      ≥ 1 line, i.e. ``max(a_start, b_start) ≤ min(a_end, b_end)``.
    * Returns the non-negative gap (``b_start - a_end`` when ``a`` is
      to the left of ``b``, symmetric on the other side) when they do
      not overlap.
    * Returns ``None`` (undefined) when either range is missing
      endpoints — the spec routes such findings to the side-only bucket.

    Range endpoints in either order are tolerated (a swapped
    ``[end, start]`` is normalized) so a fixture bug that emits the
    range in reverse does not silently produce a negative distance.
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


def _category_match(c: OverlapFinding, o: OverlapFinding) -> bool:
    """Spec §4: ``null`` matches anything; otherwise byte-exact equality.

    Until the schema introduces a ``category`` field this is always
    ``True`` because every input has ``category is None``. The matcher
    retains the rule so the call site doesn't have to special-case the
    transition when the schema lands.
    """
    if c.category is None or o.category is None:
        return True
    return c.category == o.category


def _severity_match(c: OverlapFinding, o: OverlapFinding) -> bool:
    """Spec §4: severity-equality is an *annotation*, not a gate.

    Two findings with ``null`` severity (or any single ``null``)
    produce ``severity_match == False`` — the matcher only flips the
    flag true when both sides emit the same explicit ``P*`` tag. This
    matches ``compare_findings._match_payload`` so callers reading the
    annotation get identical semantics.
    """
    return (
        c.severity is not None
        and o.severity is not None
        and c.severity == o.severity
    )


# ---------------------------------------------------------------------------
# Greedy 1:1 matching engine
# ---------------------------------------------------------------------------


def match_findings(
    codex: Sequence[OverlapFinding],
    openrouter: Sequence[OverlapFinding],
    *,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> OverlapResult:
    """Greedy 1:1 proximity match (spec §6).

    Algorithm:

    1. Build the candidate set: every ``(c, o)`` pair where both
       ``path`` values are non-empty and equal, the category rule
       passes, and ``line_distance(c, o) ≤ tolerance``.
    2. Sort candidates ascending by
       ``(line_distance, codex.finding_id, openrouter.finding_id)``.
       Severity is **not** in the sort key (spec §5).
    3. Walk in order; claim a pair when neither finding has been used
       yet. Drop the candidate otherwise.
    4. Findings not claimed land in their side's complement bucket,
       preserving the original input order.

    The result is bit-stable across reruns and across reorderings of
    the input lists (spec §7).
    """
    if tolerance < 0:
        raise ValueError(
            f"tolerance must be >= 0 (got {tolerance!r}); negative tolerances "
            f"would silently exclude every pair, which is never the intent."
        )

    # Step 1: candidate set (spec §6.1).
    candidates: list[tuple[int, str, str, OverlapFinding, OverlapFinding]] = []
    for c in codex:
        if not c.path:
            continue
        for o in openrouter:
            if not o.path or o.path != c.path:
                continue
            if not _category_match(c, o):
                continue
            d = line_distance(c.line_start, c.line_end, o.line_start, o.line_end)
            if d is None or d > tolerance:
                continue
            candidates.append((d, c.finding_id, o.finding_id, c, o))

    # Step 2: deterministic sort (spec §6.2 / §6.3). The (d, cid, oid)
    # triple is total — there is no remaining ambiguity after the
    # OpenRouter id step, because each candidate is a distinct
    # (codex_id, openrouter_id) pair by construction.
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))

    # Step 3: greedy walk.
    used_codex: set[str] = set()
    used_or: set[str] = set()
    matches: list[OverlapMatch] = []
    for d, _cid, _oid, c, o in candidates:
        if c.finding_id in used_codex or o.finding_id in used_or:
            continue
        matches.append(
            OverlapMatch(
                codex=c,
                openrouter=o,
                line_distance=d,
                severity_match=_severity_match(c, o),
                category_match=_category_match(c, o),
            )
        )
        used_codex.add(c.finding_id)
        used_or.add(o.finding_id)

    # Step 4: complements preserve input order (spec §7 guarantee 4 —
    # lossless: every non-matchable finding lands here too).
    codex_only = [c for c in codex if c.finding_id not in used_codex]
    or_only = [o for o in openrouter if o.finding_id not in used_or]
    return OverlapResult(
        matches=matches,
        codex_only=codex_only,
        openrouter_only=or_only,
    )

"""Tests for the core overlap matching engine (Sub-AC 3.2).

The engine lives in ``eval/overlap_score.py`` and is the single
implementation of the predicate / greedy-matching algorithm specified
in ``eval/overlap_spec.md``. ``eval/compare_findings.py`` delegates to
it; ``score_codex_baseline.py`` will too. These tests pin the
contract every consumer relies on:

* path equality (byte-exact) gates matching,
* line-range proximity uses closed-interval distance with a default
  ± 3 line tolerance,
* assignment is greedy 1:1 and deterministic across reorderings,
* tie-breaks are ``(line_distance, codex_finding_id,
  openrouter_finding_id)`` — severity does **not** enter,
* severity is carried as an annotation on the match record,
* findings without a usable path or both line endpoints are
  non-matchable and surface in the side-only buckets,
* category equivalence is currently a no-op (no schema field) but the
  rule is wired and tested for the future-extension path.

The spec's normative §6.4 worked example is replayed verbatim here so
any drift from the document fails a test rather than being absorbed
silently into the implementation.
"""

from __future__ import annotations

import pytest

from eval.overlap_score import (
    LINE_PROXIMITY_TOLERANCE,
    OverlapFinding,
    OverlapMatch,
    OverlapResult,
    line_distance,
    match_findings,
)


def _f(
    fid: str,
    *,
    path: str | None = "src/foo.py",
    start: int | None = 10,
    end: int | None = None,
    severity: str | None = None,
    category: str | None = None,
) -> OverlapFinding:
    """Compact finding factory; defaults exercise the happy path."""
    return OverlapFinding(
        finding_id=fid,
        path=path,
        line_start=start,
        line_end=start if end is None else end,
        severity=severity,
        category=category,
    )


# ---------------------------------------------------------------------------
# line_distance — proximity helper
# ---------------------------------------------------------------------------


class TestLineDistance:
    def test_overlap_returns_zero(self):
        # 10-12 vs 11-11 — overlap by 1 line.
        assert line_distance(10, 12, 11, 11) == 0

    def test_touching_endpoints_overlap(self):
        # 10-12 vs 12-15 — share line 12.
        assert line_distance(10, 12, 12, 15) == 0

    def test_non_overlapping_returns_gap_a_left(self):
        # 10-12 vs 18-20 — gap = 18 - 12 = 6.
        assert line_distance(10, 12, 18, 20) == 6

    def test_non_overlapping_returns_gap_b_left(self):
        # 18-20 vs 10-12 — symmetric, also 6.
        assert line_distance(18, 20, 10, 12) == 6

    def test_adjacent_returns_one(self):
        # 10-12 vs 13-14 — gap of 1 line (no shared line).
        assert line_distance(10, 12, 13, 14) == 1

    def test_swapped_endpoints_normalized(self):
        # A fixture bug that emits [end, start] should not produce a
        # negative distance — the helper normalizes the range.
        assert line_distance(12, 10, 14, 13) == 1

    @pytest.mark.parametrize(
        "args",
        [
            (None, 12, 11, 11),
            (10, None, 11, 11),
            (10, 12, None, 11),
            (10, 12, 11, None),
        ],
    )
    def test_missing_endpoint_returns_none(self, args):
        assert line_distance(*args) is None


# ---------------------------------------------------------------------------
# match_findings — basic predicate behavior
# ---------------------------------------------------------------------------


class TestMatchPredicate:
    def test_path_equal_in_tolerance_matches(self):
        c = _f("c.f0001", start=10, end=12)
        o = _f("or.f0001", start=11, end=11)
        result = match_findings([c], [o])
        assert len(result.matches) == 1
        m = result.matches[0]
        assert m.codex is c
        assert m.openrouter is o
        assert m.line_distance == 0

    def test_path_mismatch_disqualifies(self):
        c = _f("c.f0001", path="src/a.py")
        o = _f("or.f0001", path="src/b.py")
        result = match_findings([c], [o])
        assert result.matches == []
        assert result.codex_only == [c]
        assert result.openrouter_only == [o]

    def test_path_byte_exact_no_normalization(self):
        # ``./foo.py`` vs ``foo.py`` are NOT considered equal — spec §3.1.
        c = _f("c.f0001", path="./src/foo.py")
        o = _f("or.f0001", path="src/foo.py")
        result = match_findings([c], [o])
        assert result.matches == []

    def test_distance_above_tolerance_disqualifies(self):
        c = _f("c.f0001", start=10, end=10)
        o = _f("or.f0001", start=14, end=14)  # gap of 4 > tolerance(3)
        result = match_findings([c], [o])
        assert result.matches == []

    def test_distance_at_tolerance_boundary_matches(self):
        # gap of exactly 3 — included by ``<=`` semantics in spec §3.2.
        c = _f("c.f0001", start=10, end=10)
        o = _f("or.f0001", start=13, end=13)
        result = match_findings([c], [o])
        assert len(result.matches) == 1
        assert result.matches[0].line_distance == 3

    def test_custom_tolerance_zero_requires_overlap(self):
        c = _f("c.f0001", start=10, end=10)
        o = _f("or.f0001", start=11, end=11)  # gap of 1
        result = match_findings([c], [o], tolerance=0)
        assert result.matches == []
        # But true overlap survives tolerance=0:
        c2 = _f("c.f0002", start=10, end=12)
        o2 = _f("or.f0002", start=11, end=11)
        result2 = match_findings([c2], [o2], tolerance=0)
        assert len(result2.matches) == 1

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            match_findings([], [], tolerance=-1)


# ---------------------------------------------------------------------------
# match_findings — non-matchable findings (spec §2)
# ---------------------------------------------------------------------------


class TestNonMatchable:
    def test_missing_path_codex_routes_to_codex_only(self):
        c = _f("c.f0001", path=None)
        o = _f("or.f0001")
        result = match_findings([c], [o])
        assert result.matches == []
        assert c in result.codex_only
        assert o in result.openrouter_only

    def test_empty_path_codex_disqualified(self):
        c = _f("c.f0001", path="")
        o = _f("or.f0001")
        result = match_findings([c], [o])
        assert result.matches == []

    def test_missing_line_start_codex_disqualified(self):
        c = _f("c.f0001", start=None, end=None)
        o = _f("or.f0001", start=10, end=10)
        result = match_findings([c], [o])
        assert result.matches == []
        assert c in result.codex_only
        assert o in result.openrouter_only


# ---------------------------------------------------------------------------
# match_findings — greedy 1:1 assignment (spec §6.2 / §7)
# ---------------------------------------------------------------------------


class TestGreedyAssignment:
    def test_each_finding_used_at_most_once(self):
        # Two Codex findings within tolerance of one OpenRouter finding.
        # Only the closer pair should win; the farther one becomes
        # codex_only.
        c1 = _f("c.f0001", start=10, end=10)  # distance 0 to o.f0001
        c2 = _f("c.f0002", start=12, end=12)  # distance 2 to o.f0001
        o = _f("or.f0001", start=10, end=10)
        result = match_findings([c1, c2], [o])
        assert len(result.matches) == 1
        assert result.matches[0].codex.finding_id == "c.f0001"
        assert [c.finding_id for c in result.codex_only] == ["c.f0002"]
        assert result.openrouter_only == []

    def test_lossless_total_count_preserved(self):
        # Spec §7 guarantee 4: every input finding is accounted for.
        codex = [_f(f"c.f{i:04d}", start=100 + i * 50) for i in range(3)]
        openrouter = [_f(f"or.f{i:04d}", start=100 + i * 50) for i in range(3)]
        result = match_findings(codex, openrouter)
        seen = {f.finding_id for f in result.codex_only}
        seen |= {m.codex.finding_id for m in result.matches}
        assert seen == {f.finding_id for f in codex}
        seen_or = {f.finding_id for f in result.openrouter_only}
        seen_or |= {m.openrouter.finding_id for m in result.matches}
        assert seen_or == {f.finding_id for f in openrouter}

    def test_path_disjoint_never_match(self):
        # Even at line_distance == 0, findings on different paths never
        # match (spec §7 guarantee 5).
        c = _f("c.f0001", path="src/a.py", start=10, end=10)
        o = _f("or.f0001", path="src/b.py", start=10, end=10)
        result = match_findings([c], [o])
        assert result.matches == []


# ---------------------------------------------------------------------------
# match_findings — determinism and tie-break (spec §6.3)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_reordering_inputs_preserves_match_set(self):
        c1 = _f("c.f0001", start=10, end=10)
        c2 = _f("c.f0002", start=20, end=20)
        o1 = _f("or.f0001", start=10, end=10)
        o2 = _f("or.f0002", start=20, end=20)
        forward = match_findings([c1, c2], [o1, o2])
        reversed_ = match_findings([c2, c1], [o2, o1])
        # Match content is identical (closest pair claimed first), but
        # the side-only ordering follows the *input* order, so we
        # compare the match content explicitly.
        ids_forward = sorted(
            (m.codex.finding_id, m.openrouter.finding_id)
            for m in forward.matches
        )
        ids_reversed = sorted(
            (m.codex.finding_id, m.openrouter.finding_id)
            for m in reversed_.matches
        )
        assert ids_forward == ids_reversed

    def test_idempotent(self):
        c = _f("c.f0001", start=10, end=10)
        o = _f("or.f0001", start=11, end=11)
        first = match_findings([c], [o])
        second = match_findings([c], [o])
        assert first == second

    def test_tie_break_by_codex_id_then_openrouter_id(self):
        # Two equidistant pairs sharing one OpenRouter finding —
        # the lexicographically first Codex id should win.
        # c.f0002 sorts after c.f0001, both at distance 0.
        c1 = _f("c.f0001", start=10, end=10)
        c2 = _f("c.f0002", start=10, end=10)
        o = _f("or.f0001", start=10, end=10)
        result = match_findings([c2, c1], [o])  # input order swapped
        assert len(result.matches) == 1
        assert result.matches[0].codex.finding_id == "c.f0001"

    def test_severity_does_not_affect_tie_break(self):
        # Spec §5 / §6.3: severity is intentionally NOT in the sort
        # key. Two equidistant OpenRouter findings, one severity-aligned
        # with Codex and one not, must resolve by finding_id alone.
        c = _f("c.f0001", start=10, end=10, severity="P0")
        # ``or.f0001`` is severity-mismatched but has the lower id.
        # ``or.f0002`` matches Codex's severity. Tie-break must pick
        # ``or.f0001`` regardless.
        o1 = _f("or.f0001", start=10, end=10, severity="P3")
        o2 = _f("or.f0002", start=10, end=10, severity="P0")
        result = match_findings([c], [o1, o2])
        assert len(result.matches) == 1
        assert result.matches[0].openrouter.finding_id == "or.f0001"
        assert result.matches[0].severity_match is False

    def test_closer_pair_beats_id_order(self):
        # ``c.f0001`` is in tolerance of both ``or.f0001`` (gap 0) and
        # ``or.f0002`` (gap 0). But the closer pair claims first.
        # Set up a real distance gap to make the assertion interesting:
        c = _f("c.f0001", start=10, end=10)
        o_far = _f("or.f0001", start=13, end=13)  # distance 3
        o_near = _f("or.f0002", start=10, end=10)  # distance 0
        result = match_findings([c], [o_far, o_near])
        assert len(result.matches) == 1
        # Closer pair wins despite ``or.f0001`` having the smaller id.
        assert result.matches[0].openrouter.finding_id == "or.f0002"
        assert result.matches[0].line_distance == 0


# ---------------------------------------------------------------------------
# match_findings — severity / category annotations (spec §4, §5)
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_severity_match_true_when_both_p0(self):
        c = _f("c.f0001", severity="P0")
        o = _f("or.f0001", severity="P0")
        m = match_findings([c], [o]).matches[0]
        assert m.severity_match is True

    def test_severity_match_false_when_different(self):
        c = _f("c.f0001", severity="P0")
        o = _f("or.f0001", severity="P2")
        m = match_findings([c], [o]).matches[0]
        assert m.severity_match is False

    def test_severity_match_false_when_one_null(self):
        c = _f("c.f0001", severity="P0")
        o = _f("or.f0001", severity=None)
        m = match_findings([c], [o]).matches[0]
        assert m.severity_match is False

    def test_severity_mismatch_still_matches(self):
        # Spec §4 / methodology §3: severity mismatch is annotated, not
        # gated.
        c = _f("c.f0001", severity="P0")
        o = _f("or.f0001", severity="P3")
        result = match_findings([c], [o])
        assert len(result.matches) == 1

    def test_category_null_treated_as_match(self):
        # Until the schema field exists, every input has category=None,
        # which the spec §4 rule treats as matching anything.
        c = _f("c.f0001", category=None)
        o = _f("or.f0001", category=None)
        m = match_findings([c], [o]).matches[0]
        assert m.category_match is True

    def test_category_explicit_match_passes(self):
        c = _f("c.f0001", category="security")
        o = _f("or.f0001", category="security")
        m = match_findings([c], [o]).matches[0]
        assert m.category_match is True

    def test_category_explicit_mismatch_disqualifies(self):
        # The future-schema rule (spec §4.1) disqualifies pairs with
        # both categories present and different. Until the field
        # exists this branch is unreachable from production data, but
        # the behavior is wired today so the predicate is consistent.
        c = _f("c.f0001", category="security")
        o = _f("or.f0001", category="style")
        result = match_findings([c], [o])
        assert result.matches == []

    def test_category_one_null_one_present_still_matches(self):
        # Spec §4.1: ``null`` matches anything.
        c = _f("c.f0001", category=None)
        o = _f("or.f0001", category="security")
        result = match_findings([c], [o])
        assert len(result.matches) == 1


# ---------------------------------------------------------------------------
# Spec §6.4 — normative worked example
# ---------------------------------------------------------------------------


class TestSpecWorkedExample:
    """Replay the §6.4 worked example verbatim. If this fails, either
    the implementation drifted from the spec or the spec was edited
    without updating the matcher — both are bugs."""

    def test_worked_example_replay(self):
        codex = [
            _f("c.f0001", start=10, end=12, severity="P1"),
            _f("c.f0002", start=18, end=20, severity="P2"),
            _f("c.f0003", start=100, end=100, severity="P0"),
        ]
        openrouter = [
            _f("or.f0001", start=11, end=11, severity="P1"),
            _f("or.f0002", start=19, end=19, severity="P2"),
            _f("or.f0003", start=21, end=22, severity="P3"),
            _f("or.f0004", start=200, end=200, severity="P1"),
        ]
        result = match_findings(codex, openrouter)

        assert [(m.codex.finding_id, m.openrouter.finding_id, m.line_distance)
                for m in result.matches] == [
            ("c.f0001", "or.f0001", 0),
            ("c.f0002", "or.f0002", 0),
        ]
        assert [c.finding_id for c in result.codex_only] == ["c.f0003"]
        assert [o.finding_id for o in result.openrouter_only] == [
            "or.f0003",
            "or.f0004",
        ]


# ---------------------------------------------------------------------------
# Public surface sanity
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_default_tolerance_constant(self):
        assert LINE_PROXIMITY_TOLERANCE == 3

    def test_overlap_result_shape(self):
        result = match_findings([], [])
        assert isinstance(result, OverlapResult)
        assert result.matches == []
        assert result.codex_only == []
        assert result.openrouter_only == []

    def test_overlap_match_shape(self):
        c = _f("c.f0001", severity="P0")
        o = _f("or.f0001", severity="P0")
        m = match_findings([c], [o]).matches[0]
        assert isinstance(m, OverlapMatch)
        # frozen dataclass
        with pytest.raises(Exception):
            m.line_distance = 99  # type: ignore[misc]

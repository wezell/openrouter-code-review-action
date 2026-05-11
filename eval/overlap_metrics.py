#!/usr/bin/env python3
"""Per-PR overlap metrics: precision, recall, F1 (Sub-AC 3.3).

This module turns the geometric overlap match (Sub-AC 3.2 engine in
``eval/overlap_score.py``, fixture wiring in
``eval/compare_findings.py``) into the classic information-retrieval
triple — precision, recall, F1 — treating the Codex baseline as
ground truth and the OpenRouter candidate as the prediction:

* **TP** (true positive)  — matched pair (Codex finding paired to an
  OpenRouter finding within ± tolerance lines on the same path).
* **FP** (false positive) — OpenRouter finding with no Codex match
  (``openrouter_only``).
* **FN** (false negative) — Codex finding with no OpenRouter match
  (``codex_only``).

Metrics:

* ``precision = TP / (TP + FP)`` — share of OpenRouter findings that
  align with the baseline.
* ``recall    = TP / (TP + FN)`` — share of baseline findings the
  candidate recovered.
* ``f1       = 2·P·R / (P + R)`` — harmonic mean.

When a denominator is zero (e.g. zero predictions or zero baseline
findings) the corresponding metric is emitted as ``null`` so a
reviewer reads "not measurable" rather than "0%". F1 is ``null``
whenever either of its inputs is ``null`` or the sum is zero.

Inputs (per PR)
---------------

The baseline / candidate fixtures live under
``tests/fixtures/{baseline,candidate}/<fixture_pr_id>/`` (Sub-ACs 2.2
& 2.3); the eval-runs naming convention from
``eval/sample-prs.yaml`` uses ``dotcms-core-<number>``. We map between
the two by stripping the ``dotcms-core-`` prefix and prepending
``pr-`` (e.g. ``dotcms-core-35449`` → ``pr-35449``).

Output
------

::

    eval/runs/<pr_id>/overlap.json

The path matches the directory the bake-off already uses for the
Codex run artifact (``codex.json``) so reviewers can sit the metrics
record alongside the raw run.

Status semantics
----------------

Mirrors :mod:`eval.compare_findings`:

* ``ready``           — both sides captured at the same head_sha.
                        Metrics are reported.
* ``head_sha_drift``  — both captured but on different head_shas. The
                        match counts are emitted but the headline
                        metrics are forced to ``null`` because the
                        diffs reviewed differ.
* ``codex_only`` / ``openrouter_only`` / ``pending`` — only one side
                        (or neither) is captured. Metrics are
                        ``null``; counts surface what is present.

Usage
-----

::

    python -m eval.overlap_metrics                              # all
    python -m eval.overlap_metrics --pr-id dotcms-core-35449    # one
    python -m eval.overlap_metrics --check                      # CI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.compare_findings import (
    PRComparison,
    compare_pr,
)
from eval.overlap_score import LINE_PROXIMITY_TOLERANCE

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "eval" / "runs"
SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "eval/overlap_metrics.py"

# Eval-runs directory naming convention pinned by ``eval/sample-prs.yaml``
# (``id: dotcms-core-<number>``). Fixture trees use ``pr-<number>``.
_RUNS_PR_PREFIX = "dotcms-core-"
_FIXTURE_PR_PREFIX = "pr-"


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


def runs_id_to_fixture_id(runs_id: str) -> str:
    """Translate ``dotcms-core-35449`` → ``pr-35449``.

    Raises ``ValueError`` for unrecognized prefixes so a typo in a CLI
    argument fails loudly instead of silently mapping to an empty
    fixture tree.
    """
    if not runs_id.startswith(_RUNS_PR_PREFIX):
        raise ValueError(
            f"runs id {runs_id!r} does not start with "
            f"{_RUNS_PR_PREFIX!r}; expected ``{_RUNS_PR_PREFIX}<number>``"
        )
    suffix = runs_id[len(_RUNS_PR_PREFIX):]
    if not suffix:
        raise ValueError(f"runs id {runs_id!r} is missing the numeric suffix")
    return f"{_FIXTURE_PR_PREFIX}{suffix}"


def fixture_id_to_runs_id(fixture_id: str) -> str:
    """Inverse of :func:`runs_id_to_fixture_id`."""
    if not fixture_id.startswith(_FIXTURE_PR_PREFIX):
        raise ValueError(
            f"fixture id {fixture_id!r} does not start with "
            f"{_FIXTURE_PR_PREFIX!r}; expected ``{_FIXTURE_PR_PREFIX}<number>``"
        )
    suffix = fixture_id[len(_FIXTURE_PR_PREFIX):]
    if not suffix:
        raise ValueError(f"fixture id {fixture_id!r} is missing the numeric suffix")
    return f"{_RUNS_PR_PREFIX}{suffix}"


# ---------------------------------------------------------------------------
# Metrics math
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapMetrics:
    """Precision/recall/F1 derived from match counts.

    All three metrics are ``None`` when their classical denominator is
    zero so the JSON serializer surfaces ``null`` (versus a misleading
    ``0.0``).
    """

    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None


def _safe_ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def compute_metrics(tp: int, fp: int, fn: int) -> OverlapMetrics:
    """Compute (precision, recall, F1) from raw match counts.

    * ``precision = tp / (tp + fp)``
    * ``recall    = tp / (tp + fn)``
    * ``f1        = 2 · p · r / (p + r)``

    Any metric whose denominator collapses to zero is reported as
    ``None``; F1 is ``None`` whenever either input is ``None`` or the
    sum is zero (``2·0·0/(0+0)`` is undefined). Negative inputs raise
    ``ValueError`` because they cannot represent a valid count.
    """
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError(
            f"counts must be non-negative (tp={tp}, fp={fp}, fn={fn})"
        )
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision is None or recall is None:
        f1: float | None = None
    elif precision + recall <= 0:
        f1 = 0.0
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return OverlapMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
    )


# ---------------------------------------------------------------------------
# Comparison → record
# ---------------------------------------------------------------------------


def _metrics_payload(m: OverlapMetrics) -> dict[str, Any]:
    return {
        "true_positive": m.tp,
        "false_positive": m.fp,
        "false_negative": m.fn,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
    }


def build_overlap_record(
    cmp: PRComparison,
    *,
    runs_id: str,
    generated_at: str,
    tolerance: int,
) -> dict[str, Any]:
    """Build the JSON record persisted at ``eval/runs/<pr_id>/overlap.json``.

    For statuses other than ``ready`` (the only state where the
    OpenRouter side has been captured at the same head_sha as the
    Codex baseline) the headline metrics are forced to ``null``: the
    raw ``true_positive`` / ``false_positive`` / ``false_negative``
    counts are still emitted so a reviewer can see how each side's
    placeholder/partial state surfaced, but the precision/recall/F1
    triple stays unmeasurable.
    """
    tp = len(cmp.matches)
    fp = len(cmp.openrouter_only)
    fn = len(cmp.codex_only)

    measurable = cmp.status == "ready"
    if measurable:
        metrics = compute_metrics(tp, fp, fn)
    else:
        metrics = OverlapMetrics(
            tp=tp, fp=fp, fn=fn,
            precision=None, recall=None, f1=None,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "pr_id": runs_id,
        "fixture_pr_id": cmp.pr_id,
        "status": cmp.status,
        "tolerance_lines": tolerance,
        "head_sha_baseline": cmp.head_sha_baseline,
        "head_sha_candidate": cmp.head_sha_candidate,
        "metrics": _metrics_payload(metrics),
        "counts": {
            "codex_total": len(cmp.codex_findings),
            "openrouter_total": len(cmp.openrouter_findings),
            "matched_pairs": tp,
            "codex_only": fn,
            "openrouter_only": fp,
        },
        "models": {
            "codex": cmp.codex_model,
            "openrouter": cmp.openrouter_model,
        },
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
    }


# ---------------------------------------------------------------------------
# Persistence + drift check
# ---------------------------------------------------------------------------


def _overlap_path(runs_id: str, *, runs_root: Path | None = None) -> Path:
    base = runs_root if runs_root is not None else RUNS_ROOT
    return base / runs_id / "overlap.json"


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


def write_overlap(
    runs_id: str,
    *,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    runs_root: Path | None = None,
    generated_at: str | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> Path:
    """Compute and persist ``eval/runs/<runs_id>/overlap.json``."""
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fixture_id = runs_id_to_fixture_id(runs_id)
    cmp = compare_pr(
        fixture_id,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        tolerance=tolerance,
    )
    record = build_overlap_record(
        cmp,
        runs_id=runs_id,
        generated_at=generated_at,
        tolerance=tolerance,
    )
    path = _overlap_path(runs_id, runs_root=runs_root)
    _atomic_write(path, _serialize(record))
    return path


def check_overlap(
    runs_ids: Sequence[str],
    *,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    runs_root: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[str]:
    """Return drift messages — empty list when on-disk == rebuild."""
    drift: list[str] = []
    for runs_id in runs_ids:
        try:
            fixture_id = runs_id_to_fixture_id(runs_id)
        except ValueError as exc:
            drift.append(f"{runs_id}: {exc}")
            continue
        cmp = compare_pr(
            fixture_id,
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            tolerance=tolerance,
        )
        fresh = build_overlap_record(
            cmp,
            runs_id=runs_id,
            generated_at="<ignored-for-check>",
            tolerance=tolerance,
        )
        path = _overlap_path(runs_id, runs_root=runs_root)
        if not path.is_file():
            drift.append(f"{runs_id}: missing overlap.json at {path}")
            continue
        try:
            with path.open(encoding="utf-8") as f:
                on_disk = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            drift.append(f"{runs_id}: cannot read overlap.json ({exc})")
            continue
        if not isinstance(on_disk, Mapping):
            drift.append(f"{runs_id}: overlap.json is not an object")
            continue
        if _strip_volatile(on_disk) != _strip_volatile(fresh):
            drift.append(f"{runs_id}: on-disk overlap.json differs from rebuild")
    return drift


# ---------------------------------------------------------------------------
# Discovery + CLI
# ---------------------------------------------------------------------------


def discover_runs_ids(*, runs_root: Path | None = None) -> list[str]:
    """Every directory under ``eval/runs/`` that looks like a PR run."""
    base = runs_root if runs_root is not None else RUNS_ROOT
    if not base.is_dir():
        return []
    out: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(_RUNS_PR_PREFIX):
            continue
        out.append(child.name)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.overlap_metrics",
        description=(
            "Compute per-PR overlap precision/recall/F1 against the Codex "
            "baseline (Sub-AC 3.3). Writes eval/runs/<pr_id>/overlap.json."
        ),
    )
    parser.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help=(
            "Compute only this run id (e.g. ``dotcms-core-35449``). May be "
            "repeated."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild and exit non-zero if the on-disk overlap "
            "record differs from the rebuild. Use in CI."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=LINE_PROXIMITY_TOLERANCE,
        help=(
            f"± lines tolerance for proximity matching "
            f"(default: {LINE_PROXIMITY_TOLERANCE})."
        ),
    )
    args = parser.parse_args(argv)

    discovered = discover_runs_ids()
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
        print(
            "[ok] no eval/runs/<pr_id>/ directories found; nothing to do",
            file=sys.stderr,
        )
        return 0

    if args.check:
        drift = check_overlap(targets, tolerance=args.tolerance)
        if drift:
            sys.stderr.write(
                "overlap metrics drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m eval.overlap_metrics`\n"
            )
            return 1
        print(f"overlap metrics in sync ({len(targets)} PRs)")
        return 0

    written: list[Path] = []
    for runs_id in targets:
        written.append(write_overlap(runs_id, tolerance=args.tolerance))
    print(f"wrote {len(written)} overlap.json record(s) under eval/runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sample-wide overlap rollup (Sub-AC 3.4).

Per-PR overlap metrics live in ``eval/runs/<pr_id>/overlap.json``
(Sub-AC 3.3). This module rolls those across the full sample defined
by ``eval/sample-prs.yaml`` and produces a single ``eval/overlap_aggregate.json``
record so a reviewer can read sample-wide precision/recall plus
per-category breakdowns without walking every per-PR file.

What "category" means here
--------------------------

The findings schema (``eval/findings-schema.md`` and §1 envelope used
by both fixture trees) does not carry a free-form ``category`` field.
The closest discriminator that survives normalization is the
**severity** bucket extracted by ``eval/compare_findings.py`` from the
``[P0]..[P3]`` title prefix or numeric ``priority`` field. We treat
those buckets — ``P0``, ``P1``, ``P2``, ``P3``, ``unknown`` — as the
category axis for the rollup. If a future schema revision adds a
``category`` field, this module is the single place to extend.

Sample-wide metrics
-------------------

We compute precision/recall/F1 in two flavors:

* **Micro** — sum TP/FP/FN across every ``ready`` PR, then apply the
  classic IR formulas. Treats every finding as one observation. This
  is the headline number a reviewer looks at first.
* **Macro** — average per-PR precision/recall/F1 across PRs whose
  status is ``ready`` AND whose denominator is non-null. Treats every
  PR as one observation regardless of finding count, so a single
  high-volume PR cannot dominate the headline.

Both flavors collapse to ``null`` when no PR is ``ready`` yet (or when
no PR contributed a measurable number for a given metric) so a
reviewer reads "not measurable" rather than "0%".

Per-category breakdown
----------------------

For each severity bucket we emit:

* ``codex_total``       — Codex findings with this severity across all
                          ``ready`` PRs (recall denominator).
* ``openrouter_total``  — OpenRouter findings with this severity across
                          all ``ready`` PRs (precision denominator).
* ``matched_codex``     — matched pairs whose **Codex** side carries
                          this severity (recall numerator).
* ``matched_openrouter``— matched pairs whose **OpenRouter** side
                          carries this severity (precision numerator).
* ``recall``            — matched_codex / codex_total (recall from
                          Codex perspective).
* ``precision``         — matched_openrouter / openrouter_total
                          (precision from OpenRouter perspective).

We split the numerator by side because severity disagreement is a
real signal: a Codex P0 paired with an OpenRouter P3 contributes one
match to ``P0.matched_codex`` and one to ``P3.matched_openrouter``,
not one to either bucket alone. ``severity_match_total`` and the
match rate stay the cross-side check.

Inputs
------

* ``eval/sample-prs.yaml``                  — sample registry; if
                                              missing/unreadable we
                                              fall back to discovering
                                              every PR present in the
                                              fixture trees so the
                                              rollup still runs in
                                              fresh checkouts.
* ``tests/fixtures/baseline/<pr-id>/codex-findings.json``
* ``tests/fixtures/candidate/<pr-id>/openrouter-findings.json``

Output (committed alongside the other eval artifacts)
-----------------------------------------------------

``eval/overlap_aggregate.json``

The record is **idempotent**: ``--check`` rebuilds in memory and exits
non-zero if the on-disk artifact drifts from the rebuild, so CI can
keep the rollup in sync with whichever side last got recaptured.

Usage
-----

::

    python -m eval.overlap_aggregate         # rebuild
    python -m eval.overlap_aggregate --check # CI drift check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.compare_findings import (
    BASELINE_ROOT,
    CANDIDATE_ROOT,
    PRComparison,
    compare_pr,
    discover_pr_ids,
)
from eval.overlap_metrics import (
    fixture_id_to_runs_id,
    runs_id_to_fixture_id,
)
from eval.overlap_score import LINE_PROXIMITY_TOLERANCE

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_REGISTRY_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"
AGGREGATE_PATH = REPO_ROOT / "eval" / "overlap_aggregate.json"
SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "eval/overlap_aggregate.py"

# The severity axis used for per-category breakdowns. Mirrors
# ``compare_findings._by_severity`` so rebuild-after-edit stays
# consistent. Order is fixed for stable JSON output.
_SEVERITY_BUCKETS: tuple[str, ...] = ("P0", "P1", "P2", "P3", "unknown")


# ---------------------------------------------------------------------------
# Sample registry loading
# ---------------------------------------------------------------------------


def _read_sample_registry(path: Path) -> list[str]:
    """Return the list of ``dotcms-core-<n>`` ids from ``sample-prs.yaml``.

    We avoid taking a YAML dependency for this module: the registry is
    flat enough that a tiny line-scan over ``id: dotcms-core-<n>`` is
    sufficient and means this rollup runs in any Python environment
    that the rest of the eval tooling already supports.
    """
    if not path.is_file():
        return []
    ids: list[str] = []
    try:
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("#") or not line:
                    continue
                # Match ``id: dotcms-core-NNNNN`` and quoted variants.
                if not line.startswith(("id:", "- id:")):
                    continue
                _, _, value = line.partition("id:")
                value = value.strip().strip("'\"")
                if value.startswith("dotcms-core-"):
                    ids.append(value)
    except OSError:
        return []
    # Preserve registry order, dedupe.
    seen: set[str] = set()
    out: list[str] = []
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    return out


def _resolve_targets(
    *,
    registry_path: Path | None,
    baseline_root: Path | None,
    candidate_root: Path | None,
) -> tuple[list[str], str]:
    """Return ``(runs_ids, source)`` describing the sample to roll up.

    Preference order:

    1. ``eval/sample-prs.yaml`` — the canonical 10-PR registry.
    2. Discovery over the fixture trees, intersected with sample
       prefixes, so a fresh checkout still produces a meaningful
       rollup.
    """
    registry = registry_path if registry_path is not None else SAMPLE_REGISTRY_PATH
    runs_ids = _read_sample_registry(registry)
    if runs_ids:
        return runs_ids, "registry"
    fixture_ids = discover_pr_ids(
        baseline_root=baseline_root, candidate_root=candidate_root
    )
    discovered: list[str] = []
    for fid in fixture_ids:
        try:
            discovered.append(fixture_id_to_runs_id(fid))
        except ValueError:
            continue
    return discovered, "fixture_discovery"


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------


def _safe_ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _f1_from(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall <= 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


@dataclass(frozen=True)
class _PerPRRow:
    """Compact per-PR snapshot used by both totals + macro averaging."""

    runs_id: str
    fixture_id: str
    status: str
    head_sha_baseline: str | None
    head_sha_candidate: str | None
    codex_total: int
    openrouter_total: int
    matched_pairs: int
    codex_only: int
    openrouter_only: int
    severity_match: int
    precision: float | None
    recall: float | None
    f1: float | None


def _per_pr_metrics(cmp: PRComparison) -> tuple[float | None, float | None, float | None]:
    """Per-PR P/R/F1, only meaningful when ``status == "ready"``."""
    if cmp.status != "ready":
        return None, None, None
    tp = len(cmp.matches)
    fp = len(cmp.openrouter_only)
    fn = len(cmp.codex_only)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return precision, recall, _f1_from(precision, recall)


def _severity_match_count(cmp: PRComparison) -> int:
    n = 0
    for m in cmp.matches:
        cs = m.codex.severity
        os_ = m.openrouter.severity
        if cs is not None and os_ is not None and cs == os_:
            n += 1
    return n


def _build_row(runs_id: str, cmp: PRComparison) -> _PerPRRow:
    p, r, f1 = _per_pr_metrics(cmp)
    return _PerPRRow(
        runs_id=runs_id,
        fixture_id=cmp.pr_id,
        status=cmp.status,
        head_sha_baseline=cmp.head_sha_baseline,
        head_sha_candidate=cmp.head_sha_candidate,
        codex_total=len(cmp.codex_findings),
        openrouter_total=len(cmp.openrouter_findings),
        matched_pairs=len(cmp.matches),
        codex_only=len(cmp.codex_only),
        openrouter_only=len(cmp.openrouter_only),
        severity_match=_severity_match_count(cmp),
        precision=p,
        recall=r,
        f1=f1,
    )


def _bucket(sev: str | None) -> str:
    if sev in _SEVERITY_BUCKETS:
        return sev  # type: ignore[return-value]
    return "unknown"


def _per_category(comparisons: Sequence[PRComparison]) -> dict[str, dict[str, Any]]:
    """Roll up per-severity totals + matched pairs across ``ready`` PRs.

    Non-``ready`` PRs are excluded so the precision/recall numbers
    here align with the headline micro-metrics — the diff being
    reviewed has to match on both sides for a comparison to be
    meaningful.
    """
    out: dict[str, dict[str, Any]] = {
        sev: {
            "codex_total": 0,
            "openrouter_total": 0,
            "matched_codex": 0,
            "matched_openrouter": 0,
        }
        for sev in _SEVERITY_BUCKETS
    }
    for cmp in comparisons:
        if cmp.status != "ready":
            continue
        for f in cmp.codex_findings:
            out[_bucket(f.severity)]["codex_total"] += 1
        for f in cmp.openrouter_findings:
            out[_bucket(f.severity)]["openrouter_total"] += 1
        for m in cmp.matches:
            out[_bucket(m.codex.severity)]["matched_codex"] += 1
            out[_bucket(m.openrouter.severity)]["matched_openrouter"] += 1
    # Compute per-bucket precision/recall now that the totals are in.
    for sev in _SEVERITY_BUCKETS:
        bucket = out[sev]
        bucket["recall"] = _safe_ratio(
            bucket["matched_codex"], bucket["codex_total"]
        )
        bucket["precision"] = _safe_ratio(
            bucket["matched_openrouter"], bucket["openrouter_total"]
        )
    return out


def _macro_average(values: Iterable[float | None]) -> float | None:
    seq = [v for v in values if v is not None]
    if not seq:
        return None
    return round(sum(seq) / len(seq), 4)


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def _row_payload(row: _PerPRRow) -> dict[str, Any]:
    return {
        "pr_id": row.runs_id,
        "fixture_pr_id": row.fixture_id,
        "status": row.status,
        "head_sha_baseline": row.head_sha_baseline,
        "head_sha_candidate": row.head_sha_candidate,
        "codex_findings": row.codex_total,
        "openrouter_findings": row.openrouter_total,
        "matched_pairs": row.matched_pairs,
        "codex_only": row.codex_only,
        "openrouter_only": row.openrouter_only,
        "severity_match": row.severity_match,
        "precision": row.precision,
        "recall": row.recall,
        "f1": row.f1,
    }


def build_aggregate_record(
    *,
    targets: Sequence[str],
    comparisons: Sequence[PRComparison],
    generated_at: str,
    tolerance: int,
    target_source: str,
) -> dict[str, Any]:
    """Assemble the JSON object persisted at ``eval/overlap_aggregate.json``.

    ``targets`` is the planned PR list (registry or discovered);
    ``comparisons`` is the parallel ``PRComparison`` list. The two
    must align positionally so a reviewer can see at-a-glance which
    sample slot each row belongs to.
    """
    if len(targets) != len(comparisons):
        raise ValueError(
            f"targets ({len(targets)}) must align with comparisons "
            f"({len(comparisons)})"
        )

    rows = [_build_row(runs_id, cmp) for runs_id, cmp in zip(targets, comparisons)]

    status_counts: dict[str, int] = {}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    sev_match_total = 0
    codex_total = 0
    openrouter_total = 0
    ready_count = 0
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
        codex_total += row.codex_total
        openrouter_total += row.openrouter_total
        sev_match_total += row.severity_match
        if row.status == "ready":
            ready_count += 1
            micro_tp += row.matched_pairs
            micro_fp += row.openrouter_only
            micro_fn += row.codex_only

    # Micro = pool everything, then divide.
    if ready_count > 0:
        micro_precision = _safe_ratio(micro_tp, micro_tp + micro_fp)
        micro_recall = _safe_ratio(micro_tp, micro_tp + micro_fn)
        micro_f1 = _f1_from(micro_precision, micro_recall)
    else:
        micro_precision = micro_recall = micro_f1 = None

    # Macro = average per-PR rates, restricted to ready PRs whose
    # denominator was non-null (otherwise the metric is undefined for
    # that PR and contributing 0 would distort the average).
    ready_rows = [row for row in rows if row.status == "ready"]
    macro_precision = _macro_average(row.precision for row in ready_rows)
    macro_recall = _macro_average(row.recall for row in ready_rows)
    macro_f1 = _macro_average(row.f1 for row in ready_rows)

    # Severity-match rate: share of matched pairs whose Codex and
    # OpenRouter severity agree, across ready PRs. ``null`` when no
    # match was made.
    sev_match_rate = _safe_ratio(sev_match_total, micro_tp) if ready_count else None

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
        "tolerance_lines": tolerance,
        "sample_size": len(rows),
        "ready_count": ready_count,
        "target_source": target_source,
        "status_counts": status_counts,
        "totals": {
            "codex_findings": codex_total,
            "openrouter_findings": openrouter_total,
            "matched_pairs": micro_tp,
            "codex_only": micro_fn,
            "openrouter_only": micro_fp,
            "severity_match": sev_match_total,
        },
        "metrics": {
            "micro": {
                "true_positive": micro_tp,
                "false_positive": micro_fp,
                "false_negative": micro_fn,
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "averaged_over_prs": len(ready_rows),
            },
            "severity_match_rate": sev_match_rate,
        },
        "per_category": _per_category(comparisons),
        "prs": [_row_payload(row) for row in rows],
    }


# ---------------------------------------------------------------------------
# Persistence + drift
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


def _comparisons_for(
    targets: Sequence[str],
    *,
    baseline_root: Path | None,
    candidate_root: Path | None,
    tolerance: int,
) -> list[PRComparison]:
    """Translate runs ids → fixture ids, then load the comparison."""
    out: list[PRComparison] = []
    for runs_id in targets:
        fixture_id = runs_id_to_fixture_id(runs_id)
        out.append(
            compare_pr(
                fixture_id,
                baseline_root=baseline_root,
                candidate_root=candidate_root,
                tolerance=tolerance,
            )
        )
    return out


def write_aggregate(
    *,
    registry_path: Path | None = None,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    aggregate_path: Path | None = None,
    generated_at: str | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> Path:
    """Compute and persist ``eval/overlap_aggregate.json``."""
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    targets, source = _resolve_targets(
        registry_path=registry_path,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )
    comparisons = _comparisons_for(
        targets,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        tolerance=tolerance,
    )
    record = build_aggregate_record(
        targets=targets,
        comparisons=comparisons,
        generated_at=generated_at,
        tolerance=tolerance,
        target_source=source,
    )
    target_path = aggregate_path if aggregate_path is not None else AGGREGATE_PATH
    _atomic_write(target_path, _serialize(record))
    return target_path


def check_aggregate(
    *,
    registry_path: Path | None = None,
    baseline_root: Path | None = None,
    candidate_root: Path | None = None,
    aggregate_path: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[str]:
    """Return drift messages — empty list when on-disk == rebuild."""
    drift: list[str] = []
    targets, source = _resolve_targets(
        registry_path=registry_path,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
    )
    comparisons = _comparisons_for(
        targets,
        baseline_root=baseline_root,
        candidate_root=candidate_root,
        tolerance=tolerance,
    )
    fresh = build_aggregate_record(
        targets=targets,
        comparisons=comparisons,
        generated_at="<ignored-for-check>",
        tolerance=tolerance,
        target_source=source,
    )
    target_path = aggregate_path if aggregate_path is not None else AGGREGATE_PATH
    if not target_path.is_file():
        drift.append(f"missing aggregate at {target_path}")
        return drift
    try:
        with target_path.open(encoding="utf-8") as f:
            on_disk = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        drift.append(f"cannot read aggregate ({exc})")
        return drift
    if not isinstance(on_disk, Mapping):
        drift.append("aggregate is not a JSON object")
        return drift
    if _strip_volatile(on_disk) != _strip_volatile(fresh):
        drift.append("on-disk overlap_aggregate.json differs from rebuild")
    return drift


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.overlap_aggregate",
        description=(
            "Roll up per-PR overlap metrics into a single sample-wide "
            "record at eval/overlap_aggregate.json (Sub-AC 3.4)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild and exit non-zero if the on-disk "
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

    if args.check:
        drift = check_aggregate(tolerance=args.tolerance)
        if drift:
            sys.stderr.write(
                "overlap aggregate drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m eval.overlap_aggregate`\n"
            )
            return 1
        print("overlap aggregate in sync")
        return 0

    path = write_aggregate(tolerance=args.tolerance)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

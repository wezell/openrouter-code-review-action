"""Threshold gate for the sample-wide overlap rollup (Sub-AC 4.1).

The overlap pipeline (``eval.overlap_score`` → ``eval.overlap_metrics``
→ ``eval.overlap_aggregate``) computes, for every sample PR, how well
OpenRouter's findings reproduce the Codex baseline. Sub-AC 4.1 requires
CI to **fail** when that sample-wide reproduction drops below the
threshold encoded in ``seed.yaml``:

    "Review findings overlap ≥80% with Codex baseline on sample PRs."

"Overlap with Codex baseline" maps to **recall** in IR terms: of the
findings Codex flagged, what share did OpenRouter reproduce? We default
the gate to ``micro.recall`` (pooled TP / (TP + FN) across every
``ready`` PR), with knobs to switch metric or override the threshold
when a future Sub-AC introduces a separate guarantee.

Behavior
--------

* Default — ``micro.recall ≥ 0.80`` over the rollup at
  ``eval/overlap_aggregate.json``. Exit 0 when satisfied; exit 1 (with
  a one-line stderr explanation) when violated.
* When the aggregate has **no** ``ready`` PR the metric is ``null``.
  We do **not** silently pass that case — CI is meant to fail when the
  gate is unmeasurable, because the bake-off requires populated
  baselines. ``--allow-unmeasurable`` is the explicit override (used
  while fixtures are still placeholders) and emits a warning.
* The gate never recomputes the aggregate. ``eval.overlap_score``
  (``--check`` in CI) is the up-to-date guarantee; this module only
  reads the persisted artifact so failures point cleanly at "regen the
  aggregate" vs. "OpenRouter regressed".

CI wiring
---------

``.github/workflows/ci.yml`` chains:

    python -m eval.overlap_score          # rebuild aggregate
    python -m eval.overlap_score --check  # drift guard
    python -m eval.check_overlap_threshold

so the gate runs after the rollup is known to be in sync with the
fixtures, and a regression on the OpenRouter side is the only thing
that can flip the gate red.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATE_PATH = REPO_ROOT / "eval" / "overlap_aggregate.json"

# seed.yaml AC 1: overlap ≥ 80% with Codex baseline.
DEFAULT_THRESHOLD = 0.80

# Default metric — micro recall. "Overlap with Codex baseline" in seed
# language means "share of Codex findings reproduced", which is recall.
DEFAULT_METRIC = "micro.recall"

_VALID_METRICS = (
    "micro.recall",
    "micro.precision",
    "micro.f1",
    "macro.recall",
    "macro.precision",
    "macro.f1",
)


class ThresholdResult:
    """Outcome of one gate evaluation. Public for tests."""

    __slots__ = ("metric", "value", "threshold", "ready_count", "passed", "reason")

    def __init__(
        self,
        *,
        metric: str,
        value: float | None,
        threshold: float,
        ready_count: int,
        passed: bool,
        reason: str,
    ) -> None:
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.ready_count = ready_count
        self.passed = passed
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"ThresholdResult(metric={self.metric!r}, value={self.value!r}, "
            f"threshold={self.threshold!r}, ready_count={self.ready_count!r}, "
            f"passed={self.passed!r}, reason={self.reason!r})"
        )


def _resolve_metric(record: Mapping[str, Any], metric: str) -> float | None:
    """Pull the requested metric out of the rollup, or ``None`` if absent."""
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    flavor, _, axis = metric.partition(".")
    bucket = metrics.get(flavor)
    if not isinstance(bucket, Mapping):
        return None
    raw = bucket.get(axis)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def evaluate(
    record: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    metric: str = DEFAULT_METRIC,
    allow_unmeasurable: bool = False,
) -> ThresholdResult:
    """Run the gate against a loaded aggregate record.

    Decision rules:

    * ``ready_count == 0`` → unmeasurable. Pass only when the caller
      explicitly opts in via ``allow_unmeasurable``.
    * Metric resolves to ``None`` (unmeasurable for this metric) →
      same as above.
    * Otherwise pass iff ``value >= threshold``.
    """
    ready_count_raw = record.get("ready_count", 0)
    ready_count = int(ready_count_raw) if isinstance(ready_count_raw, int) else 0
    value = _resolve_metric(record, metric)

    if value is None:
        if allow_unmeasurable:
            return ThresholdResult(
                metric=metric,
                value=None,
                threshold=threshold,
                ready_count=ready_count,
                passed=True,
                reason=(
                    f"{metric} unmeasurable (ready_count={ready_count}); allow-unmeasurable set"
                ),
            )
        return ThresholdResult(
            metric=metric,
            value=None,
            threshold=threshold,
            ready_count=ready_count,
            passed=False,
            reason=(
                f"{metric} unmeasurable (ready_count={ready_count}); "
                f"populate sample fixtures or pass --allow-unmeasurable"
            ),
        )

    passed = value + 1e-9 >= threshold  # epsilon: avoid float-equality flake
    reason = (
        f"{metric}={value:.4f} {'>=' if passed else '<'} {threshold:.4f} "
        f"(ready_count={ready_count})"
    )
    return ThresholdResult(
        metric=metric,
        value=value,
        threshold=threshold,
        ready_count=ready_count,
        passed=passed,
        reason=reason,
    )


def load_aggregate(path: Path) -> Mapping[str, Any]:
    """Read and parse the aggregate JSON record."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.check_overlap_threshold",
        description=(
            "Fail when the sample-wide overlap aggregate falls below the "
            "Sub-AC 4.1 threshold (default: micro.recall >= 0.80)."
        ),
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=AGGREGATE_PATH,
        help=(
            f"Path to overlap_aggregate.json (default: {AGGREGATE_PATH.relative_to(REPO_ROOT)})."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum value for the gated metric (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--metric",
        choices=_VALID_METRICS,
        default=DEFAULT_METRIC,
        help=(
            "Which aggregate metric to gate on. 'overlap with Codex' maps "
            "to micro.recall (default)."
        ),
    )
    parser.add_argument(
        "--allow-unmeasurable",
        action="store_true",
        help=(
            "Pass the gate when no PR is 'ready' (fixtures still placeholder). "
            "Use during bring-up; CI should leave this off once the bake-off "
            "sample is populated."
        ),
    )
    args = parser.parse_args(argv)

    if not (0.0 <= args.threshold <= 1.0):
        parser.error(f"--threshold must be in [0.0, 1.0] (got {args.threshold})")

    if not args.aggregate.is_file():
        sys.stderr.write(
            f"missing aggregate at {args.aggregate}; run `python -m eval.overlap_score` first\n"
        )
        return 2

    try:
        record = load_aggregate(args.aggregate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"cannot read aggregate {args.aggregate}: {exc}\n")
        return 2

    result = evaluate(
        record,
        threshold=args.threshold,
        metric=args.metric,
        allow_unmeasurable=args.allow_unmeasurable,
    )

    if result.passed:
        if result.value is None:
            # Soft pass via --allow-unmeasurable. Surface clearly.
            sys.stderr.write(f"WARNING: {result.reason}\n")
        print(f"overlap gate PASS: {result.reason}")
        return 0

    sys.stderr.write(f"overlap gate FAIL: {result.reason}\n")
    return 1


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

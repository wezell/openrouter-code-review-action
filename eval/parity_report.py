"""Consolidated human-readable parity report (Sub-AC 4.2).

The bake-off pipeline emits three machine-readable artifact tiers:

1. ``eval/runs/<pr-id>/overlap.json`` — per-PR precision/recall/F1
   (Sub-AC 3.3).
2. ``eval/overlap_aggregate.json`` — sample-wide rollup with micro/macro
   metrics and per-severity buckets (Sub-AC 3.4).
3. ``tests/fixtures/parity_report/<pr-id>.json`` plus
   ``tests/fixtures/parity_report/_summary.json`` — Codex-vs-OpenRouter
   delta with status, severity match, and unmatched-finding lists
   (Sub-AC 2.4).

Sub-AC 4.2 needs a *single* human-readable artifact a reviewer can open
without pivoting through three trees. This module is that consolidator.
It reads from all three tiers, merges them per-PR, and writes:

* ``eval/parity_report.md`` — Markdown the CI job uploads via
  ``actions/upload-artifact``. The same file embeds a top-line gate
  status, the aggregate rollup, and a per-PR table.
* ``eval/parity_report.json`` — the same data in JSON so downstream
  tooling (badges, dashboards) can keep parsing instead of scraping.

The generator is **idempotent** — ``--check`` rebuilds in memory and
exits non-zero if the on-disk artifacts drift from the rebuild, so CI
can keep this report in sync with the underlying tiers.

This module never recomputes overlap or matches; it is purely a
consolidator. If a tier is missing the report degrades gracefully:
missing per-PR overlap rows surface as ``unmeasurable``, missing
aggregate falls back to a "rollup not generated" banner.

CLI
---

::

    python -m eval.parity_report             # write Markdown + JSON
    python -m eval.parity_report --check     # CI drift check
    python -m eval.parity_report --threshold 0.80 --metric micro.recall
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

from eval.check_overlap_threshold import (
    DEFAULT_METRIC,
    DEFAULT_THRESHOLD,
    ThresholdResult,
)
from eval.check_overlap_threshold import (
    evaluate as evaluate_threshold,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "eval" / "runs"
AGGREGATE_PATH = REPO_ROOT / "eval" / "overlap_aggregate.json"
PARITY_REPORT_ROOT = REPO_ROOT / "tests" / "fixtures" / "parity_report"
PARITY_SUMMARY_PATH = PARITY_REPORT_ROOT / "_summary.json"

REPORT_MD_PATH = REPO_ROOT / "eval" / "parity_report.md"
REPORT_JSON_PATH = REPO_ROOT / "eval" / "parity_report.json"

SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "eval/parity_report.py"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, Mapping):
        return None
    return doc


def _format_float(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_signed(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PRRow:
    """Per-PR row consolidating all three artifact tiers."""

    pr_id: str
    fixture_pr_id: str | None
    status: str
    head_sha_baseline: str | None
    head_sha_candidate: str | None
    codex_findings: int
    openrouter_findings: int
    matched_pairs: int
    codex_only: int
    openrouter_only: int
    delta_total: int
    severity_match: int
    precision: float | None
    recall: float | None
    f1: float | None
    overlap_ratio_codex: float | None
    overlap_ratio_openrouter: float | None
    codex_model: str | None
    openrouter_model: str | None


def _resolve_fixture_pr_id(pr_id: str, parity_pr: Mapping[str, Any] | None) -> str | None:
    """parity_report uses ``pr-NNNNN`` ids; runs use ``dotcms-core-NNNNN``."""
    if parity_pr is not None:
        pid = parity_pr.get("pr_id")
        if isinstance(pid, str):
            return pid
    # Fall back to deriving from the runs id ``<repo>-<num>`` → ``pr-<num>``.
    if "-" in pr_id:
        suffix = pr_id.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return f"pr-{suffix}"
    return None


def _by_fixture_id(rows: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    if not rows:
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pid = row.get("pr_id")
        if isinstance(pid, str):
            out[pid] = row
    return out


def _discover_pr_ids(runs_root: Path) -> list[str]:
    if not runs_root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "overlap.json").is_file():
            continue
        out.append(child.name)
    return out


def _build_rows(
    *,
    runs_root: Path,
    aggregate: Mapping[str, Any] | None,
    parity_summary: Mapping[str, Any] | None,
) -> list[PRRow]:
    pr_ids = _discover_pr_ids(runs_root)
    aggregate_prs = _by_fixture_id(aggregate.get("prs") if isinstance(aggregate, Mapping) else None)
    parity_prs = _by_fixture_id(
        parity_summary.get("prs") if isinstance(parity_summary, Mapping) else None
    )

    rows: list[PRRow] = []
    for pr_id in pr_ids:
        overlap_path = runs_root / pr_id / "overlap.json"
        overlap_doc = _read_json(overlap_path) or {}

        # Cross-reference into parity report by fixture pr_id.
        fixture_pr_id = overlap_doc.get("fixture_pr_id")
        if not isinstance(fixture_pr_id, str):
            fixture_pr_id = _resolve_fixture_pr_id(pr_id, None)
        parity_row = parity_prs.get(fixture_pr_id) if fixture_pr_id else None
        aggregate_row = aggregate_prs.get(fixture_pr_id) if fixture_pr_id else None

        counts = overlap_doc.get("counts") if isinstance(overlap_doc, Mapping) else None
        counts = counts if isinstance(counts, Mapping) else {}
        metrics = overlap_doc.get("metrics") if isinstance(overlap_doc, Mapping) else None
        metrics = metrics if isinstance(metrics, Mapping) else {}
        models = overlap_doc.get("models") if isinstance(overlap_doc, Mapping) else None
        models = models if isinstance(models, Mapping) else {}

        codex_total = int(counts.get("codex_total") or 0)
        or_total = int(counts.get("openrouter_total") or 0)
        matched = int(counts.get("matched_pairs") or 0)
        codex_only = int(counts.get("codex_only") or 0)
        or_only = int(counts.get("openrouter_only") or 0)

        precision = _opt_float(metrics.get("precision"))
        recall = _opt_float(metrics.get("recall"))
        f1 = _opt_float(metrics.get("f1"))

        # Parity report carries the human-friendly overlap ratios + severity match.
        ratio_codex: float | None = None
        ratio_or: float | None = None
        severity_match = 0
        if parity_row is not None:
            severity_match = (
                int(parity_row.get("severity_match", 0) or 0)
                if isinstance(parity_row.get("severity_match"), (int, float))
                else 0
            )
        # Per-PR parity JSON (richer than _summary.json) carries the ratios.
        parity_pr_doc = (
            _read_json(PARITY_REPORT_ROOT / f"{fixture_pr_id}.json") if fixture_pr_id else None
        )
        if isinstance(parity_pr_doc, Mapping):
            overlap = parity_pr_doc.get("overlap")
            if isinstance(overlap, Mapping):
                ratio_codex = _opt_float(overlap.get("overlap_ratio_codex_baseline"))
                ratio_or = _opt_float(overlap.get("overlap_ratio_openrouter_candidate"))
                sm = overlap.get("severity_match_count")
                if isinstance(sm, (int, float)):
                    severity_match = int(sm)

        status = _resolve_status(overlap_doc, parity_row, aggregate_row)

        rows.append(
            PRRow(
                pr_id=pr_id,
                fixture_pr_id=fixture_pr_id,
                status=status,
                head_sha_baseline=_opt_str(overlap_doc.get("head_sha_baseline")),
                head_sha_candidate=_opt_str(overlap_doc.get("head_sha_candidate")),
                codex_findings=codex_total,
                openrouter_findings=or_total,
                matched_pairs=matched,
                codex_only=codex_only,
                openrouter_only=or_only,
                delta_total=or_total - codex_total,
                severity_match=severity_match,
                precision=precision,
                recall=recall,
                f1=f1,
                overlap_ratio_codex=ratio_codex,
                overlap_ratio_openrouter=ratio_or,
                codex_model=_opt_str(models.get("codex")),
                openrouter_model=_opt_str(models.get("openrouter")),
            )
        )
    return rows


def _opt_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _resolve_status(
    overlap_doc: Mapping[str, Any] | None,
    parity_row: Mapping[str, Any] | None,
    aggregate_row: Mapping[str, Any] | None,
) -> str:
    for source in (overlap_doc, parity_row, aggregate_row):
        if isinstance(source, Mapping):
            status = source.get("status")
            if isinstance(status, str) and status:
                return status
    return "unknown"


# ---------------------------------------------------------------------------
# Aggregate / threshold extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateView:
    sample_size: int
    ready_count: int
    status_counts: dict[str, int]
    micro_precision: float | None
    micro_recall: float | None
    micro_f1: float | None
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    severity_match_rate: float | None
    per_category: dict[str, dict[str, Any]]


def _aggregate_view(aggregate: Mapping[str, Any] | None) -> AggregateView:
    if not isinstance(aggregate, Mapping):
        return AggregateView(
            sample_size=0,
            ready_count=0,
            status_counts={},
            micro_precision=None,
            micro_recall=None,
            micro_f1=None,
            macro_precision=None,
            macro_recall=None,
            macro_f1=None,
            severity_match_rate=None,
            per_category={},
        )
    metrics_raw = aggregate.get("metrics")
    metrics: Mapping[str, Any] = metrics_raw if isinstance(metrics_raw, Mapping) else {}
    micro_raw = metrics.get("micro")
    micro: Mapping[str, Any] = micro_raw if isinstance(micro_raw, Mapping) else {}
    macro_raw = metrics.get("macro")
    macro: Mapping[str, Any] = macro_raw if isinstance(macro_raw, Mapping) else {}
    sc_raw = aggregate.get("status_counts")
    status_counts: dict[str, int] = {}
    if isinstance(sc_raw, Mapping):
        for key, value in sc_raw.items():
            if isinstance(key, str) and isinstance(value, int):
                status_counts[key] = value
    pc_raw = aggregate.get("per_category")
    per_category: dict[str, dict[str, Any]] = {}
    if isinstance(pc_raw, Mapping):
        for key, value in pc_raw.items():
            if isinstance(key, str) and isinstance(value, Mapping):
                per_category[key] = dict(value)
    return AggregateView(
        sample_size=int(aggregate.get("sample_size", 0) or 0)
        if isinstance(aggregate.get("sample_size"), int)
        else 0,
        ready_count=int(aggregate.get("ready_count", 0) or 0)
        if isinstance(aggregate.get("ready_count"), int)
        else 0,
        status_counts=status_counts,
        micro_precision=_opt_float(micro.get("precision")),
        micro_recall=_opt_float(micro.get("recall")),
        micro_f1=_opt_float(micro.get("f1")),
        macro_precision=_opt_float(macro.get("precision")),
        macro_recall=_opt_float(macro.get("recall")),
        macro_f1=_opt_float(macro.get("f1")),
        severity_match_rate=_opt_float(metrics.get("severity_match_rate")),
        per_category=per_category,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(
    *,
    rows: Sequence[PRRow],
    aggregate: AggregateView,
    threshold_result: ThresholdResult,
    generated_at: str,
) -> str:
    out: list[str] = []
    badge = "PASS" if threshold_result.passed else "FAIL"
    out.append("# Codex-vs-OpenRouter Parity Report")
    out.append("")
    out.append(f"_Generated at: `{generated_at}`_")
    out.append("")
    out.append(f"**Threshold gate:** {badge} — {threshold_result.reason}")
    out.append("")

    out.append("## Sample Rollup")
    out.append("")
    out.append(f"- Sample size: {aggregate.sample_size}")
    out.append(f"- Ready PRs (both sides captured at same head_sha): {aggregate.ready_count}")
    if aggregate.status_counts:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(aggregate.status_counts.items()))
        out.append(f"- Status breakdown: {breakdown}")
    out.append("")
    out.append("| Metric | Micro (pooled) | Macro (per-PR mean) |")
    out.append("| --- | --- | --- |")
    out.append(
        f"| Precision | {_format_float(aggregate.micro_precision)} "
        f"| {_format_float(aggregate.macro_precision)} |"
    )
    out.append(
        f"| Recall    | {_format_float(aggregate.micro_recall)} "
        f"| {_format_float(aggregate.macro_recall)} |"
    )
    out.append(
        f"| F1        | {_format_float(aggregate.micro_f1)} | {_format_float(aggregate.macro_f1)} |"
    )
    out.append("")
    if aggregate.severity_match_rate is not None:
        out.append(
            f"Severity match rate (matched pairs only): "
            f"{_format_pct(aggregate.severity_match_rate)}"
        )
        out.append("")

    if aggregate.per_category:
        out.append("## Per-severity Breakdown")
        out.append("")
        out.append(
            "| Severity | Codex total | OpenRouter total | Matched (codex) "
            "| Matched (or) | Precision | Recall |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for key in ("P0", "P1", "P2", "P3", "unknown"):
            bucket = aggregate.per_category.get(key)
            if not isinstance(bucket, Mapping):
                continue
            out.append(
                "| {sev} | {ct} | {ot} | {mc} | {mo} | {p} | {r} |".format(
                    sev=key,
                    ct=bucket.get("codex_total", 0),
                    ot=bucket.get("openrouter_total", 0),
                    mc=bucket.get("matched_codex", 0),
                    mo=bucket.get("matched_openrouter", 0),
                    p=_format_float(_opt_float(bucket.get("precision"))),
                    r=_format_float(_opt_float(bucket.get("recall"))),
                )
            )
        out.append("")

    out.append("## Per-PR Details")
    out.append("")
    if not rows:
        out.append("_No PRs discovered under `eval/runs/`._")
        out.append("")
    else:
        out.append(
            "| PR | Status | Codex | OpenRouter | Matched | Codex-only | OR-only "
            "| Δtotal | Recall | Precision | F1 | Sev-match |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            out.append(
                f"| `{row.pr_id}` | {row.status} | {row.codex_findings} | {row.openrouter_findings} | {row.matched_pairs} | {row.codex_only} | {row.openrouter_only} | {_format_signed(row.delta_total)} | "
                f"{_format_float(row.recall)} | {_format_float(row.precision)} | {_format_float(row.f1)} | {row.severity_match} |"
            )
        out.append("")

    out.append("## Source Artifacts")
    out.append("")
    out.append("- Per-PR overlap: `eval/runs/<pr-id>/overlap.json`")
    out.append("- Aggregate rollup: `eval/overlap_aggregate.json`")
    out.append("- Codex-vs-OpenRouter delta: `tests/fixtures/parity_report/`")
    out.append("")
    out.append(f"_Generator: `{GENERATOR_REL_PATH}` (schema v{SCHEMA_VERSION})._")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# JSON sibling (machine-readable consolidation)
# ---------------------------------------------------------------------------


def _row_payload(row: PRRow) -> dict[str, Any]:
    return {
        "pr_id": row.pr_id,
        "fixture_pr_id": row.fixture_pr_id,
        "status": row.status,
        "head_sha_baseline": row.head_sha_baseline,
        "head_sha_candidate": row.head_sha_candidate,
        "codex_findings": row.codex_findings,
        "openrouter_findings": row.openrouter_findings,
        "matched_pairs": row.matched_pairs,
        "codex_only": row.codex_only,
        "openrouter_only": row.openrouter_only,
        "delta_total": row.delta_total,
        "severity_match": row.severity_match,
        "precision": row.precision,
        "recall": row.recall,
        "f1": row.f1,
        "overlap_ratio_codex_baseline": row.overlap_ratio_codex,
        "overlap_ratio_openrouter_candidate": row.overlap_ratio_openrouter,
        "codex_model": row.codex_model,
        "openrouter_model": row.openrouter_model,
    }


def _aggregate_payload(view: AggregateView) -> dict[str, Any]:
    return {
        "sample_size": view.sample_size,
        "ready_count": view.ready_count,
        "status_counts": dict(view.status_counts),
        "metrics": {
            "micro": {
                "precision": view.micro_precision,
                "recall": view.micro_recall,
                "f1": view.micro_f1,
            },
            "macro": {
                "precision": view.macro_precision,
                "recall": view.macro_recall,
                "f1": view.macro_f1,
            },
            "severity_match_rate": view.severity_match_rate,
        },
        "per_category": dict(view.per_category),
    }


def _threshold_payload(result: ThresholdResult) -> dict[str, Any]:
    return {
        "metric": result.metric,
        "threshold": result.threshold,
        "value": result.value,
        "passed": result.passed,
        "ready_count": result.ready_count,
        "reason": result.reason,
    }


def build_consolidated_report(
    *,
    runs_root: Path = RUNS_ROOT,
    aggregate_path: Path = AGGREGATE_PATH,
    parity_summary_path: Path = PARITY_SUMMARY_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    metric: str = DEFAULT_METRIC,
    allow_unmeasurable: bool = False,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return ``(json_payload, markdown)`` without writing to disk."""
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    aggregate = _read_json(aggregate_path)
    parity_summary = _read_json(parity_summary_path)

    rows = _build_rows(
        runs_root=runs_root,
        aggregate=aggregate,
        parity_summary=parity_summary,
    )
    view = _aggregate_view(aggregate)
    threshold_result = evaluate_threshold(
        aggregate or {},
        threshold=threshold,
        metric=metric,
        allow_unmeasurable=allow_unmeasurable,
    )

    json_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR_REL_PATH,
        "generated_at": generated_at,
        "threshold_gate": _threshold_payload(threshold_result),
        "aggregate": _aggregate_payload(view),
        "prs": [_row_payload(r) for r in rows],
        "sources": {
            "runs_root": str(runs_root.relative_to(REPO_ROOT))
            if runs_root.is_relative_to(REPO_ROOT)
            else str(runs_root),
            "aggregate": str(aggregate_path.relative_to(REPO_ROOT))
            if aggregate_path.is_relative_to(REPO_ROOT)
            else str(aggregate_path),
            "parity_summary": str(parity_summary_path.relative_to(REPO_ROOT))
            if parity_summary_path.is_relative_to(REPO_ROOT)
            else str(parity_summary_path),
        },
    }
    markdown = _render_markdown(
        rows=rows,
        aggregate=view,
        threshold_result=threshold_result,
        generated_at=generated_at,
    )
    return json_payload, markdown


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


def _serialize_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _strip_volatile(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    if isinstance(out, dict):
        out.pop("generated_at", None)
    return out  # type: ignore[no-any-return]


def _strip_md_volatile(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("_Generated at:"))


def write_report(
    *,
    runs_root: Path = RUNS_ROOT,
    aggregate_path: Path = AGGREGATE_PATH,
    parity_summary_path: Path = PARITY_SUMMARY_PATH,
    md_path: Path = REPORT_MD_PATH,
    json_path: Path = REPORT_JSON_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    metric: str = DEFAULT_METRIC,
    allow_unmeasurable: bool = False,
    generated_at: str | None = None,
) -> tuple[Path, Path]:
    payload, markdown = build_consolidated_report(
        runs_root=runs_root,
        aggregate_path=aggregate_path,
        parity_summary_path=parity_summary_path,
        threshold=threshold,
        metric=metric,
        allow_unmeasurable=allow_unmeasurable,
        generated_at=generated_at,
    )
    _atomic_write(json_path, _serialize_json(payload))
    _atomic_write(md_path, markdown if markdown.endswith("\n") else markdown + "\n")
    return md_path, json_path


def check_report(
    *,
    runs_root: Path = RUNS_ROOT,
    aggregate_path: Path = AGGREGATE_PATH,
    parity_summary_path: Path = PARITY_SUMMARY_PATH,
    md_path: Path = REPORT_MD_PATH,
    json_path: Path = REPORT_JSON_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    metric: str = DEFAULT_METRIC,
    allow_unmeasurable: bool = False,
) -> list[str]:
    fresh_payload, fresh_md = build_consolidated_report(
        runs_root=runs_root,
        aggregate_path=aggregate_path,
        parity_summary_path=parity_summary_path,
        threshold=threshold,
        metric=metric,
        allow_unmeasurable=allow_unmeasurable,
        generated_at="<ignored-for-check>",
    )
    drift: list[str] = []
    on_disk_json = _read_json(json_path)
    if on_disk_json is None:
        drift.append(f"missing JSON report at {json_path}")
    elif _strip_volatile(on_disk_json) != _strip_volatile(fresh_payload):
        drift.append(f"{json_path}: JSON differs from rebuild")
    if not md_path.is_file():
        drift.append(f"missing Markdown report at {md_path}")
    else:
        on_disk_md = md_path.read_text(encoding="utf-8")
        if _strip_md_volatile(on_disk_md) != _strip_md_volatile(fresh_md):
            drift.append(f"{md_path}: Markdown differs from rebuild")
    return drift


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.parity_report",
        description=(
            "Consolidate per-PR overlap metrics, the aggregate rollup, and "
            "the Codex-vs-OpenRouter delta into a single human-readable "
            "Markdown artifact (plus a JSON sibling). Sub-AC 4.2."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild in memory and exit non-zero if the "
            "on-disk artifacts differ from the rebuild. Use in CI."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold for the gate banner (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help=f"Metric flavor for the gate banner (default: {DEFAULT_METRIC}).",
    )
    parser.add_argument(
        "--allow-unmeasurable",
        action="store_true",
        help="Treat an unmeasurable gate as PASS in the banner.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPORT_MD_PATH,
        help=f"Markdown output path (default: {REPORT_MD_PATH}).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPORT_JSON_PATH,
        help=f"JSON output path (default: {REPORT_JSON_PATH}).",
    )
    args = parser.parse_args(argv)

    if args.check:
        drift = check_report(
            md_path=args.md_out,
            json_path=args.json_out,
            threshold=args.threshold,
            metric=args.metric,
            allow_unmeasurable=args.allow_unmeasurable,
        )
        if drift:
            sys.stderr.write(
                "consolidated parity report drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m eval.parity_report`\n"
            )
            return 1
        print("consolidated parity report in sync")
        return 0

    md_path, json_path = write_report(
        md_path=args.md_out,
        json_path=args.json_out,
        threshold=args.threshold,
        metric=args.metric,
        allow_unmeasurable=args.allow_unmeasurable,
    )
    print(f"wrote {md_path} and {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

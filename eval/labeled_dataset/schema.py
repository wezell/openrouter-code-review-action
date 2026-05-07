"""Typed loader, validator, and schema enforcement for the labeled dataset.

This module is the gate between ``eval/labeled_dataset/dataset.yaml`` and
every consumer (tests, scoring scripts, future labelers). It:

* Loads the YAML into a typed :class:`LabeledDataset` so callers don't
  pass around dicts.
* Validates the schema — required fields, allowed values for ``label``,
  ``severity``, ``class``, ``line_range_status`` — and raises
  :class:`DatasetValidationError` with a precise message on the first
  failure.
* Cross-references each entry's ``pr_id`` and ``head_sha`` against
  ``eval/sample-prs.yaml`` so registry drift breaks loudly.

The allowed-value sets are pinned here, not pulled from the YAML
``*_taxonomy`` blocks. The taxonomy blocks in YAML are documentation
for human labelers; this module is the authoritative contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "eval" / "labeled_dataset" / "dataset.yaml"
SAMPLE_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"


# ---------------------------------------------------------------------------
# Allowed values (the authoritative contract)
# ---------------------------------------------------------------------------

ALLOWED_LABELS: frozenset[str] = frozenset({"TP", "FP"})
"""Methodology §2.1. ``DUP`` and ``CARRY`` are run-artifact labels and
do not appear in the ground-truth benchmark dataset."""

ALLOWED_SEVERITIES: frozenset[str] = frozenset({"P0", "P1", "P2", "P3"})
"""From ``prompts/review.md``."""

ALLOWED_CLASSES: frozenset[str] = frozenset(
    {
        "bug_correctness",
        "security",
        "performance",
        "refactor_regression",
        "missing_test",
        "hallucination",
        "pre_existing",
        "style_nit",
        "speculation",
        "out_of_scope",
    }
)
"""Finding class taxonomy. Mirrors the ``class_taxonomy`` block in
``dataset.yaml``; pinned here so a stray new class can't slip in
without an explicit code change."""

ALLOWED_LINE_RANGE_STATUSES: frozenset[str] = frozenset({"confirmed", "unverified"})
"""``confirmed`` requires a non-null ``line_range``; ``unverified``
must have ``line_range: null``."""

ALLOWED_SOURCES: frozenset[str] = frozenset(
    {
        "retrospective_diff_review",
        "merged_pr_metadata",
        "reviewer_comment_thread",
        "bake_off_diagnostic",
    }
)
"""Where the entry's evidence came from. Used to weight confidence
when adjudicating disagreement."""

ALLOWED_CONFIDENCES: frozenset[str] = frozenset({"high", "medium", "low"})


# ---------------------------------------------------------------------------
# Errors and types
# ---------------------------------------------------------------------------


class DatasetValidationError(ValueError):
    """Raised when ``dataset.yaml`` fails schema or cross-reference checks."""


LineRangeStatus = Literal["confirmed", "unverified"]


@dataclass(frozen=True)
class LineRange:
    """Inclusive line range in the target file, ``[start, end]``."""

    start: int
    end: int

    def overlaps(self, other: LineRange, *, tolerance: int = 3) -> bool:
        """Methodology §4.5 rule 2: overlap by ≥1 line OR within ±tolerance."""
        if self.end < other.start - tolerance:
            return False
        if self.start > other.end + tolerance:
            return False
        return True


@dataclass(frozen=True)
class LabeledFinding:
    """A single ground-truth labeled finding."""

    id: str
    pr_id: str
    head_sha: str
    label: str
    severity: str
    finding_class: str  # `class` is reserved; field aliased on load
    title: str
    rationale: str
    match_keywords: tuple[str, ...]
    source: str
    confidence: str
    path_hint: str | None
    line_range: LineRange | None
    line_range_status: str

    def is_tp(self) -> bool:
        return self.label == "TP"

    def is_fp(self) -> bool:
        return self.label == "FP"


@dataclass(frozen=True)
class LabeledDataset:
    """The whole labeled dataset, loaded and validated."""

    version: str
    dataset_id: str
    generated_at: str
    source_sample: str
    methodology: str
    findings: tuple[LabeledFinding, ...]
    removed: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    def by_pr(self, pr_id: str) -> tuple[LabeledFinding, ...]:
        return tuple(f for f in self.findings if f.pr_id == pr_id)

    def true_positives(self) -> tuple[LabeledFinding, ...]:
        return tuple(f for f in self.findings if f.is_tp())

    def false_positives(self) -> tuple[LabeledFinding, ...]:
        return tuple(f for f in self.findings if f.is_fp())

    def pr_ids(self) -> tuple[str, ...]:
        # Preserve declaration order (sample-prs.yaml ordering).
        seen: list[str] = []
        for f in self.findings:
            if f.pr_id not in seen:
                seen.append(f.pr_id)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(path: Path | None = None, *, sample_path: Path | None = None) -> LabeledDataset:
    """Load and validate the labeled dataset.

    Both paths default to the canonical in-repo locations; tests can
    override either to exercise validation against fixtures.
    """
    dataset_doc = _load_yaml_mapping(path or DATASET_PATH, kind="dataset")
    sample_doc = _load_yaml_mapping(sample_path or SAMPLE_PATH, kind="sample registry")
    sample_index = _index_sample(sample_doc, source=sample_path or SAMPLE_PATH)
    return validate_dataset(dataset_doc, sample_index=sample_index)


def validate_dataset(
    doc: Mapping[str, Any], *, sample_index: Mapping[str, str]
) -> LabeledDataset:
    """Validate a parsed dataset mapping against the schema and the sample registry.

    ``sample_index`` maps ``pr_id -> head_sha`` for cross-reference.
    """
    version = _require_str(doc, "version")
    dataset_id = _require_str(doc, "dataset_id")
    generated_at = _require_str(doc, "generated_at")
    source_sample = _require_str(doc, "source_sample")
    methodology = _require_str(doc, "methodology")

    findings_raw = doc.get("findings")
    if not isinstance(findings_raw, list):
        raise DatasetValidationError("dataset 'findings' must be a list")
    if not findings_raw:
        raise DatasetValidationError(
            "dataset 'findings' must not be empty — a labeled dataset with no entries "
            "cannot benchmark anything"
        )

    seen_ids: set[str] = set()
    findings: list[LabeledFinding] = []
    for index, raw in enumerate(findings_raw):
        if not isinstance(raw, Mapping):
            raise DatasetValidationError(f"findings[{index}] must be a mapping")
        finding = _validate_finding(raw, index=index, sample_index=sample_index)
        if finding.id in seen_ids:
            raise DatasetValidationError(
                f"findings[{index}]: duplicate id {finding.id!r}; ids must be unique"
            )
        seen_ids.add(finding.id)
        findings.append(finding)

    removed_raw = doc.get("removed") or []
    if not isinstance(removed_raw, list):
        raise DatasetValidationError("dataset 'removed' must be a list when present")
    removed = tuple(item for item in removed_raw if isinstance(item, Mapping))

    return LabeledDataset(
        version=version,
        dataset_id=dataset_id,
        generated_at=generated_at,
        source_sample=source_sample,
        methodology=methodology,
        findings=tuple(findings),
        removed=removed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml_mapping(path: Path, *, kind: str) -> Mapping[str, Any]:
    if not path.exists():
        raise DatasetValidationError(f"{kind} not found at {path}")
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise DatasetValidationError(f"{kind} at {path} must be a YAML mapping at the top level")
    return doc


def _index_sample(doc: Mapping[str, Any], *, source: Path) -> Mapping[str, str]:
    """Build a ``pr_id -> head_sha`` map from the sample registry."""
    raw = doc.get("prs") or []
    if not isinstance(raw, list):
        raise DatasetValidationError(f"{source}: 'prs' must be a list")
    out: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        pr_id = entry.get("id")
        head_sha = entry.get("head_sha")
        if isinstance(pr_id, str) and isinstance(head_sha, str):
            out[pr_id] = head_sha
    return out


def _validate_finding(
    raw: Mapping[str, Any], *, index: int, sample_index: Mapping[str, str]
) -> LabeledFinding:
    where = f"findings[{index}]"

    finding_id = _require_str(raw, "id", where=where)
    if not finding_id.startswith("gt-"):
        raise DatasetValidationError(
            f"{where}: id {finding_id!r} must start with 'gt-' so benchmark ids are "
            "distinguishable from per-run finding ids"
        )

    pr_id = _require_str(raw, "pr_id", where=where)
    head_sha = _require_str(raw, "head_sha", where=where)
    expected_head = sample_index.get(pr_id)
    if expected_head is None:
        raise DatasetValidationError(
            f"{where}: pr_id {pr_id!r} is not in eval/sample-prs.yaml — either add it "
            "to the registry or fix the typo"
        )
    if head_sha != expected_head:
        raise DatasetValidationError(
            f"{where}: head_sha {head_sha!r} does not match the registry entry for "
            f"{pr_id} ({expected_head!r}). The dataset must always score against "
            "the pinned head_sha (methodology §1.1)."
        )

    label = _require_one_of(raw, "label", ALLOWED_LABELS, where=where)
    severity = _require_one_of(raw, "severity", ALLOWED_SEVERITIES, where=where)
    finding_class = _require_one_of(raw, "class", ALLOWED_CLASSES, where=where)
    source = _require_one_of(raw, "source", ALLOWED_SOURCES, where=where)
    confidence = _require_one_of(raw, "confidence", ALLOWED_CONFIDENCES, where=where)
    line_range_status = _require_one_of(
        raw, "line_range_status", ALLOWED_LINE_RANGE_STATUSES, where=where
    )

    title = _require_str(raw, "title", where=where)
    rationale = _require_str(raw, "rationale", where=where)

    path_hint_raw = raw.get("path_hint")
    if path_hint_raw is not None and not isinstance(path_hint_raw, str):
        raise DatasetValidationError(f"{where}: path_hint must be a string or null")

    keywords_raw = raw.get("match_keywords")
    if not isinstance(keywords_raw, list) or not all(isinstance(k, str) for k in keywords_raw):
        raise DatasetValidationError(
            f"{where}: match_keywords must be a list of strings"
        )
    if not keywords_raw:
        # match_keywords is the universal anchor — even FP-trap entries
        # with no specific file target need distinctive strings the
        # matcher can use to associate model output with this entry.
        raise DatasetValidationError(
            f"{where}: match_keywords must contain at least one string so the "
            "§4.5 matcher has a keyword anchor"
        )

    line_range = _validate_line_range(raw.get("line_range"), where=where)
    if line_range_status == "confirmed" and line_range is None:
        raise DatasetValidationError(
            f"{where}: line_range_status is 'confirmed' but line_range is null. "
            "A confirmed entry must pin its line numbers."
        )
    if line_range_status == "unverified" and line_range is not None:
        raise DatasetValidationError(
            f"{where}: line_range_status is 'unverified' but line_range is set. "
            "Either confirm the entry or null out the line range."
        )

    # match_keywords is now a hard-required non-empty anchor (above), so
    # path_hint and line_range may both be null on FP-trap entries that
    # the matcher should associate by keyword + pr_id alone.

    return LabeledFinding(
        id=finding_id,
        pr_id=pr_id,
        head_sha=head_sha,
        label=label,
        severity=severity,
        finding_class=finding_class,
        title=title,
        rationale=rationale,
        match_keywords=tuple(keywords_raw),
        source=source,
        confidence=confidence,
        path_hint=path_hint_raw if isinstance(path_hint_raw, str) else None,
        line_range=line_range,
        line_range_status=line_range_status,
    )


def _validate_line_range(value: Any, *, where: str) -> LineRange | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetValidationError(f"{where}: line_range must be null or a mapping")
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise DatasetValidationError(
            f"{where}: line_range.start and line_range.end must be integers"
        )
    if start < 1 or end < 1:
        raise DatasetValidationError(
            f"{where}: line_range start/end must be ≥ 1 (file lines are 1-indexed)"
        )
    if end < start:
        raise DatasetValidationError(
            f"{where}: line_range.end ({end}) must be ≥ line_range.start ({start})"
        )
    return LineRange(start=start, end=end)


def _require_str(raw: Mapping[str, Any], key: str, *, where: str = "dataset") -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{where}: missing required string field {key!r}")
    return value


def _require_one_of(
    raw: Mapping[str, Any], key: str, allowed: Iterable[str], *, where: str
) -> str:
    value = raw.get(key)
    allowed_set = frozenset(allowed)
    if not isinstance(value, str) or value not in allowed_set:
        allowed_sorted = ", ".join(sorted(allowed_set))
        raise DatasetValidationError(
            f"{where}: {key!r} must be one of {{{allowed_sorted}}}; got {value!r}"
        )
    return value

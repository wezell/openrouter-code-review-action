"""Schema and integrity tests for the ground-truth labeled dataset.

Sub-AC 2.3.1. These tests pin the contract enforced by
``eval/labeled_dataset/schema.py`` and verify the checked-in
``eval/labeled_dataset/dataset.yaml`` actually loads, validates,
and cross-references the sample registry without drift.

The dataset is the long-lived benchmark for the parity bake-off
(seed AC 1 + AC 2). Schema regressions here would silently corrupt
every future model swap's overlap and FP-rate scores — the tests
here are the gate that prevents that.
"""

from __future__ import annotations

import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.labeled_dataset import (  # noqa: E402
    ALLOWED_CLASSES,
    ALLOWED_LABELS,
    ALLOWED_LINE_RANGE_STATUSES,
    ALLOWED_SEVERITIES,
    DATASET_PATH,
    LabeledDataset,
    LineRange,
    load_dataset,
    validate_dataset,
)
from eval.labeled_dataset.schema import (  # noqa: E402
    SAMPLE_PATH,
    DatasetValidationError,
    _index_sample,
    _load_yaml_mapping,
)

# ---------------------------------------------------------------------------
# Fixtures: a minimal in-memory dataset and matching sample registry
# ---------------------------------------------------------------------------

_GOOD_SAMPLE_INDEX = {"sample-pr-001": "deadbeefcafe1234"}


def _make_finding(**overrides):
    base = {
        "id": "gt-001-001",
        "pr_id": "sample-pr-001",
        "head_sha": "deadbeefcafe1234",
        "label": "TP",
        "severity": "P1",
        "class": "bug_correctness",
        "path_hint": "src/app.py",
        "line_range": {"start": 10, "end": 12},
        "line_range_status": "confirmed",
        "title": "P1: example bug",
        "rationale": "The diff introduces an off-by-one.",
        "match_keywords": ["off-by-one"],
        "source": "retrospective_diff_review",
        "confidence": "high",
    }
    base.update(overrides)
    return base


def _make_dataset_doc(findings):
    return {
        "version": "1.0",
        "dataset_id": "test-dataset",
        "generated_at": "2026-05-06",
        "source_sample": "eval/sample-prs.yaml",
        "methodology": "eval/baseline-methodology.md",
        "findings": list(findings),
    }


# ---------------------------------------------------------------------------
# Real dataset (the file we ship)
# ---------------------------------------------------------------------------


def test_real_dataset_loads_and_validates():
    """The shipped dataset.yaml must always be loadable and valid."""
    dataset = load_dataset()
    assert isinstance(dataset, LabeledDataset)
    assert dataset.findings, "shipped dataset must not be empty"
    assert dataset.dataset_id


def test_real_dataset_covers_every_sample_pr():
    """Every PR in the sample registry must have at least one labeled entry.

    A sample PR with zero ground-truth entries is a hole in the
    benchmark — it produces neither overlap nor FP-rate signal.
    """
    dataset = load_dataset()
    sample_doc = _load_yaml_mapping(SAMPLE_PATH, kind="sample registry")
    sample_index = _index_sample(sample_doc, source=SAMPLE_PATH)
    pr_ids_in_dataset = set(dataset.pr_ids())
    pr_ids_in_sample = set(sample_index)
    missing = pr_ids_in_sample - pr_ids_in_dataset
    assert not missing, (
        f"sample PRs missing from dataset.yaml: {sorted(missing)}. "
        "Every sample PR must have at least one labeled entry."
    )


def test_real_dataset_has_both_tp_and_fp_entries():
    """Bench-marking parity needs both axes: overlap (TP) and FP discipline."""
    dataset = load_dataset()
    assert dataset.true_positives(), "dataset must include at least one TP entry"
    assert dataset.false_positives(), "dataset must include at least one FP entry"


def test_real_dataset_finding_ids_are_unique():
    dataset = load_dataset()
    ids = [f.id for f in dataset.findings]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Allowed-value contract (the authoritative schema)
# ---------------------------------------------------------------------------


def test_allowed_labels_are_tp_and_fp_only():
    # DUP and CARRY are run-artifact labels (methodology §2.1), not
    # ground-truth labels — they must NOT be allowed in this dataset.
    assert ALLOWED_LABELS == {"TP", "FP"}


def test_allowed_severities_match_prompt():
    assert ALLOWED_SEVERITIES == {"P0", "P1", "P2", "P3"}


def test_allowed_classes_cover_methodology_taxonomy():
    # Sanity: every class in the YAML taxonomy must be in the code allow-list.
    expected_subset = {
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
    assert expected_subset.issubset(ALLOWED_CLASSES)


def test_allowed_line_range_statuses():
    assert ALLOWED_LINE_RANGE_STATUSES == {"confirmed", "unverified"}


# ---------------------------------------------------------------------------
# validate_dataset — happy path and each failure mode
# ---------------------------------------------------------------------------


def test_validate_dataset_accepts_minimal_valid_doc():
    doc = _make_dataset_doc([_make_finding()])
    dataset = validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)
    assert dataset.findings[0].label == "TP"


def test_validate_dataset_rejects_empty_findings():
    doc = _make_dataset_doc([])
    with pytest.raises(DatasetValidationError, match="must not be empty"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_unknown_pr_id():
    doc = _make_dataset_doc([_make_finding(pr_id="not-in-registry")])
    with pytest.raises(DatasetValidationError, match="not in eval/sample-prs.yaml"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_head_sha_drift():
    doc = _make_dataset_doc([_make_finding(head_sha="0000000000000000")])
    with pytest.raises(DatasetValidationError, match="does not match the registry entry"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_id_without_gt_prefix():
    doc = _make_dataset_doc([_make_finding(id="finding-001")])
    with pytest.raises(DatasetValidationError, match="must start with 'gt-'"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_duplicate_ids():
    doc = _make_dataset_doc([_make_finding(), _make_finding()])
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


@pytest.mark.parametrize("bad_label", ["DUP", "CARRY", "tp", "true_positive", ""])
def test_validate_dataset_rejects_invalid_label(bad_label):
    doc = _make_dataset_doc([_make_finding(label=bad_label)])
    with pytest.raises(DatasetValidationError, match="'label'"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


@pytest.mark.parametrize("bad_severity", ["P4", "low", "critical", "p1"])
def test_validate_dataset_rejects_invalid_severity(bad_severity):
    doc = _make_dataset_doc([_make_finding(severity=bad_severity)])
    with pytest.raises(DatasetValidationError, match="'severity'"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_invalid_class():
    doc = _make_dataset_doc([_make_finding(**{"class": "made_up_class"})])
    with pytest.raises(DatasetValidationError, match="'class'"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_confirmed_with_null_line_range():
    doc = _make_dataset_doc(
        [_make_finding(line_range=None, line_range_status="confirmed")]
    )
    with pytest.raises(DatasetValidationError, match="must pin its line numbers"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_unverified_with_set_line_range():
    doc = _make_dataset_doc(
        [
            _make_finding(
                line_range={"start": 5, "end": 7}, line_range_status="unverified"
            )
        ]
    )
    with pytest.raises(DatasetValidationError, match="confirm the entry or null out"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_inverted_line_range():
    doc = _make_dataset_doc(
        [
            _make_finding(
                line_range={"start": 20, "end": 10}, line_range_status="confirmed"
            )
        ]
    )
    with pytest.raises(DatasetValidationError, match="must be ≥ line_range.start"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_zero_or_negative_lines():
    doc = _make_dataset_doc(
        [
            _make_finding(
                line_range={"start": 0, "end": 5}, line_range_status="confirmed"
            )
        ]
    )
    with pytest.raises(DatasetValidationError, match="must be ≥ 1"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_accepts_keyword_only_anchor():
    """FP-trap entries with no specific file may anchor by keywords + pr_id alone."""
    doc = _make_dataset_doc(
        [
            _make_finding(
                path_hint=None,
                line_range=None,
                line_range_status="unverified",
                match_keywords=["spurious-complaint-keyword"],
            )
        ]
    )
    dataset = validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)
    assert dataset.findings[0].path_hint is None
    assert dataset.findings[0].line_range is None
    assert dataset.findings[0].match_keywords == ("spurious-complaint-keyword",)


def test_validate_dataset_rejects_entry_with_empty_keywords():
    bad = _make_finding(path_hint=None, line_range=None, line_range_status="unverified")
    bad["match_keywords"] = []
    doc = _make_dataset_doc([bad])
    with pytest.raises(DatasetValidationError, match="at least one string"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_rejects_non_string_keywords():
    bad = _make_finding()
    bad["match_keywords"] = ["ok", 42]
    doc = _make_dataset_doc([bad])
    with pytest.raises(DatasetValidationError, match="match_keywords"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


# ---------------------------------------------------------------------------
# LineRange.overlaps — methodology §4.5 rule 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_range", "b_range", "expected"),
    [
        # Direct overlap
        ((10, 20), (15, 25), True),
        # Touching exactly
        ((10, 20), (20, 30), True),
        # Within ±3 lines (the methodology tolerance)
        ((10, 20), (22, 25), True),
        ((10, 20), (5, 7), True),
        # Outside the tolerance
        ((10, 20), (30, 40), False),
        ((10, 20), (1, 5), False),
    ],
)
def test_line_range_overlap_with_methodology_tolerance(a_range, b_range, expected):
    a = LineRange(start=a_range[0], end=a_range[1])
    b = LineRange(start=b_range[0], end=b_range[1])
    assert a.overlaps(b) is expected
    # Symmetric — important so the matcher can call it from either side.
    assert b.overlaps(a) is expected


def test_line_range_overlap_zero_tolerance_rejects_close_misses():
    a = LineRange(start=10, end=20)
    b = LineRange(start=22, end=25)
    assert a.overlaps(b, tolerance=3) is True
    assert a.overlaps(b, tolerance=0) is False


# ---------------------------------------------------------------------------
# LabeledDataset views
# ---------------------------------------------------------------------------


def test_dataset_by_pr_returns_only_matching_entries():
    f1 = _make_finding(id="gt-001-001", pr_id="sample-pr-001")
    f2 = _make_finding(
        id="gt-002-001",
        pr_id="sample-pr-002",
        head_sha="cafef00dcafef00d",
    )
    sample_index = {
        "sample-pr-001": "deadbeefcafe1234",
        "sample-pr-002": "cafef00dcafef00d",
    }
    doc = _make_dataset_doc([f1, f2])
    dataset = validate_dataset(doc, sample_index=sample_index)
    assert [f.id for f in dataset.by_pr("sample-pr-001")] == ["gt-001-001"]
    assert [f.id for f in dataset.by_pr("sample-pr-002")] == ["gt-002-001"]
    assert dataset.by_pr("nonexistent") == ()


def test_dataset_tp_fp_partitions_are_disjoint_and_total():
    findings = [
        _make_finding(id="gt-001-001", label="TP"),
        _make_finding(id="gt-001-002", label="FP"),
    ]
    doc = _make_dataset_doc(findings)
    dataset = validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)
    tp_ids = {f.id for f in dataset.true_positives()}
    fp_ids = {f.id for f in dataset.false_positives()}
    assert tp_ids.isdisjoint(fp_ids)
    assert tp_ids | fp_ids == {f.id for f in dataset.findings}


def test_dataset_pr_ids_preserves_first_occurrence_order():
    sample_index = {
        "sample-pr-aa": "1111111111111111",
        "sample-pr-bb": "2222222222222222",
    }
    doc = _make_dataset_doc(
        [
            _make_finding(
                id="gt-001-001", pr_id="sample-pr-bb", head_sha="2222222222222222"
            ),
            _make_finding(
                id="gt-001-002", pr_id="sample-pr-aa", head_sha="1111111111111111"
            ),
            _make_finding(
                id="gt-001-003", pr_id="sample-pr-bb", head_sha="2222222222222222"
            ),
        ]
    )
    dataset = validate_dataset(doc, sample_index=sample_index)
    assert dataset.pr_ids() == ("sample-pr-bb", "sample-pr-aa")


# ---------------------------------------------------------------------------
# load_dataset — end-to-end with a tmp file
# ---------------------------------------------------------------------------


def test_load_dataset_from_path(tmp_path: Path):
    sample = tmp_path / "sample.yaml"
    # Use a hex-mixed SHA so PyYAML can't parse it as a number — the
    # registry always carries real SHAs (40-char hex), and the
    # validator requires both id and head_sha to be strings.
    fake_sha = "abcd1234ef567890" * 2 + "1234abcd"  # 40-char string
    assert len(fake_sha) == 40
    sample.write_text(
        textwrap.dedent(
            f"""\
            prs:
              - id: temp-pr-001
                head_sha: {fake_sha}
            """
        )
    )
    dataset_file = tmp_path / "dataset.yaml"
    finding = _make_finding(pr_id="temp-pr-001", head_sha=fake_sha)
    doc = _make_dataset_doc([finding])
    import yaml as _yaml

    dataset_file.write_text(_yaml.safe_dump(doc, sort_keys=False))
    loaded = load_dataset(path=dataset_file, sample_path=sample)
    assert loaded.findings[0].pr_id == "temp-pr-001"


def test_load_dataset_real_path_constant_points_to_shipped_file():
    """Defends against accidental relocations of the shipped dataset."""
    assert DATASET_PATH.exists(), (
        f"expected shipped dataset at {DATASET_PATH}; if you moved it, update "
        "eval/labeled_dataset/schema.py::DATASET_PATH"
    )


# ---------------------------------------------------------------------------
# Defensive: tampering with the doc never silently slips through
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_missing_required_top_level_field():
    doc = _make_dataset_doc([_make_finding()])
    del doc["dataset_id"]
    with pytest.raises(DatasetValidationError, match="dataset_id"):
        validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)


def test_validate_dataset_does_not_mutate_input():
    doc = _make_dataset_doc([_make_finding()])
    snapshot = deepcopy(doc)
    validate_dataset(doc, sample_index=_GOOD_SAMPLE_INDEX)
    assert doc == snapshot

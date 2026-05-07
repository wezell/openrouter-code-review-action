"""Ground-truth labeled dataset for the Codex → OpenRouter parity bake-off.

Sub-AC 2.3.1 deliverable: a checked-in dataset of historical PR review
findings annotated with TP/FP labels, plus a typed loader and validator.

The dataset itself lives at ``eval/labeled_dataset/dataset.yaml`` and is
loaded through :func:`load_dataset`. The schema (entry shape, allowed
labels, allowed severities, allowed classes) is enforced by
:func:`validate_dataset` and exercised by ``tests/test_labeled_dataset.py``.

Key design choices
------------------

* **Methodology-anchored.** Label set and FP definition come straight
  out of ``eval/baseline-methodology.md`` §2-§3. This module does not
  invent new label semantics; it just enforces that on-disk entries
  conform.

* **Cross-references the sample registry.** Every entry's ``pr_id``
  must appear in ``eval/sample-prs.yaml``, and the ``head_sha`` must
  match. The validator surfaces drift between the dataset and the
  registry as a hard error so silent rebases of the registry can't
  desync the labels.

* **Allows unverified line ranges.** Anchoring labels to exact line
  numbers requires inspecting each PR's diff. Until that's done,
  entries set ``line_range: null`` and ``line_range_status: unverified``;
  the validator accepts both states. This lets the dataset be useful
  immediately while supporting incremental refinement.
"""

from __future__ import annotations

from .schema import (
    ALLOWED_CLASSES,
    ALLOWED_LABELS,
    ALLOWED_LINE_RANGE_STATUSES,
    ALLOWED_SEVERITIES,
    ALLOWED_SOURCES,
    DATASET_PATH,
    LabeledDataset,
    LabeledFinding,
    LineRange,
    LineRangeStatus,
    load_dataset,
    validate_dataset,
)

__all__ = [
    "ALLOWED_CLASSES",
    "ALLOWED_LABELS",
    "ALLOWED_LINE_RANGE_STATUSES",
    "ALLOWED_SEVERITIES",
    "ALLOWED_SOURCES",
    "DATASET_PATH",
    "LabeledDataset",
    "LabeledFinding",
    "LineRange",
    "LineRangeStatus",
    "load_dataset",
    "validate_dataset",
]

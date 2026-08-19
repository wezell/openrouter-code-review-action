"""Sub-AC 1.4 — baseline fixture loader / validator.

Later regression-comparison ACs (parity bake-off scoring, anchor-validity
checks, posting-pipeline replays) all need to read the curated PR
fixtures under ``tests/fixtures/baseline/<pr-id>/`` *as a dataset* — not
as three loose files sprinkled across the repo. This module is the
single entry point those ACs call to:

* discover every fixture PR (without re-implementing yaml parsing)
* parse the three on-disk artifacts (``diff.patch``, ``metadata.json``,
  ``codex-findings.json``)
* assert well-formedness (schema, required fields, non-empty findings
  when the capture is ``"captured"``) before the dataset is consumed

The loader deliberately raises a single ``BaselineFixtureError`` with a
``pr_id``-prefixed message so the failing fixture is obvious from a CI
log line — the goal is "the dataset is healthy or pytest tells you
exactly which file to fix", not silent corruption that surfaces as a
mysterious bake-off regression three commits later.

Capture-status semantics
------------------------

Per ``eval/baseline-methodology.md`` §2.3, a Codex baseline mirror can
be in one of three states:

* ``"captured"`` — real Codex findings are pinned. ``review_run_result``
  must be present and ``findings`` must be a non-empty list of
  well-formed finding objects. This is what the parity bake-off scores.
* ``"pending"`` — placeholder for a fixture whose live capture has not
  yet run (e.g. ``GITHUB_TOKEN`` was not available).
  ``review_run_result`` must be ``null`` and ``capture.notes`` must
  explain why so future capture runs are not surprised.
* ``"failed"`` — a live capture attempt errored out. Same shape as
  ``"pending"`` — ``review_run_result`` is ``null`` and
  ``capture.notes`` carries the failure reason.

The "non-empty findings" half of Sub-AC 1.4 is enforced **only** for
``"captured"`` fixtures. Forcing it on placeholders would tie regression
testing to the presence of a live ``GITHUB_TOKEN`` in CI, which the
methodology explicitly forbids.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "baseline"
DEFAULT_SAMPLE_PATH = REPO_ROOT / "evaluation" / "bakeoff" / "sample-prs.yml"

# Capture states the methodology recognises. Anything outside this set
# is rejected by the validator as a corrupt mirror.
ALLOWED_CAPTURE_STATES: frozenset[str] = frozenset({"captured", "pending", "failed"})

# Capture states where ``review_run_result`` MUST be present + non-empty.
# Other states require ``review_run_result is None`` so the placeholder
# contract is unambiguous.
SCORABLE_CAPTURE_STATES: frozenset[str] = frozenset({"captured"})

# Top-level keys every codex-findings.json mirror must carry. Mirrors
# the contract pinned by ``eval/mirror_codex_baseline.py``.
REQUIRED_CODEX_KEYS: frozenset[str] = frozenset(
    {
        "pr_id",
        "run",
        "head_sha",
        "model",
        "reasoning_effort",
        "web_search_mode",
        "capture",
        "review_run_result",
        "posted",
        "source",
    }
)

# Required fields on each PR metadata.json (Sub-AC 1.2 contract).
REQUIRED_METADATA_KEYS: frozenset[str] = frozenset(
    {"number", "baseRefOid", "headRefOid", "mergeCommit", "files", "changedFiles"}
)

# Required fields on each captured Codex finding (the matcher reads
# ``code_location`` + ``title``/``body``/``priority`` directly, see
# ``eval/score_codex_baseline.py::_normalize_finding``).
REQUIRED_FINDING_KEYS: frozenset[str] = frozenset({"title", "body", "priority", "code_location"})


class BaselineFixtureError(AssertionError):
    """Raised when a baseline fixture violates the Sub-AC 1.4 contract.

    Subclasses ``AssertionError`` so pytest renders the message inline
    without a traceback wall — the failing ``pr_id`` is in ``args[0]``
    and is the first thing a maintainer sees.
    """


@dataclass(frozen=True)
class BaselineFixture:
    """One PR's worth of baseline fixture data, fully parsed.

    Consumers (parity bake-off, anchor-validity tests, posting-pipeline
    replays) read these fields directly — they are not expected to
    re-open the underlying JSON files. ``codex_findings`` is a list of
    raw finding dicts (not normalized) because the scorer module owns
    the normalization shape and we don't want to fork that contract.
    """

    pr_id: str
    pr_number: int
    fixture_dir: Path
    diff_text: str
    metadata: dict[str, Any]
    codex_payload: dict[str, Any]
    capture_status: str
    head_sha: str
    review_run_result: dict[str, Any] | None
    codex_findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        """True when the fixture carries scorable Codex findings."""
        return self.capture_status in SCORABLE_CAPTURE_STATES and bool(self.codex_findings)


def _fail(pr_id: str, message: str) -> BaselineFixtureError:
    return BaselineFixtureError(f"{pr_id}: {message}")


def _read_json(path: Path, *, pr_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise _fail(pr_id, f"missing {path.name}")
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise _fail(pr_id, f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise _fail(pr_id, f"{path.name} top-level value must be a JSON object")
    return dict(doc)


def _validate_metadata(pr_id: str, pr_number: int, meta: Mapping[str, Any]) -> None:
    missing = REQUIRED_METADATA_KEYS - meta.keys()
    if missing:
        raise _fail(pr_id, f"metadata.json missing keys {sorted(missing)}")
    if meta["number"] != pr_number:
        raise _fail(
            pr_id,
            f"metadata.json number={meta['number']} but fixture pr_id implies {pr_number}",
        )
    for sha_field in ("baseRefOid", "headRefOid"):
        sha = meta[sha_field]
        if not (isinstance(sha, str) and len(sha) == 40):
            raise _fail(
                pr_id,
                f"metadata.json {sha_field!r} must be a 40-char SHA, got {sha!r}",
            )
    merge = meta["mergeCommit"]
    if not isinstance(merge, Mapping) or not isinstance(merge.get("oid"), str):
        raise _fail(pr_id, "metadata.json mergeCommit.oid must be a string")
    if len(merge["oid"]) != 40:
        raise _fail(pr_id, "metadata.json mergeCommit.oid must be a 40-char SHA")
    files = meta["files"]
    if not isinstance(files, list) or not files:
        raise _fail(pr_id, "metadata.json files must be a non-empty list")
    if meta["changedFiles"] != len(files):
        raise _fail(
            pr_id,
            f"metadata.json changedFiles={meta['changedFiles']} disagrees with len(files)={len(files)}",
        )


def _validate_codex_payload(pr_id: str, payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_CODEX_KEYS - payload.keys()
    if missing:
        raise _fail(pr_id, f"codex-findings.json missing keys {sorted(missing)}")
    if payload["pr_id"] != pr_id:
        raise _fail(
            pr_id,
            f"codex-findings.json pr_id={payload['pr_id']!r} does not match fixture id",
        )
    if payload["run"] != "codex":
        raise _fail(
            pr_id,
            f"codex-findings.json run={payload['run']!r}; expected 'codex'",
        )
    capture = payload["capture"]
    if not isinstance(capture, Mapping):
        raise _fail(pr_id, "codex-findings.json capture must be an object")
    status = capture.get("status")
    if status not in ALLOWED_CAPTURE_STATES:
        raise _fail(
            pr_id,
            f"capture.status={status!r} not in {sorted(ALLOWED_CAPTURE_STATES)}",
        )
    if not isinstance(capture.get("runner_version"), str):
        raise _fail(pr_id, "capture.runner_version must be a string")
    head_sha = payload["head_sha"]
    if not (isinstance(head_sha, str) and len(head_sha) == 40):
        raise _fail(pr_id, f"head_sha must be a 40-char SHA, got {head_sha!r}")
    posted = payload["posted"]
    if not isinstance(posted, Mapping):
        raise _fail(pr_id, "posted must be an object")
    source = payload["source"]
    if not isinstance(source, Mapping):
        raise _fail(pr_id, "source must be an object")
    for field_name in ("produced_by", "mirrored_by", "eval_artifact"):
        if not isinstance(source.get(field_name), str):
            raise _fail(pr_id, f"source.{field_name} must be a string")


def _validate_finding(pr_id: str, idx: int, finding: Any) -> None:
    if not isinstance(finding, Mapping):
        raise _fail(pr_id, f"review_run_result.findings[{idx}] must be an object")
    missing = REQUIRED_FINDING_KEYS - finding.keys()
    if missing:
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}] missing keys {sorted(missing)}",
        )
    if not isinstance(finding["title"], str) or not finding["title"].strip():
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}].title must be a non-empty string",
        )
    if not isinstance(finding["body"], str):
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}].body must be a string",
        )
    priority = finding["priority"]
    if priority is not None and not (isinstance(priority, int) and 0 <= priority <= 3):
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}].priority must be int in [0,3] or null",
        )
    code_location = finding["code_location"]
    if not isinstance(code_location, Mapping):
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}].code_location must be an object",
        )
    if not isinstance(code_location.get("absolute_file_path"), str):
        raise _fail(
            pr_id,
            f"review_run_result.findings[{idx}].code_location.absolute_file_path must be a string",
        )


def _validate_review_run_result(
    pr_id: str, capture_status: str, payload: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    review = payload["review_run_result"]

    if capture_status not in SCORABLE_CAPTURE_STATES:
        # Pending / failed: must be a strict null placeholder.
        if review is not None:
            raise _fail(
                pr_id,
                f"capture.status={capture_status!r} requires review_run_result=null "
                f"(got {type(review).__name__}); placeholder mirrors must not "
                "embed synthesized findings",
            )
        capture = payload["capture"]
        if not isinstance(capture.get("notes"), str) or not capture["notes"].strip():
            raise _fail(
                pr_id,
                f"capture.status={capture_status!r} requires capture.notes "
                "to explain why no findings are present",
            )
        return None, []

    # Captured: review_run_result must be a dict carrying a non-empty
    # ``findings`` list. This is the half of Sub-AC 1.4 that pins the
    # "non-empty findings" guarantee for downstream regression compares.
    if not isinstance(review, Mapping):
        raise _fail(
            pr_id,
            "capture.status='captured' requires review_run_result to be an object",
        )
    findings_raw = review.get("findings")
    if not isinstance(findings_raw, list):
        raise _fail(
            pr_id,
            "captured review_run_result.findings must be a list",
        )
    if not findings_raw:
        raise _fail(
            pr_id,
            "captured review_run_result.findings must be non-empty "
            "(use capture.status='pending' for placeholder mirrors)",
        )
    for idx, item in enumerate(findings_raw):
        _validate_finding(pr_id, idx, item)
    return dict(review), [dict(f) for f in findings_raw]


def load_baseline_fixture(
    pr_id: str,
    *,
    fixtures_dir: Path | None = None,
) -> BaselineFixture:
    """Load and validate a single baseline fixture.

    Raises ``BaselineFixtureError`` (a subclass of ``AssertionError``)
    with a ``<pr_id>: <reason>`` message when any contract is violated.
    """
    base = fixtures_dir or DEFAULT_FIXTURES_DIR
    pr_dir = base / pr_id
    if not pr_dir.is_dir():
        raise _fail(pr_id, f"fixture directory {pr_dir} does not exist")

    diff_path = pr_dir / "diff.patch"
    if not diff_path.is_file():
        raise _fail(pr_id, "missing diff.patch")
    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    if not diff_text.startswith("diff --git "):
        raise _fail(pr_id, "diff.patch is not a unified diff (must start with 'diff --git ')")

    if not pr_id.startswith("pr-"):
        raise _fail(pr_id, f"pr_id must follow 'pr-<NUMBER>' convention, got {pr_id!r}")
    try:
        pr_number = int(pr_id.removeprefix("pr-"))
    except ValueError as exc:
        raise _fail(pr_id, f"pr_id suffix must be an integer: {exc}") from exc

    metadata = _read_json(pr_dir / "metadata.json", pr_id=pr_id)
    _validate_metadata(pr_id, pr_number, metadata)

    codex_payload = _read_json(pr_dir / "codex-findings.json", pr_id=pr_id)
    _validate_codex_payload(pr_id, codex_payload)

    capture_status = codex_payload["capture"]["status"]
    review_run_result, codex_findings = _validate_review_run_result(
        pr_id, capture_status, codex_payload
    )

    # The codex baseline replays reviews against the merge commit (not
    # the PR branch tip), so ``codex.head_sha`` must match
    # ``metadata.mergeCommit.oid``. ``headRefOid`` is the branch tip and
    # is used by the fixture-regen tooling, not by review replay.
    merge_sha_meta = metadata["mergeCommit"]["oid"]
    head_sha_codex = codex_payload["head_sha"]
    if merge_sha_meta != head_sha_codex:
        raise _fail(
            pr_id,
            "review-replay SHA disagrees: "
            f"metadata.mergeCommit.oid={merge_sha_meta!r} codex.head_sha={head_sha_codex!r}",
        )

    return BaselineFixture(
        pr_id=pr_id,
        pr_number=pr_number,
        fixture_dir=pr_dir,
        diff_text=diff_text,
        metadata=metadata,
        codex_payload=codex_payload,
        capture_status=capture_status,
        head_sha=head_sha_codex,
        review_run_result=review_run_result,
        codex_findings=codex_findings,
    )


def list_fixture_pr_ids(
    *,
    sample_path: Path | None = None,
) -> list[str]:
    """Return ``['pr-<N>', ...]`` for every PR in the bake-off sample.

    Reads the canonical sample registry instead of globbing the
    fixtures directory so a stray on-disk directory is detected as a
    drift, not silently treated as a fixture.
    """
    path = sample_path or DEFAULT_SAMPLE_PATH
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping):
        raise BaselineFixtureError(f"{path}: top-level document must be a mapping")
    prs = doc.get("prs") or []
    if not isinstance(prs, list):
        raise BaselineFixtureError(f"{path}: 'prs' must be a list")
    out: list[str] = []
    for entry in prs:
        if not isinstance(entry, Mapping) or "number" not in entry:
            raise BaselineFixtureError(f"{path}: each prs[] entry must be a mapping with 'number'")
        out.append(f"pr-{int(entry['number'])}")
    return out


def iter_baseline_fixtures(
    *,
    fixtures_dir: Path | None = None,
    sample_path: Path | None = None,
) -> Iterator[BaselineFixture]:
    """Yield every baseline fixture, validated, in sample-registry order."""
    for pr_id in list_fixture_pr_ids(sample_path=sample_path):
        yield load_baseline_fixture(pr_id, fixtures_dir=fixtures_dir)


def load_all_baseline_fixtures(
    *,
    fixtures_dir: Path | None = None,
    sample_path: Path | None = None,
) -> list[BaselineFixture]:
    """Eagerly load and validate every baseline fixture.

    The eager form is what the parity bake-off tooling will call: it
    surfaces *every* fixture defect at once instead of stopping at the
    first failure, which makes batch fixture refreshes much easier to
    reason about.
    """
    fixtures: list[BaselineFixture] = []
    errors: list[str] = []
    for pr_id in list_fixture_pr_ids(sample_path=sample_path):
        try:
            fixtures.append(load_baseline_fixture(pr_id, fixtures_dir=fixtures_dir))
        except BaselineFixtureError as exc:
            errors.append(str(exc))
    if errors:
        joined = "\n  - " + "\n  - ".join(errors)
        raise BaselineFixtureError(
            f"{len(errors)} baseline fixture(s) failed Sub-AC 1.4 validation:{joined}"
        )
    return fixtures

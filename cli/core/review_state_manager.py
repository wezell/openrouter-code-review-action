"""High-level state lookup and update API keyed by PR identifier.

Sub-AC 4.1.3 — this is the **service layer** that sits on top of the
file-backed :class:`~cli.core.review_state_store.ReviewStateStore` (Sub-AC
4.1.2) and the schema/contract layer in
:mod:`cli.core.review_state` (Sub-AC 4.1.1). It is the single place the
review workflow goes when it needs to answer two operational questions:

1. *Do we already have prior review state for this PR?* (lookup)
2. *Where do we record what just happened?* (update)

The store knows how to read/write a file given a key. The manager knows
how to do the right thing when the file is missing, when a force-push has
moved the PR HEAD to a new SHA, and when a review run has produced a new
batch of findings that needs to land somewhere durable.

Concept map
-----------

* ``ReviewStateManager`` — the lookup/update façade. Bound to a
  :class:`~cli.core.review_state_store.ReviewStateStore`. All workflow
  callers go through this class, never directly through the store.
* ``ReviewStateLookupResult`` — the structured return value of
  :meth:`ReviewStateManager.lookup_or_initialize`. Records *which*
  branch produced the state (current SHA hit, previous SHA fall-back, or
  fresh-initialized) so the caller can log the decision and decide
  whether incremental review is safe.
* ``LookupSource`` — string-typed enum-ish constants used by
  ``ReviewStateLookupResult.source``. Open string instead of an Enum so
  the field can round-trip through JSON without a custom encoder.

Why initialization is **non-raising**?
The seed contract says first-time PRs must work the same way as
returning PRs — no special-casing in the workflow. Mirroring the
"force-push falls back to fresh full review" rule, a missing-state
lookup returns an empty :class:`PriorReviewState` envelope rather than
throwing, so the workflow code is always handed a usable state object.
The ``initialized=True`` flag is the signal to the caller that this is a
fresh start, not a continuation.

Why the manager keeps the *previous* SHA out of the
:class:`~cli.core.review_state.PriorReviewKey`?
The key is identity. Two runs with different SHAs are different keys and
need to remain distinct so secondary-restore lookups work. The
"previous head SHA" is a *hint* used only at lookup time to find a
neighbour record; it does not become part of the persisted identity.

Why ``record_review_run`` always rewrites the full envelope?
Findings, carried-forward IDs, and the summary comment ID move together
— the next run wants a single consistent snapshot. Partial updates would
let the persisted state drift out of sync (e.g. new findings written
without the new summary comment ID) and the next run would have to do
reconciliation. One atomic write is simpler and matches how
``ReviewStateStore.save`` already works under the hood.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .review_state import (
    REVIEW_STATE_SCHEMA_VERSION,
    PersistedReviewFinding,
    PriorReviewKey,
    PriorReviewState,
)
from .review_state_store import (
    DEFAULT_STATE_DIRECTORY_NAME,
    ReviewStateStore,
    WrittenStateRecord,
)

# ---------------------------------------------------------------------------
# Lookup-source labels
# ---------------------------------------------------------------------------

#: The lookup hit a record already keyed against the *current* head SHA.
#: This is the steady-state "another review run on the same SHA"
#: scenario — e.g. a workflow re-run with no new commits.
LOOKUP_SOURCE_CURRENT_SHA: Final[str] = "current_sha"

#: The current SHA had no record but the caller-supplied previous SHA did.
#: This is the *incremental review continuation* path: the PR HEAD has
#: moved forward, but the prior reviewed SHA is still an ancestor and we
#: can re-use the cached findings as a starting point.
LOOKUP_SOURCE_PREVIOUS_SHA: Final[str] = "previous_sha"

#: Neither the current nor the previous SHA had a record on disk; the
#: manager initialised a fresh empty envelope so the workflow can
#: proceed with a full first-time review.
LOOKUP_SOURCE_INITIALIZED: Final[str] = "initialized"


@dataclass(frozen=True)
class ReviewStateLookupResult:
    """Outcome of a :meth:`ReviewStateManager.lookup_or_initialize` call.

    The state object is always populated — callers never see ``None``.
    What changes between the three branches is :attr:`source` (and, by
    construction, :attr:`initialized`):

    * ``LOOKUP_SOURCE_CURRENT_SHA`` — direct hit on ``key``;
      :attr:`initialized` is ``False``.
    * ``LOOKUP_SOURCE_PREVIOUS_SHA`` — hit on the previous-SHA
      neighbour key; :attr:`initialized` is ``False`` and
      :attr:`fallback_key` carries the older key the data came from.
      The :attr:`state` returned is the *original* prior state (still
      keyed under the older SHA) — re-keying it to the current SHA is
      the workflow's responsibility because that decision depends on
      ancestry checks the manager intentionally does not do.
    * ``LOOKUP_SOURCE_INITIALIZED`` — no record existed; :attr:`state`
      is :meth:`PriorReviewState.empty` for ``key`` and
      :attr:`initialized` is ``True``.
    """

    state: PriorReviewState
    source: str
    initialized: bool
    fallback_key: PriorReviewKey | None = None


class ReviewStateManager:
    """Service-layer façade over :class:`ReviewStateStore`.

    Provides the two operations the review workflow actually needs:

    * :meth:`lookup_or_initialize` — find existing prior state by PR
      identifier, falling back to a previous-SHA neighbour, and
      otherwise minting an empty envelope so first-time PRs always get
      a usable :class:`PriorReviewState`.
    * :meth:`record_review_run` — persist a new state envelope after a
      review run completes, atomically replacing the prior record (if
      any) under the same key.

    The manager intentionally does not perform git ancestry checks,
    cache-hit telemetry, or summary-comment metadata parsing. Those
    decisions live in the workflow layer; the manager just gives the
    workflow a consistent state object to work with.
    """

    def __init__(self, store: ReviewStateStore) -> None:
        self.store = store

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------

    @classmethod
    def from_runner_temp(
        cls,
        runner_temp: str | Path | None,
        *,
        directory_name: str = DEFAULT_STATE_DIRECTORY_NAME,
    ) -> ReviewStateManager:
        """Build a manager rooted at ``$RUNNER_TEMP/<directory_name>``.

        Convenience pass-through to
        :meth:`ReviewStateStore.from_runner_temp` so the workflow layer
        can construct the manager without ever touching the store class
        directly.
        """
        store = ReviewStateStore.from_runner_temp(
            runner_temp, directory_name=directory_name
        )
        return cls(store)

    @classmethod
    def from_base_dir(cls, base_dir: str | Path) -> ReviewStateManager:
        """Build a manager pointed at an explicit base directory.

        Used by unit tests (where ``RUNNER_TEMP`` does not exist) and by
        operator tooling that wants to inspect a specific cache
        location without the runner-temp fall-back logic.
        """
        return cls(ReviewStateStore(base_dir))

    # ---------------------------------------------------------------
    # Lookup / initialization
    # ---------------------------------------------------------------

    def lookup_or_initialize(
        self,
        key: PriorReviewKey,
        *,
        previous_head_sha: str | None = None,
        generated_at: str | None = None,
    ) -> ReviewStateLookupResult:
        """Look up state for ``key``; mint an empty envelope on miss.

        Search order:

        1. **Current key** — direct lookup against ``key`` (the current
           run's PR identifier including the current head SHA).
        2. **Previous SHA neighbour** — if ``previous_head_sha`` was
           supplied and differs from ``key.reviewed_head_sha``, look up
           the same ``(repo, pr, model)`` tuple keyed against the older
           SHA. This is what enables incremental review continuation
           after a fast-forward push.
        3. **Initialize empty** — return
           :meth:`PriorReviewState.empty` for ``key``. ``generated_at``
           is forwarded into the envelope so first-time records carry a
           timestamp without forcing the caller to wrap a
           ``with_findings`` call.

        Empty/whitespace ``previous_head_sha`` is treated the same as
        ``None`` so callers can pass through environment-variable
        strings without sanitising first.
        """
        primary_state = self.store.load(key)
        if primary_state is not None:
            return ReviewStateLookupResult(
                state=primary_state,
                source=LOOKUP_SOURCE_CURRENT_SHA,
                initialized=False,
                fallback_key=None,
            )

        normalized_previous = _normalise_previous_sha(previous_head_sha)
        if (
            normalized_previous is not None
            and normalized_previous != key.reviewed_head_sha
        ):
            fallback_key = key.with_sha(normalized_previous)
            fallback_state = self.store.load(fallback_key)
            if fallback_state is not None:
                return ReviewStateLookupResult(
                    state=fallback_state,
                    source=LOOKUP_SOURCE_PREVIOUS_SHA,
                    initialized=False,
                    fallback_key=fallback_key,
                )

        return ReviewStateLookupResult(
            state=PriorReviewState.empty(key, generated_at=generated_at),
            source=LOOKUP_SOURCE_INITIALIZED,
            initialized=True,
            fallback_key=None,
        )

    def has_state_for(self, key: PriorReviewKey) -> bool:
        """Whether a record exists on disk for ``key``.

        Thin pass-through to :meth:`ReviewStateStore.exists` so callers
        that only need a boolean (e.g. the action's "cache hit"
        telemetry output) do not have to import the store directly.
        """
        return self.store.exists(key)

    # ---------------------------------------------------------------
    # Update
    # ---------------------------------------------------------------

    def record_review_run(
        self,
        key: PriorReviewKey,
        *,
        findings: Iterable[PersistedReviewFinding] = (),
        carried_forward_comment_ids: Iterable[str] = (),
        summary_comment_id: int | None = None,
        generated_at: str | None = None,
        notes: Iterable[str] = (),
    ) -> WrittenStateRecord:
        """Persist a freshly produced state envelope under ``key``.

        Always writes a complete envelope — there is no "patch only the
        new findings" path on purpose. Callers carry forward whatever
        prior fields they want by passing them explicitly (typically
        ``carried_forward_comment_ids`` from the previous run plus
        whatever the current run added).

        The atomic write is delegated to
        :meth:`ReviewStateStore.save`, which uses
        :func:`cli.core.filesystem.write_text_atomic` so a runner that
        loses power mid-write cannot leave a half-written record the
        next run will choke on.
        """
        state = PriorReviewState(
            schema_version=REVIEW_STATE_SCHEMA_VERSION,
            key=key,
            findings=tuple(findings),
            carried_forward_comment_ids=tuple(carried_forward_comment_ids),
            summary_comment_id=summary_comment_id,
            generated_at=generated_at,
            notes=tuple(notes),
        )
        return self.store.save(state)

    def record_state(self, state: PriorReviewState) -> WrittenStateRecord:
        """Persist a pre-built :class:`PriorReviewState` directly.

        Useful when the workflow has already constructed (or carried
        forward) a state envelope — e.g. when re-keying a previous-SHA
        fall-back result onto the new head SHA via
        :meth:`migrate_to_sha` before saving.
        """
        return self.store.save(state)

    def migrate_to_sha(
        self,
        state: PriorReviewState,
        new_head_sha: str,
        *,
        generated_at: str | None = None,
    ) -> PriorReviewState:
        """Re-key ``state`` to a new head SHA without writing to disk.

        Used by the incremental-review path: the manager hands the
        workflow a state record keyed against the *previous* SHA, the
        workflow validates ancestry, and then asks the manager to
        re-stamp it with the new SHA so the next ``record_review_run``
        lands under the current key. Findings, carried-forward IDs,
        summary comment ID, and notes are copied verbatim;
        ``generated_at`` is replaced if explicitly supplied so callers
        can refresh the timestamp at re-key time.
        """
        rekeyed_key = state.key.with_sha(new_head_sha)
        return PriorReviewState(
            schema_version=state.schema_version,
            key=rekeyed_key,
            findings=state.findings,
            carried_forward_comment_ids=state.carried_forward_comment_ids,
            summary_comment_id=state.summary_comment_id,
            generated_at=generated_at if generated_at is not None else state.generated_at,
            notes=state.notes,
        )

    # ---------------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------------

    def discard(self, key: PriorReviewKey) -> bool:
        """Remove the persisted record for ``key`` if present.

        Pass-through to :meth:`ReviewStateStore.delete` exposed through
        the manager so a workflow can drop a known-bad record (e.g.
        after detecting an irrecoverable schema mismatch) without
        importing the store layer.
        """
        return self.store.delete(key)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_previous_sha(value: str | None) -> str | None:
    """Strip and empty-string the supplied SHA hint."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "LOOKUP_SOURCE_CURRENT_SHA",
    "LOOKUP_SOURCE_INITIALIZED",
    "LOOKUP_SOURCE_PREVIOUS_SHA",
    "ReviewStateLookupResult",
    "ReviewStateManager",
]

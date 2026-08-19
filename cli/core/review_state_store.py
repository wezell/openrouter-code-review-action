"""Storage backend for prior-review state across GitHub Action runs.

This module is the **persistence layer** that sits between the in-memory
:class:`~cli.core.review_state.PriorReviewState` schema and the disk
representation that survives across GitHub Action runs (via the
``actions/cache`` round-trip on ``RUNNER_TEMP``).

It deliberately stays a thin file/JSON layer instead of pulling in a
database or sidecar daemon: GitHub Actions runners are ephemeral, the
cache contract is a tarball, and a single-file-per-key layout is what the
``actions/cache`` action restores most reliably.

Concept map
-----------

* ``ReviewStateStore`` — the persistence object. Bound to a base directory
  (typically ``$RUNNER_TEMP/openrouter-review-state``) that is mounted
  back into the next run by ``actions/cache``. Every persisted state is
  one file: ``<base_dir>/<cache_key>.json``. Two runs sharing the same
  :class:`PriorReviewKey` round-trip through the same file path.
* ``DEFAULT_STATE_DIRECTORY_NAME`` — the conventional directory name
  carved out under ``RUNNER_TEMP``. Distinct from the legacy
  ``dotbot-review-state`` directory so a post-migration run cannot
  accidentally pick up a stale review thread cache.
* ``ReviewStateNotFound`` — raised by ``load(...)`` only when the caller
  passes ``required=True``. The default surface returns ``None`` for
  missing files so the workflow can fall back to a fresh full review on
  cache miss without having to ``try``/``except`` for the common case.

Why a fresh module instead of extending ``review_state.py``?
``review_state.py`` is the **schema/contract** layer — pure data and
JSON round-tripping, no I/O. Mixing disk I/O into it would force the
schema module to grow a filesystem dependency and would muddle the
"can I trust this payload's shape?" question with the "can I trust this
file's bytes?" question. Keeping them separate means a future
re-platforming of the storage backend (e.g. moving to a GitHub
Releases artifact, an S3 object, a sqlite db) only touches this file
and not the dataclass surface every other module imports.

Why one file per key instead of one big index file?
Concurrent writers are not a concern (GitHub Actions schedules at most
one workflow run per PR head SHA + model + repo at a time), but partial
write recovery and incremental restore are. With a single index file,
any corruption invalidates the entire cache and forces a fresh review on
*every* PR; with per-key files, a corrupted entry only forces a fresh
review on the affected PR. ``actions/cache`` also tarballs the directory
either way, so we lose nothing by going per-file.

Why corruption tolerance defaults to ``True``?
The seed constraint says "force-push that breaks reviewed-SHA ancestry
falls back to fresh full review". Treating a corrupted cache file the
same way — return ``None`` and fall back to a fresh review — is the
operationally safe default. Callers that explicitly want to surface a
corruption can pass ``tolerate_corruption=False`` and handle the
``ReviewContractError``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ReviewContractError, ReviewResumeError
from .filesystem import write_text_atomic
from .review_state import (
    OPENROUTER_REVIEW_CACHE_PREFIX,
    PriorReviewKey,
    PriorReviewState,
)

#: Conventional name for the directory under ``RUNNER_TEMP`` that holds
#: persisted review state. Picked once and pinned because the GitHub
#: Actions cache step keys against this exact path.
DEFAULT_STATE_DIRECTORY_NAME = "openrouter-review-state"

#: Filename suffix for persisted review-state files. Per-file format is
#: pretty-printed JSON so operators can inspect the cache by hand.
STATE_FILE_SUFFIX = ".json"


class ReviewStateNotFound(ReviewResumeError):
    """Raised by ``ReviewStateStore.load`` when ``required=True`` and the
    requested key has no record on disk.

    Distinct from :class:`ReviewContractError` so the workflow layer can
    distinguish "no prior state for this key, do a fresh review" from
    "prior state exists but is unreadable".
    """


@dataclass(frozen=True)
class WrittenStateRecord:
    """Result of a successful ``ReviewStateStore.save`` call.

    Carries the path the state was written to and the cache key derived
    from the state's :class:`PriorReviewKey`. The cache key is what the
    GitHub Actions ``actions/cache`` step uses to upload the directory,
    so returning it here lets the caller emit it as a step output without
    having to recompute it.
    """

    path: Path
    cache_key: str


class ReviewStateStore:
    """File-backed store for :class:`PriorReviewState` records.

    The store maps a :class:`PriorReviewKey` to a single JSON file under
    ``base_dir``. Files are named ``<cache_key>.json`` so the on-disk
    layout is a flat directory, not a nested tree, which keeps the
    ``actions/cache`` round-trip predictable.

    Read/write operations are designed for the "one writer, one reader,
    many cache misses" pattern of GitHub Actions: writes go through
    :func:`write_text_atomic` so a runner that loses power mid-write
    cannot leave a half-written file the next run will choke on; reads
    default to graceful degradation so a corrupted entry is treated the
    same as a missing entry.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------

    @classmethod
    def from_runner_temp(
        cls,
        runner_temp: str | Path | None,
        *,
        directory_name: str = DEFAULT_STATE_DIRECTORY_NAME,
    ) -> ReviewStateStore:
        """Build a store rooted at ``$RUNNER_TEMP/<directory_name>``.

        ``runner_temp`` may be ``None`` or empty (e.g. running outside a
        GitHub Actions runner); in that case we anchor the store at the
        current working directory so unit tests and local invocations
        still work without special-casing.
        """
        if runner_temp is None or runner_temp == "":
            anchor = Path.cwd().resolve()
        else:
            anchor = Path(runner_temp).resolve()
        return cls(anchor / directory_name)

    # ---------------------------------------------------------------
    # Path arithmetic
    # ---------------------------------------------------------------

    def path_for(self, key: PriorReviewKey) -> Path:
        """Compute the on-disk file path that backs a given key.

        The path is deterministic (no timestamp, no hash) so the next
        run can compute the same path purely from its
        :class:`PriorReviewKey` without having to enumerate the directory.
        """
        return self.base_dir / f"{key.cache_key()}{STATE_FILE_SUFFIX}"

    def ensure_directory(self) -> Path:
        """Create the base directory if it does not yet exist.

        Returns the resolved base directory path. Idempotent — safe to
        call before every write or once at workflow startup.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        return self.base_dir

    # ---------------------------------------------------------------
    # Read operations
    # ---------------------------------------------------------------

    def exists(self, key: PriorReviewKey) -> bool:
        """Whether a state record exists on disk for ``key``."""
        return self.path_for(key).is_file()

    def load(
        self,
        key: PriorReviewKey,
        *,
        required: bool = False,
        tolerate_corruption: bool = True,
    ) -> PriorReviewState | None:
        """Read the persisted state for ``key`` from disk.

        Behavior matrix:

        * Missing file + ``required=False`` → returns ``None``.
        * Missing file + ``required=True`` → raises
          :class:`ReviewStateNotFound`.
        * Empty file (zero bytes) — ``actions/cache`` is known to
          truncate when it runs out of disk — is treated the same as
          missing: returns ``None`` (or raises if ``required`` is set).
        * Corrupt JSON / schema mismatch + ``tolerate_corruption=True``
          → returns ``None`` so the workflow falls back to a fresh
          review. The corrupt file is left in place so an operator can
          inspect it.
        * Corrupt JSON / schema mismatch + ``tolerate_corruption=False``
          → raises :class:`ReviewContractError`.
        """
        path = self.path_for(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if required:
                raise ReviewStateNotFound(
                    f"No prior review state for {key.cache_key()!r} at {path}"
                )
            return None

        if not raw.strip():
            # Truncated cache restore — treat as missing.
            if required:
                raise ReviewStateNotFound(
                    f"Prior review state file for {key.cache_key()!r} is empty"
                )
            return None

        try:
            return PriorReviewState.from_json(raw)
        except ReviewContractError:
            if tolerate_corruption:
                return None
            raise

    def load_path(
        self,
        path: str | Path,
        *,
        tolerate_corruption: bool = True,
    ) -> PriorReviewState | None:
        """Read a state record by absolute file path.

        Used when the caller already knows the file location (e.g. when
        iterating ``list_paths``) and does not want to recompute the
        :class:`PriorReviewKey`.
        """
        target = Path(path)
        try:
            raw = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        if not raw.strip():
            return None

        try:
            return PriorReviewState.from_json(raw)
        except ReviewContractError:
            if tolerate_corruption:
                return None
            raise

    # ---------------------------------------------------------------
    # Write operations
    # ---------------------------------------------------------------

    def save(self, state: PriorReviewState) -> WrittenStateRecord:
        """Persist ``state`` to disk atomically and return where it landed.

        The base directory is created on demand; writes use
        :func:`write_text_atomic` so a failure mid-write does not corrupt
        the existing record.
        """
        self.ensure_directory()
        target = self.path_for(state.key)
        write_text_atomic(target, state.to_json())
        return WrittenStateRecord(path=target, cache_key=state.cache_key)

    def delete(self, key: PriorReviewKey) -> bool:
        """Remove the persisted record for ``key``.

        Returns ``True`` if a file was removed, ``False`` if no record
        existed. Never raises on missing files — deletes are idempotent.
        """
        path = self.path_for(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    # ---------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------

    def list_paths(self) -> list[Path]:
        """All persisted state file paths under the base directory.

        Returns an empty list if the base directory does not exist.
        Files are sorted by name for deterministic test/log output.
        Only files matching the ``openrouter-review-v1-*.json`` naming
        convention are returned, so a stray ``.dotbot-review-v1-*`` file
        from the legacy action will not be reported.
        """
        if not self.base_dir.is_dir():
            return []
        prefix = f"{OPENROUTER_REVIEW_CACHE_PREFIX}-"
        return sorted(
            entry
            for entry in self.base_dir.iterdir()
            if entry.is_file()
            and entry.name.startswith(prefix)
            and entry.name.endswith(STATE_FILE_SUFFIX)
        )

    def iter_states(
        self,
        *,
        tolerate_corruption: bool = True,
    ) -> Iterator[PriorReviewState]:
        """Yield every parseable state record under the base directory.

        Corrupt entries are skipped when ``tolerate_corruption`` is
        ``True`` (the default) so a single bad file does not abort
        bulk-discovery operations.
        """
        for path in self.list_paths():
            state = self.load_path(path, tolerate_corruption=tolerate_corruption)
            if state is not None:
                yield state


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def load_review_state(
    base_dir: str | Path,
    key: PriorReviewKey,
    *,
    required: bool = False,
    tolerate_corruption: bool = True,
) -> PriorReviewState | None:
    """One-shot read for callers that don't want to keep a store handle.

    Equivalent to ``ReviewStateStore(base_dir).load(key, ...)``.
    """
    return ReviewStateStore(base_dir).load(
        key,
        required=required,
        tolerate_corruption=tolerate_corruption,
    )


def save_review_state(
    base_dir: str | Path,
    state: PriorReviewState,
) -> WrittenStateRecord:
    """One-shot atomic write for callers that don't want to keep a store handle.

    Equivalent to ``ReviewStateStore(base_dir).save(state)``.
    """
    return ReviewStateStore(base_dir).save(state)


def serialize_state_to_json(state: PriorReviewState, *, indent: int | None = 2) -> str:
    """Render a state record to JSON without touching the filesystem.

    Useful for uploading the state to a non-filesystem backend (e.g. a
    GitHub artifact, an S3 bucket) without going through the
    ``ReviewStateStore`` directory layout. Returns the same byte-for-byte
    output that ``ReviewStateStore.save`` would have written.
    """
    return state.to_json(indent=indent)


def deserialize_state_from_json(
    raw: str,
    *,
    tolerate_corruption: bool = False,
) -> PriorReviewState | None:
    """Inverse of :func:`serialize_state_to_json`.

    Parses a JSON payload into a :class:`PriorReviewState`. Returns
    ``None`` on parse failure when ``tolerate_corruption`` is ``True``;
    otherwise re-raises the underlying :class:`ReviewContractError` so
    callers that want to fail loudly can.
    """
    try:
        return PriorReviewState.from_json(raw)
    except ReviewContractError:
        if tolerate_corruption:
            return None
        raise


# ---------------------------------------------------------------------------
# JSON load helper retained for symmetry with ``review_state``
# ---------------------------------------------------------------------------


def _strict_json_loads(raw: str) -> object:
    """Internal helper kept for forward compatibility.

    Centralised so a future migration to e.g. ``orjson`` only touches
    one call site. Currently delegates to the stdlib ``json`` module so
    the action does not pull in another dependency.
    """
    return json.loads(raw)


__all__ = [
    "DEFAULT_STATE_DIRECTORY_NAME",
    "STATE_FILE_SUFFIX",
    "ReviewStateNotFound",
    "ReviewStateStore",
    "WrittenStateRecord",
    "deserialize_state_from_json",
    "load_review_state",
    "save_review_state",
    "serialize_state_to_json",
]

#!/usr/bin/env python3
"""Normalize OpenRouter review-run output into the parity-bake-off schema.

Sub-AC 2.3 deliverable. The OpenRouter+aider review path produces raw
review-run output in ``eval/runs/<eval-id>/openrouter.json`` (written by
``eval.run_openrouter_baseline``). This module provides the canonical
*normalizer* that:

1. Validates / re-shapes the raw output against the §1 envelope from
   ``eval/findings-schema.md`` (the schema defined in Sub-AC 2.1).
2. Wraps it in the per-run artifact wrapper from §4 of the same doc.
3. Mirrors the result into the test-fixture tree at
   ``tests/fixtures/candidate/<pr-id>/openrouter-findings.json`` so the
   parity-evidence comparison in Sub-AC 2.4 can be replayed from a
   fresh git clone without re-running the live action.

The normalizer is the OpenRouter analogue of
``eval.mirror_codex_baseline``: same ``pr_id`` translation
(``dotcms-core-<NUMBER>`` → ``pr-<NUMBER>``), same atomic write,
same ``--check`` mode for CI drift detection. The crucial extra step is
**schema validation** — the candidate output is run through
``cli.core.models.validate_review_payload`` before it is mirrored, so a
schema regression in the action surfaces here rather than silently
corrupting the bake-off scoring inputs.

Usage::

    # Library
    from eval.normalize_openrouter_findings import normalize_review_run_result
    normalized = normalize_review_run_result(raw_dict)

    # CLI — mirror every captured PR
    python -m eval.normalize_openrouter_findings
    python -m eval.normalize_openrouter_findings --check       # exit 1 on drift
    python -m eval.normalize_openrouter_findings --pr-id pr-35509
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_RUNS = REPO_ROOT / "eval" / "runs"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "candidate"


# ---------------------------------------------------------------------------
# Pure normalizer — raw review run → §1 envelope (schema-validated)
# ---------------------------------------------------------------------------


def normalize_review_run_result(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize raw OpenRouter review output to the §1 envelope.

    ``payload`` is the inner ``review_run_result`` from the action's
    output (the value returned by the model when ``response_format`` is
    the strict review schema). ``None`` is returned unchanged so callers
    can pipe through "pending" placeholders without special-casing.

    The function:

    * Validates ``payload`` against ``REVIEW_OUTPUT_SCHEMA`` via the
      production parser (`validate_review_payload`). Any schema
      violation raises ``ReviewContractError`` here rather than at
      bake-off scoring time.
    * Re-emits the parsed object via ``ReviewRunResult.as_dict`` so the
      output has the canonical key set / ordering, even if the raw
      payload was missing nullable fields like ``side`` (which legacy
      fixtures predating Sub-AC 3.1 omit).
    """
    if payload is None:
        return None
    # Local import — keeps `eval/` importable in environments that do
    # not have the `cli/` package on the path (e.g. doc tooling).
    from cli.core.models import validate_review_payload  # noqa: PLC0415

    parsed = validate_review_payload(payload)
    return parsed.as_dict()


# ---------------------------------------------------------------------------
# Wrapper construction — §4 per-run artifact shape
# ---------------------------------------------------------------------------


_WRAPPER_REQUIRED_KEYS = (
    "pr_id",
    "run",
    "head_sha",
    "model",
    "reasoning_effort",
    "web_search_mode",
    "prior_review_state",
    "review_run_result",
    "posted",
    "capture",
)


def _eval_id_to_fixture_id(eval_id: str) -> str:
    """``dotcms-core-35567`` → ``pr-35567`` (fixture-tree convention)."""
    if not eval_id.startswith("dotcms-core-"):
        raise ValueError(f"unexpected eval id {eval_id!r}; expected 'dotcms-core-<NUMBER>'")
    number = eval_id[len("dotcms-core-") :]
    if not number.isdigit():
        raise ValueError(f"unexpected eval id suffix {eval_id!r}")
    return f"pr-{number}"


def _load_openrouter_artifact(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top-level document must be an object")
    return data


def _build_fixture_payload(eval_artifact: Mapping[str, Any], fixture_id: str) -> dict[str, Any]:
    """Translate the eval artifact into the fixture-tree wrapper.

    The wrapper shape mirrors §4 of ``eval/findings-schema.md``:

    * ``pr_id`` is rewritten to the fixture convention so a test can
      look up the file by ``pr-<NUMBER>`` like every other fixture.
    * ``review_run_result`` is **re-validated** through
      ``normalize_review_run_result`` so the candidate fixtures cannot
      drift away from the schema even if the upstream artifact does.
    * A ``source`` block is added (mirror-only, never present in
      ``eval/runs/<id>/openrouter.json``) so a reader of the fixture
      can locate the upstream artifact without grepping.
    """
    missing = [k for k in _WRAPPER_REQUIRED_KEYS if k not in eval_artifact]
    if missing:
        raise SystemExit(f"openrouter eval artifact missing required keys: {sorted(missing)}")
    payload: dict[str, Any] = dict(eval_artifact)
    payload["pr_id"] = fixture_id

    raw_rrr = payload.get("review_run_result")
    if raw_rrr is not None and not isinstance(raw_rrr, Mapping):
        raise SystemExit("openrouter eval artifact 'review_run_result' must be an object or null")
    payload["review_run_result"] = normalize_review_run_result(raw_rrr)

    eval_pr_id = (
        eval_artifact.get("pr_id")
        if isinstance(eval_artifact.get("pr_id"), str)
        else f"dotcms-core-{fixture_id.removeprefix('pr-')}"
    )
    payload["source"] = {
        "eval_artifact": str(
            (EVAL_RUNS / str(eval_pr_id) / "openrouter.json").relative_to(REPO_ROOT)
        ),
        "produced_by": "eval.run_openrouter_baseline",
        "mirrored_by": "eval.normalize_openrouter_findings",
    }
    return payload


def _fixture_path(fixture_id: str) -> Path:
    return FIXTURE_ROOT / fixture_id / "openrouter-findings.json"


# ---------------------------------------------------------------------------
# CLI / mirror loop
# ---------------------------------------------------------------------------


def normalize_one(eval_dir: Path, *, check_only: bool) -> tuple[str, bool]:
    """Normalize and mirror one ``eval/runs/<id>/openrouter.json``.

    Returns ``(fixture_id, changed)``. ``changed`` is true when the
    on-disk fixture differs from what would be written.
    """
    artifact_path = eval_dir / "openrouter.json"
    eval_artifact = _load_openrouter_artifact(artifact_path)
    eval_id = str(eval_artifact.get("pr_id") or eval_dir.name)
    fixture_id = _eval_id_to_fixture_id(eval_id)
    fixture_path = _fixture_path(fixture_id)

    payload = _build_fixture_payload(eval_artifact, fixture_id)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if fixture_path.exists():
        existing = fixture_path.read_text(encoding="utf-8")
        if existing == rendered:
            return fixture_id, False

    if check_only:
        print(
            f"[drift] {fixture_path.relative_to(REPO_ROOT)}: would update",
            file=sys.stderr,
        )
        return fixture_id, True

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(rendered, encoding="utf-8")
    return fixture_id, True


def discover_eval_dirs() -> list[Path]:
    """Find every ``eval/runs/dotcms-core-*`` dir with an openrouter.json."""
    if not EVAL_RUNS.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(EVAL_RUNS.iterdir()):
        if not child.is_dir() or not child.name.startswith("dotcms-core-"):
            continue
        if not (child / "openrouter.json").is_file():
            continue
        out.append(child)
    return out


def _resolve_targets(pr_ids: list[str] | None) -> list[Path]:
    dirs = discover_eval_dirs()
    if not pr_ids:
        return dirs
    by_id: dict[str, Path] = {}
    for d in dirs:
        # Accept both eval id and fixture id for ergonomics.
        eval_id = d.name
        fixture_id = _eval_id_to_fixture_id(eval_id)
        by_id[eval_id] = d
        by_id[fixture_id] = d
    missing = [pid for pid in pr_ids if pid not in by_id]
    if missing:
        raise SystemExit(
            f"unknown pr_id(s): {', '.join(missing)}. Known: "
            f"{', '.join(sorted({_eval_id_to_fixture_id(d.name) for d in dirs}))}"
        )
    # Preserve user order, dedup.
    seen: set[str] = set()
    out: list[Path] = []
    for pid in pr_ids:
        d = by_id[pid]
        if d.name in seen:
            continue
        seen.add(d.name)
        out.append(d)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.normalize_openrouter_findings",
        description=(
            "Normalize eval/runs/<id>/openrouter.json against the Sub-AC 2.1 "
            "schema and mirror the result into "
            "tests/fixtures/candidate/<pr-id>/openrouter-findings.json."
        ),
    )
    parser.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help=(
            "Mirror only this PR id (eval id `dotcms-core-<N>` or "
            "fixture id `pr-<N>` accepted). May be repeated."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any fixture would be updated.",
    )
    args = parser.parse_args(argv)

    targets = _resolve_targets(args.pr_ids)
    if not targets:
        # No openrouter captures yet is *not* a failure: the action's
        # baseline is captured asynchronously to schema work, and this
        # script must remain importable / runnable on a fresh checkout.
        print(
            "[ok] no eval/runs/dotcms-core-*/openrouter.json found; nothing to normalize",
            file=sys.stderr,
        )
        return 0

    drifted: list[str] = []
    written: list[str] = []
    for eval_dir in targets:
        fixture_id, changed = normalize_one(eval_dir, check_only=args.check)
        if changed:
            (drifted if args.check else written).append(fixture_id)

    if args.check:
        if drifted:
            print(
                f"[fail] {len(drifted)} fixture(s) drifted: {', '.join(drifted)}",
                file=sys.stderr,
            )
            return 1
        print(f"[ok] {len(targets)} fixture(s) match", file=sys.stderr)
        return 0

    print(
        f"[ok] normalized {len(targets)} eval artifact(s); updated {len(written)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

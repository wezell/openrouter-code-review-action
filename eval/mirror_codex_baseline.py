#!/usr/bin/env python3
"""Mirror the captured Codex baselines into the test-fixture tree.

Sub-AC 1.3 deliverable. The Codex baseline runs live in
``eval/runs/<id>/codex.json`` (machine-readable artifacts written by
``eval.run_codex_baseline``). Sub-AC 1.3 also requires the same raw
findings output to be addressable from the test-suite-facing fixture
tree at ``tests/fixtures/baseline/<pr-id>/codex-findings.json`` so the
parity bake-off and the anchor/posting tests can read from a single
checked-in path keyed by the same ``pr-<NUMBER>`` id used by every
other fixture file.

The mirror is **idempotent and lossless**: it copies the source JSON
verbatim into the fixture path, only translating the ``id`` field
between conventions (``dotcms-core-35567`` → ``pr-35567``). Re-running
the script overwrites stale fixtures and is safe to invoke whenever
``eval/run_codex_baseline.py`` produces a fresher capture.

Usage::

    python -m eval.mirror_codex_baseline
    python -m eval.mirror_codex_baseline --check        # exit 1 if drifted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_RUNS = REPO_ROOT / "eval" / "runs"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "baseline"


def _eval_id_to_fixture_id(eval_id: str) -> str:
    """``dotcms-core-35567`` → ``pr-35567`` (the fixture-tree convention)."""
    if not eval_id.startswith("dotcms-core-"):
        raise ValueError(
            f"unexpected eval id {eval_id!r}; expected 'dotcms-core-<NUMBER>'"
        )
    number = eval_id[len("dotcms-core-") :]
    if not number.isdigit():
        raise ValueError(f"unexpected eval id suffix {eval_id!r}")
    return f"pr-{number}"


def _load_codex_artifact(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top-level document must be an object")
    return data


def _build_fixture_payload(eval_artifact: dict[str, Any], fixture_id: str) -> dict[str, Any]:
    """Translate an ``eval/runs/<id>/codex.json`` artifact for fixture use.

    The payload shape is preserved 1:1 so downstream scoring code can
    treat both copies interchangeably; only ``pr_id`` is rewritten to
    the fixture-tree convention so a test can look the file up by the
    same ``pr-<NUMBER>`` id every other fixture uses.
    """
    payload = dict(eval_artifact)
    payload["pr_id"] = fixture_id
    payload["source"] = {
        "eval_artifact": str(
            (
                EVAL_RUNS
                / f"dotcms-core-{fixture_id.removeprefix('pr-')}"
                / "codex.json"
            ).relative_to(REPO_ROOT)
        ),
        "produced_by": "eval.run_codex_baseline",
        "mirrored_by": "eval.mirror_codex_baseline",
    }
    return payload


def _fixture_path(fixture_id: str) -> Path:
    return FIXTURE_ROOT / fixture_id / "codex-findings.json"


def mirror_one(eval_dir: Path, *, check_only: bool) -> tuple[str, bool]:
    eval_artifact = _load_codex_artifact(eval_dir / "codex.json")
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
    if not EVAL_RUNS.is_dir():
        raise SystemExit(f"{EVAL_RUNS}: directory not found")
    out: list[Path] = []
    for child in sorted(EVAL_RUNS.iterdir()):
        if not child.is_dir() or not child.name.startswith("dotcms-core-"):
            continue
        if not (child / "codex.json").is_file():
            continue
        out.append(child)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.mirror_codex_baseline",
        description=(
            "Mirror eval/runs/<id>/codex.json into "
            "tests/fixtures/baseline/<pr-id>/codex-findings.json."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 1 if any fixture would be updated.",
    )
    args = parser.parse_args(argv)

    dirs = discover_eval_dirs()
    if not dirs:
        raise SystemExit(
            f"no eval/runs/dotcms-core-* directories with codex.json under {EVAL_RUNS}"
        )

    drifted: list[str] = []
    written: list[str] = []
    for eval_dir in dirs:
        fixture_id, changed = mirror_one(eval_dir, check_only=args.check)
        if changed:
            (drifted if args.check else written).append(fixture_id)

    if args.check:
        if drifted:
            print(
                f"[fail] {len(drifted)} fixture(s) drifted: {', '.join(drifted)}",
                file=sys.stderr,
            )
            return 1
        print(f"[ok] {len(dirs)} fixture(s) match", file=sys.stderr)
        return 0

    print(
        f"[ok] mirrored {len(dirs)} eval artifact(s); updated {len(written)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

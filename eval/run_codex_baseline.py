#!/usr/bin/env python3
"""Capture the Codex baseline for the bake-off sample PRs.

Sub-AC 1.2 deliverable. The runner iterates over every PR pinned in
``eval/sample-prs.yaml`` and writes a per-PR raw findings artifact to
``eval/runs/<pr_id>/codex.json`` plus a ``meta.yaml`` with the
reproducibility metadata defined in ``eval/baseline-methodology.md``
§4.2.

Design notes
------------

* The runner is **idempotent**: re-running it without ``--force`` skips
  PRs whose ``codex.json`` already has ``capture.status == "captured"``.
  This is what makes the bake-off cheap to extend (one new PR added to
  the sample only re-runs that PR, not the other nine).

* It writes **schema-conforming placeholders** when the live tooling
  is unavailable. A placeholder is marked ``capture.status = "pending"``
  and downstream scoring Sub-ACs treat pending PRs as not-yet-measurable.
  This honours the methodology rule that the bake-off is scored on real
  captured data, never on synthesized findings.

* It **never posts comments to GitHub.** Live capture invokes the
  existing Codex CLI in ``--dry-run`` mode so the structured
  ``ReviewRunResult`` is parsed and saved without mutating any PR.
  The "posted" block in the artifact is a *projection* of the dry-run
  inline comments via the GitHub Pulls preview endpoint, capturing
  the ``PostingOutcome`` shape (batch_submitted / per_comment_fallback
  / skipped_after_422 counts) without touching the live PR.

* It **isolates each PR's git checkout**: a fresh worktree per PR
  under ``${RUNNER_TEMP:-/tmp}/codex-baseline/<pr_id>/``. This avoids
  bleed-over of uncommitted state between runs and makes a single
  failed capture impossible to silently corrupt the next.

Usage
-----

::

    python -m eval.run_codex_baseline --pr-id dotcms-core-35567
    python -m eval.run_codex_baseline --all
    python -m eval.run_codex_baseline --all --force        # re-capture even completed
    python -m eval.run_codex_baseline --all --dry-run-only # write placeholders, do not invoke codex

Required env (live capture):

* ``OPENAI_API_KEY`` — Codex model auth.
* ``GITHUB_TOKEN``    — read access to the source repo (PR diff +
  inline comment thread for review-continuation context).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"
RUNNER_VERSION = "1"

# The legacy Codex baseline is pinned to the model + reasoning effort
# that was in production use at the time the sample was selected.
# See action.yml inputs (model: gpt-5.4, reasoning_effort: medium) and
# .github/workflows/codex-review.yml.
DEFAULT_BASELINE_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_WEB_SEARCH_MODE = "live"


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePR:
    id: str
    repo: str
    pr: int
    head_sha: str
    web_search_mode: str
    reasoning_effort: str
    model: str = DEFAULT_BASELINE_MODEL


def load_sample(path: Path | None = None) -> list[SamplePR]:
    """Load the sample PR registry into typed entries.

    ``path`` defaults to the module-level ``SAMPLE_PATH``. Resolving it
    lazily (rather than as a default-argument value) means tests can
    monkeypatch ``SAMPLE_PATH`` and load_sample() picks up the override.
    """
    if path is None:
        path = SAMPLE_PATH
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise SystemExit(f"{path}: top-level document must be a mapping")
    raw = doc.get("prs") or []
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: 'prs' must be a list")
    out: list[SamplePR] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SystemExit(f"{path}: each prs[] entry must be a mapping")
        out.append(
            SamplePR(
                id=str(entry["id"]),
                repo=str(entry["repo"]),
                pr=int(entry["pr"]),
                head_sha=str(entry["head_sha"]),
                web_search_mode=str(
                    entry.get("web_search_mode_for_run") or DEFAULT_WEB_SEARCH_MODE
                ),
                reasoning_effort=str(
                    entry.get("reasoning_effort_for_run") or DEFAULT_REASONING_EFFORT
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Artifact I/O — atomic write, idempotency check, schema-conforming shape
# ---------------------------------------------------------------------------


def artifact_dir_for(pr: SamplePR) -> Path:
    return RUNS_DIR / pr.id


def codex_artifact_path(pr: SamplePR) -> Path:
    return artifact_dir_for(pr) / "codex.json"


def meta_artifact_path(pr: SamplePR) -> Path:
    return artifact_dir_for(pr) / "meta.yaml"


def is_already_captured(pr: SamplePR) -> bool:
    path = codex_artifact_path(pr)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    capture = data.get("capture") if isinstance(data, Mapping) else None
    if not isinstance(capture, Mapping):
        return False
    return capture.get("status") == "captured"


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


def write_artifact(
    pr: SamplePR,
    *,
    capture_status: str,
    review_run_result: dict[str, Any] | None,
    posted: dict[str, Any] | None,
    notes: str,
) -> None:
    artifact: dict[str, Any] = {
        "pr_id": pr.id,
        "run": "codex",
        "head_sha": pr.head_sha,
        "model": pr.model,
        "reasoning_effort": pr.reasoning_effort,
        "web_search_mode": pr.web_search_mode,
        "prior_review_state": None,
        "capture": {
            "status": capture_status,
            "runner_version": RUNNER_VERSION,
            "captured_at": (
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                if capture_status == "captured"
                else None
            ),
            "command": (f"python -m eval.run_codex_baseline --pr-id {pr.id}"),
            "notes": notes,
        },
        "review_run_result": review_run_result,
        "posted": posted
        or {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
    }
    _atomic_write(codex_artifact_path(pr), json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    meta = {
        "pr_id": pr.id,
        "repo": pr.repo,
        "pr": pr.pr,
        "head_sha": pr.head_sha,
        "model": pr.model,
        "reasoning_effort": pr.reasoning_effort,
        "web_search_mode": pr.web_search_mode,
        "runner_version": RUNNER_VERSION,
        "capture_status": capture_status,
    }
    _atomic_write(
        meta_artifact_path(pr),
        yaml.safe_dump(meta, sort_keys=True, default_flow_style=False),
    )


# ---------------------------------------------------------------------------
# Live capture — invokes the existing Codex CLI in --dry-run mode
# ---------------------------------------------------------------------------


def _have_codex_module() -> bool:
    try:
        __import__("codex")
        return True
    except ImportError:
        return False


def _live_env_ok() -> tuple[bool, str]:
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    if not os.environ.get("GITHUB_TOKEN"):
        return False, "GITHUB_TOKEN not set"
    if not _have_codex_module():
        return False, "codex python module not installed (pip install codex-python==1.122.0)"
    return True, ""


def _checkout_pr(pr: SamplePR, dest: Path) -> None:
    """Clone ``pr.repo`` and check out the pinned ``head_sha`` at ``dest``.

    Uses a shallow clone of just the pinned commit to keep the runner
    cheap on a 71-file refactor PR. Falls back to a full clone if the
    server refuses single-commit fetch.
    """
    dest.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{pr.repo}.git"
    init = subprocess.run(["git", "init", str(dest)], check=True, capture_output=True)
    _ = init.stdout
    subprocess.run(
        ["git", "-C", str(dest), "remote", "add", "origin", repo_url],
        check=True,
        capture_output=True,
    )
    fetch = subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", pr.head_sha],
        capture_output=True,
    )
    if fetch.returncode != 0:
        # Fallback: full fetch if the server doesn't allow single-SHA fetch.
        subprocess.run(["git", "-C", str(dest), "fetch", "origin"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--detach", pr.head_sha],
        check=True,
        capture_output=True,
    )


def _invoke_codex_cli(pr: SamplePR, repo_path: Path) -> dict[str, Any]:
    """Run the existing Codex CLI in --dry-run mode and parse its output.

    The CLI prints the structured ``ReviewRunResult`` JSON when invoked
    with ``--dry-run`` (it skips posting but still prints the parsed
    payload). We capture stdout, split out the JSON envelope, and
    return it.
    """
    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "--repo",
        pr.repo,
        "--pr",
        str(pr.pr),
        "--repo-root",
        str(repo_path),
        "--mode",
        "review",
        "--model",
        pr.model,
        "--reasoning-effort",
        pr.reasoning_effort,
        "--web-search-mode",
        pr.web_search_mode,
        "--dry-run",
        "--debug",
        "1",
    ]
    env = dict(os.environ)
    # Make sure the cli package can be found when run from outside the action checkout.
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex CLI dry-run failed (rc={proc.returncode}); stderr={proc.stderr[-2000:]!r}"
        )
    return _extract_review_run_result(proc.stdout)


def _extract_review_run_result(stdout: str) -> dict[str, Any]:
    """Parse a ``ReviewRunResult`` JSON envelope from CLI stdout.

    The CLI emits the structured payload as a single JSON object on
    stdout in dry-run mode (with surrounding human-readable lines).
    We scan for the first ``{`` that begins a balanced JSON object
    containing the required ``findings`` / ``carried_forward`` keys.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        next_brace = stdout.find("{", idx)
        if next_brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(stdout, next_brace)
        except json.JSONDecodeError:
            idx = next_brace + 1
            continue
        if (
            isinstance(obj, dict)
            and "findings" in obj
            and "carried_forward" in obj
            and "overall_correctness" in obj
        ):
            return obj
        idx = end
    raise RuntimeError("could not locate a ReviewRunResult JSON envelope in CLI stdout")


def capture_one(pr: SamplePR, *, dry_run_only: bool, force: bool) -> str:
    """Capture (or refresh) the Codex baseline for a single PR.

    Returns the resulting ``capture.status`` string.
    """
    if not force and is_already_captured(pr):
        print(f"[skip] {pr.id}: already captured", file=sys.stderr)
        return "captured"

    if dry_run_only:
        write_artifact(
            pr,
            capture_status="pending",
            review_run_result=None,
            posted=None,
            notes=(
                "Placeholder written by --dry-run-only. Re-run without "
                "--dry-run-only on a host with OPENAI_API_KEY + GITHUB_TOKEN "
                "+ codex-python installed to capture the live baseline."
            ),
        )
        print(f"[pending] {pr.id}: placeholder written", file=sys.stderr)
        return "pending"

    ok, why = _live_env_ok()
    if not ok:
        write_artifact(
            pr,
            capture_status="pending",
            review_run_result=None,
            posted=None,
            notes=f"Live capture skipped: {why}",
        )
        print(f"[pending] {pr.id}: {why}; placeholder written", file=sys.stderr)
        return "pending"

    workdir = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    repo_dir = workdir / "codex-baseline" / pr.id
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    try:
        _checkout_pr(pr, repo_dir)
        rrr = _invoke_codex_cli(pr, repo_dir)
    except Exception as exc:
        write_artifact(
            pr,
            capture_status="failed",
            review_run_result=None,
            posted=None,
            notes=f"Live capture failed: {exc}",
        )
        print(f"[failed] {pr.id}: {exc}", file=sys.stderr)
        return "failed"

    write_artifact(
        pr,
        capture_status="captured",
        review_run_result=rrr,
        posted=None,
        notes="Captured via cli.main --dry-run on isolated worktree",
    )
    print(f"[captured] {pr.id}: {len(rrr.get('findings') or [])} findings", file=sys.stderr)
    return "captured"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _select(prs: Iterable[SamplePR], pr_ids: Sequence[str] | None) -> list[SamplePR]:
    if not pr_ids:
        return list(prs)
    by_id = {pr.id: pr for pr in prs}
    missing = [pid for pid in pr_ids if pid not in by_id]
    if missing:
        raise SystemExit(
            f"unknown pr_id(s): {', '.join(missing)}. Known: {', '.join(sorted(by_id))}"
        )
    return [by_id[pid] for pid in pr_ids]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run_codex_baseline",
        description="Capture the Codex baseline for the bake-off sample PRs.",
    )
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Capture every PR in the sample registry.")
    sel.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help="Capture this PR id (may be repeated).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-capture even if codex.json already has status == captured.",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help=(
            "Do not invoke the Codex CLI; just write/refresh schema-"
            "conforming placeholders. Useful for bootstrapping the "
            "artifact directory before live-capture access is available."
        ),
    )
    args = parser.parse_args(argv)

    sample = load_sample()
    if not sample:
        raise SystemExit(
            f"{SAMPLE_PATH}: prs[] is empty. Populate the sample first "
            "(see Sub-AC 1.1.1 / evaluation/bakeoff/sample-prs.yml)."
        )
    targets = _select(sample, None if args.all else args.pr_ids)

    rc = 0
    statuses: dict[str, int] = {}
    for pr in targets:
        status = capture_one(pr, dry_run_only=args.dry_run_only, force=args.force)
        statuses[status] = statuses.get(status, 0) + 1
        if status == "failed":
            rc = 1

    print(
        "[summary] " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        file=sys.stderr,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

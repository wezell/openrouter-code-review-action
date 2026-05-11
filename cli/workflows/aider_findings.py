"""Translate review findings into aider edit instructions.

Sub-AC 7.2.2: this module is the bridge between the review pipeline's
structured ``ReviewFinding`` objects (and the unresolved review threads
GitHub returns for re-runs) and the ``aider`` CLI wrapper in
:mod:`cli.clients.aider_client`. The wrapper expects three things:

1. A natural-language ``message`` describing what to change.
2. A list of files aider should add to its chat context (file scoping —
   aider only edits files it has been shown).
3. Any `````suggestion`` blocks the reviewer included, hoisted out of
   the finding body so the model does not have to re-derive them.

Design constraints
------------------
* **Pure / side-effect-free.** This module only manipulates strings and
  dataclasses; it never touches git, the filesystem, or the network.
  That keeps it trivially unit-testable and lets the workflow layer
  decide how to combine the output with branch state.
* **No backwards-compat with Codex's ``apply_patch`` instructions.** The
  prompt is shaped for aider — it tells the model to edit files in
  place using aider's own tooling and to emit `````python``-style
  diff suggestions only when the user explicitly asked. We do **not**
  re-emit the legacy ``apply_patch`` rubric.
* **Schema lives in the body.** OpenRouter's review schema does not
  carve out a dedicated ``suggestion`` field; suggestion blocks are
  embedded in ``finding.body``. We extract them with the same fence
  rule used by ``cli.review.comment_validator`` (`````suggestion`` …
  ```````) so any reviewer-flagged suggestion automatically flows
  through to act mode.
* **File scoping deduplicates while preserving order.** aider tolerates
  duplicate ``--file`` args but emits warnings; we feed it a clean
  list. The first-seen order is preserved so the most-relevant file
  (the one the reviewer flagged first) lands at the top of aider's
  chat context.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..core.models import ReviewFinding

# Matches a fenced ```suggestion … ``` block. We use DOTALL so the body
# of the suggestion (which may include newlines) is captured greedily
# *until* the next closing fence on its own line. The leading fence
# must be the start of a line so prose mentions of "```suggestion" do
# not accidentally trigger extraction.
_SUGGESTION_BLOCK_RE = re.compile(
    r"(?m)^[ \t]*```suggestion[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?^[ \t]*```[ \t]*$",
    re.DOTALL,
)

# Hard cap on the prompt length we hand to aider. Aider streams the
# prompt as a single user message; very long prompts blow context
# windows and cost money. ``ReviewWorkflow`` already truncates oversize
# finding bodies during posting, so this cap is a defense-in-depth
# safety net rather than a primary truncator.
_PROMPT_CHAR_LIMIT = 24_000

# Per-finding body cap so a single noisy finding cannot dominate the
# prompt budget. Mirrors the GitHub inline-comment payload limit.
_FINDING_BODY_CHAR_LIMIT = 3_500


@dataclass(frozen=True)
class FindingSuggestion:
    """A single ```suggestion block extracted from a review finding."""

    file_path: str
    start_line: int
    end_line: int
    replacement: str

    @property
    def is_multi_line(self) -> bool:
        return self.end_line > self.start_line


@dataclass(frozen=True)
class AiderEditPlan:
    """Output of :func:`build_aider_edit_plan`.

    ``message`` is the prompt aider runs as a one-shot. ``files`` is the
    deduplicated, repo-relative file list aider should ``--add`` to its
    chat context. ``suggestions`` carries every ```suggestion block we
    pulled out of the finding bodies, in case the workflow layer wants
    to log them separately or feed them into a smaller targeted call.
    """

    message: str
    files: tuple[str, ...]
    suggestions: tuple[FindingSuggestion, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.message.strip() or not self.files


# ---------------------------------------------------------------------------
# Suggestion extraction
# ---------------------------------------------------------------------------


def extract_suggestion_blocks(body: str) -> list[str]:
    """Return every ```suggestion block payload found in ``body``.

    The returned list contains the *inner* text of each suggestion fence
    (no leading/trailing fence lines), in source order. Blocks with an
    empty body are dropped — GitHub renders an empty suggestion as a
    deletion which is rarely what the reviewer intended.
    """
    if not body:
        return []
    return [
        match.group("body")
        for match in _SUGGESTION_BLOCK_RE.finditer(body)
        if match.group("body").strip()
    ]


def extract_finding_suggestions(finding: ReviewFinding) -> list[FindingSuggestion]:
    """Lift suggestion blocks out of ``finding.body`` into structured records."""
    blocks = extract_suggestion_blocks(finding.body)
    if not blocks:
        return []
    location = finding.code_location
    return [
        FindingSuggestion(
            file_path=_normalize_path(location.absolute_file_path),
            start_line=location.start_line,
            end_line=location.end_line,
            replacement=block.rstrip("\n"),
        )
        for block in blocks
    ]


# ---------------------------------------------------------------------------
# File scoping
# ---------------------------------------------------------------------------


def collect_finding_files(findings: Iterable[ReviewFinding]) -> list[str]:
    """Return the deduplicated, first-seen-order file scope for ``findings``.

    Paths are normalised (``./`` stripped, leading slash removed) so the
    aider wrapper can pass them straight through to the subprocess. Any
    finding with a missing or whitespace-only path is skipped — those
    cannot be addressed by aider without a target file anyway.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for finding in findings:
        path = _normalize_path(finding.code_location.absolute_file_path)
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _normalize_path(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    candidate = raw.strip()
    if not candidate:
        return ""
    # Strip a leading ``./`` so paths are repo-relative and dedupe
    # correctly. We deliberately do NOT call ``Path.resolve()`` — that
    # would touch the filesystem; this module is pure.
    if candidate.startswith("./"):
        candidate = candidate[2:]
    if candidate.startswith("/"):
        candidate = candidate.lstrip("/")
    # PurePath normalisation collapses ``a/./b`` and ``a//b`` without
    # consulting the FS.
    return Path(candidate).as_posix()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def render_finding_block(
    finding: ReviewFinding,
    *,
    index: int,
    body_limit: int = _FINDING_BODY_CHAR_LIMIT,
) -> str:
    """Render a single ``ReviewFinding`` as an XML-ish block for the prompt.

    The block carries the file path, line range, title, and full body
    (truncated at ``body_limit`` to keep the prompt within budget). We
    keep suggestion fences inline so aider sees the exact replacement
    the reviewer proposed; the fence makes them easy for the model to
    recognise.
    """
    location = finding.code_location
    path = _normalize_path(location.absolute_file_path) or "<unknown>"
    start = location.start_line
    end = location.end_line if location.end_line >= start else start
    body = finding.body.strip()
    if len(body) > body_limit:
        body = body[:body_limit].rstrip() + "\n\n… (truncated)"

    title = finding.title.strip() or "(untitled finding)"
    priority = "" if finding.priority is None else f' priority="{finding.priority}"'

    return (
        f'<finding index="{index}" path="{path}" '
        f'start_line="{start}" end_line="{end}"{priority}>\n'
        f"<title>{title}</title>\n"
        f"<body>\n{body}\n</body>\n"
        "</finding>"
    )


def build_aider_message(
    findings: Sequence[ReviewFinding],
    *,
    extra_instructions: str | None = None,
    char_limit: int = _PROMPT_CHAR_LIMIT,
) -> str:
    """Compose the natural-language ``--message`` payload for aider.

    The prompt has three sections, in order:

    1. ``<act_overview>`` — what aider's job is.
    2. ``<extra_instructions>`` — caller-supplied free text (optional).
    3. ``<findings>`` — every finding rendered with file/line/body.

    The output is hard-capped at ``char_limit`` — if rendering all
    findings would exceed it, later findings are dropped with a note so
    the model knows the list was truncated. Truncation order matches
    the input order, so callers can pre-sort by priority if desired.
    """
    overview = (
        "<act_overview>\n"
        "You are aider, running in non-interactive act mode for a code review.\n"
        "Apply the smallest reasonable diff that resolves every finding listed below.\n"
        "Only edit files inside the chat context. Do not change unrelated code.\n"
        "If a finding includes a ```suggestion``` block, treat it as the reviewer's\n"
        "preferred replacement for the indicated line range — apply it verbatim\n"
        "unless it would obviously break the build.\n"
        "Do not run git commit or git push; the host workflow handles that after\n"
        "your edits land on disk.\n"
        "</act_overview>\n"
    )

    sections: list[str] = [overview]

    instructions = (extra_instructions or "").strip()
    if instructions:
        sections.append("<extra_instructions>\n" + instructions + "\n</extra_instructions>\n")

    if not findings:
        sections.append("<findings>\n(no findings supplied)\n</findings>\n")
        return "".join(sections)

    rendered_blocks: list[str] = []
    truncated = False
    running = sum(len(part) for part in sections) + len("<findings>\n</findings>\n")
    for index, finding in enumerate(findings, start=1):
        block = render_finding_block(finding, index=index)
        # +1 for the joining newline between blocks.
        if running + len(block) + 1 > char_limit and rendered_blocks:
            truncated = True
            break
        rendered_blocks.append(block)
        running += len(block) + 1

    body = "<findings>\n" + "\n".join(rendered_blocks) + "\n"
    if truncated:
        body += (
            f"<note>Truncated after {len(rendered_blocks)} of {len(findings)} findings "
            "to fit the prompt budget. Re-run act mode after addressing these to pick "
            "up the rest.</note>\n"
        )
    body += "</findings>\n"
    sections.append(body)

    return "".join(sections)


def build_aider_edit_plan(
    findings: Sequence[ReviewFinding],
    *,
    extra_instructions: str | None = None,
    char_limit: int = _PROMPT_CHAR_LIMIT,
) -> AiderEditPlan:
    """One-shot translation of review findings into an :class:`AiderEditPlan`.

    The returned plan is shaped for direct consumption by
    :class:`cli.clients.aider_client.AiderClient`:

    * ``plan.message`` → ``AiderClient.run(..., message=...)``
    * ``plan.files``   → ``AiderClient.run(..., files=...)``

    ``plan.suggestions`` is informational — the wrapper does not need
    it, but the workflow layer can log it or surface it in the PR
    reply for transparency.
    """
    files = tuple(collect_finding_files(findings))
    suggestions: list[FindingSuggestion] = []
    for finding in findings:
        suggestions.extend(extract_finding_suggestions(finding))
    message = build_aider_message(
        findings,
        extra_instructions=extra_instructions,
        char_limit=char_limit,
    )
    return AiderEditPlan(
        message=message,
        files=files,
        suggestions=tuple(suggestions),
    )

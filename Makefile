.PHONY: help fmt lint type qa hooks hooks-run eval-overlap eval-overlap-check eval-parity-report eval-parity-report-check

PY_SRC := cli

# Sub-AC 4.1: allow CI to soft-pass while sample fixtures are still
# placeholder. Set ALLOW_UNMEASURABLE=0 (or unset) once the bake-off
# baselines are populated.
ALLOW_UNMEASURABLE ?= 1
OVERLAP_THRESHOLD ?= 0.80
OVERLAP_METRIC ?= micro.recall
ALLOW_UNMEASURABLE_FLAG := $(if $(filter 1 true yes,$(ALLOW_UNMEASURABLE)),--allow-unmeasurable,)

help:
	@echo "Targets:"
	@echo "  fmt              - Format code (ruff format)"
	@echo "  lint             - Lint and autofix (ruff check --fix)"
	@echo "  type             - Type-check (mypy)"
	@echo "  qa               - Run fmt, lint, and type"
	@echo "  hooks            - Install pre-commit hooks"
	@echo "  hooks-run        - Run pre-commit on all files"
	@echo "  eval-overlap     - Regenerate overlap pipeline + enforce 80% gate"
	@echo "  eval-overlap-check - Drift-check overlap pipeline + enforce gate"
	@echo "  eval-parity-report - Build consolidated Markdown+JSON parity report (Sub-AC 4.2)"
	@echo "  eval-parity-report-check - Drift-check the consolidated parity report"

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy .

fmt:
	uv run ruff format $(PY_SRC)

type:
	uv run mypy $(PY_SRC)

qa: fmt lint type

hooks:
	uv run pre-commit install

hooks-run:
	uv run pre-commit run --all-files

# Sub-AC 4.1: end-to-end overlap pipeline + threshold gate.
# 1. Regenerate per-PR + aggregate artifacts so the gate reads fresh data.
# 2. Drift-check guards committed artifacts vs. fixtures.
# 3. Threshold gate fails CI when sample-wide overlap < 80%.
eval-overlap:
	uv run python -m eval.overlap_score
	uv run python -m eval.overlap_score --check
	uv run python -m eval.check_overlap_threshold \
		--threshold $(OVERLAP_THRESHOLD) \
		--metric $(OVERLAP_METRIC) \
		$(ALLOW_UNMEASURABLE_FLAG)

# Read-only variant for CI: skip regeneration, just enforce both gates.
eval-overlap-check:
	uv run python -m eval.overlap_score --check
	uv run python -m eval.check_overlap_threshold \
		--threshold $(OVERLAP_THRESHOLD) \
		--metric $(OVERLAP_METRIC) \
		$(ALLOW_UNMEASURABLE_FLAG)

# Sub-AC 4.2: consolidated parity report. Merges per-PR overlap, the
# aggregate rollup, and the Codex-vs-OpenRouter delta into a single
# Markdown artifact (eval/parity_report.md) plus a JSON sibling
# (eval/parity_report.json) that CI uploads via actions/upload-artifact.
eval-parity-report:
	uv run python -m eval.parity_report \
		--threshold $(OVERLAP_THRESHOLD) \
		--metric $(OVERLAP_METRIC) \
		$(ALLOW_UNMEASURABLE_FLAG)

eval-parity-report-check:
	uv run python -m eval.parity_report --check \
		--threshold $(OVERLAP_THRESHOLD) \
		--metric $(OVERLAP_METRIC) \
		$(ALLOW_UNMEASURABLE_FLAG)

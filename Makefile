# Makefile for deep-research
#
# Quality/security suite:
#   make lint      - ruff check on package + tests (matches CI)
#   make typecheck - mypy on the package (matches CI)
#   make test      - pytest with coverage (term + xml + html) + end-to-end smoke test
#   make codeql    - build a CodeQL database for the package and analyze it (SARIF)
#   make opengrep  - run OpenGrep static analysis (JSON)
#   make scan      - codeql + opengrep
#   make all       - lint + typecheck + test + scan (full suite)
#   make clean     - remove generated reports and caches
#
# Lint, typecheck, and test failures DO fail the build (same as CI).
# Security scans (codeql, opengrep) are informational and never fail the target.

# bash is required for the ANSI-C quoted LGTM_INDEX_FILTERS below.
SHELL := /bin/bash

PYTHON      ?= .venv/bin/python
PKG         := deep_research
TESTS       := tests
REPORTS_DIR := reports

RUFF        ?= $(PYTHON) -m ruff
MYPY        ?= $(PYTHON) -m mypy

CODEQL         ?= codeql
CODEQL_DB      := $(REPORTS_DIR)/codeql-db
CODEQL_SARIF   := $(REPORTS_DIR)/codeql.sarif
CODEQL_QUERIES := codeql/python-queries

OPENGREP        ?= opengrep
OPENGREP_CONFIG := p/python
OPENGREP_JSON   := $(REPORTS_DIR)/opengrep.json

.DEFAULT_GOAL := help
.PHONY: all lint typecheck test coverage smoke codeql opengrep scan clean help

help:
	@echo "deep-research quality/security targets:"
	@echo "  make lint      - ruff check on $(PKG)/ and $(TESTS)/"
	@echo "  make typecheck - mypy on $(PKG)/"
	@echo "  make test      - pytest + coverage (term/xml/html) and the smoke test"
	@echo "  make codeql    - CodeQL database build + analysis -> $(CODEQL_SARIF)"
	@echo "  make opengrep  - OpenGrep static analysis -> $(OPENGREP_JSON)"
	@echo "  make scan      - codeql + opengrep"
	@echo "  make all       - lint + typecheck + test + scan (full suite)"
	@echo "  make clean     - remove $(REPORTS_DIR)/ and test/coverage caches"

$(REPORTS_DIR):
	@mkdir -p $(REPORTS_DIR)

# --- Lint -------------------------------------------------------------------
lint:
	$(RUFF) check $(PKG)/ $(TESTS)/

# --- Type check -------------------------------------------------------------
typecheck:
	$(MYPY) $(PKG)/

# --- Tests + coverage -------------------------------------------------------
# `coverage` is a convenience alias for `test`.
test coverage: | $(REPORTS_DIR)
	$(PYTHON) -m pytest \
		--cov=$(PKG) \
		--cov-report=term-missing \
		--cov-report=xml:$(REPORTS_DIR)/coverage.xml \
		--cov-report=html:$(REPORTS_DIR)/htmlcov
	@echo ">> Coverage written to $(REPORTS_DIR)/coverage.xml and $(REPORTS_DIR)/htmlcov/"

# End-to-end smoke test only (drives Coordinator.stream() against mocked OWUI).
# It is part of the full `make test` run; this target runs just that module.
smoke:
	$(PYTHON) -m pytest tests/test_smoke_e2e.py -v

# --- CodeQL -----------------------------------------------------------------
# Index only our package (LGTM_INDEX_FILTERS keeps .venv / vendored files out of
# the database), then run the standard Python security+quality query suite.
codeql: | $(REPORTS_DIR)
	@echo ">> Building CodeQL database (scoped to $(PKG)/) ..."
	LGTM_INDEX_FILTERS=$$'exclude:**/*\ninclude:$(PKG)/**' \
		$(CODEQL) database create $(CODEQL_DB) \
			--language=python --source-root=. --overwrite --threads=0
	@echo ">> Analyzing with $(CODEQL_QUERIES) ..."
	$(CODEQL) database analyze $(CODEQL_DB) $(CODEQL_QUERIES) \
		--download --format=sarif-latest --output=$(CODEQL_SARIF) --threads=0
	@$(PYTHON) -c 'import json; r=json.load(open("$(CODEQL_SARIF)"))["runs"][0]; print(">> CodeQL:", len(r.get("results", [])), "result(s) ->", "$(CODEQL_SARIF)")'

# --- OpenGrep ---------------------------------------------------------------
# Leading "-" so findings (non-zero exit) do not fail the target.
opengrep: | $(REPORTS_DIR)
	@echo ">> Running OpenGrep ($(OPENGREP_CONFIG)) ..."
	-$(OPENGREP) scan --config $(OPENGREP_CONFIG) --quiet \
		--json --output $(OPENGREP_JSON) $(PKG)
	@test -f $(OPENGREP_JSON) \
		&& $(PYTHON) -c 'import json; d=json.load(open("$(OPENGREP_JSON)")); print(">> OpenGrep:", len(d.get("results", [])), "finding(s),", len(d.get("errors", [])), "error(s) ->", "$(OPENGREP_JSON)")' \
		|| echo ">> OpenGrep: no JSON output produced"

# --- Aggregates -------------------------------------------------------------
scan: codeql opengrep

all: lint typecheck test scan
	@echo ">> All checks complete. Reports in $(REPORTS_DIR)/"

clean:
	rm -rf $(REPORTS_DIR) .coverage .coverage.* .pytest_cache

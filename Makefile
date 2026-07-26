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
# Release:
#   make release       - tag the current main with the pyproject version and push
#   make release-check - run the release preflight checks without tagging
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

RELEASE_REMOTE ?= gitlab
VERSION        := $(shell awk -F'"' '/^version = / {print $$2; exit}' pyproject.toml)
RELEASE_TAG    := v$(VERSION)

.DEFAULT_GOAL := help
.PHONY: all lint typecheck test coverage smoke codeql opengrep scan clean help \
        release release-check

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
	@echo ""
	@echo "release targets:"
	@echo "  make release-check - preflight only: verify main is clean, synced and untagged"
	@echo "  make release       - tag $(RELEASE_TAG) on $(RELEASE_REMOTE) and push it"

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

# --- Release ----------------------------------------------------------------
# Releases are tag-driven. Bump `version` in pyproject.toml inside the merge
# request; once it is merged, run `make release` on main.
#
# The tag is DERIVED from pyproject.toml rather than typed by hand. That is the
# point of this target: the docker workflows read the version out of
# pyproject.toml at the tagged ref, so a hand-typed tag that disagrees with the
# file would publish images whose version silently contradicts their tag.
#
# Pushing the tag to GitLab mirrors it to GitHub, where docker-mcp-server.yml
# and docker-openapi-tool.yml trigger on `push: tags: ["v*.*.*"]` and publish
# mcp-$(VERSION) and tool-$(VERSION).
#
# Override the remote with: make release RELEASE_REMOTE=origin
#
# Note: no tests run here. release-check requires local main to equal
# $(RELEASE_REMOTE)/main, so the CI that already ran against that exact commit
# is the relevant signal; re-running it locally would only duplicate it.

release-check:
	@if [ -z "$(VERSION)" ]; then \
		echo "!! could not read version from pyproject.toml"; exit 1; fi
	@if ! git remote get-url $(RELEASE_REMOTE) >/dev/null 2>&1; then \
		echo "!! no remote '$(RELEASE_REMOTE)' (override with RELEASE_REMOTE=...)"; exit 1; fi
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$branch" != "main" ]; then \
		echo "!! on '$$branch'; releases are cut from main"; exit 1; fi
	@if ! git diff-index --quiet HEAD --; then \
		echo "!! working tree is dirty; commit or stash first"; exit 1; fi
	@git fetch -q $(RELEASE_REMOTE) main
	@if [ "$$(git rev-parse HEAD)" != "$$(git rev-parse $(RELEASE_REMOTE)/main)" ]; then \
		echo "!! local main differs from $(RELEASE_REMOTE)/main; pull or push first"; exit 1; fi
	@if git rev-parse -q --verify "refs/tags/$(RELEASE_TAG)" >/dev/null; then \
		echo "!! tag $(RELEASE_TAG) already exists locally"; exit 1; fi
	@if [ -n "$$(git ls-remote --tags $(RELEASE_REMOTE) refs/tags/$(RELEASE_TAG))" ]; then \
		echo "!! tag $(RELEASE_TAG) already exists on $(RELEASE_REMOTE)"; exit 1; fi
	@echo ">> ready to release $(RELEASE_TAG) from $$(git rev-parse --short HEAD)"

release: release-check
	git tag -a "$(RELEASE_TAG)" -m "Release $(RELEASE_TAG)"
	@# If the push fails the local tag must go, otherwise it is left behind and the
	@# next `make release` trips its own "tag already exists locally" guard, turning
	@# a transient network error into a confusing manual cleanup.
	@if ! git push $(RELEASE_REMOTE) "$(RELEASE_TAG)"; then \
		echo "!! push failed; removing local tag $(RELEASE_TAG) so the release can be retried"; \
		git tag -d "$(RELEASE_TAG)"; \
		exit 1; \
	fi
	@echo ">> pushed $(RELEASE_TAG); GitHub will publish mcp-$(VERSION) and tool-$(VERSION)"

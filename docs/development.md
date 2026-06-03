# Development

## Local setup

```bash
git clone <this repo> && cd open-webui-deep-research-tool
uv sync --all-groups
# or: python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install pytest pytest-asyncio respx hypothesis ruff mypy pre-commit
.venv/bin/pre-commit install
```

## Day-to-day

```bash
# Tests
.venv/bin/pytest

# End-to-end smoke test (all 9 phases, every OWUI endpoint mocked with respx)
make smoke
# = .venv/bin/python -m pytest tests/test_smoke_e2e.py -v

# Lint + format
.venv/bin/ruff check deep_research/ tests/
.venv/bin/ruff format deep_research/ tests/

# Type check
.venv/bin/mypy deep_research/

# Security scans (also run in CI)
opengrep scan --config=auto deep_research/
codeql database create /tmp/dr-db --language=python --source-root=. --overwrite
codeql database analyze /tmp/dr-db codeql/python-queries --format=sarif-latest --output=/tmp/dr.sarif
```

The smoke test is the fastest way to catch full-pipeline regressions —
narrower unit tests miss bugs that only show up under the full coroutine
flow. It drives `Coordinator.run(sink=...)` (the exact coroutine the
OpenAPI `POST /research` endpoint uses) end-to-end with every OWUI REST
endpoint mocked via `respx`, and runs as part of `pytest`.

## Repository layout

- `deep_research/` — the active package.
- `CLAUDE.md` — agent context, including the OWUI REST response-shape
  table and concurrency contract notes.
- `tests/` — unit tests for caches, text utilities, the EventBus, the env
  loader, the adapter (with `respx` mocks), the Coordinator (inflight
  dedupe + lifecycle), and budget windows.
- `deploy/` — the local manual-test Compose stack (see
  [`deploy/README.md`](../deploy/README.md)).

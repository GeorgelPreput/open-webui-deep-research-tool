# Deep Research for Open WebUI

A multi-cycle web research engine that plans an outline, drives iterative search/fetch/compress cycles against Open WebUI's built-in web search and retrieval endpoints, and produces a synthesised, citation-verified report with a persisted knowledge base.

The same core ships in four runtimes:

| Runtime | When to use it |
|---|---|
| **OWUI Function** | You run Open WebUI yourself and want Deep Research to appear as a model in the chat dropdown. |
| **OWUI Pipelines plugin** | You can't drop new Functions into the OWUI container but you do run a sidecar Pipelines container. |
| **OpenAPI Tool server** | Anything that can call HTTP + SSE — non-OWUI clients, internal services, CI jobs. |
| **MCP server** | Anything that speaks Streamable-HTTP MCP — Claude Desktop, Cline, an MCP-aware orchestrator. |

All four runtimes call back into Open WebUI over its REST API (`/api/chat/completions`, `/api/embeddings`, `/api/v1/retrieval/*`, `/api/v1/files/*`, `/api/v1/knowledge/*`, `/api/v1/chats/*`). There are no `open_webui.*` Python imports anywhere in the package — Open WebUI is treated purely as a remote service.

> Core algorithms (semantic eigendecomposition, multi-cycle research dimensions, preference direction vectors, gap-vector exploration) are adapted from [atineiatte/deep-research-at-home](https://github.com/atineiatte/deep-research-at-home). The packaging, runtime split, REST adapter, event bus, and persistence layer are this project's contributions.

---

## What it actually does

A run goes through nine phases:

1. **Rehydrate** — load checkpoint from `chat.deepResearch`, provision a per-conversation OWUI knowledge base, restore previous state.
2. **Outline feedback continuation** — if the previous turn was waiting for user feedback on the outline, fold the feedback into the topic graph.
3. **Initial queries** — detect follow-up vs fresh research, generate seed search queries, kick off the first cycle. If `interactive_research` is on and this is a fresh run, emit the proposed outline and stop here for one turn to let the user adjust.
4. **Outline** — produce a synthesis outline (the structure of the final report) from the research outline and initial results.
5. **Cycles** — the core loop. Each cycle: rank under-covered topics by gap vector + trajectory, generate fresh queries, fetch/process N URLs (REST extraction first, paywall-spoof fallback for PDFs, archive.org rescue on 403), filter by similarity to a learned preference direction vector, run quality filtering through a small LLM. Stops between `min_cycles` and `max_cycles` based on coverage.
6. **Compress** — stepped semantic compression of the results corpus, eigendecomposition-driven so chunks aligned with the research trajectory survive.
7. **Synthesise** — section-by-section content generation against the synthesis outline, with inline citations correlated to the master source table.
8. **Front/back matter** — titles, abstract, introduction, conclusion, review pass with edit application.
9. **Finalise** — assemble the full report, run citation verification, persist final report to the KB, export research data if enabled.

The whole run emits structured progress events (status pill, message chunk, embed iframe) over an `EventBus` which the entrypoint translates to OWUI's `__event_emitter__` payloads, an SSE stream, or whatever the host speaks.

---

## Requirements

- **Open WebUI** ≥ 0.9.5 recommended (older works too, with degraded progress UI — see [Compatibility](#compatibility)).
- An OWUI admin API key (`sk-...`) with permission to embeddings, chat completions, retrieval, files, knowledge, and chat persistence.
- A search provider configured in OWUI Settings → Web Search.
- An embedding model registered in OWUI Settings → Models (used for similarity, eigendecomposition, query-trajectory math).
- Python 3.11+ if you're running the OpenAPI Tool / MCP / Pipelines runtimes yourself.

---

## Installation

### 1. OWUI Function (in-container)

The OWUI Function runtime needs the `deep_research/` package importable from inside the OWUI container.

**Option A — install the wheel into OWUI's container:**

```bash
# Inside the OWUI container (or via a custom Dockerfile that extends OWUI):
pip install /path/to/deep-research-0.2.0-py3-none-any.whl
```

Then in OWUI: **Admin Panel → Functions → New Function → upload `deep_research/entrypoints/owui_function/pipe.py`** (just the shim file — its imports resolve from the wheel installed above).

**Option B — flatten the package into a single file** using a tool like the `inline-python-modules` skill. This produces one `.py` file you upload to OWUI Functions verbatim. Useful for hosted-OWUI instances where you can't `pip install` into the container.

After install, the Function appears as a model called `Deep Research` in the chat model dropdown.

### 2. OWUI Pipelines plugin (sidecar container)

Spin up the Pipelines sidecar pointed at your OWUI:

```yaml
# docker-compose.yml fragment
services:
  pipelines:
    image: ghcr.io/open-webui/pipelines:main
    ports:
      - "9099:9099"
    environment:
      PIPELINES_API_KEY: "0p3n-w3bu!"
      DR_OWUI_BASE_URL: "http://open-webui:8080"
      DR_OWUI_API_KEY: "sk-..."
    volumes:
      - ./deep_research:/app/pipelines/pipelines/deep_research
      - ./deep_research/entrypoints/owui_pipeline/pipeline.py:/app/pipelines/pipelines/deep_research_pipeline.py
      - pipelines-data:/app/pipelines/data
volumes:
  pipelines-data:
```

In OWUI: **Settings → Connections → Add OpenAI-compatible** pointing at `http://pipelines:9099`. The Deep Research pipeline shows up in the model dropdown.

### 3. OpenAPI Tool server (Docker)

Standalone REST service that exposes `POST /research` (SSE stream) and `GET /health`.

```bash
docker build -t deep-research-openapi -f deep_research/entrypoints/openapi_tool/Dockerfile .
docker run -p 8000:8000 \
  -e DR_OWUI_BASE_URL=http://host.docker.internal:8080 \
  -e DR_OWUI_API_KEY=sk-... \
  -e DR_DATA_DIR=/data/deep_research \
  -v dr-data:/data/deep_research \
  deep-research-openapi
```

Smoke test:

```bash
curl -N http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Summarise the state of post-quantum cryptography in 2026."}'
```

### 4. MCP server (Docker, Streamable-HTTP)

```bash
docker build -t deep-research-mcp -f deep_research/entrypoints/mcp/Dockerfile .
docker run -p 9000:9000 \
  -e DR_OWUI_BASE_URL=http://host.docker.internal:8080 \
  -e DR_OWUI_API_KEY=sk-... \
  -v dr-data:/data/deep_research \
  deep-research-mcp
```

Then in Claude Desktop / Cline / your MCP client, register `http://localhost:9000` as a Streamable-HTTP MCP server. A single tool, `deep_research(prompt, conversation_id=None)`, becomes available.

### 5. Library use (no runtime shim)

```python
import asyncio
from deep_research import Coordinator, Valves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig

async def main():
    coord = Coordinator(valves=Valves(), config=RuntimeConfig(data_dir="/tmp/dr"))
    await coord.start()
    try:
        async def sink(event): print(event)
        report = await coord.run(
            user=RunUser(id="me", name="me"),
            conversation_id="local-1",
            chat_id=None,
            token=StaticToken("sk-..."),
            prompt="How does Mamba compare to Transformer for long-context tasks?",
            history=[],
            sink=sink,
        )
        print(report.content)
    finally:
        await coord.close()

asyncio.run(main())
```

---

## Configuration

Every knob is grouped on the `Valves` Pydantic model. In OWUI runtimes (Function / Pipelines) the groups render as nested forms in the admin UI. In Docker runtimes (OpenAPI Tool / MCP), valves are populated from environment variables with the `DR_<GROUP>_<FIELD>` convention, e.g. `DR_MODELS_RESEARCH_MODEL=gemma3:27b`.

### Valve groups

| Group | Fields |
|---|---|
| `models` | `research_model`, `synthesis_model`, `quality_filter_model`, `research_context_window` (override; `None` = auto-detect from `/api/v1/models/list`), `synthesis_context_window`, `temperature`, `synthesis_temperature` |
| `cycles` | `min_cycles` (10), `max_cycles` (15), `gap_exploration_weight` (0.4), `trajectory_momentum` (0.6), `followup_weight` (0.5) |
| `web` | `search_results_per_query` (3), `successful_results_per_query` (1), `extra_results_per_query` (3), `repeats_before_expansion` (3), `max_result_tokens` (4000), `domain_priority`, `content_priority`, `quality_filter_enabled` (True), `quality_similarity_threshold` (0.60), `fetch_concurrency` (4), `search_concurrency` (2) |
| `compression` | `chunk_level` (2), `compression_level` (4), `stepped_synthesis_compression` (True) |
| `persistence` | `export_research_data` (True), `interactive_research` (True), `user_preference_throughout` (True) |
| `events` | `enable_progress_embed` (True), `flush_interval_ms` (400), `quiet_chat_mode` (True) |
| `advanced` | `query_weight` (0.5), `llm_concurrency` (4), `embedding_concurrency` (8), `executor_workers` (2), `http_timeout_seconds` (600), `http_max_retries` (3) |
| `logging` | `level` (`INFO`), `format` (`text` \| `json`), `include_tracebacks` (True) |

Engineering knobs that aren't user-facing (PDF page caps, extraction-quality thresholds, eigendecomposition radii, etc.) live in `deep_research/config/constants.py` — edit the source if you really need to tune them.

### Logging

Deep Research configures its own logger tree (`deep_research.*`) at every entry point — OpenAPI server, OWUI Function, Pipelines plugin, and MCP server. Output goes to `stderr` and does not propagate into the host's root logger, so it coexists with uvicorn / OWUI logging without duplicates.

Configure via env (shorthand) or valves (long form). Precedence: explicit valve > env > default.

| Setting | Env (shorthand) | Env (valve form) | Valve | Default | Notes |
|---|---|---|---|---|---|
| Level | `DR_LOG_LEVEL` | `DR_LOGGING_LEVEL` | `logging.level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| Format | `DR_LOG_FORMAT` | `DR_LOGGING_FORMAT` | `logging.format` | `text` | `text` (human-readable) or `json` (one JSON object per line) |
| Tracebacks | `DR_LOG_INCLUDE_TRACEBACKS` | `DR_LOGGING_INCLUDE_TRACEBACKS` | `logging.include_tracebacks` | `true` | Set false to strip stack traces while keeping the error message |

Every record carries the active `conversation_id`, `chat_id`, `run_id`, and `request_id` (propagated via `contextvars`, surfaced as record fields in both formats). API keys and bearer tokens are redacted to first 4 + last 4 (`sk-a…0xyz`) wherever they would otherwise appear in logs; values under 9 characters render as `********`.

At `DEBUG` level you get:
- Coordinator startup with model IDs and concurrency settings.
- Each of the 9 research phases — `Phase start: <name>` / `Phase done: <name> elapsed_s=…` / `Phase failed: <name>` with traceback.
- Every OWUI HTTP call: method, path, model id, body keys, status, elapsed time. Non-2xx responses include a 500-char truncated body — exactly the place the K8s-sidecar `400: 'NoneType' object has no attribute 'startswith'` failures surface.
- Each retry attempt from `adapter/retry.py` with attempt number, delay, classified reason, and exception type.

Example for the OpenAPI Tool runtime:

```bash
DR_LOG_LEVEL=DEBUG DR_LOG_FORMAT=text \
  DR_OWUI_BASE_URL=http://owui:8080 DR_OWUI_API_KEY=sk-xxxxxxxx \
  uvicorn deep_research.entrypoints.openapi_tool.server:app --port 8000
```

Sample of one failing chat-completions call:

```
2026-06-01 12:00:05,002 INFO deep_research.orchestrator [conv=api_abc run=r-9f req=req-77] Run started: conversation_id=api_abc prompt_chars=128
2026-06-01 12:00:05,003 INFO deep_research.orchestrator [conv=api_abc run=r-9f req=req-77] Phase start: outline_feedback
2026-06-01 12:00:05,051 DEBUG deep_research.adapter.client [conv=api_abc run=r-9f req=req-77] HTTP POST /api/chat/completions body_keys=['model', 'messages', 'stream', 'temperature'] model=gemma3:12b
2026-06-01 12:00:05,402 WARNING deep_research.adapter.client [conv=api_abc run=r-9f req=req-77] HTTP POST /api/chat/completions -> 400 model=gemma3:12b elapsed_s=0.35 body={"detail":"'NoneType' object has no attribute 'startswith'"}
```

For structured ingestion (Loki, ELK, Datadog) set `DR_LOG_FORMAT=json` — each line is a JSON object with `ts`, `level`, `logger`, `message`, `conversation_id`, `chat_id`, `run_id`, `request_id`, and optional `exception`.

### Required environment variables (Docker runtimes only)

| Var | Default | Notes |
|---|---|---|
| `DR_OWUI_BASE_URL` | `http://localhost:8080` | URL the runtime uses to reach Open WebUI. |
| `DR_OWUI_API_KEY` | _empty_ | Admin `sk-...` token. Must own the chats you want to persist into. See [Chat persistence caveat](#chat-persistence-caveat). |
| `DR_DATA_DIR` | `/data/deep_research` | Where vocab embeddings, transformation caches, and checkpoints are written. Mount a volume. |

The OWUI Function runtime uses the caller's `Authorization` header by default and falls back to `DR_OWUI_API_KEY` only when no header is present.

### Model ID format

Use the **exact ID OWUI registers in Settings → Models**, not the value shown in Documents → Embedding Model. Models behind an OpenAI-compatible connection with a configured `prefix_id` show up as `{prefix_id}.{model_id}`. Ollama and Infinity-style providers expose IDs containing slashes — keep the slash.

If you set a non-existent ID in `models.research_model` you'll get `Model not found` from `/api/embeddings` on the first cycle. The fix is always: list registered IDs with `GET /api/models` and copy verbatim.

---

## Compatibility

### OWUI version matrix

| OWUI version | Status |
|---|---|
| **≥ 0.9.5** | Fully supported. The progress embed uses the [`replace` flag for `embeds` events](https://github.com/open-webui/open-webui/commit/aa51ce482c161fb423767c41e8166197dce2d11b), so each conversation has exactly one live progress iframe instead of N stacked snapshots on reload. |
| **0.9.2 – 0.9.4** | Functional. The `replace` flag is silently ignored, so reloading a finished run will show stacked progress embeds. Live progress during a run is unaffected. |
| **< 0.9.2** | Untested. Async-ORM-aware endpoints in this version range should work via REST, but the progress embed format may render differently. |

### Chat persistence caveat

Open WebUI's `GET /api/v1/chats/{id}` filters by `chat.user_id == authenticated_user_id`. **Admin role does not bypass this filter.** If the `DR_OWUI_API_KEY` belongs to user A and the user invoking Deep Research is user B, persistence to `chat.deepResearch` silently fails. The research itself still runs.

Mitigation: in the OWUI Function runtime, the caller's Authorization header is passed through, so chat persistence works correctly. In Pipelines / OpenAPI Tool / MCP runtimes the API key is fixed, so persistence only works for chats owned by that key's user.

---

## How a research run looks

A typical "fresh" run with `interactive_research=True` (default):

1. **Turn 1 — user asks a question.** The pipe runs initial queries, produces a research outline, emits it as a message, and stops with a `waiting_for_outline_feedback` flag set on conversation state. The progress embed shows "Awaiting outline confirmation".
2. **Turn 2 — user replies.** Empty / "looks good" / "please proceed" = continue as-is. Otherwise the feedback is parsed by the LLM, mapped to outline edits (add topic, remove topic, refine subtopic), and folded into the outline. Then the full research → synthesis → review pipeline runs to completion in this same turn.
3. **Final assistant message** contains the report. If `events.quiet_chat_mode=True` (default), only the final report shows in the assistant message; if `False`, intermediate cycle summaries are streamed as message chunks too.
4. **Per-conversation knowledge base** is provisioned on first turn and attached to the chat. Every selected source is persisted as a markdown file with the original URL, full extracted text, and metadata. The final report is also persisted, so a separate model can RAG against the whole research corpus.
5. **Follow-up turns** (same conversation, after a report exists) are detected as `post_report_user_qa` and answered against the KB instead of starting a fresh research run.

If `interactive_research=False`, step 1 runs straight through into research cycles in the first turn.

---

## Architecture in one diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Open WebUI (any 0.9.2+ instance)                                   │
│  • /api/chat/completions   • /api/v1/retrieval/*                    │
│  • /api/embeddings         • /api/v1/files/*                        │
│                            • /api/v1/knowledge/*                    │
└─────────────────▲───────────────────────────────────────────────────┘
                  │ httpx (OWUIClient — the SOLE host adapter)
                  │
┌─────────────────┴───────────────────────────────────────────────────┐
│                                                                     │
│   deep_research/                                                    │
│   ├── adapter/    OWUIClient, httpx, retries, semaphores            │
│   ├── core/       caches, state manager, types, errors, text utils  │
│   ├── config/     Valves, constants, env loader                     │
│   ├── semantics/  embeddings, vocabulary, eigendecomp, dimensions,  │
│   │               trajectory, preference, similarity                │
│   ├── budget/     token counting, context windows, packing          │
│   ├── compression/ local-similarity, eigendecomp, repeated, stepped │
│   ├── web/        search, classify, fetch, html/pdf extract,        │
│   │               paywall (UA spoof + cookies, archive.org rescue)  │
│   ├── research/   query gen, ranking, relevance, outline-feedback,  │
│   │               grouping, per-query cycle driver                  │
│   ├── synthesis/  outline, sections+citations, verify, review,      │
│   │               titles+abstract                                   │
│   ├── persistence/ chat-state checkpoints, KB ensure/attach/upload, │
│   │                source markdown, post-report QA                  │
│   ├── orchestrator/ Coordinator + 9 phases                          │
│   ├── progress/   event bus, snapshot, progress-embed HTML          │
│   └── entrypoints/                                                  │
│       ├── owui_function/pipe.py        # OWUI Function shim         │
│       ├── owui_pipeline/pipeline.py    # OWUI Pipelines shim        │
│       ├── openapi_tool/server.py       # FastAPI + SSE              │
│       └── mcp/server.py                # FastMCP Streamable-HTTP    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Every request that hits a runtime instantiates a fresh `RunContext` (per-call, never shared). The Coordinator is process-shared, the OWUIClient HTTP session is process-shared, the embedding/transformation/vocabulary caches are process-shared and lock-protected. Two concurrent `.run()` calls for the same `(user_id, conversation_id)` are rejected with `AlreadyRunningError` — the same dedupe semantics as the original `pipe.py`.

---

## Development

```bash
git clone <this repo> && cd open-webui-deep-research-tool
uv sync --all-groups
# or: python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install pytest pytest-asyncio respx hypothesis ruff mypy pre-commit
.venv/bin/pre-commit install

# Tests
.venv/bin/pytest

# Lint + format
.venv/bin/ruff check deep_research/ tests/
.venv/bin/ruff format deep_research/ tests/

# Type check
.venv/bin/mypy deep_research/

# Security scans (optional, both also run in CI)
opengrep scan --config=auto deep_research/
codeql database create /tmp/dr-db --language=python --source-root=. --overwrite
codeql database analyze /tmp/dr-db codeql/python-queries --format=sarif-latest --output=/tmp/dr.sarif
```

### Repository layout

- `deep_research/` — the package.
- `pipe.py`, `deep_research_pipeline.py` — frozen pre-refactor sources kept as historical reference until the new package has been deployed and observed in production. Do not edit. They are excluded from ruff/mypy.
- `REFACTOR_PLAN.md` — the multi-LLM migration plan that produced this refactor. Useful for understanding why a file is structured the way it is.
- `CLAUDE.md` — agent context, including the OWUI REST response-shape table.
- `tests/` — unit tests for caches, text utilities, the EventBus, the env loader, the adapter (with `respx` mocks), the Coordinator (inflight dedupe + lifecycle), and budget windows.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Model not found` on first embedding call | `models.research_model` doesn't match a registered OWUI model ID — copy from Settings → Models exactly (with any `prefix_id.` prefix). |
| Function appears in OWUI but every run fails with `OWUIClient not started` | The Function wasn't instantiated through the shim path — check that the wheel installed cleanly and that `from deep_research import Coordinator` resolves inside the OWUI container. |
| Pipelines runtime hangs on first request | `DR_OWUI_BASE_URL` unreachable from the Pipelines container; check intra-network DNS (use the OWUI service name, not `localhost`). |
| Stacked progress embeds on chat reload | OWUI < 0.9.5; the `replace` flag is ignored. Upgrade OWUI or live with the visual clutter on reload (live runs are unaffected). |
| Research starts but no sources appear after several cycles | OWUI Web Search isn't configured (Settings → Web Search) or the configured engine is rate-limited. Status pill should show the failure mode. |
| KB persistence "silently does nothing" | `DR_OWUI_API_KEY` belongs to a user that doesn't own the chat. See [Chat persistence caveat](#chat-persistence-caveat). |
| `httpx.ConnectError` on PDF fetches | The paywall-spoof path uses direct `httpx` (not OWUI). Check egress firewall rules and that the target host responds to non-OWUI traffic. |

---

## License & attribution

This project: see [LICENSE](LICENSE).

Core deep-research algorithms (multi-cycle research dimensions, semantic eigendecomposition, preference direction vectors, gap-vector exploration) are adapted from [atineiatte/deep-research-at-home](https://github.com/atineiatte/deep-research-at-home). Where mathematical behaviour was preserved 1:1 from that source per the refactor plan, the implementation reflects that origin.

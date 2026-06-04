# Deep Research for Open WebUI

A multi-cycle web research engine that plans an outline, drives iterative
search/fetch/compress cycles against Open WebUI's built-in web search and
retrieval endpoints, and produces a synthesised, citation-verified report
with a persisted knowledge base.

The same core ships in three runtimes:

| Runtime | When to use it |
|---|---|
| **OWUI Function** | You run Open WebUI yourself and want Deep Research to appear as a model in the chat dropdown. |
| **OpenAPI Tool server** | A JSON-over-HTTP tool surface designed for Open WebUI's tool-server integration; also usable from any HTTP client. |
| **MCP server** | Anything that speaks Streamable-HTTP MCP — Claude Desktop, Cline, an MCP-aware orchestrator. |

All three runtimes talk to **three independently configured services**: an
OpenAI-compatible chat LLM provider, an OpenAI-compatible embeddings
provider, and Open WebUI (for retrieval, files, knowledge base, and chat
persistence). Each surface has its own base URL and bearer token, so chat
and embeddings can point at the same backend or at different providers
(chat at OpenAI, embeddings at a local Ollama). There are no
`open_webui.*` Python imports anywhere in the package — all three
services are pure remote HTTP APIs.

> Core algorithms (semantic eigendecomposition, multi-cycle research
> dimensions, preference direction vectors, gap-vector exploration) are
> adapted from
> [atineiatte/deep-research-at-home](https://github.com/atineiatte/deep-research-at-home).
> The packaging, runtime split, REST adapter, event bus, and persistence
> layer are this project's contributions.

---

## Requirements

- **Open WebUI** ≥ 0.9.5 recommended (older works with degraded progress
  UI — see [docs/compatibility.md](./docs/compatibility.md)).
- An OWUI API key (`sk-...`). **`DR_OWUI_API_KEY` must be an admin
  token** if you run the **OpenAPI Tool Server** with writeback enabled
  (the default) — the server uses it to post writeback events to OWUI's
  per-message `/event` endpoint on behalf of arbitrary users, and that
  endpoint only admin-bypasses chat ownership for admin tokens. A
  non-admin key works for the OWUI Function and MCP runtimes, or for
  the OpenAPI Tool Server with `DR_JOBS_WRITEBACK_ENABLED=false`. The
  OpenAPI Tool Server refuses to start if `DR_OWUI_API_KEY` is unset
  while writeback is enabled. Either way the token needs permission to
  embeddings, chat completions, retrieval, files, knowledge, and chat
  persistence.
- A search provider configured in OWUI Settings → Web Search.
- An embedding model registered in OWUI Settings → Models.
- Python 3.11+ if you're running the OpenAPI Tool or MCP runtimes
  yourself.

---

## Quick start

The six env vars you'll set in every Docker runtime:

```bash
DR_OWUI_BASE_URL=http://owui:8080
DR_OWUI_API_KEY=sk-owui-admin-key
DR_LLM_BASE_URL=http://owui:8080/openai          # or your chat provider
DR_LLM_API_KEY=sk-llm-key
DR_EMBEDDINGS_BASE_URL=http://owui:8080/openai   # or a separate embedding provider
DR_EMBEDDINGS_API_KEY=sk-emb-key
```

Chat and embeddings can share a backend or be split. For production layout
(separate keys per surface, K8s/Helm manifests, throttling for low-TPM
embedding providers), see [docs/deployment.md](./docs/deployment.md).

> ⚠ **Admin-key requirement (OpenAPI Tool Server only).** The placeholder
> `sk-owui-admin-key` above is intentional: with writeback enabled (the
> default), this token must have OWUI's `admin` role. The server refuses
> to start if `DR_OWUI_API_KEY` is unset, and `GET /health` surfaces a
> structured warning when the key is set but not admin. See
> [OpenAPI Tool server § live writeback](#2-openapi-tool-server-docker).

### 1. OWUI Function (in-container)

The OWUI Function runtime needs the `deep_research/` package importable
from inside the OWUI container.

**Option A — install the wheel:**

```bash
# Inside the OWUI container (or via a custom Dockerfile that extends OWUI):
pip install /path/to/deep-research-0.2.0-py3-none-any.whl
```

Then in OWUI: **Admin Panel → Functions → New Function → upload
`deep_research/entrypoints/owui_function/pipe.py`** (just the shim file —
its imports resolve from the installed wheel).

**Option B — flatten the package into one file** with the
`inline-python-modules` skill and upload that single `.py` verbatim.
Useful for hosted OWUI instances where you can't `pip install` into the
container.

After install, the Function appears as a model called `Deep Research` in
the chat model dropdown.

### 2. OpenAPI Tool server (Docker)

Standalone REST service designed to plug into Open WebUI as a
[tool server](https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/open-webui).
The server exposes a two-call **start → user replies → submit feedback** workflow
plus a self-polling live-progress iframe.

| Operation | Path | Purpose |
|---|---|---|
| `start_research_job` | `POST /research_jobs` | Kicks off a research run. Returns a `job_id` immediately while the engine runs in the background. The response includes a `user_facing_instruction` the LLM must emit verbatim. |
| `submit_research_feedback` | `POST /research_jobs/{job_id}/feedback` | Forwards the user's slash-command reply (`/k 1,3,5`, `/r 2,4`, `/continue`, or freeform text) and resumes the engine. |
| `get_research_job` | `GET /research_jobs/{job_id}` | JSON snapshot. `phase` + `progress` while running, `report_markdown` once `completed`. |
| `cancel_research_job` | `POST /research_jobs/{job_id}/cancel` | Best-effort cancellation; the engine bails at the next phase boundary. |
| _(live view)_ | `GET /live_view/{job_id}` | Renders the progress iframe HTML (view-token authenticated). |
| _(live view JSON)_ | `GET /live_view/{job_id}/status` | JSON snapshot used by the iframe's polling loop. |
| _(health)_ | `GET /health` | Liveness check; returns `config_warnings` as JSON (admin-token misconfiguration, missing public base URL, OWUI-side header-forwarding misconfig, etc.). |

```bash
docker build -t deep-research-openapi -f deep_research/entrypoints/openapi_tool/Dockerfile .
docker run -p 8000:8000 \
  -e DR_OWUI_BASE_URL=http://host.docker.internal:8080 \
  -e DR_OWUI_API_KEY=sk-owui-admin-key \
  -e DR_LLM_BASE_URL=http://host.docker.internal:8080/openai \
  -e DR_LLM_API_KEY=sk-llm-key \
  -e DR_EMBEDDINGS_BASE_URL=http://host.docker.internal:8080/openai \
  -e DR_EMBEDDINGS_API_KEY=sk-emb-key \
  -e DR_DATA_DIR=/data/deep_research \
  -e DR_OPENAPI_PUBLIC_BASE_URL=https://research-tool.example.com \
  -v dr-data:/data/deep_research \
  deep-research-openapi
```

`DR_OPENAPI_PUBLIC_BASE_URL` is the URL the **user's browser** uses to
reach this server (distinct from `DR_OWUI_BASE_URL`, which OWUI itself
uses). The live-progress iframe needs it to construct an absolute
polling URL; if unset, the server falls back to the incoming request's
host header, which usually works in development but breaks when OWUI
calls in via an internal cluster name.

Smoke test:

```bash
curl -s -X POST http://localhost:8000/research_jobs \
  -H 'Authorization: Bearer sk-owui-admin-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Summarise the state of post-quantum cryptography in 2026."}' \
  | jq
# → {"job_id":"...", "status":"running", "next_action":"await_user_selection",
#    "user_facing_instruction":"..."}

# Poll for the outline-feedback gate
curl -s http://localhost:8000/research_jobs/$JOB_ID \
  -H 'Authorization: Bearer sk-owui-admin-key' | jq

# Submit a selection
curl -s -X POST http://localhost:8000/research_jobs/$JOB_ID/feedback \
  -H 'Authorization: Bearer sk-owui-admin-key' \
  -H 'Content-Type: application/json' \
  -d '{"selection":"/continue"}' | jq

# Eventually
curl -s http://localhost:8000/research_jobs/$JOB_ID \
  -H 'Authorization: Bearer sk-owui-admin-key' | jq '.report_markdown'
```

**Use from Open WebUI**: Settings → Tools → Add tool server → point it
at `http(s)://<host>/openapi.json`. Once registered, open the
tool-server icon in the chat composer and toggle **Deep Research** on
for the current chat. The tool prompts the LLM to emit a verbatim
slash-command instruction so the user knows how to drive the outline-
feedback step.

**Live writeback to chat content (recommended)** requires two things:

1. **`DR_OWUI_API_KEY` set to an OWUI admin token.** Writeback POSTs to
   OWUI's per-message `/event` endpoint on behalf of arbitrary users; a
   non-admin token can only write its own chats. The server refuses to
   start without the key (with writeback enabled); a non-admin key
   surfaces as a `OWUI_API_KEY_NOT_ADMIN` warning on `GET /health`.
2. **`ENABLE_FORWARD_USER_INFO_HEADERS=true` on the OWUI container** so
   `X-OpenWebUI-Chat-Id` / `X-OpenWebUI-Message-Id` reach the tool
   server. Without these headers, jobs record `chat_id=None` and
   writeback is silently disabled (Phase 1 verbatim-instruction
   fallback still works). The tool server detects the missing headers
   on the first authenticated request and surfaces the warning code
   `OWUI_HEADERS_NOT_FORWARDED` via `/health`.

When both are in place, the topic list and final report land directly
in the assistant message as the engine produces them. The LLM no longer
needs to repeat the topic list — the user sees it in chat in real time,
types their `/k 1,3,5` reply, and the final report appears in the next
tool-call message. Status pills and side-panel citations also stream in
via the same channel. The mechanism is OWUI's per-message
`POST /api/v1/chats/{id}/messages/{message_id}/event` endpoint, which
admin-bypasses chat ownership (so the tool server's admin key can write
to any user's chat).

Set `DR_JOBS_WRITEBACK_ENABLED=false` to disable writeback explicitly;
in that mode a non-admin `DR_OWUI_API_KEY` is fine and the iframe-only
UX still works.

**Embedding-quota tip**: if KB ingestion is the bottleneck (low-TPM
embedding key), set `DR_PERSISTENCE_DISABLE_KB_PERSISTENCE=true` to
skip uploads to the OWUI KB. Research becomes ephemeral (no rehydrate,
no post-report KB Q&A) but the in-chat report still lands.

### 3. MCP server (Docker, Streamable-HTTP)

```bash
docker build -t deep-research-mcp -f deep_research/entrypoints/mcp/Dockerfile .
docker run -p 9000:9000 \
  -e DR_OWUI_BASE_URL=http://host.docker.internal:8080 \
  -e DR_OWUI_API_KEY=sk-owui-admin-key \
  -e DR_LLM_BASE_URL=http://host.docker.internal:8080/openai \
  -e DR_LLM_API_KEY=sk-llm-key \
  -e DR_EMBEDDINGS_BASE_URL=http://host.docker.internal:8080/openai \
  -e DR_EMBEDDINGS_API_KEY=sk-emb-key \
  -v dr-data:/data/deep_research \
  deep-research-mcp
```

Then in Claude Desktop / Cline / your MCP client, register
`http://localhost:9000` as a Streamable-HTTP MCP server. A single tool,
`deep_research(prompt, conversation_id=None)`, becomes available.

### 4. Library use (no runtime shim)

```python
import asyncio
from deep_research import Coordinator, Valves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig

async def main():
    coord = Coordinator(valves=Valves(), config=RuntimeConfig(
        data_dir="/tmp/dr",
        llm_base_url="http://localhost:11434",
        llm_api_key="ollama",
        embeddings_base_url="http://localhost:11434",
        embeddings_api_key="ollama",
    ))
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

## Documentation

| Doc | When to read it |
|---|---|
| [docs/deployment.md](./docs/deployment.md) | Deploying to production — three-secret K8s/Helm patterns, env-var ownership matrix, pre-flight checks, security guidance |
| [docs/configuration.md](./docs/configuration.md) | Full valve reference (every group, every field), model ID format, logging configuration |
| [docs/troubleshooting.md](./docs/troubleshooting.md) | Symptom → cause → check → fix for the common failure modes |
| [docs/architecture.md](./docs/architecture.md) | The 9 research phases, what a run looks like to the user, package layout |
| [docs/compatibility.md](./docs/compatibility.md) | OWUI version matrix, chat persistence caveat |
| [docs/development.md](./docs/development.md) | Clone, test, lint, security scans, repo layout |
| [deploy/README.md](./deploy/README.md) | Local manual-test Compose stack |

---

## License & attribution

This project: see [LICENSE](LICENSE).

Core deep-research algorithms (multi-cycle research dimensions, semantic
eigendecomposition, preference direction vectors, gap-vector exploration)
are adapted from
[atineiatte/deep-research-at-home](https://github.com/atineiatte/deep-research-at-home).
Where mathematical behaviour was preserved 1:1 from that source per the
refactor plan, the implementation reflects that origin.

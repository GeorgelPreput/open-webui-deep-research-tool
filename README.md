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
- An OWUI admin API key (`sk-...`) with permission to embeddings, chat
  completions, retrieval, files, knowledge, and chat persistence.
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
Exposes three operations plus a health check, all returning JSON.

| Operation | Path | Purpose |
|---|---|---|
| `research` | `POST /research` | Synchronous run. Returns the full markdown report + structured citations. Blocking for the entire run. |
| `start_research_job` | `POST /research_jobs` | Same shape as `research`, but returns a `job_id` immediately. Use when the run may exceed your tool-call HTTP timeout. |
| `get_research_job` | `GET /research_jobs/{job_id}` | Poll status. Returns `pending` / `running` / `completed` / `failed` with the same result payload once done. |
| _(health)_ | `GET /health` | Liveness check. |

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
  -v dr-data:/data/deep_research \
  deep-research-openapi
```

Smoke test:

```bash
curl -s http://localhost:8000/research \
  -H 'Authorization: Bearer sk-owui-admin-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Summarise the state of post-quantum cryptography in 2026."}' \
  | jq
```

**Use from Open WebUI**: Settings → Tools → Add tool server → point it at
`http(s)://<host>/openapi.json`. Once registered, open the tool-server
icon in the chat composer and toggle **Deep Research** on for the current
chat. Prefer `research` for prompts that complete inside your
`AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` window; switch to
`start_research_job` + `get_research_job` polling for long runs.

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

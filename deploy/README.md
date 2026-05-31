# Local manual-test stack

A throwaway Docker Compose stack to eyeball the Deep Research engine across its
runtimes in a browser. It brings up:

| Service | URL | What it is |
|---|---|---|
| `open-webui` | http://localhost:3000 | The Open WebUI you test in |
| `pipelines` | http://localhost:9099 | OWUI Pipelines sidecar with the `deep_research` pipeline preloaded |
| `openapi-tool` | http://localhost:8000 | OpenAPI tool server (`/docs`, `POST /research`, `/health`) |
| `mcp` | http://localhost:9000 | MCP server (streamable-HTTP at `/mcp`) |

> The **OWUI Function** runtime (`pipe.py`) is *not* a container — it runs inside
> OWUI itself. To test it, paste `pipe.py` into OWUI > Admin > Functions. This
> stack covers the other three runtimes.

## 1. Start

```bash
cp .env.example .env          # then edit (see below)
docker compose up --build     # first build pulls images + compiles wheels (~few min)
```

Open http://localhost:3000 and create the first admin account.

## 2. Give Open WebUI a brain

The engine drives OWUI for LLM, embeddings, and web search, so OWUI must be able
to do those things:

- **Models** — OWUI > Admin Settings > Connections. Add an **Ollama** server
  (uncomment `OLLAMA_BASE_URL` in `docker-compose.yml` to reach a host Ollama via
  `http://host.docker.internal:11434`) or an **OpenAI-compatible** connection.
  Pull/enable a chat model and an **embedding** model.
- **Web search** — OWUI > Admin Settings > Web Search: enable an engine
  (e.g. DuckDuckGo needs no key) so the research crawl returns results.
- Note the exact model IDs from OWUI > Admin Settings > Models. If they aren't
  the engine defaults (`gemma3:12b`, `gemma3:27b`, `nomic-embed-text`), set
  `DR_MODELS_*` in `docker-compose.yml` for `openapi-tool`/`mcp` (see comments
  there) and in the Pipelines valves (step 4) for the pipeline.

## 3. Create the callback API key

The `openapi-tool`, `mcp`, and `pipelines` services call back into OWUI and need
an admin key:

1. OWUI > Settings > Account > API Keys > create one.
2. Put it in `.env` as `DR_OWUI_API_KEY=sk-...`.
3. `docker compose up -d` to restart the three caller services with the key.

## 4. Wire each runtime

**Pipelines** — OWUI > Admin Settings > Connections > add an **OpenAI-compatible**
connection: URL `http://pipelines:9099`, API key = `PIPELINES_API_KEY` (default
`0p3n-w3bu!`). A model named **`deep_research_pipeline.deep-research`** (id
`deep_research`) appears in the model picker. Open valves on that connection to
set `OWUI_API_KEY` / model IDs if needed. Start a chat with it.

**OpenAPI tool server** — the reliable check is the built-in Swagger UI at
http://localhost:8000/docs → `POST /research` with `{"prompt": "..."}` (the
response is an SSE event stream). To attach it to OWUI: Settings > Tools > add a
tool server with URL `http://openapi-tool:8000` (OWUI fetches `/openapi.json`).
*Caveat:* `/research` streams via SSE, which OWUI's tool-call UI may not render
incrementally — `/docs` or `curl` is the most direct functional test.

```bash
curl -N http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain the Mamba state-space architecture"}'
```

**MCP server** — streamable-HTTP at `http://mcp:9000/mcp` (from inside the
compose network) or `http://localhost:9000/mcp` (from the host). Add it as an MCP
tool in an MCP-capable client, or in OWUI's MCP/External-Tools settings. It
exposes one tool, `deep_research(prompt, conversation_id?)`.

## Ports & data

Host ports: `3000` (OWUI), `8000` (OpenAPI), `9000` (MCP), `9099` (Pipelines).
State lives in named volumes (`openwebui-data`, `pipelines-data`, `openapi-data`,
`mcp-data`).

## Teardown

```bash
docker compose down            # stop (keep data)
docker compose down -v         # stop and delete the volumes
```

## Notes

- A research run only works once OWUI has a working chat model, embedding model,
  and web search — otherwise the engine starts but every LLM/search call fails.
- All three caller services reach OWUI at `http://open-webui:8080` on the compose
  network; that's why `DR_OWUI_BASE_URL` is hardcoded to it in the compose file.

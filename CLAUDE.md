# Deep Research pipe — investigation notes

## What this repo is

Two-file repo. Both files implement the same deep research engine (multi-cycle web research with semantic compression, eigendecomposition-based dimension tracking, KB persistence, and citation verification) but target different OWUI deployment environments:

| File | Class | Target | Entry point |
|---|---|---|---|
| `pipe.py` | `Pipe` | OWUI **Function** (runs inside OWUI container, shares its asyncio loop and `app.state`) | `async def pipe(self, body, __user__, __event_emitter__, ...)` |
| `deep_research_pipeline.py` | `Pipeline` | OWUI **Pipelines** plugin (runs in a separate Pipelines container, calls OWUI via REST) | `def pipe(self, user_message, model_id, messages, body) -> Iterator[str]` |

`pipe.py` is **read-only reference** — do not modify it. All active development happens in `deep_research_pipeline.py`.

---

## Architecture: `deep_research_pipeline.py`

### Key structural differences from `pipe.py`

| Concern | `pipe.py` (OWUI Function) | `deep_research_pipeline.py` (Pipelines plugin) |
|---|---|---|
| LLM calls | `generate_chat_completions` from `open_webui.main` | `OWUIClient.chat_completions` → POST `/api/chat/completions` |
| Embeddings | `self.__request__.app.state.config.RAG_EMBEDDING_MODEL` + direct EmbeddingFunction call | `OWUIClient.embeddings` → POST `/api/embeddings` |
| Web search | `_try_openwebui_search` using OWUI internal search function | `OWUIClient.web_search` → POST `/api/v1/retrieval/process/web/search` |
| Document extraction | `_build_document_loader_kwargs` + OWUI loaders | `OWUIClient.process_web_url` / `process_file` → REST endpoints |
| KB operations | Direct ORM: `Knowledges.insert_new_knowledge()`, `Files.insert_new_file()` | `OWUIClient.create_kb` / `upload_file` / `add_file_to_kb` → REST |
| Chat persistence | Direct ORM: `Chats.get_chat_by_id_and_user_id()` | `OWUIClient.get_chat` / `update_chat` — **only works for chats owned by the API key's user** (no admin bypass for `get_chat_by_id_and_user_id`) |
| Status/progress UI | `__event_emitter__({"type": "status", ...})` → status pills in OWUI UI | `<details type="reasoning">` blocks yielded into the SSE stream |
| Progress embed widget | HTML iframe rendered via `render_progress_embed_html` + `refresh_progress_embed` | Downgraded to plain-text summary inside the reasoning block |
| Entry point signature | `async def pipe(...)` — awaitable coroutine | `def pipe(...) -> Iterator[str]` — sync, must return an iterator |
| Asyncio environment | Runs inside OWUI's existing event loop | Runs in a fresh per-call event loop in a daemon thread |
| OWUI imports | `from open_webui.models.chats import Chats`, `from open_webui.main import generate_chat_completions`, etc. | No OWUI imports — pure stdlib + third-party |

### File layout (approximate line numbers — search by class/function name, not line)

```
deep_research_pipeline.py
├── Module docstring + frontmatter (requirements pip list)
├── OWUIClient                         # Async REST client (~L77–559)
│   ├── __init__ / start / close
│   ├── _request (retry + backoff core)
│   ├── chat_completions               # streaming SSE parser
│   ├── embeddings
│   ├── web_search
│   ├── process_web_url
│   ├── process_text
│   ├── upload_file (multipart)
│   ├── process_file
│   ├── get_file
│   ├── create_kb / get_kb / add_file_to_kb
│   ├── query_collection
│   ├── get_chat / update_chat
│   └── list_models
├── _BridgeSink                        # Thread-safe async→sync bridge (~L562–588)
├── _ReasoningBlock                    # <details type="reasoning"> builder (~L591–632)
├── EmbeddingCache / TransformationCache / ResearchStateManager  # shared (~L635–782)
├── _PipelineCallLocal                 # threading.local subclass; defines per-call slots (~L785)
├── _tls_prop                          # module helper: builds threadlocal-backed property (~L815)
├── Pipeline                           # Pipelines plugin shell
│   ├── Valves (BaseModel)             # ~60 valves from pipe.py + 9 new ones
│   ├── client/_sink/_reasoning/...    # 15 property descriptors → self._tls.<name>
│   ├── __init__                       # type="manifold", pipelines list, self._tls=_PipelineCallLocal()
│   ├── on_startup                     # Creates DATA_DIR only — NO client (wrong loop)
│   ├── on_shutdown                    # Shuts down executor
│   ├── on_valves_updated              # No-op (client is per-call)
│   ├── _build_client                  # Factory; caller must start() in the right loop
│   ├── pipe()                         # Sync entrypoint — spawns thread, returns iter(sink)
│   ├── _run_research_async            # Async: creates client, runs engine, pushes result
│   ├── _push_status_line              # Time-based reasoning flush
│   ├── _flush_reasoning               # Finalizes current <details> block
│   ├── _compat_event_emitter          # Translates OWUI event dicts for compat shims
│   └── [all methods from pipe.py]    # Lifted verbatim; only I/O surface changed
```

### New valves (not in `pipe.py`)

| Valve | Default | Purpose |
|---|---|---|
| `OWUI_BASE_URL` | `http://localhost:8080` | OWUI instance to call back into |
| `OWUI_API_KEY` | `""` | Admin `sk-...` key. Must own the chats for persistence to work |
| `EMBEDDING_MODEL` | `""` | OWUI embedding model ID. If empty, OWUI uses its default |
| `DATA_DIR` | `/app/pipelines/data/deep-research` | Vocab embedding cache, checkpoints |
| `OWUI_REQUEST_TIMEOUT_S` | `600` | Per-request aiohttp timeout |
| `OWUI_MAX_CONCURRENT` | `8` | Semaphore cap on parallel OWUI calls |
| `OWUI_MAX_RETRIES` | `3` | Retry count for retriable errors (5xx, connect errors) |
| `STATUS_AS_REASONING` | `True` | Emit status as `<details type="reasoning">` vs. plain text |
| `REASONING_FLUSH_SECONDS` | `4.0` | Max seconds before force-flushing an open reasoning block |

### Valve type changes from `pipe.py`

`DOMAIN_PRIORITY`, `CONTENT_PRIORITY`, `OWUI_API_KEY`, `EMBEDDING_MODEL` are `Optional[str]` in the Pipelines version (not `str`). OWUI admin UI sends empty fields as JSON `null`; Pydantic rejects bare `str` for null but accepts `Optional[str]`.

### Valve model ID format

Model IDs for `RESEARCH_MODEL`, `SYNTHESIS_MODEL`, and `EMBEDDING_MODEL` must match **exactly** what is registered in `app.state.MODELS` — the main OWUI models registry, NOT the RAG embedding config.

**How IDs are formed:**
- If the OpenAI-compatible connection has a `prefix_id` set (e.g., `myprefix`), models are stored as `{prefix_id}.{model_id}` (dot separator), e.g., `myprefix.my-llm-model`.
- If no prefix is configured, models are stored by their raw ID from the API, e.g., `my-llm-model`.
- Some providers (Infinity Embedding, HuggingFace) return model IDs with slashes, e.g., `org/embedding-model` — these are stored with the slash as-is.

**Common error**: copying the embedding model ID from Admin Panel > Documents. That panel shows the RAG config format (`{engine}/{model}`, e.g., `openai/embedding-model`) which is **not** the model's registered ID. Use the ID from Settings > Models instead.

**"Model not found" on `/api/embeddings`**: means the model ID is not in `app.state.MODELS`. Check that:
1. The model is **enabled** in OWUI Admin > Models (disabled models are excluded from `app.state.MODELS`).
2. The model ID matches exactly — call `GET /api/models` to list all registered IDs.

---

## Critical implementation patterns

### Per-call OWUIClient — why it's required

`aiohttp.ClientSession` is bound to the event loop it was created in. The Pipelines framework calls `pipe()` from multiple threads (one per request). Each call spawns a daemon thread with its own `asyncio.new_event_loop()`. If a shared session from `on_startup()` were used, it would raise `Future attached to a different loop`.

**The rule:** never create an aiohttp session in `on_startup`. Always call `_build_client()` at the start of `_run_research_async`, inside the per-call event loop.

```python
# In the worker thread:
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)   # required for libraries that call get_event_loop()
loop.run_until_complete(self._run_research_async(...))
```

```python
# In _run_research_async:
self.client = self._build_client()
await self.client.start()
try:
    ...
finally:
    await self.client.close()
    self.client = None
```

### Async-to-sync bridge (`_BridgeSink`)

`pipe()` must return `Iterator[str]` (sync). The engine is entirely async. Pattern:

1. Create `_BridgeSink(queue.Queue(maxsize=64))` with a sentinel object.
2. Spawn daemon thread: `thread_target` creates a new event loop, runs `_run_research_async`, puts SENTINEL when done.
3. `pipe()` returns `iter(sink)` — `_BridgeSink.__iter__` pulls from the queue, stopping at SENTINEL.

Any string put into the sink becomes a chunk in OWUI's SSE stream, rendered immediately as markdown.

### `<details type="reasoning">` rendering — exact format required

OWUI's marked extension (`extension.ts:31`) tokenizes the block with this regex:

```
/^<details(\s+[^>]*)?>\n/
```

The `\n` after `>` is **mandatory** — the tokenizer will not match without it. Similarly, the summary regex requires `\n` after `</summary>`. `_ReasoningBlock.render()` must produce exactly:

```python
f"\n\n<details {attrs}>\n"
f"<summary>{html.escape(summary)}</summary>\n"
f"{body}\n"
f"</details>\n\n"
```

- Leading `\n\n` — ensures the block starts on a fresh paragraph (otherwise inline text runs into it).
- `\n` after opening `>` — required by tokenizer.
- `\n` after `</summary>` — required by tokenizer.
- Trailing `\n\n` — paragraph break after the block.

Attributes used: `type="reasoning"`, `done="true"|"false"`, `duration="X.X"` (seconds, shown as "Thought for X seconds" when done).

Consecutive `<details type="reasoning">` blocks are grouped by OWUI's `ConsecutiveDetailsGroup.svelte` into a single collapsible accordion. Each research phase (planning, cycle N, synthesis, citation verify) should open as `done="false"` and close with `done="true"`.

### QUIET_CHAT_MODE — return value must be pushed to sink

In OWUI Functions, the return value of `pipe()` is auto-appended to the chat as the assistant message. In Pipelines, the return value is discarded — only yielded strings reach the user.

`_run_research` (the async engine) returns a `comprehensive_answer` string in QUIET mode. `_run_research_async` must capture it and push it to the sink:

```python
result = await self._run_research(body=body, __user__=user_dict)
self._flush_reasoning(done=True)
if isinstance(result, str) and result.strip():
    self._sink_put(result)
```

### Time-based reasoning flush (`REASONING_FLUSH_SECONDS`)

Without this, all status lines accumulate in memory and only flush at message events or end-of-run — the user sees no output for 9+ minutes. `_push_status_line` checks `time.monotonic() - self._block_opened_at` and flushes early if the block has been open longer than `REASONING_FLUSH_SECONDS` (default 4.0s).

---

## OWUI REST response shapes (verified against `.venv/lib/python3.11/site-packages/open_webui/`)

These were wrong in the initial migration and required fixes. Trust these, not the endpoint docstrings:

| Endpoint | Response shape | Notes |
|---|---|---|
| `POST /api/v1/retrieval/query/collection` | `{distances: [[...]], documents: [[...]], metadatas: [[...]]}` | Lists-of-lists (one inner list per query). Unpack: `documents[0]`, `metadatas[0]`, `distances[0]` |
| `POST /api/v1/retrieval/process/file` | `{status, collection_name, filename, content}` | `content` is at top level, not under `data` |
| `POST /api/v1/retrieval/process/web` | `{status, content}` | `content` at top level when `process=False`; no `documents`/`docs` key |
| `POST /api/v1/files/` (upload) | `{id, filename, meta: {content_type, size}, data: {content}}` | `data.content` has extracted text if `process=true` |
| `GET /api/v1/files/{id}` | same shape as upload response | |
| `POST /api/chat/completions` (stream) | Server-Sent Events, each line: `data: {"choices":[{"delta":{"content":"..."}}]}` | Finish chunk has `finish_reason` set; last line is `data: [DONE]` |

### Chat persistence limitation

`GET /api/v1/chats/{id}` calls `Chats.get_chat_by_id_and_user_id(id, user_id)` — it filters by `user_id` matching the authenticated user. Admin role does **not** bypass this filter. If the `OWUI_API_KEY` belongs to user A and the chat was created by user B, `get_chat` returns 404. Persistence silently fails for chats not owned by the API key user. There is no workaround short of using the chat owner's key.

---

## Bugs found and fixed during the Pipelines migration

1. **`_kb_search` wrong response shape** — was calling `result.get("results")`. Fixed to unpack `result.get("documents", [[]])[0]` etc. (lists-of-lists).

2. **`_primary_document_extract` wrong content field** — was reading `proc.get("data").get("content")`. Fixed to `proc.get("content")` (top-level field in `process/file` response), with fallback chain to `upload_file` response `data.content` and `get_file` `data.content`.

3. **`generate_completion` hardcoded `stream=False`** — was ignoring caller's `stream` parameter. Fixed to pass through.

4. **`_primary_web_extract` wrong field** — was looking for `documents`/`docs` keys. Fixed to read `result.get("content", "")`.

5. **`user` field in `chat_completions`** — was sending `{"id": user_id}` dict; OpenAI spec requires string. Fixed to send `user_id` string directly.

6. **`messages` not injected into body** — Pipelines framework passes `messages` as a separate argument, but the engine reads it from `body["messages"]`. Added defensive injection at the start of `_run_research_async`.

7. **`DOMAIN_PRIORITY` / `CONTENT_PRIORITY` null validation error** — OWUI admin UI sends empty optional fields as JSON `null`; bare `str` Pydantic type rejects null. Fixed by changing both to `Optional[str]`.

8. **`<details>` block not rendered** — marked tokenizer regex requires `\n` immediately after the opening `>`. Old renderer put `<summary>` on the same line. Fixed by adding the required newlines (see rendering format above).

9. **"Future attached to a different loop"** — aiohttp session created in `on_startup()` was on the framework's main loop; worker threads use their own loops. Fixed by: removing session creation from `on_startup`, creating per-call `OWUIClient` in `_run_research_async`, adding `asyncio.set_event_loop(loop)` in each worker thread.

10. **No streaming output for 9 minutes** — status lines accumulated in memory, only flushed at message events or end-of-run. Fixed with `REASONING_FLUSH_SECONDS` time-based flush in `_push_status_line`.

11. **Final report not shown in QUIET_CHAT_MODE** — `_run_research` returns the report string; Pipelines framework discards return values. Fixed by capturing it in `_run_research_async` and pushing to sink after the last reasoning flush.

12. **Executor not shut down** — `ThreadPoolExecutor` leaked on shutdown. Fixed by adding `self.executor.shutdown(wait=False, cancel_futures=True)` in `on_shutdown`.

---

## Deployment configuration

### Container setup

The Pipelines container needs:
- `PIPELINES_API_KEY` env var (default `0p3n-w3bu!`) — used by OWUI to authenticate with Pipelines.
- `OWUI_API_KEY` env var (or set via OWUI admin UI > Valves) — an admin `sk-...` key for Pipelines to call back into OWUI.
- Volume mount at `/app/pipelines/data` (or `DATA_DIR`) for vocab embedding cache (~50–500 MB depending on model dimensions).

Intra-pod deployment (Pipelines container in the same K8s pod as OWUI): `OWUI_BASE_URL=http://localhost:8080`. Docker Compose: `OWUI_BASE_URL=http://open-webui:8080`.

### OWUI connection

In OWUI: Settings > Connections > add OpenAI-compatible connection pointing at the Pipelines container (e.g. `http://pipelines:9099`). The model `deep_research_pipeline.deep-research` will appear in the model list.

### Reasoning block flushing

`REASONING_FLUSH_SECONDS=4.0` is calibrated for interactive use. Setting it lower gives more granular streaming but more frequent block boundaries. Setting it to `0` disables the cap (flush only at explicit `done=True` events).

---

## OWUI runtime architecture (verified by reading installed source)

Source available locally at `.venv/lib/python3.11/site-packages/open_webui/` and inside the pod at `/app/backend/open_webui/`.

### Shared aiohttp session pool
- `open_webui/utils/session_pool.py` — singleton `aiohttp.ClientSession` shared across the entire OWUI process.
- Lazily created by `get_session()`. Closed *only* by `close_session()`.
- `close_session()` is called from exactly one place: `main.py:740-742`, inside the FastAPI `lifespan` context manager **after `yield`** — i.e. only on application shutdown.
- **Implication**: a `Closed shared aiohttp session pool` log line means uvicorn is shutting down. It is *not* a per-request cleanup signal.

### LLM call routing (pipe.py / OWUI Function path)
Pipe calls `generate_chat_completions` (imported from `open_webui.main`). Chain:
- `main.generate_chat_completions` → `utils/chat.generate_chat_completion` → `routers/openai.generate_chat_completion`
- For OpenAI-compatible models, that resolves the model via `request.app.state.OPENAI_MODELS[model_id]['urlIdx']` to:
  - `request.app.state.config.OPENAI_API_BASE_URLS[idx]`
  - `request.app.state.config.OPENAI_API_KEYS[idx]`
  - `request.app.state.config.OPENAI_API_CONFIGS[str(idx)]` (auth_type, azure flag, etc.)
- Equivalent for Ollama: `app.state.OLLAMA_MODELS` + `config.OLLAMA_BASE_URLS` / `OLLAMA_API_CONFIGS`.

### How aiohttp errors surface to a calling pipe
In `routers/openai.py::generate_chat_completion`, the bottom-level `except Exception` re-raises any aiohttp error (incl. `ServerDisconnectedError`) as `HTTPException(status_code=400, detail='Open WebUI: Server Connection Error')`. That HTTPException propagates back out of the pipe's `await generate_chat_completions(...)` call as a regular Python exception (catchable in the pipe).

---

## The "LLM disconnect crashes the pipe" investigation (pipe.py era)

### Symptom
When litellm flakes mid-research, the user sees:
```
ERROR | open_webui.routers.openai:generate_chat_completion - Server disconnected
ERROR | open_webui.main:process_chat - Error processing chat payload: Open WebUI: Server Connection Error
INFO  | open_webui.utils.session_pool:close_session - Closed shared aiohttp session pool
WARNING | open_webui.utils.middleware:response_handler - Task was cancelled!
INFO  | open_webui.main:process_chat - Chat processing was cancelled
```
…and the OWUI pod is restarted by Kubernetes. The user's research session is lost.

### Correct diagnosis
The cascade is a **clean uvicorn shutdown**, not an in-app cancellation triggered by the LLM error:
- `close_session` is only called from the FastAPI lifespan shutdown hook (verified: exactly one caller at `main.py:740-742`).
- `kubectl describe pod` showed `Last State: Terminated, Reason: Completed, Exit Code: 0`. Graceful uvicorn exit (SIGTERM → lifespan teardown → exit 0).
- `middleware.py:5028` is a *handler* for `CancelledError`; it doesn't *cause* the cancellation.

Chain: `something sends SIGTERM → uvicorn lifespan shutdown → close_session + all in-flight tasks cancelled → exit 0 → K8s restart per restartPolicy`.

### Likely root causes (not confirmed)
1. **Liveness probe timeout** — `compute_semantic_eigendecomposition`, KMeans, vocab embedding lookups block the asyncio event loop past a tight probe timeout.
2. **Memory pressure** — same heavy work + cached embeddings; eviction can show as `Reason: Completed` rather than `OOMKilled`.

### Diagnostic checks
```bash
kubectl get deploy <owui-deploy> -n <ns> -o yaml | grep -A10 -E 'livenessProbe|readinessProbe|resources'
kubectl top pod <owui-pod> -n <ns> --containers
kubectl get pod <owui-pod> -n <ns> -w
```

**Resolution:** moving the pipe to the Pipelines container (separate pod) eliminates the OWUI liveness probe concern entirely — the heavy CPU work no longer runs inside the OWUI uvicorn worker.

---

## Key takeaways for future Claude sessions

1. **`close_session` in OWUI logs == OWUI is shutting down.** Don't chase it as an in-app error path.
2. **`await asyncio.sleep(...)` inside a retry block is a no-op when the parent task is being cancelled** — `CancelledError` fires immediately.
3. **Bypassing `generate_chat_completions` loses OWUI semantics** — custom models, pipe-as-model routing, filters, access controls. Don't propose this.
4. **Pod-restart problems are infrastructure problems**, not pipe code problems. Look at probes / resources first.
5. **Never create an aiohttp session in `on_startup` for a Pipelines plugin** — it binds to the framework's main loop; worker threads use their own loops. Always create per-call.
6. **The marked extension tokenizer is strict** — `\n` after `<details ...>` and after `</summary>` are mandatory or the block renders as raw HTML.
7. **OWUI REST response shapes differ from what you'd guess** — always verify against the installed source in `.venv/lib/python3.11/site-packages/open_webui/routers/`. The shapes in the plan doc and comments can be wrong; the source is authoritative.
8. **Chat persistence requires API key ownership** — there is no admin bypass for `get_chat_by_id_and_user_id`. The `OWUI_API_KEY` user must own the chat.
9. **QUIET_CHAT_MODE return values must be explicitly pushed to the sink** — Pipelines discards `pipe()` return values; only yielded strings reach OWUI.
10. **`/api/embeddings` uses the chat models registry, not the RAG config** — `app.state.MODELS` is checked; the embedding model must be enabled in Admin > Models and its ID must match exactly. Disabled models are invisible to this endpoint even if configured in Documents settings.
11. **"Model not found" on `/api/chat/completions` or `/api/embeddings`** — check two things: (a) the model ID in the valve uses the correct prefix format for this OWUI instance, and (b) the model is enabled in Admin > Models.

---

## `pipe.py` code orientation (for reference only)

- All LLM calls funnel through `Pipe.generate_completion` (~L7506). 25+ call sites; mix of critical (synthesis, query generation, outline) and fail-soft (citation verify, group titles, smoothing, abstract).
- Pipe accesses OWUI internals via `self.__request__.app.state` — see `_openwebui_extraction_available` (~L4403), `load_vocabulary_embeddings` (~L1651), `_build_document_loader_kwargs` (~L4498).
- Direct aiohttp usage: `fetch_content` (~L5117), `_fallback_search` (~L6605).
- State persistence: `_load_persisted_dr_state` / `_save_persisted_dr_state` / `_checkpoint` (~L740–790).
- Entry point: `async def pipe(self, body, __user__, __event_emitter__, ...)` at ~L11652. Returns `comprehensive_answer` string in QUIET mode (auto-appended as assistant message by OWUI).

## Useful pod-side commands

```bash
# OWUI source
ls /app/backend/open_webui/utils/
sed -n '720,770p' /app/backend/open_webui/main.py     # lifespan handler
grep -rn "close_session\|session_pool" /app/backend/open_webui/

# Pipelines
docker logs <pipelines-container> -f --tail=100
```

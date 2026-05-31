# Deep Research pipe — investigation notes

## Minimum supported Open WebUI version

`pipe.py` targets **Open WebUI >= 0.9.2** (the async-ORM line). Every
`await Chats.*` / `await Knowledges.*` / `await Files.*` / `await
upload_file_handler` / `await process_file` site, and the
`open_webui.internal.db.get_async_db_context` import, only resolve on
OWUI 0.9.0+. Running the pipe against 0.8.x will raise `TypeError:
object ... can't be used in 'await' expression` or `ImportError` on
the first persistence/KB call.

## What this repo is

Two-file repo. Both files implement the same deep research engine (multi-cycle web research with semantic compression, eigendecomposition-based dimension tracking, KB persistence, and citation verification) but target different OWUI deployment environments:

| File | Class | Target | Entry point |
|---|---|---|---|
| `pipe.py` | `Pipe` | OWUI **Function** (runs inside OWUI container, shares its asyncio loop and `app.state`) | `async def pipe(self, body, __user__, __event_emitter__, ...)` |
| `deep_research_pipeline.py` | `Pipeline` | OWUI **Pipelines** plugin (runs in a separate Pipelines container, calls OWUI via REST) | `def pipe(self, user_message, model_id, messages, body) -> Iterator[str]` |

`pipe.py` is the **active production code** — it ships as an OWUI Function. `deep_research_pipeline.py` was an attempted port to the OWUI Pipelines runtime; that approach proved a dead end (see "Why the Pipelines port was abandoned" below) and is kept only as historical reference. **Do not put new fixes in `deep_research_pipeline.py`.**

### Why the Pipelines port was abandoned

The Pipelines runtime forces a per-call thread + new-event-loop pattern, decouples the plugin from OWUI's request lifecycle, and prevents direct ORM access for chat persistence (the REST chat endpoints filter by API-key user, so persistence silently fails for chats not owned by that user). Cumulative friction made it not worth maintaining as a parallel implementation. The documentation below is preserved as a reference for how that port worked; do not treat it as a target for new code.

---

## Concurrency contract (`pipe.py`)

OWUI instantiates `Pipe` once and calls `pipe()` concurrently for every user request. Concurrent invocations are **separate `asyncio.Task` instances on the same event loop** — not separate threads. The concurrency model in `pipe.py` is built on three rules.

**Per-call state lives in module-level `contextvars.ContextVar` instances and is surfaced on `Pipe` via property descriptors.** The list (top of `pipe.py`): event emitter and call hook, `__user__`, `__model__`, `__request__`, `conversation_id`, `chat_id`, `is_pdf_content`, `research_date`, `trajectory_accumulator`, `_seen_subtopics`, `_seen_sections`. Each is declared as a ContextVar at module top and read/written through a `_ctxvar_prop` descriptor. `pipe()` initialises every slot at entry. A bare `self.foo = bar` in `Pipe.__init__` or `pipe()` for new per-call state will silently leak between concurrent users — add a new ContextVar instead.

**Process-shared state lives on `Pipe` directly, guarded by `asyncio.Lock`.** This is limited to: `valves`, `state_manager`, `embedding_cache`, `transformation_cache`, `vocabulary_cache`, `vocabulary_embeddings`, `executor`, plus the locks (`_inflight_lock`, `_vocab_load_lock`, and the per-cache `_lock`) and the inflight set (`_inflight`). `asyncio.Lock` is the right primitive because `pipe.py` runs in OWUI's single event loop. `threading.Lock` is unnecessary; `threading.local` would be **wrong** — all concurrent `pipe()` coroutines share the same thread, so threadlocal slots would be shared, not isolated.

**ContextVars do NOT propagate across `loop.run_in_executor`.** Callables submitted to `self.executor` run in pool worker threads whose ContextVar slots are unrelated to the calling Task's context. Never read per-call attributes (`self.__user__`, `self.conversation_id`, etc.) inside a function passed to `run_in_executor`; pass per-call values in as explicit closure arguments. At present this rule is observed — every executor callable in `pipe.py` captures only local args (`_run_load`, `extract_with_pypdf`, `extract_with_pdfplumber`, `extract_with_bs4`). Audit any new callbacks you add against this rule.

**Same-conversation entry dedupe.** `pipe()` rejects a second invocation for a `conversation_id` already in `self._inflight`. Two concurrent calls on the same conversation would share the `ResearchStateManager` dict and corrupt state through interleaved read-modify-write across `await` boundaries; rather than retrofitting a lock around 60+ state mutation sites, the second invocation is rejected at entry with a notice.

---

## Architecture: `deep_research_pipeline.py` (historical, abandoned)

**Abandoned.** This pipeline is no longer maintained; fixes go in `pipe.py`. The sections below describe how the port worked and what was learned from it. Do not extend or maintain this code.

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

## Current `pipe.py` implementation details worth remembering

### Cache implementation

- Both in-process vector caches now share a byte-bounded LRU base class, `_LRUBytesBoundedCache`.
- `EmbeddingCache` and `TransformationCache` store dense `numpy.float32` arrays internally and materialize fresh Python `list[float]` values on cache hits. This keeps memory bounded while preserving the external caller contract.
- Cache keys for text inputs must be process-stable: use `_stable_text_key(text)` rather than Python `hash(...)`. This matters for both plain embedding lookups and transformation cache keys.
- Cache values are intentionally snapshotted on write and materialized on read. Do not return internal array references or reuse caller-owned mutable inputs directly.
- Cache eviction is true LRU, not FIFO. Reads update recency via `move_to_end(...)` semantics.
- Cache sizing is controlled by valves: `EMBEDDING_CACHE_MAX_MB` and `TRANSFORMATION_CACHE_MAX_MB`. These are read in `Pipe.__init__`; applying a new value requires a process restart.
- Cache stats now reflect actual hits/misses/evictions and byte usage. If you extend stats, keep the semantics aligned with real cache behavior.

### Persistent and per-conversation state shape

- `ResearchStateManager` storage for `completed_topics` and `irrelevant_topics` is JSON-safe `list`, not `set`. When `_run_pipeline` needs set operations, it converts those lists to local `set(...)` working copies and converts back to `list(...)` on write-back.
- The persisted deep-research state saved into chat metadata is still a JSON-shaped structure. New fields intended for persistence must stay JSON-serializable.
- `conversation_id` is the real conversation identity key. Fresh-state provisioning happens when that id is first seen; there is no longer any `len(messages) <= 2` shortcut for deciding whether a conversation is new.
- Post-report QA depends on preserving state across short follow-up turns. Any future “new conversation” heuristic based on message count, task type, or short-turn shape is likely wrong unless it keys off actual conversation identity.
- `research_completed` remains meaningful for follow-up detection, but `emit_status` must derive UI status from the current `done` flag only.

### Completion and retry behavior

- `Pipe.generate_completion` must distinguish model failure from model output. Terminal failures raise `CompletionError`; they are not converted into completion-shaped payloads.
- Transient retry classification lives in `_classify_transient_completion_error(...)`. It prefers exception type and HTTP status code inspection, with substring matching only as a fallback.
- OWUI connection-wrapper failures can surface as `HTTPException` carrying `detail='Open WebUI: Server Connection Error'`; the retry classifier explicitly knows about that wrapper.
- The front-of-pipeline planning stages are treated as hard dependencies. If initial query or outline generation raises `CompletionError`, the run should abort cleanly rather than continue with placeholder text.

### Vocabulary and disk-cache paths

- Deep-research disk caches now live under OWUI's resolved `CACHE_DIR`, inside `CACHE_DIR / "deep_research"`, not under a hardcoded `/app/backend/data/...` path.
- That applies to both the vocabulary text cache and vocabulary-embedding `.npz` files.
- If you add new on-disk caches for `pipe.py`, prefer sibling paths under `CACHE_DIR / "deep_research"` so they follow OWUI's resolved data-dir behavior.
- Vocabulary lazy loading is process-shared and protected by `_vocab_load_lock`; keep any new lazy-init logic for shared vocabulary assets inside that locking pattern.

### Executor and async patterns

- Inside coroutines, use `asyncio.get_running_loop()`, not `asyncio.get_event_loop()`.
- `run_in_executor` callables must receive all per-call inputs explicitly, because `ContextVar` state does not propagate into executor threads.
- `asyncio.to_thread(...)` is not a drop-in replacement for existing `run_in_executor(self.executor, ...)` sites when the code relies on the custom executor's concurrency profile.

### Fetching / extraction implementation details

- `fetch_content` now routes HTML URLs only through `_try_primary_web_flow`. If Open WebUI extraction is unavailable/fails/quality-rejected, `fetch_content` returns the OWUI extraction failure string (`Error fetching content: OWUI extraction failed for <url>`); there is no legacy HTML direct-fetch fallback path.
- The retained legacy web-fetch machinery (fake UA, spoofed headers/cookies, per-domain rate limiting via `domain_session_map`) is now scoped to `_fetch_pdf_via_legacy_download` for PDF-classified URLs.
- `fake_useragent.UserAgent()` is initialized once at module import time, with a static fallback UA list if import or provider construction fails. Do not move that work into per-request paths.
- Domain cookie state used by `_fetch_pdf_via_legacy_download` is treated as mapping-shaped data; the old alternate conversion path is gone because `SimpleCookie` already behaves like a `dict` for the relevant usage here.
- Same-domain session reuse in `_fetch_pdf_via_legacy_download` depends on the existing `domain_session_map` entry being initialized before cookie reuse. Preserve that control-flow assumption if you refactor the method.

### Small but meaningful code-shape rules

- `Pipe.get_research_model` no longer exists; read `self.valves.RESEARCH_MODEL` directly. `get_synthesis_model()` still exists because it contains real fallback logic.
- Redundant in-function imports already covered by module-top imports were cleaned out. If you add imports inside a function, there should be a concrete reason such as lazy loading, circular-import avoidance, or optional dependency handling.

## Key takeaways for future Claude sessions

1. **`close_session` in OWUI logs == OWUI is shutting down.** Don't chase it as an in-app error path.
2. **`await asyncio.sleep(...)` inside a retry block is a no-op when the parent task is being cancelled** — `CancelledError` fires immediately.
3. **Bypassing `generate_chat_completions` loses OWUI semantics** — custom models, pipe-as-model routing, filters, access controls. Don't propose this.
4. **Pod-restart problems are infrastructure problems**, not pipe code problems. Look at probes / resources first.

---

## `pipe.py` code orientation (for reference only)

- All LLM calls funnel through `Pipe.generate_completion` (~L7506). 25+ call sites; mix of critical (synthesis, query generation, outline) and fail-soft (citation verify, group titles, smoothing, abstract).
- Pipe accesses OWUI internals via `self.__request__.app.state` — see `_openwebui_extraction_available` (~L4403), `load_vocabulary_embeddings` (~L1651), `_build_document_loader_kwargs` (~L4498).
- Direct aiohttp usage: `fetch_content` (~L5117).
- Web search path is OWUI-only: `search_web` computes the result budget once and passes `total_results` to `_try_openwebui_search`, which returns `(results, failure_reason)` so logs distinguish no-results from OWUI search failure.
- State persistence: `_load_persisted_dr_state` / `_save_persisted_dr_state` / `_checkpoint` (~L740–790).
- Entry point: `async def pipe(self, body, __user__, __event_emitter__, ...)` at ~L11652. Returns `comprehensive_answer` string in QUIET mode (auto-appended as assistant message by OWUI).

## `deep_research/` package — implementation notes

These document non-obvious behavior in the refactored `deep_research/` package
(the active code; `pipe.py` is the frozen pre-refactor monolith).

### Citation marker replacement is first-occurrence only

In `orchestrator/phases/synthesize.py`, after sections are generated, raw inline
citation markers are rewritten to numbered references against the
`global_citation_map`:

```python
modified = modified.replace(raw, f"[{global_citation_map[url]}]", 1)
```

The trailing `1` is deliberate — it replaces **only the first occurrence** of
`raw` per citation entry, not all of them. `raw` is `cit.get("raw_text")`, the
free-text citation fragment the LLM emitted; it can be a short phrase that also
appears verbatim in running body text. An unbounded `.replace()` would rewrite
those legitimate text occurrences into stray `[N]` markers and corrupt the prose.

**Trade-off / known limitation:** if the *same* citation marker legitimately
appears multiple times in one section and every instance should be numbered, this
will only tag the first. That is the accepted behavior — prefer under-tagging to
corrupting body text. If a future change needs all-occurrence tagging, do it with
a bounded regex that matches only the marker shape, not a blanket
`str.replace(raw, ...)`.

### Model-response parsing goes through `response_text()`

`core/text.py::response_text(response)` is the single safe accessor for OWUI chat
completion responses. It returns `""` for any None / malformed / empty response
instead of raising. Every site that needs the assistant content must use it — do
**not** reintroduce raw `response["choices"][0]["message"]["content"]` subscripts,
which crash the whole run on a single bad model reply.

### No fabricated embedding vectors

When `get_embedding()` returns nothing, callers (`orchestrator/phases/cycles.py`,
`orchestrator/phases/initial_queries.py`) **skip the query with a logged warning**
rather than injecting a placeholder vector. The old `[0.0] * 384` fallback was
doubly wrong: a zero vector makes cosine similarity divide by zero (→ NaN that
poisons all downstream ranking), and 384 ≠ the real embedding dimension
(e.g. `nomic-embed-text` is 768), causing shape mismatches. `similarity.py`
additionally guards every normalize/dot via `_safe_normalize()` (returns `None`
for zero/empty/non-finite vectors), so any stray bad vector yields a `0.0`
similarity instead of NaN. Do not reintroduce dimension-hardcoded fallbacks.

### `ctx.state` is the manager, not the conversation dict

`RunContext.state` is a `ResearchStateManager`. To read/write per-conversation
state you MUST go through `ctx.state.get_state(ctx.conversation_id)` (returns the
live dict you can mutate in place) or `ctx.state.update_state(conv_id, key, val)`.
The manager has **no** `__getitem__`/`__setitem__`/`.get`, so `ctx.state[key]`,
`ctx.state[key] = v`, and `ctx.state.get(...)` all raise at runtime. The
synthesis modules had ~31 such sites (`state = ctx.state` followed by
`state.get(...)`, plus `ctx.state[key] = ...`) that crashed the whole synthesis
phase on every run; all now use `state = ctx.state.get_state(ctx.conversation_id)`
and mutate that dict. Unit tests missed this because the test fixture happened to
exercise only the manager API — see the smoke test note below.

### `RuntimeConfig.data_dir` is coerced to `pathlib.Path`

The runtime shims pass `DR_DATA_DIR` as a **str**, but the vocabulary disk-cache
code does `ctx.config.data_dir / "deep_research"`. `RuntimeConfig.__init__` now
coerces `self.data_dir = Path(data_dir)` so that works. Don't assume `data_dir`
is a str elsewhere.

### Vocabulary uses TWO locks (deadlock hazard)

`load_vocabulary_embeddings()` calls `load_vocabulary()` *while holding its own
lock*. `asyncio.Lock` is **not reentrant**, so they must use distinct locks
(`_vocab_emb_load_lock` and `_vocab_load_lock`). A single shared lock deadlocks
the whole run (hangs forever). Keep them separate.

### Local smoke test (no live OWUI required)

`tests/test_smoke_e2e.py` drives `Coordinator.stream()` (the exact coroutine the
OpenAPI `/research` endpoint uses) end-to-end through all 9 phases with every OWUI
REST endpoint mocked via `respx`. It asserts: (a) the prompt reaches the outbound
LLM call, (b) the KB is named from the prompt, (c) a same-conversation follow-up
answers from the KB without a new search crawl, (d) a substantial report is
produced, (e) no error-level events are emitted. It now runs **in-process** as
part of the normal `pytest` suite (a module-scoped fixture runs the two-turn
scenario once via `asyncio.run`; individual behaviors are separate test
functions), so the synthesis/research/web paths it drives are counted under
coverage. Run the whole suite with `.venv/bin/python -m pytest`, or just this
module with `make smoke` (= `pytest tests/test_smoke_e2e.py -v`). This is the fast
way to catch full-pipeline regressions that the narrower unit tests miss — every
bug in the four notes above was caught by this end-to-end scenario, not by the
narrower per-function tests.

## Useful pod-side commands

```bash
# OWUI source
ls /app/backend/open_webui/utils/
sed -n '720,770p' /app/backend/open_webui/main.py     # lifespan handler
grep -rn "close_session\|session_pool" /app/backend/open_webui/

# Pipelines
docker logs <pipelines-container> -f --tail=100
```

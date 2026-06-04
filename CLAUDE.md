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

A multi-cycle web research engine (semantic compression,
eigendecomposition-based dimension tracking, KB persistence,
citation verification) packaged as the `deep_research/` Python
package. The package ships in three runtimes, all of them
shims over the same `Coordinator.run()` coroutine:

| Entrypoint | Target | Shape |
|---|---|---|
| `deep_research/entrypoints/owui_function/pipe.py` | OWUI **Function** (runs inside OWUI container) | `async def pipe(self, body, __user__, __event_emitter__, ...)` |
| `deep_research/entrypoints/openapi_tool/server.py` | **OpenAPI Tool Server** (separate container, REST + iframe) | FastAPI app with `/research_jobs`, `/live_view`, etc. |
| `deep_research/entrypoints/mcp/server.py` | **MCP server** (Streamable-HTTP) | Single FastMCP tool `deep_research(prompt, conversation_id?)` |

The OWUI Pipelines plugin was removed (OWUI itself flagged
Pipelines as legacy); the `owui_pipeline/` directory no longer
exists.

### OpenAPI Tool Server runtime — Phase 1 v2 endpoint surface

The OpenAPI Tool Server was rewritten around a **two-call** workflow:

  - `POST /research_jobs` starts a job and returns immediately with a
    `job_id` plus `user_facing_instruction` (verbatim text the LLM
    must surface so the user knows the slash-command grammar).
  - `POST /research_jobs/{job_id}/feedback` forwards the user's
    `/k 1,3,5` / `/r 2,4` / `/continue` (or freeform) reply and
    resumes the engine.
  - `GET /research_jobs/{job_id}` JSON snapshot.
  - `POST /research_jobs/{job_id}/cancel` cooperative cancellation
    via a `CancellationToken` checked at every phase boundary.
  - `GET /live_view/{job_id}` HTML iframe — per-job view tokens,
    sha256-hashed at rest, self-polling.
  - `GET /live_view/{job_id}/status` JSON snapshot used by the iframe.

Job state lives in a durable `aiosqlite` store
(`deep_research/entrypoints/openapi_tool/jobs.py`). A server restart
no longer drops in-flight runs (running engine state in
`ResearchStateManager` IS lost, but the JobRecord survives so
clients see a usable `failed` state instead of a 404).

The engine's suspend/resume model wasn't extended for this — the
runner just calls `Coordinator.run()` twice on the same
`conversation_id`. State manager keeps the outline-feedback data
between calls. See `entrypoints/openapi_tool/runner.py`.

`target_message_id` lives on `RunContext` (defaults `None`; only the
OpenAPI runner populates it). It rebinds on each
`submit_research_feedback` call so the Phase 2 writeback channel
posts to the *new* tool-call assistant message, not the prior one.

### `/event` endpoint admin-bypass

OWUI's per-message endpoint
`POST /api/v1/chats/{id}/messages/{message_id}/event` is distinct
from the chat-update endpoints documented in the "Chat persistence
limitation" section below. Unlike `GET /api/v1/chats/{id}`, the
`/event` endpoint **does** allow an admin token to post events to
chats owned by other users. This is what makes the Phase 2
writeback channel viable for the OpenAPI Tool Server (which holds an
admin `DR_OWUI_API_KEY`).

The accepted short event names that OWUI persists to the chat-message
row are: `status`, `message`, `replace`, `embeds`, `files`, `source`
/ `citation`. Long-name aliases (`chat:message:embeds` etc.)
broadcast but do NOT persist.

### Phase 2 writeback wiring

The OpenAPI Tool Server uses a **second** `OWUIClient` bound to a
static admin `DR_OWUI_API_KEY`. It's instantiated by
`Coordinator.start(writeback_token=...)` and exposed as
`Coordinator.writeback_client`. The user-request `OWUIClient` keeps
its per-request `ContextTokenProvider` for everything else — only the
`/event` POST path needs the admin token, because only that endpoint
admin-bypasses chat ownership.

The `OutboxWorker` (`entrypoints/openapi_tool/outbox.py`) is the
durable writeback queue. It lives in the same sqlite file as
`research_jobs` so a single `DR_DATA_DIR` volume gives durability for
both. The runner translates each engine event to an `OutboxRow` via
`_event_to_outbox`:

  - `StatusEvent` → `status` (dedupe_key uses a monotonic per-job counter)
  - `MessageEvent` → `replace` (the topic list at the gate; the LLM
    no longer needs to reproduce it)
  - `EmbedEvent` → `embeds` (engine-emitted iframe HTML)
  - `CitationEvent` → `source` (side-panel citations, emitted per
    bibliography entry in `phases/finalize.py`)

Two writebacks happen *outside* the sink:

  - **Bootstrap embed** at `start_job` and again at `submit_feedback`
    (rebinds to the new tool-call message). Posts the iframe HTML with
    a `replace: true` flag, baked with the per-job view token and the
    `DR_OPENAPI_PUBLIC_BASE_URL`-resolved status poll URL.
  - **Final writeback** after `coord.run()` returns: a `replace` with
    `report.content`, followed by an `embeds` with `[]` to clear the
    iframe.

The drain order is `next_attempt_at ASC, rowid ASC` — `rowid` is
sqlite's auto-increment, so rows enqueued in the same second still
deliver in insertion order. UUIDs as `outbox_id` are useless for
ordering and the original `outbox_id ASC` secondary sort produced
nondeterministic delivery sequence.

Worker loop spawn is opt-out: `OutboxWorker.start(spawn_loop=False)`
opens the sqlite connection without spawning the background drain.
The tests use this mode so `drain_once()` is deterministic; production
uses `spawn_loop=True` (the default).

Skip conditions in `_event_to_outbox` / bootstrap / final-writeback:
`chat_id is None`, `target_message_id is None`, or
`chat_id.startswith("local:")`. The `local:` prefix is OWUI's
ephemeral-chat marker — the `/event` endpoint accepts the POST but
the event is dropped, so we don't waste an HTTP round-trip.

### Phase 2 deferred items and decisions

These are intentional non-implementations from the Phase 2 rollout —
documented so a future session doesn't try to "fix" something that's
already the deliberate behaviour, and so the rationale is preserved
when someone does want to revisit.

**No `build_initial_snapshot(record)` helper.** The Phase 2 plan named
this factory but the implementation inlines the equivalent dict-merge
into `_enqueue_bootstrap_embed` at
`deep_research/entrypoints/openapi_tool/runner.py:443-468` instead.
The snapshot consumed by the bootstrap iframe is exactly the same one
the live-view iframe polls for, already cached in
`self._snapshots[job_id]`; we just stamp `{"query": record.prompt}` on
top before rendering. Extracting a separate factory for one caller is
premature abstraction. If a second caller appears (e.g., a future
"resume after restart" path that rehydrates the snapshot from
`JobRecord` fields), extract it then.

**MCP cancellation is implicit, not wired to a FastMCP hook.**
FastMCP 3.x's `Context` doesn't expose a direct cancel signal a tool
function can subscribe to. The MCP runtime relies on anyio cancelling
the enclosing task when the client sends a cancel notification, which
surfaces as `asyncio.CancelledError` inside the tool function. The
`deep_research` MCP entrypoint
(`deep_research/entrypoints/mcp/server.py`) catches that, calls
`cancel_token.cancel()` on its locally-held `CancellationToken`, and
re-raises. Net behaviour: cancellation unwinds the engine, but the
phase-boundary checks in `ctx.raise_if_cancelled()` only fire if the
engine is between awaits when cancel arrives (the natural `CancelledError`
from anyio handles the await-blocked case directly). This is the
closest mapping FastMCP 3.x permits; revisit if a future FastMCP
release exposes a first-class cancel hook in `Context`.

**`PERSISTED_EVENT_TYPES` accepts both `source` and `citation`.**
OWUI's docs treat these as aliases for the side-panel citation event.
Phase 2 settles on emitting `source` (from `CitationEvent.to_dict()`
in `deep_research/progress/events.py`) and the
`OWUIClient.post_message_event` validator at
`deep_research/adapter/client.py:21-23` permits either. If OWUI ever
splits the two into distinct semantics (currently they look like
historical aliases), pick one and remove the other from the validator
set. The runner's `_event_to_outbox` should stay pinned to whichever
side wins.

**Citation/source mapping has dedicated tests; do not add defensive
filters.** `_event_to_outbox`'s CitationEvent branch at
`deep_research/entrypoints/openapi_tool/runner.py:481-491` is pinned by
five tests:
`tests/test_openapi_outbox.py::test_source_payload_round_trip`,
`tests/test_openapi_writeback_e2e.py::test_citation_event_maps_to_source_row`,
`...::test_multiple_citations_with_same_url_deduplicate`,
`...::test_runner_emits_source_not_citation`, and
`...::test_runner_does_not_filter_empty_url_citation`. Two pinned contracts
are non-obvious and the tests will fail loudly if accidentally reverted:
(a) URL alone is the dedupe identity; same URL + different snippet = the
first emission wins via `INSERT OR IGNORE`. (b) The runner passes
CitationEvent through even with empty `url`; the sole gatekeeper is
`deep_research/orchestrator/phases/finalize.py:83-85`. Do not add a
"defensive" empty-URL filter to the runner — splitting responsibility
across two sites makes future maintenance ambiguous.

`CitationEvent.snippet` is populated from
`master_source_table[url].content_preview` via `BibliographyEntry.snippet`
(set in `deep_research/synthesis/citations.py::generate_bibliography`).
Every CitationEvent emitted from `finalize.py:89` carries up to a 500-char
excerpt that renders as the `document` array in OWUI's side-panel
citation. The 500-char cap is enforced at source-registration time, NOT
at emit time — do not re-truncate. Sites that populate `content_preview`:

  - `deep_research/web/fetch.py:226` — primary HTML/PDF cache path
  - `deep_research/web/fetch.py:398` — archived-source registration
    (Wayback-style)
  - `deep_research/web/paywall.py:247` — paywall-stripper PDF path
  - `deep_research/web/search.py:175` — search-result registration
  - `deep_research/synthesis/sections.py:452` — URLs cited in a section
    but not previously fetched; populates `content_preview=""` by design
    (no fresh source text at synthesis time). Surfaces in OWUI as a
    citation with an empty `document` array — same shape as the
    pre-wiring fallback.

**Outbox cancellation on shutdown is not flushed.** When the lifespan
shutdown handler runs `OutboxWorker.stop()`, any rows whose
`delivered_at IS NULL` stay in the table. On the next process start,
the worker picks them up and continues delivery — this is durability
by design (a process crash mid-writeback resumes correctly). The
trade-off is that a *clean* shutdown leaves rows undelivered until
the next start; we don't drain-on-stop. Acceptable because the rows
target message_ids that still exist in OWUI's chat table on restart.
Don't add a defensive "drain on stop" call — it would block shutdown
on OWUI availability.

**Multi-process / multi-replica OpenAPI Tool Server is out of scope
for v1.** Both the `JobStore` (`research_jobs` table) and the
`OutboxWorker` (`owui_outbox` table) live in a single sqlite file at
`{DR_DATA_DIR}/jobs.sqlite`. Two server processes writing to the same
sqlite file produce `database is locked` errors at scale; sqlite
serialises writes via file locks and busy_timeout only buys time. The
deeper issue: `Coordinator._state_manager` and the four
`CacheBundle` caches are process-shared *within* one process. Two
server processes have two independent copies, so a request that hit
process #1 for `start_research_job` and process #2 for
`submit_research_feedback` would resume on the wrong replica with an
empty state manager — the engine would either invent fresh outline
data or fail. Horizontal scaling needs (a) PostgreSQL or equivalent
for the two tables and (b) either sticky sessions keyed on
`conversation_id` or moving the engine state into the shared store.
Neither is wired today; the runtime documents itself as single-process.

(Note: same-process concurrency across different users / different
`conversation_id`s is fully supported and is *not* the same problem.
`ResearchStateManager` dispenses per-conversation dicts and every
`Coordinator.run()` call builds a fresh `RunContext` — no
cross-conversation taint inside a single process. The earlier
implementation's contamination bug was caused by per-call state being
held on module/Pipe-level attributes; the new package has no such
sites by construction.)

**Slash-command grammar lives in three places.** The user-facing
slash-command grammar (`/k`, `/keep`, `/r`, `/remove`, `/continue`,
`/c`, plus range syntax `5-7`) is referenced by three independent
sites:

  1. `deep_research/research/outline_feedback.py::render_outline_prompt`
     — the in-chat prompt the user sees and types against.
  2. `deep_research/research/outline_feedback.py::process_outline_feedback_continuation`
     — the parser that consumes the user's reply.
  3. `deep_research/entrypoints/openapi_tool/schemas.py::StartResearchResponse.user_facing_instruction`
     — the *pre-outline* teaser the LLM is told to emit verbatim
     immediately after calling `start_research_job`.

Any change to the grammar — rename, add, or remove a command — MUST
touch all three sites in the same commit. Sites #1 and #2 are pinned
together by `tests/test_outline_prompt.py` and the parser's own
regex (`outline_feedback.py:249-250`); site #3 is pinned by
`tests/test_openapi_jobs.py:99-100, 110-116`.

Do NOT try to consolidate #3 into a single source of truth with #1.
They serve different audiences (LLM re-emission vs direct-to-user
chat content) and different timings (#3 fires before the outline
exists; #1 fires when it's ready). The intentional duplication is
documented here so a future cleanup pass doesn't accidentally
collapse them.

The `/q` and `/quit` cancel commands live in **two** sites, not three,
because they're never parsed by the engine's
`process_outline_feedback_continuation` as a normal command — they
raise `asyncio.CancelledError` if they reach the parser at all (the
LLM is expected to route them to `cancel_research_job` directly). The
two sites:

  1. `deep_research/entrypoints/openapi_tool/server.py::CANCEL_DESCRIPTION`
     — the tool-description text the LLM reads.
  2. `deep_research/entrypoints/openapi_tool/schemas.py::StartResearchResponse.user_facing_instruction`
     — the user-facing teaser the LLM is told to emit verbatim.

**`local:` chats are refused at `start_research_job`.** The OpenAPI
Tool Server returns 409 with code `unsaved_chat_unsupported` for any
`X-OpenWebUI-Chat-Id` starting with `local:` (OWUI's ephemeral chat
marker). Writeback to a `local:` chat would silently no-op — OWUI's
`/event` returns 200 but drops the event because the chat doesn't
persist — and starting a multi-minute research run invisible to the
user is bad value. The LLM is instructed (via the appended paragraph
in `START_DESCRIPTION`) to relay the refusal message and ask the user
to send a brief message in the chat to promote it from
`local:<random>` to a persisted UUID. The `_writeback_target` skip
in `deep_research/entrypoints/openapi_tool/runner.py:64-65` stays as
defence-in-depth for any pre-upgrade JobRecord still in
`jobs.sqlite` and to keep the existing
`tests/test_openapi_writeback_e2e.py::test_writeback_skipped_when_chat_id_is_local`
invariant honest.

**Terminal writebacks no longer clear the iframe.** Both successful
completion and cancellation go through `_enqueue_terminal_writeback`
in `deep_research/entrypoints/openapi_tool/runner.py`, which posts a
`status` pill (`done=true`) and a `replace` with the message body.
The iframe (last `refresh_progress_embed` snapshot, typically showing
the categorised topic dashboard) is preserved. Prior behaviour cleared
the iframe with `embeds: []`; that destroyed the user's progress
reference and is no longer done. The matching writeback test pinned in
`tests/test_openapi_writeback_e2e.py::test_writeback_sequence_through_full_lifecycle`
explicitly asserts no `embeds: []` clear is enqueued.

**`runner.cancel` handles two paths.** When the engine task is
actively running, cancellation flows through the token → phase
boundary → `asyncio.CancelledError` → handler in `_run_initial` /
`_run_feedback`. When the task is already `.done()` (paused at the
outline-feedback gate), the handler won't fire; `runner.cancel`
updates the phase + enqueues the writeback inline. Both paths land at
the same `_enqueue_terminal_writeback` helper. Pinned by
`tests/test_openapi_runner.py::test_cancel_at_gate_marks_cancelled`
and the matching e2e
`tests/test_openapi_writeback_e2e.py::test_gate_cancel_posts_status_and_replace`.

**Bootstrap iframe is no longer posted in the preliminary phase.**
`start_job` does not call `_enqueue_bootstrap_embed`; the iframe is
first posted on `submit_feedback` (the moment the engine moves into
research and the live snapshot becomes meaningful). The only
user-visible content in the preliminary tool-call message is the
topic-list `MessageEvent → replace`.

**Config warning code taxonomy.** The OpenAPI Tool Server runs a
startup configuration audit
(`deep_research/entrypoints/openapi_tool/config_audit.py`) plus one
runtime detector inside `start_research_job`. Results are cached on
`app.state.config_warnings` and surfaced via `GET /health` as JSON
(`config_warnings: list[{code, severity, message, remediation}]`).
The stable `code` values:

  - `MISSING_OWUI_API_KEY` — `DR_OWUI_API_KEY` unset while
    `valves.jobs.writeback_enabled=True`. The lifespan also fails
    fast (raises `RuntimeError`) on this exact condition before the
    audit runs; the audit retains the check so a future ops escape
    hatch that disables the fail-fast still surfaces the warning.
  - `OWUI_API_KEY_NOT_ADMIN` — probe (`OWUIClient.get_session_user()`
    → `GET /api/v1/auths/`) returned `role != "admin"`. Writeback
    POSTs would 401 at runtime; the server still starts.
  - `OWUI_API_KEY_PROBE_FAILED` — the admin probe raised. Could be
    transient infrastructure (OWUI not yet up); does NOT fail-fast.
  - `MISSING_PUBLIC_BASE_URL` — `DR_OPENAPI_PUBLIC_BASE_URL` unset.
    Severity `info` (degraded UX, not broken) — the iframe falls
    back to the inbound request's host header.
  - `OWUI_HEADERS_NOT_FORWARDED` — detected at runtime (not startup):
    the first authenticated request to `start_research_job` arrived
    without `X-OpenWebUI-Chat-Id`. One-shot per process via
    `app.state._forward_headers_warned`. Means OWUI hasn't set
    `ENABLE_FORWARD_USER_INFO_HEADERS=true`.

Probe endpoint is `GET /api/v1/auths/`, not `GET /api/v1/users/`:
`auths/` admits any valid token and returns the holder's `role`
field, so we can check `role == "admin"` directly without inferring
admin status from 401/403 codes. The matching adapter method is
`OWUIClient.get_session_user()` in `deep_research/adapter/client.py`.

**The audit is intentionally not re-run on every `/health` call.**
K8s readinessProbe defaults to `periodSeconds=10`, so re-probing OWUI
on each call would add ~360 calls/hour per pod purely for cosmetic
warnings, and a transient OWUI outage would CrashLoopBackOff the tool
server. The cached result is stale only after token rotation or OWUI
config changes, which normally trigger a Secret rotation → pod
restart anyway.

**Fail-fast scope is narrow on purpose.** Only `MISSING_OWUI_API_KEY`
(combined with `writeback_enabled=True`) raises at startup.
`OWUI_API_KEY_NOT_ADMIN` and `OWUI_API_KEY_PROBE_FAILED` do not
fail-fast because the former can't be checked synchronously before
the event loop is up and the latter is often a transient infra
issue. `MISSING_PUBLIC_BASE_URL` is informational only.

---

## Concurrency contract (the engine)

OWUI instantiates `Pipe` once and calls `pipe()` concurrently for every user request. Concurrent invocations are **separate `asyncio.Task` instances on the same event loop** — not separate threads. The concurrency model in `pipe.py` is built on three rules.

**Per-call state lives in module-level `contextvars.ContextVar` instances and is surfaced on `Pipe` via property descriptors.** The list (top of `pipe.py`): event emitter and call hook, `__user__`, `__model__`, `__request__`, `conversation_id`, `chat_id`, `is_pdf_content`, `research_date`, `trajectory_accumulator`, `_seen_subtopics`, `_seen_sections`. Each is declared as a ContextVar at module top and read/written through a `_ctxvar_prop` descriptor. `pipe()` initialises every slot at entry. A bare `self.foo = bar` in `Pipe.__init__` or `pipe()` for new per-call state will silently leak between concurrent users — add a new ContextVar instead.

**Process-shared state lives on `Pipe` directly, guarded by `asyncio.Lock`.** This is limited to: `valves`, `state_manager`, `embedding_cache`, `transformation_cache`, `vocabulary_cache`, `vocabulary_embeddings`, `executor`, plus the locks (`_inflight_lock`, `_vocab_load_lock`, and the per-cache `_lock`) and the inflight set (`_inflight`). `asyncio.Lock` is the right primitive because `pipe.py` runs in OWUI's single event loop. `threading.Lock` is unnecessary; `threading.local` would be **wrong** — all concurrent `pipe()` coroutines share the same thread, so threadlocal slots would be shared, not isolated.

**ContextVars do NOT propagate across `loop.run_in_executor`.** Callables submitted to `self.executor` run in pool worker threads whose ContextVar slots are unrelated to the calling Task's context. Never read per-call attributes (`self.__user__`, `self.conversation_id`, etc.) inside a function passed to `run_in_executor`; pass per-call values in as explicit closure arguments. At present this rule is observed — every executor callable in `pipe.py` captures only local args (`_run_load`, `extract_with_pypdf`, `extract_with_pdfplumber`, `extract_with_bs4`). Audit any new callbacks you add against this rule.

**Same-conversation entry dedupe.** `pipe()` rejects a second invocation for a `conversation_id` already in `self._inflight`. Two concurrent calls on the same conversation would share the `ResearchStateManager` dict and corrupt state through interleaved read-modify-write across `await` boundaries; rather than retrofitting a lock around 60+ state mutation sites, the second invocation is rejected at entry with a notice.

---

## Valve model ID format

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

`tests/test_smoke_e2e.py` drives `Coordinator.run(sink=...)` (the exact
coroutine the OpenAPI `POST /research` endpoint uses) end-to-end through all
9 phases with every OWUI REST endpoint mocked via `respx`. Events emitted
through the pipeline are captured by a list-appending sink for the assertions.
It asserts: (a) the prompt reaches the outbound
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
```

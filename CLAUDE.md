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
    `job_id`, a per-job `view_token` (the cleartext live-view token;
    sha256-hashed at rest), and `user_facing_instruction` (verbatim
    text the LLM must surface so the user knows the slash-command
    grammar).
  - `POST /research_jobs/{job_id}/feedback` forwards the user's
    `/k 1,3,5` / `/r 2,4` / `/continue` (or freeform) reply and
    resumes the engine.
  - `GET /research_jobs/{job_id}` JSON snapshot.
  - `POST /research_jobs/{job_id}/cancel` cooperative cancellation
    via a `CancellationToken` checked at every phase boundary.
  - `GET /live_view/{job_id}` HTML iframe — per-job view tokens,
    sha256-hashed at rest, self-polling. View-token equality is
    compared with `hmac.compare_digest` for constant-time safety.
  - `GET /live_view/{job_id}/status` JSON snapshot used by the iframe.

The cleartext `view_token` returned on the start response is only
needed when an OpenAPI consumer renders the iframe URL itself
(`/live_view/{job_id}?token={view_token}`) — typically the
writeback-disabled or non-OWUI deployment case. In the normal
writeback-enabled flow the iframe HTML, with the cleartext baked into
the polling script's `data-bootstrap` attribute by
`progress/embed.py::render_progress_embed_html`, is posted to the chat
row directly via the writeback channel, and chat-history replay works
without the LLM having to retain the token.

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

**Per-job lock invariant (`JobRunner`).** All lifecycle transitions
(`start_job`, `submit_feedback`, `cancel`) for a given `job_id` are
serialised by a per-job `asyncio.Lock` held by
`JobRunner._job_locks[job_id]`. The runner uses a two-tier scheme:
`_registry_lock` (held for microseconds) covers mutations of the
`_job_locks` dict and is also the cross-job lock used by `shutdown`;
each per-job lock covers a single job's transitions. `submit_feedback`
and `cancel` follow the same **Phase A → release → Phase B →
re-acquire → Phase C** shape, releasing the lock during the engine-task
wait (Phase B) and re-acquiring it for the finalisation (Phase C) —
holding the lock across an indefinite engine wait would block
concurrent cancel. Phase C re-validates against the store-read record;
a concurrent cancel during Phase B surfaces as `FeedbackCancelledError`
in `submit_feedback` (mapped to HTTP 409
`cancelled_during_feedback`) or is bridged by `_cancel_requested` in
`cancel`. The `_cancel_requested: set[str]` intent flag prevents the
cancel-vs-natural-completion race from swallowing the user's cancel
intent: the success branches of `_run_initial` / `_run_feedback`
check the flag *before* writing COMPLETED to the store, so a cancel
arriving mid-finalisation lets the task return without setting a
terminal phase and `cancel()`'s Phase C lands CANCELLED. Per-job
state dicts (`_tasks`, `_cancellation_tokens`, `_owui_tokens`,
`_view_tokens`, `_snapshots`, `_status_dedupe_counter`,
`_cancel_requested`) are GC'd on terminal phase via a task
done-callback scheduling `_maybe_drop_job_state` through `call_soon`.
`_job_locks` is intentionally NOT GC'd: a coroutine that has already
received a lock reference but not yet acquired it could race a newcomer
that creates a fresh lock for the same `job_id`, breaking the
serialisation invariant. Lock leak is bounded by observed active
`job_id`s — small (~100 bytes per terminal job) and the server
short-circuits new calls for terminal jobs at the handler layer.

**Sqlite UNIQUE partial index as defence in depth.**
`research_jobs` has a `CREATE UNIQUE INDEX ... ON research_jobs(chat_id)
WHERE chat_id IS NOT NULL AND phase NOT IN ('completed', 'failed',
'cancelled')` enforcing "one active job per chat" at the database
layer. `JobRunner.start_job` catches `aiosqlite.IntegrityError` from
`store.create` and translates it to `ActiveJobExistsError`, which the
server handler maps to HTTP 409 `already_running`. A pre-migration
query in `JobStore.start` resolves any pre-existing duplicate active
rows (older row marked FAILED with `error_text` carrying the
`pre_migration` token) so the UNIQUE index can be created cleanly on
databases that pre-date this constraint. The index is single-process
defence: see "Multi-process / multi-replica" deferred item — two
server processes writing to the same sqlite file still serialise via
file locks and `IntegrityError` is the user-visible failure under
contention, NOT clean 409 routing.

### OWUI external-tool model — constraints that shape the v2 design

Three OWUI facts force the two-call workflow + JSON-only responses +
iframe-via-`/event` writeback shape. Documenting them here so a future
session doesn't propose "simpler" designs that hit the same walls.

**OWUI's per-call timeout `AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER` (default
~10 min, operator-set on the OWUI side)** is the hard ceiling on any
single tool-call HTTP request. A multi-minute research run cannot
block one HTTP call without risking 504. That's why
`start_research_job` and `submit_research_feedback` both return in
well under a second, with the engine running in a background
`asyncio.Task`. Don't propose "just block synchronously" — it works
for the outline phase (~10–30s) but breaks on the long research leg.

**The `(html, context)` tuple-response form is unreachable for
external OpenAPI tool servers.** OWUI's `process_tool_result` requires
both `Content-Type: text/html` AND a 2-element list/tuple body to
deliver Rich UI + structured LLM context together. But aiohttp's
`response.json()` (used in OWUI's `execute_tool_server`) is strict on
content-type: an HTML response falls back to `response.text()` and
yields a `str`, not a list. The two conditions are mutually exclusive
on the wire. This is *why* `start_research_job` returns JSON and the
iframe is posted via the `/event` channel rather than returned
directly as HTML+tuple from the tool call.

**`message.embeds` is replace-not-append in the external-tool path.**
OWUI's `Chat.svelte` sets `message.embeds = data.embeds`
unconditionally for the external-tool branch — the `replace: true`
flag the Function path sets is *irrelevant* on the OpenAPI runtime
because there's no append behaviour to opt out of. Each tool-call
message gets its own message-embeds slot anyway. Don't "fix" the
missing `replace: true` in the OpenAPI outbox writebacks; it would
change nothing.

**OWUI iframe sandbox constraints (`FullHeightIframe.svelte`).** The
iframe is mounted sandboxed `allow-scripts` *without*
`allow-same-origin` (user toggle `iframeSandboxAllowSameOrigin` off
by default). Inside the iframe: no cookies, no `localStorage`, no
`parent.fetch`. Cross-origin `fetch()` to the tool server works only
because `server.py` sets `allow_origins=["*"]` in its CORS config.
Also: `iframeSandboxAllowScripts` (default on) — if the user disables
it, both the height-postMessage script and the self-polling script
stop running. `render_progress_embed_html` has no fixed-height CSS
fallback, so a scripts-disabled iframe collapses. Anything added to
the embed must assume opaque-origin / no-storage / no-cookies;
Alpine.js-style inline frameworks are fine, anything requiring
same-origin storage isn't.

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

**Dedupe keys are content-deterministic across process restarts.** The
`_event_to_outbox` status and replace key shapes are
`{job_id}:{message_id}:status:rev{revision}:{sha256(description|int(done))[:16]}`
and
`{job_id}:{message_id}:replace:msg:rev{revision}:{sha256(content)[:16]}`
respectively; the bootstrap-embed key is
`{job_id}:{message_id}:embeds:bootstrap:{revision}` and the
engine-embed key is `{job_id}:{message_id}:embeds:engine:{revision}`.
Identical status/message content within the same `JobRecord.revision`
deduplicates by design — this aligns with `EventBus._flusher`'s
consecutive-collapse behaviour and OWUI overwriting the status pill on
each post. **Do not reintroduce a process-local sequence counter** (the
old `_status_dedupe_counter` / the never-landed `_bootstrap_seq`):
Python's builtin `hash()` is per-process salted and such counters reset
to zero on restart, so `INSERT OR IGNORE` silently stops deduping
across restarts and each counter leaks one int per completed job. The
revision segment is the restart-stable namespace — a genuine re-submit
either lands on a new `message_id` (rebind) or a bumped `revision`, so
it gets a fresh key without any counter; only a true duplicate against
the identical `(message_id, revision)` collapses. Pinned by
`tests/test_openapi_runner.py::test_status_dedupe_*` and
`::test_bootstrap_embed_dedupe_key_is_revision_based`.

Worker loop spawn is opt-out: `OutboxWorker.start(spawn_loop=False)`
opens the sqlite connection without spawning the background drain.
The tests use this mode so `drain_once()` is deterministic; production
uses `spawn_loop=True` (the default).

Skip conditions in `_event_to_outbox` / bootstrap / final-writeback:
`chat_id is None`, `target_message_id is None`, or
`chat_id.startswith("local:")`. The `local:` prefix is OWUI's
ephemeral-chat marker — the `/event` endpoint accepts the POST but
the event is dropped, so we don't waste an HTTP round-trip.

**Outbox row status enum.** `owui_outbox` has a `status` column
(`pending | retrying | delivered | abandoned`) alongside `delivered_at`.
Convention: `delivered_at IS NOT NULL` means "terminal, do not
redeliver" (so the worker's pending query stays simple); `status`
distinguishes "delivered" (POST 2xx, true success) from "abandoned"
(gave up after `max_attempts` or rejected for a non-retriable reason
like a corrupt payload JSON blob). Ops counting "successful
deliveries" MUST filter on `status = 'delivered'`, NOT on
`delivered_at IS NOT NULL`. A pre-`status` DB is migrated in-place on
`OutboxWorker.start` via `_migrate_schema`; the migration cannot
reconstruct true-vs-abandoned for legacy rows and back-fills both as
'delivered' with a one-shot info log line (PRAGMA `table_info`
gates the ALTER so re-runs are no-ops). The counts are exposed via
`OutboxWorker.count_by_status()` and surfaced live (not cached) under
the `outbox` key of `GET /health` — a local sqlite `GROUP BY` is
microseconds, so the caching rationale that applies to the OWUI admin
probe does NOT apply here.

**Outbox Retry-After honours the server's value.** The worker's
`_compute_backoff` calls `extract_retry_after_seconds(exc)` for every
transient error type the helper can read headers from (`AdapterError`,
`httpx.HTTPStatusError`, any mapping-headers duck-type). When present,
the server-supplied value replaces the default exponential delay
entirely. The two ceilings have different jobs and are NOT collapsible:

| Knob | Purpose | Default |
|---|---|---|
| `jobs.outbox_max_backoff_s` | Cap on **default** exponential delay | 60 s |
| `jobs.outbox_max_retry_after_s` | Cap on **server-supplied** Retry-After | 600 s |

The 10-min default for `outbox_max_retry_after_s` matches production
observation: runs last 40–90 min, so a 10-min deferral fits inside the
run window without delaying the chat row's user-visible terminal write
past the run itself. Do not collapse the two knobs — clamping
Retry-After down to `max_backoff_s` is the "retried too soon, throttled
harder" anti-pattern the header exists to override.

**Writeback `OWUIClient` is throttled.** The writeback client is
constructed with its own `HttpThrottle` (label `owui_writeback`, valve
group `writeback_throttle`). It is functionally distinct from
`jobs.outbox_max_retry_after_s`: the throttle gates dispatched HTTP
calls (token bucket + min-interval) at the *client* layer; the ceiling
bounds the *worker*'s `next_attempt_at` math after a failure. Both
exist because the writeback path has two distinct burst surfaces —
`/event` posts (HTTP-cheap, OWUI side) and `upload_file` KB ingestion
(downstream-triggers OWUI's own embedding pipeline, so quota-expensive
on the model provider). The throttle covers both via `OWUIClient`'s
`_request` and `upload_file` paths.

Counters: `record_attempt` / `record_success` / `record_retry` /
`record_429` are wired on the writeback throttle in `OWUIClient` via
`with_retry`'s `on_transient` / `on_exhausted` callbacks — same shape
as `LLMProviderClient`. Degraded mode is therefore live on the
writeback path if operators tune `writeback_throttle.max_delay_seconds`.

Defaults ship the writeback throttle disabled
(`max_requests_per_second=0`, `min_interval_ms=0`) so existing
deployments observe no behaviour change. Operators tune via
`DR_WRITEBACK_THROTTLE_*` env vars.

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
these tests:
`tests/test_openapi_outbox.py::test_source_payload_round_trip` (queue
contract — a hand-built `source` payload round-trips unmangled),
`tests/test_openapi_outbox.py::test_citation_event_mapped_to_source_row`
(the runner-side mapping itself: `event_type='source'`, chat/message
routing, `data` from `CitationEvent.to_dict()`), and the four
parametrised e2e cases of
`tests/test_openapi_writeback_e2e.py::test_citation_event_outbox_mapping`
— `[single-citation-maps-correctly]`,
`[same-url-deduplicates-first-wins]`,
`[event-type-is-source-not-citation]`, and
`[empty-url-passes-through-runner]` (these four replaced the former
standalone `test_citation_event_maps_to_source_row` /
`test_multiple_citations_with_same_url_deduplicate` /
`test_runner_emits_source_not_citation` /
`test_runner_does_not_filter_empty_url_citation` functions; each case's
assertion is preserved verbatim in its own helper). Two pinned contracts
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

The UNIQUE partial index on `research_jobs(chat_id) WHERE phase NOT IN
terminal` (see "OpenAPI Tool Server runtime") strengthens the
single-process invariant but does NOT make multi-process safe: two
processes racing on `INSERT` for the same `chat_id` still produce
`IntegrityError` rather than a clean 409 routing, and the in-process
per-job lock in `JobRunner` isn't shared across processes.

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

Site #3's text is stored in the module-level constant
`deep_research.entrypoints.openapi_tool.schemas.USER_FACING_INSTRUCTION`,
referenced by both the field default and the `json_schema_extra` example
within the schema module (Group 14 collapsed that intra-file
duplication; pinned by
`tests/test_openapi_jobs.py::test_user_facing_instruction_default_equals_example`).
This is still site #3 — the separation from site #1 is unchanged. (The
`local:` 409 body and its `START_DESCRIPTION` LLM-relay paragraph were
left as two strings: their remedy advice is already worded per-audience,
not byte-identical, so a shared constant would have forced a text
change.)

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

Site #1's tool-description text requires the LLM to confirm any
natural-language cancel before calling; only the unambiguous slash
commands `/q` and `/quit` are forwarded without a confirmation
question. The parser branch in
`process_outline_feedback_continuation` signals the engine's
`CancellationToken` (via `getattr(ctx, "cancellation_token", None)`)
before raising `asyncio.CancelledError`, so any post-unwind
`is_cancelled()` reader observes the cancel correctly. All slash
commands (`/k`, `/keep`, `/r`, `/remove`, `/q`, `/quit`, `/continue`,
`/c`) are case-insensitive — the parser uses `re.IGNORECASE` on the
keep/remove patterns and `.lower()` on the single-word commands.

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

The helper takes a `kind: Literal["final", "cancelled"]` argument;
both kinds share a single `terminal` dedupe suffix, so a cancel
arriving during a successful final-writeback enqueue cannot produce a
mismatched status/body pair — `INSERT OR IGNORE` lets whichever
event-type row inserts first survive. Pinned by
`tests/test_openapi_writeback_e2e.py::test_terminal_writeback_dedupes_when_final_wins`
and `..._when_cancel_wins`.

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

User-typed `/q` / `/quit` arriving via the outline-feedback parser
signals the same `CancellationToken` (via `getattr(ctx,
"cancellation_token", None)`) before raising
`asyncio.CancelledError`, so a downstream phase-boundary check sees
the cancel even when the raise unwinds before the runner's handler.

**Bootstrap iframe is no longer posted in the preliminary phase.**
`start_job` does not call `_enqueue_bootstrap_embed`; the iframe is
first posted on `submit_feedback` (the moment the engine moves into
research and the live snapshot becomes meaningful). The only
user-visible content in the preliminary tool-call message is the
topic-list `MessageEvent → replace`.

`_enqueue_bootstrap_embed` has exactly one production caller
(`submit_feedback` Phase C) and takes no `marker` argument. Its dedupe
key is `{job_id}:{message_id}:embeds:bootstrap:{record.revision}` —
deterministic, no process-local counter (see the dedupe-key note under
"Phase 2 writeback wiring"). A genuine re-submit lands on a rebound
`message_id` (which also bumps `revision` via
`rebind_target_message`) so it enqueues a fresh iframe row; a true
duplicate against the identical `(message_id, revision)` correctly
dedupes. The snapshot read of `self._snapshots[job_id]` is **not**
lock-guarded and must not be: the sink's `_update_snapshot` and this
read are both synchronous (no `await` between the read and the
`{**snapshot, ...}` copy), so on the single event loop they cannot
interleave; and `_enqueue_bootstrap_embed` already runs inside the
caller's per-job lock, so acquiring any per-job `asyncio.Lock` here
would deadlock (not reentrant). Do not "fix" the missing lock.

**Config warning code taxonomy.** The OpenAPI Tool Server runs a
startup configuration audit
(`deep_research/entrypoints/openapi_tool/config_audit.py`) plus one
runtime detector inside `start_research_job`. Results are cached on
`app.state.config_warnings` and surfaced via `GET /health` as JSON
(`config_warnings: list[{code, severity, message, remediation}]`).
The stable `code` values:

  - `MISSING_OWUI_API_KEY` — `DR_OWUI_API_KEY` unset while
    `valves.jobs.writeback_enabled=True`. The lifespan fails fast
    (raises `RuntimeError`) on this exact condition and the server
    refuses to start. The startup log line carries this code; it is
    NOT emitted into `app.state.config_warnings`, so `/health` will
    never return it. (The audit module no longer re-checks this
    condition — the lifespan fail-fast is the canonical site.)
  - `OWUI_API_KEY_NOT_ADMIN` — probe (`OWUIClient.get_session_user()`
    → `GET /api/v1/auths/`) returned `role != "admin"`. Writeback
    POSTs would 401 at runtime; the server still starts.
  - `OWUI_API_KEY_PROBE_FAILED` — the admin probe raised, returned a
    non-dict body (mapped via `AdapterError`), or did not return
    within the 30s lifespan budget. Could be transient infrastructure
    (OWUI not yet up); does NOT fail-fast.
  - `MISSING_PUBLIC_BASE_URL` — `DR_OPENAPI_PUBLIC_BASE_URL` unset.
    Severity `info` (degraded UX, not broken) — the iframe falls
    back to the inbound request's host header.
  - `CLEANUP_INTERVAL_FLOORED` — operator set
    `DR_JOBS_CLEANUP_INTERVAL_S` below 60s. Severity `info` (runtime
    is correct; the retention sweep still floors to 60s — this code
    surfaces the floored value). Emitted once at lifespan time;
    appears in `/health` until the next pod restart.
  - `OWUI_HEADERS_NOT_FORWARDED` — detected at runtime (not startup):
    the first request to `start_research_job` arrived without
    `X-OpenWebUI-Chat-Id`. One-shot per process via
    `app.state._forward_headers_warned`. Means OWUI hasn't set
    `ENABLE_FORWARD_USER_INFO_HEADERS=true`. The detector gates on
    header absence **alone**, deliberately NOT on bearer presence:
    OWUI's inbound-auth type (none/bearer/session) is orthogonal to
    header forwarding, so gating on credentials would blind the warning
    to an OWUI instance whose tool server is configured `auth: none`
    with forwarding off — the exact silent-writeback misconfig this
    detector exists to surface. The accepted trade-off is a benign
    one-shot false positive when a non-OWUI caller (operator/curl/test)
    starts a job without the header; the remediation text names the
    OWUI knob, which is only meaningful if the caller is OWUI. Pinned by
    `tests/test_openapi_jobs.py::test_headers_not_forwarded_fires_without_bearer`.

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
issue. `MISSING_PUBLIC_BASE_URL` is informational only. The startup
audit probe is bounded by a 30s `asyncio.wait_for` (see
`_run_audit_with_timeout` in `entrypoints/openapi_tool/server.py`);
a timeout falls back to enqueuing `OWUI_API_KEY_PROBE_FAILED` in
`config_warnings` so a briefly-unreachable OWUI does not stall the
lifespan or trip K8s readinessProbe.

### Runner failure logging

Three exception paths in `deep_research/entrypoints/openapi_tool/runner.py`
log instead of swallowing silently. These are positive invariants; do
not restore the silent-suppress shape during cleanup.

  - `_make_sink` — both the `_update_snapshot` call and the
    `_event_to_outbox` call are wrapped in `try/except Exception`
    with `logger.exception`. The engine continues running after a bad
    event (so one malformed event doesn't cascade into a job-wide
    writeback freeze), but each failure leaves a traceback with the
    job id and the offending event class.
  - The outer `except Exception` in `_run_initial` and `_run_feedback`
    wraps the `_store.update(phase=FAILED, ...)` call in its own
    `try/except Exception` with `logger.exception`, and `_mark_phase`
    runs unconditionally. The in-memory snapshot phase tracks the
    intended terminal phase even when the DB write itself raises
    (sqlite IO error, transient lock contention).
  - `_deserialise_history` logs via `logger.warning` on a corrupted
    blob. The log line carries blob *length*, the exception class
    name, and `str(exc)` (JSONDecodeError emits position-only
    metadata; no content from the input). Returns `[]` so the engine
    still resumes with the fallback path. **PII-safety invariant:**
    do not log any fragment of the raw blob — the test
    `test_deserialise_history_logs_on_bad_blob` pins this.

The analogous *cancellation* branches — the three
`_store.update(phase=CANCELLED, ...)` calls (`cancel()`'s Phase C plus the
`CancelledError` handlers of `_run_initial` / `_run_feedback`) — now carry
the same `try/except Exception` + `logger.exception` guard, with
`_mark_phase(CANCELLED)` running unconditionally after. This landed under
Group 3 alongside the terminal-cancel-helper extraction; the FAILED-write
and CANCELLED-write paths are now symmetric.

---

## Concurrency contract (the engine)

OWUI instantiates `Pipe` once and calls `pipe()` concurrently for every user request. Concurrent invocations are **separate `asyncio.Task` instances on the same event loop** — not separate threads. The concurrency model in `pipe.py` is built on three rules.

**Per-call state lives in module-level `contextvars.ContextVar` instances and is surfaced on `Pipe` via property descriptors.** The list (top of `pipe.py`): event emitter and call hook, `__user__`, `__model__`, `__request__`, `conversation_id`, `chat_id`, `is_pdf_content`, `research_date`, `trajectory_accumulator`, `_seen_subtopics`, `_seen_sections`. Each is declared as a ContextVar at module top and read/written through a `_ctxvar_prop` descriptor. `pipe()` initialises every slot at entry. A bare `self.foo = bar` in `Pipe.__init__` or `pipe()` for new per-call state will silently leak between concurrent users — add a new ContextVar instead.

**Process-shared state lives on `Pipe` directly, guarded by `asyncio.Lock`.** This is limited to: `valves`, `state_manager`, `embedding_cache`, `transformation_cache`, `vocabulary_cache`, `vocabulary_embeddings`, `executor`, plus the locks (`_inflight_lock`, `_vocab_load_lock`, and the per-cache `_lock`) and the inflight set (`_inflight`). `asyncio.Lock` is the right primitive because `pipe.py` runs in OWUI's single event loop. `threading.Lock` is unnecessary; `threading.local` would be **wrong** — all concurrent `pipe()` coroutines share the same thread, so threadlocal slots would be shared, not isolated.

**ContextVars do NOT propagate across `loop.run_in_executor`.** Callables submitted to `self.executor` run in pool worker threads whose ContextVar slots are unrelated to the calling Task's context. Never read per-call attributes (`self.__user__`, `self.conversation_id`, etc.) inside a function passed to `run_in_executor`; pass per-call values in as explicit closure arguments. At present this rule is observed — every executor callable in `pipe.py` captures only local args (`_run_load`, `extract_with_pypdf`, `extract_with_pdfplumber`, `extract_with_bs4`). Audit any new callbacks you add against this rule.

**Same-conversation entry dedupe.** `pipe()` rejects a second invocation for a `conversation_id` already in `self._inflight`. Two concurrent calls on the same conversation would share the `ResearchStateManager` dict and corrupt state through interleaved read-modify-write across `await` boundaries; rather than retrofitting a lock around 60+ state mutation sites, the second invocation is rejected at entry with a notice.

**`Coordinator._inflight` role across runtimes.** The `Coordinator` itself maintains an `_inflight` set keyed on `(user.id, conversation_id)` (see `deep_research/orchestrator/coordinator.py`) that raises `AlreadyRunningError` for a concurrent `Coordinator.run()` on the same conversation. This is the **primary** serialisation mechanism for the **Function path** (`pipe.py` doesn't have its own per-conversation lock around `Coordinator.run` — it relies on the engine's guard *and* its own pre-entry rejection in `_inflight`). For the **OpenAPI Tool Server runtime**, `Coordinator._inflight` is now a **defence-in-depth backstop**: the runner's per-job `asyncio.Lock` plus the sqlite UNIQUE partial index on `research_jobs(chat_id) WHERE phase NOT IN terminal` are the primary serialisation, and the engine guard catches anything that slips past them. Don't remove the `_inflight` code on the assumption it's redundant — it's load-bearing for the Function path.

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

### `JobPhase` is `StrEnum`, not `str, Enum`

`JobPhase` (`entrypoints/openapi_tool/jobs.py`) inherits from
`enum.StrEnum`, so `str(JobPhase.X)` returns the lowercase **value**
(`"queued"`), not the pre-3.11 `str, Enum` **name** form
(`"JobPhase.QUEUED"`). Every persistence site uses `.value` explicitly, so
the migration was deliberate and safe. Don't "fix" the explicit `.value`
usage with a `__str__` override — that would silently restore the old
name-form and contradict the codebase-wide convention. Writing
`f"{record.phase}"` yields the value (matches `.value`) but reads
ambiguously in logs; prefer `.value`. A warning comment sits above the
class declaration and the contract is pinned by
`tests/test_openapi_jobs_store.py::test_job_phase_str_returns_value_not_name`
— a revert to `(str, Enum)` fails there.

`RunContext.cancellation_token` (and the two `Coordinator` signatures it
flows through) is typed `CancellationToken | None` via a `TYPE_CHECKING`
import + `from __future__ import annotations`, not `Any`. The import cycle
it originally dodged was speculative (`core.cancellation` only imports
`asyncio`); the guard is kept as cheap insurance.

### `render_progress_embed_html` has two render modes

`progress/embed.py::render_progress_embed_html(snapshot, *, poll_url=None,
view_token=None)` is one function with two output flavours:

  - **Push mode** (`poll_url=None`, used by the OWUI Function runtime):
    emits the topic-categories DOM + the height-postMessage script.
    Progress updates arrive via `__event_emitter__` with
    `{"type":"embeds", "data":{"embeds":[...], "replace":True}}` —
    the iframe is recreated on each push. Caller:
    `progress/embed.py::refresh_progress_embed`.
  - **Self-poll mode** (`poll_url` + `view_token` set, used by the
    OpenAPI live view): same DOM plus an inline script that polls
    `{poll_url}?token={view_token}&since_version=N` every 2s and
    reloads the iframe on a revision bump. The iframe HTML is
    rendered once with `poll_url` baked in at job-create time so
    chat-history replay works with no server involvement. Callers:
    `entrypoints/openapi_tool/runner.py::_enqueue_bootstrap_embed`
    and the `/live_view/{job_id}` route handler in
    `entrypoints/openapi_tool/server.py`.

Same function, two code paths sharing one DOM template. Modifying one
(adding a new snapshot field, changing the CSP nonce flow, etc.) risks
silently breaking the other; assert the relevant flavour in tests when
you touch this file.

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

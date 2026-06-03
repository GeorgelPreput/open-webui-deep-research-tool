# Configuration reference

Every knob is grouped on the `Valves` Pydantic model
(`deep_research/config/valves.py`). In the OWUI Function runtime the
groups render as nested forms in the admin UI. In Docker runtimes
(OpenAPI Tool / MCP) the same fields are populated from environment
variables with the `DR_<GROUP>_<FIELD>` convention, e.g.
`DR_MODELS_RESEARCH_MODEL=gemma3:27b`.

For production env layout see [deployment.md](./deployment.md). This page
is the full field-by-field reference.

---

## LLM and embedding providers

Deep Research sends chat completions and embeddings to **two
independently configured OpenAI-compatible providers** — each with its
own base URL, bearer token, and path. They can point at the same backend
(e.g. both at OWUI's `/openai` proxy with the same key) or at different
backends (chat at OpenAI, embeddings at a local Ollama). Neither has to
live on the same host as OWUI.

### Chat provider

| Env var | Default | Required | Notes |
|---|---|---|---|
| `DR_LLM_BASE_URL` | _(none)_ | **yes** | Base URL, no trailing slash. E.g. `http://ollama:11434` or `https://api.openai.com/v1`. |
| `DR_LLM_API_KEY` | _(none)_ | **yes** | Bearer token sent on every chat completion and `/models` request. |
| `DR_LLM_CHAT_PATH` | `/chat/completions` | no | Path appended to `DR_LLM_BASE_URL`. |

### Embeddings provider

| Env var | Default | Required | Notes |
|---|---|---|---|
| `DR_EMBEDDINGS_BASE_URL` | _(none)_ | **yes** | Base URL, no trailing slash. |
| `DR_EMBEDDINGS_API_KEY` | _(none)_ | **yes** | Bearer token sent on every embeddings request. Can differ from `DR_LLM_API_KEY`. |
| `DR_EMBEDDINGS_EMBEDDINGS_PATH` | `/embeddings` | no | Path appended to `DR_EMBEDDINGS_BASE_URL`. The doubled `EMBEDDINGS_` is correct — it's `DR_<group=embeddings>_<field=embeddings_path>`. |

If any of `DR_LLM_BASE_URL`, `DR_LLM_API_KEY`, `DR_EMBEDDINGS_BASE_URL`,
or `DR_EMBEDDINGS_API_KEY` is missing at startup the `Coordinator` raises
`ValueError` immediately rather than failing mid-request.

### Example configurations

**Same backend for both (OWUI's openai proxy):**
```bash
DR_LLM_BASE_URL=http://owui:8080/openai
DR_LLM_API_KEY=sk-your-owui-key
DR_EMBEDDINGS_BASE_URL=http://owui:8080/openai
DR_EMBEDDINGS_API_KEY=sk-your-owui-key
```

**Different backends (chat at OpenAI, embeddings at local Ollama):**
```bash
DR_LLM_BASE_URL=https://api.openai.com/v1
DR_LLM_API_KEY=sk-openai-...
DR_EMBEDDINGS_BASE_URL=http://ollama:11434
DR_EMBEDDINGS_API_KEY=ollama         # Ollama ignores the key; any non-empty value works
```

---

## Valve groups

Field defaults come from `deep_research/config/valves.py`.

| Group | Fields (default) |
|---|---|
| `llm` | `base_url` (required), `api_key` (required), `chat_path` (`/chat/completions`) |
| `embeddings` | `base_url` (required), `api_key` (required), `embeddings_path` (`/embeddings`) |
| `models` | `research_model` (`gemma3:12b`), `synthesis_model` (`gemma3:27b`), `quality_filter_model` (`gemma3:4b`), `embedding_model` (`nomic-embed-text`), `research_context_window` (auto), `synthesis_context_window` (auto), `temperature` (0.7), `synthesis_temperature` (0.6) |
| `cycles` | `min_cycles` (10), `max_cycles` (15), `gap_exploration_weight` (0.4), `trajectory_momentum` (0.6), `followup_weight` (0.5) |
| `web` | `search_results_per_query` (3), `successful_results_per_query` (1), `extra_results_per_query` (3), `repeats_before_expansion` (3), `max_result_tokens` (4000), `domain_priority` (`""`), `content_priority` (`""`), `quality_filter_enabled` (True), `quality_similarity_threshold` (0.60), `fetch_concurrency` (4), `search_concurrency` (2) |
| `compression` | `chunk_level` (2), `compression_level` (4), `stepped_synthesis_compression` (True) |
| `persistence` | `export_research_data` (True), `interactive_research` (True), `user_preference_throughout` (True), `max_kb_uploads_per_cycle` (0 = unlimited), `kb_upload_delay_ms` (0), `disable_during_degraded` (False), `disable_kb_persistence` (False) |
| `events` | `enable_progress_embed` (True), `flush_interval_ms` (400), `quiet_chat_mode` (True) |
| `jobs` _(OpenAPI runtime)_ | `completed_retention_s` (30d), `failed_retention_s` (24h), `cleanup_interval_s` (1h), `sqlite_busy_timeout_ms` (5000), `writeback_enabled` (True), `outbox_poll_interval_ms` (250), `outbox_max_attempts` (10), `outbox_max_backoff_s` (60) |
| `advanced` | `query_weight` (0.5), `llm_concurrency` (4), `embedding_concurrency` (8), `executor_workers` (2), `http_timeout_seconds` (600), `http_max_retries` (3), `pdf_legacy_tls_verify` (True) |
| `llm_throttle` | `max_requests_per_second` (0 = off), `min_interval_ms` (0), `max_retries` (5), `base_delay_seconds` (1.0), `max_delay_seconds` (60.0) |
| `embeddings_throttle` | same five fields, plus `batch_max_inputs` (64) |
| `logging` | `level` (`INFO`), `format` (`text` \| `json`), `include_tracebacks` (True) |

Auto-detected context windows query `GET {DR_LLM_BASE_URL}/models` at
startup; an explicit `research_context_window` / `synthesis_context_window`
overrides the auto-detect.

Engineering knobs that aren't user-facing (PDF page caps, extraction-
quality thresholds, eigendecomposition radii, etc.) live in
`deep_research/config/constants.py` — edit the source if you really need
to tune them.

---

## OpenAPI runtime: writeback channel (Phase 2)

The OpenAPI Tool Server posts engine output (topic list, status pills,
final report, citations, live iframe) directly to OWUI's per-message
`/event` endpoint when three conditions hold:

1. **`ENABLE_FORWARD_USER_INFO_HEADERS=true` is set on the OWUI side**
   (in the OWUI container, not the tool server). Without it, OWUI
   doesn't forward `X-OpenWebUI-Chat-Id` / `X-OpenWebUI-Message-Id`
   to outbound tool calls and the runner has no binding for writeback.
2. **`DR_OWUI_API_KEY` is set to an admin token** on the tool server.
   The per-message `/event` endpoint admin-bypasses chat ownership, so
   this is the *only* OWUI endpoint a server-side key can use to write
   into chats owned by other users.
3. **`jobs.writeback_enabled=true`** (default). Set
   `DR_JOBS_WRITEBACK_ENABLED=false` to disable the channel without
   tearing down the admin key — useful for staging the rollout.

When all three hold, the topic list and final report land in the
assistant message verbatim — the LLM doesn't need to repeat them. When
any are missing, the runtime degrades to Phase 1 behaviour (the LLM
emits a `user_facing_instruction` describing the slash-command grammar).

The `outbox_*` valves tune the writeback queue:

- `outbox_poll_interval_ms` — how often the worker checks for pending
  rows when the queue is empty. Lower values reduce per-event latency
  but raise idle CPU.
- `outbox_max_attempts` — give up after this many failed POSTs to OWUI
  for a single event. The row is then marked delivered so the queue
  doesn't deadlock on a permanently broken target message.
- `outbox_max_backoff_s` — caps the exponential backoff window between
  retries; OWUI's `Retry-After` header replaces the exponential delay
  when present (also capped by this value).

---

## Model ID format

Use the **exact ID OWUI registers in Settings → Models**, not the value
shown in Documents → Embedding Model. Models behind an OpenAI-compatible
connection with a configured `prefix_id` show up as `{prefix_id}.{model_id}`
(dot separator). Ollama and Infinity-style providers expose IDs
containing slashes (`org/model`) — keep the slash.

If you set a non-existent ID in `models.research_model` you'll get
`Model not found` on the first chat call. The fix is always: list
registered IDs (`GET {DR_LLM_BASE_URL}/models`) and copy verbatim.

---

## Logging

Deep Research configures its own logger tree (`deep_research.*`) at every
entry point. Output goes to `stderr` and does **not** propagate into the
host's root logger, so it coexists with uvicorn / OWUI logging without
duplicates.

Configure via env (shorthand) or valves (long form). Precedence: explicit
valve > env > default.

| Setting | Env (shorthand) | Env (valve form) | Valve | Default | Notes |
|---|---|---|---|---|---|
| Level | `DR_LOG_LEVEL` | `DR_LOGGING_LEVEL` | `logging.level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| Format | `DR_LOG_FORMAT` | `DR_LOGGING_FORMAT` | `logging.format` | `text` | `text` (human-readable) or `json` (one JSON object per line) |
| Tracebacks | `DR_LOG_INCLUDE_TRACEBACKS` | `DR_LOGGING_INCLUDE_TRACEBACKS` | `logging.include_tracebacks` | `true` | Set false to strip stack traces while keeping the error message |

Every record carries the active `conversation_id`, `chat_id`, `run_id`,
and `request_id` (propagated via `contextvars`, surfaced as record fields
in both formats). API keys and bearer tokens are redacted to first 4 +
last 4 (`sk-a…0xyz`) wherever they would otherwise appear in logs;
values under 9 characters render as `********`. See
[deployment.md › Security](./deployment.md#security) for the implementation
pointer.

### What `DEBUG` gives you

- **Coordinator startup** — effective chat base URL, embeddings base URL,
  OWUI base URL, all paths, and both LLM and embedding keys redacted
  (`sk-a…0xyz`).
- **Each of the 9 research phases** — `Phase start: <name>` /
  `Phase done: <name> elapsed_s=…` / `Phase failed: <name>` with traceback.
- **Every HTTP call** from all three clients. `LLMProviderClient` and
  `EmbeddingProviderClient` share the `deep_research.adapter.llm_provider`
  logger (same module); `OWUIClient` logs under
  `deep_research.adapter.client`. Each line carries method, path, model
  id, body keys, status, elapsed time. Non-2xx responses include a
  500-char truncated body.
- **Each retry attempt** from `adapter/retry.py` with attempt number,
  delay, classified reason, and exception type.

### Example — OpenAPI Tool runtime in debug mode

```bash
DR_LOG_LEVEL=DEBUG DR_LOG_FORMAT=text \
  DR_OWUI_BASE_URL=http://owui:8080 DR_OWUI_API_KEY=sk-owui-key \
  DR_LLM_BASE_URL=http://owui:8080/openai DR_LLM_API_KEY=sk-llm-key \
  DR_EMBEDDINGS_BASE_URL=http://ollama:11434 DR_EMBEDDINGS_API_KEY=ollama \
  uvicorn deep_research.entrypoints.openapi_tool.server:app --port 8000
```

Sample output, one chat call, one embeddings call:

```
2026-06-01 12:00:04,891 INFO deep_research.orchestrator Coordinator starting: owui_base=http://owui:8080 llm_base=http://owui:8080/openai llm_chat=/chat/completions llm_key=sk-ll…m-key embeddings_base=http://ollama:11434 embeddings_path=/embeddings embeddings_key=oll…lama
2026-06-01 12:00:05,002 INFO deep_research.orchestrator [conv=api_abc run=r-9f req=req-77] Run started: conversation_id=api_abc prompt_chars=128
2026-06-01 12:00:05,051 DEBUG deep_research.adapter.llm_provider [conv=api_abc run=r-9f req=req-77] HTTP POST /chat/completions body_keys=['model', 'messages', 'stream', 'temperature'] model=gemma3:12b
2026-06-01 12:00:06,210 DEBUG deep_research.adapter.llm_provider [conv=api_abc run=r-9f req=req-77] HTTP POST /chat/completions -> 200 in 1.16s
2026-06-01 12:00:06,251 DEBUG deep_research.adapter.llm_provider [conv=api_abc run=r-9f req=req-77] HTTP POST /embeddings body_keys=['model', 'input'] model=nomic-embed-text
2026-06-01 12:00:06,332 DEBUG deep_research.adapter.llm_provider [conv=api_abc run=r-9f req=req-77] HTTP POST /embeddings -> 200 in 0.08s
```

For structured ingestion (Loki, ELK, Datadog) set `DR_LOG_FORMAT=json` —
each line is a JSON object with `ts`, `level`, `logger`, `message`,
`conversation_id`, `chat_id`, `run_id`, `request_id`, and optional
`exception`.

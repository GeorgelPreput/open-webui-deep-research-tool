# Troubleshooting

Each entry: **symptom → likely cause → quick check → fix**. If you reach
the bottom without a match, run the [pre-flight checks](./deployment.md#pre-flight-validation)
in order — the first one to fail localises the problem.

---

## Startup

### `ValueError: llm_base_url is required` / `llm_api_key is required`

- **Cause:** `DR_LLM_BASE_URL` or `DR_LLM_API_KEY` not set.
- **Check:** `env | grep '^DR_LLM_'` inside the container.
- **Fix:** Set both. They can equal `DR_EMBEDDINGS_BASE_URL` / `DR_EMBEDDINGS_API_KEY` (Pattern A in [deployment.md](./deployment.md)) or differ (Pattern B).

### `ValueError: embeddings_base_url is required` / `embeddings_api_key is required`

- **Cause:** `DR_EMBEDDINGS_BASE_URL` or `DR_EMBEDDINGS_API_KEY` not set.
- **Check:** `env | grep '^DR_EMBEDDINGS_'`.
- **Fix:** Set both. For Ollama or any provider that ignores auth, any non-empty string in `DR_EMBEDDINGS_API_KEY` works.

---

## Provider mismatches

### Chat works but embeddings fail with 404 on `/embeddings`

- **Cause:** `DR_EMBEDDINGS_BASE_URL` or `DR_EMBEDDINGS_EMBEDDINGS_PATH` points at something that doesn't expose the OpenAI-compatible embeddings path. Common: pointing at a chat-only host, or forgetting that the env var is `DR_EMBEDDINGS_EMBEDDINGS_PATH` (doubled) and instead setting `DR_EMBEDDINGS_PATH` (silently ignored, default `/embeddings` is used).
- **Check:**
  ```bash
  curl -fsS -X POST "$DR_EMBEDDINGS_BASE_URL$DR_EMBEDDINGS_EMBEDDINGS_PATH" \
    -H "Authorization: Bearer $DR_EMBEDDINGS_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"nomic-embed-text","input":"hi"}'
  ```
- **Fix:** Either point at a host that actually serves `/embeddings`, or set `DR_EMBEDDINGS_EMBEDDINGS_PATH` to the provider's real embedding path.

### Chat works but embeddings fail with 429

- **Cause:** Embedding provider quota is below current load. Often shows up the moment a research cycle ranks 30+ chunks at once.
- **Check:** Provider dashboard for the embedding key, plus the engine logs (`DR_LOG_LEVEL=DEBUG`) for `429` lines under `deep_research.adapter.llm_provider`.
- **Fix:** Apply [Pattern C — constrained embeddings quota](./deployment.md#pattern-c--constrained-embeddings-quota). At minimum, set `DR_ADVANCED_EMBEDDING_CONCURRENCY=1` and `DR_EMBEDDINGS_THROTTLE_MAX_REQUESTS_PER_SECOND` to half your provider's RPS limit. If OWUI's own ingestion is competing for the same key, also `DR_PERSISTENCE_DISABLE_DURING_DEGRADED=true`. If KB ingestion itself is the dominant embedding consumer, set `DR_PERSISTENCE_DISABLE_KB_PERSISTENCE=true` — research becomes ephemeral (no rehydrate, no post-report KB Q&A) but the in-chat report still lands.

### `Model not found` on first embedding (or chat) call

- **Cause:** The configured model ID isn't registered with the provider — typo, disabled model, or wrong prefix (OWUI OpenAI-connection `prefix_id`, Ollama `org/model` slash).
- **Check:**
  ```bash
  curl -fsS "$DR_LLM_BASE_URL/models"        -H "Authorization: Bearer $DR_LLM_API_KEY" | jq '.data[].id'
  curl -fsS "$DR_EMBEDDINGS_BASE_URL/models" -H "Authorization: Bearer $DR_EMBEDDINGS_API_KEY" | jq '.data[].id'
  ```
- **Fix:** Copy the ID verbatim into `DR_MODELS_RESEARCH_MODEL` / `DR_MODELS_SYNTHESIS_MODEL` / `DR_MODELS_EMBEDDING_MODEL`. Keep any `prefix.` segment and any `org/model` slash.

### 429 bursts under mixed OWUI + Deep Research load

- **Cause:** OWUI's own embedding ingestion (web search → KB pipeline) and Deep Research are both calling the embeddings provider with the same key, exhausting one shared rate-limit bucket.
- **Check:** Provider dashboard shows the surge correlated with active research runs; OWUI's logs show its own 429s under `retrieval` calls.
- **Fix:** Split keys — give Deep Research its own embeddings key (Pattern B) so each stack has its own rate-limit domain. If a single key is unavoidable, throttle Deep Research (Pattern C) and set `DR_PERSISTENCE_DISABLE_DURING_DEGRADED=true` so it backs off OWUI's KB ingest path while degraded.

---

## OWUI integration

### OWUI KB persistence "silently does nothing" but chat succeeds

- **Cause:** `DR_OWUI_API_KEY` belongs to user A, the chat being researched is owned by user B. `GET /api/v1/chats/{id}` filters by `chat.user_id == authenticated_user_id` and **admin role does not bypass this filter**. The research itself still runs; only the write-back to `chat.deepResearch` fails.
- **Check:**
  ```bash
  curl -fsS "$DR_OWUI_BASE_URL/api/v1/chats/<chat_id>" \
    -H "Authorization: Bearer $DR_OWUI_API_KEY"
  ```
  A 404 confirms the cause.
- **Fix:** Use the chat owner's API key, or accept that the OpenAPI / MCP runtimes can only persist to chats owned by their fixed key user. The OWUI **Function** runtime sidesteps this — it forwards the caller's `Authorization` header. See [Compatibility › chat persistence caveat](./compatibility.md#chat-persistence-caveat).

### OWUI returns 401 on every Deep Research API call

- **Cause:** `DR_OWUI_API_KEY` is wrong, expired, or belongs to a deactivated user.
- **Check:** `curl -fsS "$DR_OWUI_BASE_URL/api/v1/auths/" -H "Authorization: Bearer $DR_OWUI_API_KEY"` — 200 with user info if the key is good.
- **Fix:** Reissue the key in OWUI → Settings → Account → API Keys, push it into the `dr-owui-key` Secret, rolling-restart the Deployment.


---

## OWUI tool-server integration

### Tool discovered in OWUI but no usable output in the chat

- **Cause:** The v2 OpenAPI tool server uses a two-call workflow (`start_research_job` → user reply → `submit_research_feedback`) rather than a single synchronous call; if OWUI is still configured against the v1 `POST /research` endpoint, no useful output appears.
- **Check:** `curl -fsS "http://<openapi-host>/openapi.json" | jq '.paths | keys'`. You should see `/research_jobs` (POST), `/research_jobs/{job_id}` (GET), `/research_jobs/{job_id}/feedback`, `/research_jobs/{job_id}/cancel`, `/live_view/{job_id}`, and `/live_view/{job_id}/status`. The old `/research` is gone.
- **Fix:** Re-register the tool server in OWUI Settings → Tools so it picks up the new operations. Confirm the per-chat tool toggle is on (the tool-server icon in the OWUI chat composer must show **Deep Research → enabled** for the current chat).

### Tool call times out before the report is ready

- **Cause:** The v2 endpoint surface returns immediately from `start_research_job`, so the LLM call cannot time out on the research itself. If you're still seeing this, the LLM is calling `submit_research_feedback` synchronously and waiting for completion — also returns immediately and shouldn't block.
- **Check:** `kubectl logs deployment/deep-research-openapi` shows both calls returning `200` in well under a second. The actual run continues in the background.
- **Fix:** Confirm the LLM is calling the v2 operations (`start_research_job`, `submit_research_feedback`, `get_research_job`) and not the long-gone `/research`.

### Iframe shows but no live updates

- **Cause:** `DR_OPENAPI_PUBLIC_BASE_URL` is unreachable from the user's browser, so the iframe's polling script keeps failing with network errors. The iframe paints once from the initial snapshot, then can't refresh.
- **Check:** Open the browser DevTools network tab; the `/live_view/{id}/status` polling calls should be visible. If they 404 / fail DNS / hit an internal-only hostname, that's the cause.
- **Fix:** Set `DR_OPENAPI_PUBLIC_BASE_URL` to the URL the **user's browser** sees (e.g. `https://research.example.com`), not the URL OWUI uses to reach the tool server (`http://deep-research-openapi:8000`). Restart the OpenAPI deployment.

### LLM gave a summary instead of presenting the topic list

- **Cause (Phase 1):** The LLM skipped reproducing the `user_facing_instruction` field from the `start_research_job` response, even though the operation description tells it to emit it verbatim. The user never sees the slash-command grammar and the next step stalls.
- **Check:** Look at the OWUI chat — does the LLM's reply contain `/k`, `/r`, or `/continue` hints?
- **Fix (Phase 1):** Verify the operation description is current — `curl -fsS "http://<openapi-host>/openapi.json" | jq '.paths["/research_jobs"].post.description'` should mention `verbatim`. Try a stronger model or add an explicit instruction to the system prompt.
- **Fix (Phase 2):** Set `ENABLE_FORWARD_USER_INFO_HEADERS=true` on the OWUI side and restart OWUI. With Phase 2 writeback wired, the topic list lands directly in the assistant message via the `/event` channel; the LLM no longer needs to repeat anything.

---

## Compatibility / display

### Stacked progress embeds on chat reload

- **Cause:** OWUI < 0.9.5. The `replace` flag on `embeds` events is silently ignored, so reloading a finished run paints every saved progress snapshot.
- **Fix:** Upgrade OWUI ≥ 0.9.5, or accept the visual clutter (live runs are unaffected). See [Compatibility › OWUI version matrix](./compatibility.md#owui-version-matrix).

---

## Engine behaviour

### Research starts but no sources appear after several cycles

- **Cause:** OWUI's web search isn't configured, or the configured engine is rate-limited / blocked.
- **Check:** OWUI → Admin Settings → Web Search shows an engine enabled; engine logs show successful queries; the status pill on the run shows the failure mode.
- **Fix:** Enable a working search engine in OWUI (DuckDuckGo needs no key). If the engine is rate-limited, lower `DR_WEB_SEARCH_CONCURRENCY`.

### `httpx.ConnectError` on PDF fetches only

- **Cause:** The paywall-spoof fallback path uses direct `httpx` instead of OWUI's extraction (which already failed or wasn't applicable for this URL).
- **Check:** Confirm the URL is reachable from the pod: `curl -v <url>`.
- **Fix:** Egress firewall / DNS — the pod must be allowed to reach the source host directly, not only OWUI.

---

If nothing above matches, raise `DR_LOG_LEVEL=DEBUG`, reproduce, and grep
the logs for the `request_id` from the failing run. Every HTTP call is
logged with method, path, status, and elapsed time under
`deep_research.adapter.llm_provider` (LLM + embeddings) or
`deep_research.adapter.client` (OWUI).

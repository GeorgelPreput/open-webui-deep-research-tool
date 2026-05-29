# TODO.md — Deep Research tool fix plan

## Context

Marko's review of `deep_research/` produced a **catastrophe** verdict: three
guaranteed runtime crashes on the hot path, plus several systemic bugs.
Deeper investigation expanded the scope:

- `OWUIClient.chat_completions` uses keyword-only parameters (`*`), but
  **24 of 25 call sites pass arguments positionally** — every research
  cycle, synthesis call, citation check, ranking call, etc. is a
  guaranteed `TypeError` on first invocation. The single correct site
  (`web/search.py:456`) is in *dead code* (a stale duplicate of
  `research/query_gen.py`).
- `Coordinator.start()` hard-codes `base_url="http://localhost:8080"` and
  `StaticToken("")`. The per-request bearer token plumbed through
  `pipe()` is captured into a local `token` variable but **never
  installed on the client** — all calls go out with no auth.
- `RuntimeConfig` only has `data_dir`; `kb_search()` reads
  `ctx.config.EMBEDDING_MODEL` and `ctx.config.VECTOR_DB_CLIENT` which
  do not exist.
- `OWUIClient.upload_file` accepts `(content, filename, process)` and
  returns `FileUploadResponse`, but `persistence/kb.py:179` passes a
  `metadata=` kwarg that doesn't exist and treats the return value as
  if it were a `str` `file_id`.
- `get_embedding` calls the embedding endpoint with
  `ctx.valves.models.research_model` (a generative LLM, e.g.
  `gemma3:12b`). 42 call sites depend on this returning real embedding
  vectors.

This plan fixes every issue Marko raised plus the expanded findings.
Outcome: the tool actually runs end to end, KB persistence works, and
the embedding subsystem produces real vectors.

---

## Approach summary

Three phases, ordered by criticality. Phase A unblocks any run at all.
Phase B restores functional correctness of features. Phase C is hygiene.

Where there is a choice between "edit one method" and "edit dozens of
call sites" with equivalent semantics, prefer the one-edit fix
(documented inline) over the 24-site sweep. We will, however, sweep
when the call-site code itself is broken (not just a calling
convention).

---

## Phase A — Critical (must fix; no run completes without these)

### A1. Fix `chat_completions` keyword-only / positional mismatch

**Files:** `deep_research/adapter/client.py`

**Problem:** `chat_completions` and `stream_chat_completions` declare
`self, *, model, messages, ...` (line 199-207 and 222-228), making all
params keyword-only. 24 call sites across the codebase pass `model` and
`messages` positionally.

**Fix:** Remove the `*` marker from both method signatures so positional
calls work. This is a single-file change that unblocks 24 sites without
touching them.

`deep_research/adapter/client.py`:

```python
# BEFORE
async def chat_completions(
    self,
    *,
    model: str,
    messages: list[dict],
    stream: bool = False,
    temperature: float | None = None,
    chat_id: str | None = None,
) -> dict:

# AFTER (remove the lone `*` line)
async def chat_completions(
    self,
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    temperature: float | None = None,
    chat_id: str | None = None,
) -> dict:
```

Same shape change for `stream_chat_completions`: keep `model` and
`messages` positional-or-keyword, keep `temperature` keyword-only.

**Verification:** grep for `chat_completions(` in `deep_research/` and
confirm every call now resolves. Existing call sites already match the
new signature.

---

### A2. Wire OWUI base URL and per-request bearer token through to `OWUIClient`

**Files:** `deep_research/orchestrator/coordinator.py`,
`deep_research/adapter/auth.py`,
`deep_research/adapter/client.py`,
`deep_research/entrypoints/owui_function/pipe.py`,
`deep_research/entrypoints/owui_pipeline/pipeline.py` (if it touches
the client),
`deep_research/entrypoints/openapi_tool/server.py`,
`deep_research/entrypoints/mcp/server.py`.

**Problem:**
- `Coordinator.start()` (`coordinator.py:64-78`) constructs
  `OWUIClient(base_url="http://localhost:8080", token_provider=StaticToken(""), ...)`.
  No valve, env var, or config field overrides this.
- `Coordinator.run(... token=token ...)` (`coordinator.py:87-115`)
  accepts the per-request token but `_build_context` never passes it
  into the client. The client's `_token_provider` remains the empty
  `StaticToken` from startup. Every request goes out with
  `Authorization: Bearer ` (literally empty).

**Fix:**

**A2.1** Add a `base_url` field to `RuntimeConfig`:

```python
# deep_research/orchestrator/coordinator.py
class RuntimeConfig:
    def __init__(
        self,
        data_dir: str = "/tmp/deep_research",
        base_url: str = "http://localhost:8080",
    ):
        self.data_dir = data_dir
        self.base_url = base_url
```

**A2.2** Update `Coordinator.start()` to read `base_url` from
`self._config.base_url` instead of the hard-coded string.

**A2.3** Add a `ContextTokenProvider` in
`deep_research/adapter/auth.py`:

```python
# deep_research/adapter/auth.py
import contextvars
from typing import Protocol


class BearerTokenProvider(Protocol):
    async def get_token(self) -> str: ...


class StaticToken:
    def __init__(self, token: str) -> None:
        self._token = token

    async def get_token(self) -> str:
        return self._token


_current_token: contextvars.ContextVar[str] = contextvars.ContextVar(
    "deep_research_owui_token", default=""
)


class ContextTokenProvider:
    """Per-request bearer token read from a contextvar.

    Set the token at the start of a request via set_current_token(); the
    OWUIClient reads it on every outbound call. This lets us share one
    OWUIClient (and its connection pool) across many concurrent requests
    while still scoping the bearer per-call.
    """

    async def get_token(self) -> str:
        return _current_token.get()


def set_current_token(token: str) -> contextvars.Token:
    return _current_token.set(token)


def reset_current_token(reset_token: contextvars.Token) -> None:
    _current_token.reset(reset_token)
```

**A2.4** Update `Coordinator.start()` to use `ContextTokenProvider`
instead of `StaticToken("")`:

```python
from deep_research.adapter.auth import ContextTokenProvider, set_current_token, reset_current_token
# ...
self._client = OWUIClient(
    base_url=self._config.base_url,
    token_provider=ContextTokenProvider(),
    ...
)
```

**A2.5** In `Coordinator.run()`, set the contextvar from the incoming
`token` parameter for the duration of the run:

```python
async def run(self, *, user, conversation_id, chat_id, token, prompt, history, sink):
    inflight_key = f"{user.id}:{conversation_id}"
    async with self._inflight_lock:
        if inflight_key in self._inflight:
            raise AlreadyRunningError(...)
        self._inflight.add(inflight_key)

    bearer = await token.get_token()
    token_handle = set_current_token(bearer)
    try:
        ctx = await self._build_context(...)
        await ctx.events.start()
        try:
            return await self._run_phases(ctx)
        finally:
            await ctx.events.stop()
    finally:
        reset_current_token(token_handle)
        async with self._inflight_lock:
            self._inflight.discard(inflight_key)
```

**A2.6** Update entrypoints to pass `base_url` to `RuntimeConfig`:

```python
# deep_research/entrypoints/owui_function/pipe.py:34
config = RuntimeConfig(
    data_dir=os.environ.get("DR_DATA_DIR", "/tmp/deep_research"),
    base_url=os.environ.get("DR_OWUI_BASE_URL", "http://localhost:8080"),
)
```

Apply the same construction pattern to any other entrypoint that
instantiates `RuntimeConfig` (`entrypoints/owui_pipeline/pipeline.py`,
`entrypoints/openapi_tool/server.py`, `entrypoints/mcp/server.py`).
Read each entrypoint first and only modify the ones that already build
a `RuntimeConfig` — do not invent new ones.

**Verification:** Start a run, tcpdump the outbound traffic to OWUI,
confirm `Authorization: Bearer <real-token>` is present and the URL is
not `localhost` when configured otherwise.

---

### A3. Fix `OWUIClient.upload_file` caller mismatch in `kb.py`

**File:** `deep_research/persistence/kb.py`

**Problem:** `upload_markdown_to_kb` (line 161-197) calls:

```python
file_id = await ctx.client.upload_file(
    filename=filename,
    content=payload,
    metadata=upload_meta,   # <-- not a parameter on upload_file
    process=True,
)
if not file_id:             # <-- file_id is a FileUploadResponse object, always truthy
    ...
await ctx.client.add_file_to_kb(kb_id, file_id)  # <-- passes object, expects string ID
```

`OWUIClient.upload_file(self, content: bytes, filename: str, process: bool = True) -> FileUploadResponse`
has no `metadata` parameter and returns the full response object.

**Fix:** Rewrite `upload_markdown_to_kb` to match the actual client
contract. Drop the `metadata` kwarg (the OWUI upload endpoint does not
accept extra metadata — file metadata is derived server-side from the
upload). Extract `.id` from the returned `FileUploadResponse`:

```python
async def upload_markdown_to_kb(
    ctx: RunContext,
    kb_id: str,
    filename: str,
    markdown_text: str,
    metadata: dict[str, Any] | None = None,  # accepted for compatibility but unused
) -> str | None:
    payload = markdown_text.encode("utf-8", errors="replace")

    try:
        upload = await ctx.client.upload_file(payload, filename, process=True)
    except Exception as e:
        logger.warning(f"upload_file failed for {filename}: {e}")
        return None

    file_id = getattr(upload, "id", None)
    if not file_id:
        logger.warning(f"upload returned no id for {filename}")
        return None

    try:
        await ctx.client.add_file_to_kb(kb_id, file_id)
    except Exception as e:
        logger.error(f"add_file_to_kb failed (kb={kb_id}, file={file_id}): {e}")
        return None

    return file_id
```

The `metadata` parameter stays in the signature (it is passed by
`persist_selected_source` and `persist_final_report`), but it is no
longer forwarded to `upload_file`. Add a one-line note that the
parameter is accepted for forward compatibility and ignored today.

**Verification:** Run an end-to-end research flow with one source.
Check OWUI logs for a successful POST to `/api/v1/files/` (returns
file id) followed by a successful POST to
`/api/v1/knowledge/{kb_id}/file/add`.

---

### A4. Replace `kb_search`'s broken VECTOR_DB_CLIENT path with REST `query_collection`

**File:** `deep_research/persistence/kb.py` (function `kb_search`,
lines 372-417)

**Problem:** Reads `ctx.config.EMBEDDING_MODEL` and
`ctx.config.VECTOR_DB_CLIENT` from `RuntimeConfig`, which only has
`data_dir`. Guaranteed `AttributeError` on first call. Even if those
attributes existed, the code is trying to reach into OWUI's internal
vector DB client, which is not available from this process.

**Fix:** Use the existing `OWUIClient.query_collection` REST method
(already defined at `adapter/client.py:389-405`).

```python
async def kb_search(
    ctx: RunContext, kb_id: str, query: str, k: int = 6
) -> list[dict[str, Any]]:
    """Run a vector search against the research KB collection via OWUI REST."""
    if not kb_id or not query:
        return []
    try:
        resp = await ctx.client.query_collection(
            collection_names=[kb_id],
            query=query,
            k=int(k),
            hybrid=False,
        )
    except Exception as e:
        logger.warning(
            f"KB vector search failed (kb={kb_id}, q={query[:60]!r}): {e}"
        )
        return []

    # QueryCollectionResponse is a Pydantic model wrapping
    # {distances: [[...]], documents: [[...]], metadatas: [[...]]}
    # (lists-of-lists, one inner list per query).
    documents = (getattr(resp, "documents", None) or [[]])
    metadatas = (getattr(resp, "metadatas", None) or [[]])
    distances = (getattr(resp, "distances", None) or [[]])
    docs0 = documents[0] if documents else []
    metas0 = metadatas[0] if metadatas else []
    dists0 = distances[0] if distances else []

    out: list[dict[str, Any]] = []
    for i, txt in enumerate(docs0):
        meta = metas0[i] if i < len(metas0) else {}
        dist = dists0[i] if i < len(dists0) else None
        out.append({"text": txt or "", "source": meta, "distance": dist})
    return out
```

Remove the now-unused `asyncio.get_running_loop()` /
`run_in_executor` block and the `ctx.config.EMBEDDING_MODEL` /
`ctx.config.VECTOR_DB_CLIENT` reads.

Also verify the shape of `adapter/models.QueryCollectionResponse` (read
`deep_research/adapter/models.py`) matches the
`{distances, documents, metadatas}` lists-of-lists structure documented
in `CLAUDE.md`. If the Pydantic model declares different field names,
align names; don't break the model.

**Verification:** With a populated KB, call `kb_search` and confirm it
returns a list of `{text, source, distance}` dicts.

---

## Phase B — Functional correctness

### B1. Add an `embedding_model` valve and use it for embeddings

**Files:** `deep_research/config/valves.py`,
`deep_research/semantics/embeddings.py`,
`deep_research/semantics/vocabulary.py`.

**Problem:** `get_embedding` calls
`ctx.client.embeddings(ctx.valves.models.research_model, [text])`. A
generative LLM (e.g. `gemma3:12b`) is not an embedding model. 42 call
sites consume the resulting vectors. Either OWUI rejects the call, or
worse, returns garbage that passes type checks.

**Fix:**

**B1.1** Add an `embedding_model` field to `ModelsValves` in
`deep_research/config/valves.py`:

```python
class ModelsValves(BaseModel):
    research_model: str = Field("gemma3:12b", description="Primary research LLM")
    synthesis_model: str = Field("gemma3:27b", description="Synthesis LLM (optional override)")
    quality_filter_model: str = Field("gemma3:4b", description="Relevance filter LLM")
    embedding_model: str = Field(
        "nomic-embed-text",
        description="Embedding model ID, must match an OWUI-registered embedding model",
    )
    # ... rest unchanged
```

**B1.2** Update `deep_research/semantics/embeddings.py:22-23`:

```python
# BEFORE
model = ctx.valves.models.research_model
result = await ctx.client.embeddings(model, [text])

# AFTER
model = ctx.valves.models.embedding_model
result = await ctx.client.embeddings(model, [text])
```

**B1.3** Update `deep_research/semantics/vocabulary.py:174-177`
(same substitution):

```python
embedding_model = ctx.valves.models.embedding_model
```

**B1.4** While in `vocabulary.py:119`, change the cache-naming hack:

```python
# BEFORE
model_name = getattr(ctx.config, "embedding_model", "") or "default"

# AFTER
model_name = ctx.valves.models.embedding_model or "default"
```

**Verification:** Start a run, watch the OWUI logs for
`/api/embeddings` calls with a real embedding model id. Confirm the
returned vector length matches an embedding model's expected
dimensionality, not a chat model's.

---

### B2. Delete the dead `improved_query_generation` duplicate in `web/search.py`

**File:** `deep_research/web/search.py`

**Problem:** Lines 422-503 hold a near-identical copy of
`research/query_gen.py:improved_query_generation`. The only consumer
(`orchestrator/phases/cycles.py:10`) imports from `query_gen`, not
`search`. The duplicate is dead code that risks silent divergence (the
duplicate uses `model=`/`messages=` kwargs; the canonical version is
positional). After A1 both call shapes work, so neither version is
preferred for correctness — the duplicate is just dead.

**Fix:** Delete the duplicate function (the entire `async def
improved_query_generation(...)` block in `web/search.py`, including
the prompt and the fallback). Remove unused imports the deletion
leaves dangling (likely `json` if no other use; verify).

**Verification:** Grep for `improved_query_generation` — only one
definition remains, in `research/query_gen.py`. Run a research cycle:
`cycles.py` still gets its queries.

---

### B3. Fix the malformed JSON example in the canonical query-gen prompt

**File:** `deep_research/research/query_gen.py`

**Problem:** The system prompt instructs the model to return:

```json
{"queries": [
  "query": "search query 1", "topic": "related research topic",
  ...
]}
```

This is **not valid JSON** — inside the array, each element should be
an object literal wrapped in `{...}`. The model has to guess the right
shape; the fallback parser then has to handle whatever the model
produced.

**Fix:** Rewrite the example to valid JSON. In the
`query_prompt["content"]` string:

```text
Format your response as a valid JSON object with the following structure:
{
  "queries": [
    {"query": "search query 1", "topic": "related research topic"},
    {"query": "search query 2", "topic": "related research topic"},
    {"query": "search query 3", "topic": "related research topic"},
    {"query": "search query 4", "topic": "related research topic"}
  ]
}
```

**Verification:** Run a cycle; the query JSON should now parse cleanly
without falling through to the per-topic fallback.

---

### B4. Bound `ResearchStateManager` memory

**File:** `deep_research/core/state.py`

**Problem:** `ResearchStateManager.conversation_states` is an
unbounded `dict[str, dict]`. Each conversation injects 30+ keys of
list/dict state; over a long-running process this grows without limit.

**Fix:** Convert the backing dict to a bounded `OrderedDict` with LRU
eviction. On `get_state`, move the key to end; on a new conversation,
evict the oldest when the size exceeds a cap. Default cap: 256
conversations. Configurable later if needed but no valve yet.

```python
import collections
import numpy as np


class ResearchStateManager:
    MAX_CONVERSATIONS = 256

    def __init__(self):
        self.conversation_states: collections.OrderedDict[str, dict] = (
            collections.OrderedDict()
        )

    def get_state(self, conversation_id):
        if conversation_id in self.conversation_states:
            self.conversation_states.move_to_end(conversation_id)
            return self.conversation_states[conversation_id]
        self.conversation_states[conversation_id] = self._new_state()
        while len(self.conversation_states) > self.MAX_CONVERSATIONS:
            self.conversation_states.popitem(last=False)
        return self.conversation_states[conversation_id]

    def _new_state(self) -> dict:
        return {
            # ... existing 30+ keys, copied verbatim from current get_state()
        }

    def update_state(self, conversation_id, key, value):
        state = self.get_state(conversation_id)
        state[key] = value

    def reset_state(self, conversation_id):
        if conversation_id in self.conversation_states:
            del self.conversation_states[conversation_id]
```

Move the existing default-state dict into `_new_state()` unchanged; do
not alter any field names or default values.

**Verification:** Simulate 300 distinct `conversation_id`s, confirm
size stays at 256.

---

## Phase C — Hygiene

### C1. Remove `__import__("time").time()` hack

**File:** `deep_research/orchestrator/coordinator.py`

**Problem:** Line 166 uses
`started_at=__import__("time").time()` because `time` was never
imported at module top.

**Fix:** Add `import time` to the top-of-file imports (alphabetically:
after `asyncio`, before `uuid`) and change line 166 to
`started_at=time.time()`.

---

### C2. Remove `or False` tautologies in `process_search_result`

**File:** `deep_research/web/search.py`

**Problem:** Lines 246 and 330 both have:

```python
if url.endswith(".pdf") or False:
    source_type = "pdf"
```

The `or False` is dead.

**Fix:** Drop ` or False` in both places. Leave the rest of the
conditional untouched.

---

### C3. Remove dead `pass # request metadata lookup removed` stub

**File:** `deep_research/persistence/chat_state.py`

**Problem:** `resolve_chat_id` (lines 196-202) does:

```python
chat_id = md.get("chat_id") or body.get("chat_id")
if not chat_id:
    pass  # request metadata lookup removed
return chat_id
```

The `if not chat_id: pass` is a no-op left over from a deletion.

**Fix:** Remove the `if not chat_id: pass` block entirely. Function
becomes:

```python
def resolve_chat_id(ctx: RunContext, body: dict[str, Any]) -> str | None:
    md = body.get("metadata") or {}
    return md.get("chat_id") or body.get("chat_id")
```

---

### C4. Address `verify=False` on archive.org client

**File:** `deep_research/web/fetch.py:310`

**Problem:** `httpx.AsyncClient(timeout=httpx.Timeout(20.0), verify=False)`
disables TLS verification for all archive.org requests.

**Fix:** Remove `verify=False`. archive.org has a valid certificate;
verification should succeed. If a future TLS error reappears, document
the specific cert chain issue in a comment before re-disabling.

---

### C5. Refactor `process_search_result` duplication

**File:** `deep_research/web/search.py` (function `process_search_result`,
lines 133-418)

**Problem:** 280-line function with the source-table-registration
block copy-pasted in two branches (lines ~241-302 and ~327-377). The
truncation branch also re-counts tokens inside its own conditional in
a way that masks the no-truncation token math.

**Fix (smallest viable):** Extract the duplicated block into a helper
`_register_source(ctx, url, title, snippet, content_for_preview,
query, source_type, master_source_table, content_tokens,
url_selected_count, url_token_counts) -> tuple[str, str, int]`
returning `(title, source_type, tokens)`. Call it once in each branch
with the right `content_for_preview` (truncated vs. full snippet).

This is mechanical extraction — do not change semantics. After
extraction, the function body should be under 120 lines.

Keep the `persist_selected_source` write-through behavior identical
(write the *full* `snippet`, not the truncated copy, as the existing
code already does).

**Verification:** Run a full research cycle with at least one URL that
exceeds `max_result_tokens` and one that doesn't. The
`master_source_table`, `url_selected_count`, and `url_token_counts`
mutations must match the pre-refactor behavior exactly.

---

## Files touched (summary)

- `deep_research/adapter/auth.py` — add `ContextTokenProvider` and
  helpers.
- `deep_research/adapter/client.py` — drop `*` from
  `chat_completions` / `stream_chat_completions` signatures.
- `deep_research/orchestrator/coordinator.py` — add `base_url` to
  `RuntimeConfig`, wire token contextvar, `import time`.
- `deep_research/entrypoints/owui_function/pipe.py` — pass `base_url`
  into `RuntimeConfig` (env var).
- Other entrypoints (`owui_pipeline`, `openapi_tool`, `mcp`) — same
  `RuntimeConfig(base_url=...)` change *only if* they currently
  instantiate `RuntimeConfig`.
- `deep_research/persistence/kb.py` — rewrite `upload_markdown_to_kb`
  and `kb_search`.
- `deep_research/persistence/chat_state.py` — remove dead `pass`.
- `deep_research/config/valves.py` — add `embedding_model` field.
- `deep_research/semantics/embeddings.py` — use `embedding_model`.
- `deep_research/semantics/vocabulary.py` — use `embedding_model` (two
  sites).
- `deep_research/research/query_gen.py` — fix JSON example in prompt.
- `deep_research/web/search.py` — delete `improved_query_generation`
  duplicate; remove `or False`; extract `_register_source` helper.
- `deep_research/web/fetch.py` — drop `verify=False`.
- `deep_research/core/state.py` — bound `ResearchStateManager`.

---

## Verification plan (end-to-end)

1. **Static checks**
   - `grep -n "chat_completions(" deep_research/` — every call should
     now resolve (the `*` is gone, kwargs are still allowed where
     used).
   - `grep -rn "improved_query_generation" deep_research/` — exactly
     one definition (`research/query_gen.py`) and at least one import
     (`orchestrator/phases/cycles.py`).
   - `grep -rn "ctx.config.EMBEDDING_MODEL\|ctx.config.VECTOR_DB_CLIENT\|metadata=upload_meta" deep_research/`
     — zero matches.
   - `grep -rn "__import__" deep_research/` — zero matches.
   - `grep -rn "or False" deep_research/web/search.py` — zero matches.

2. **Type / lint**
   - `mypy deep_research/` (or whatever the repo uses — check
     `pyproject.toml`).
   - `ruff check deep_research/`.

3. **End-to-end run against a real OWUI**
   - Set `DR_OWUI_BASE_URL`, `DR_OWUI_API_KEY` env vars.
   - Through OWUI UI, invoke the Deep Research function on a simple
     prompt (e.g. "What is the current state of EU AI Act
     enforcement?").
   - Confirm: cycle progresses past cycle 1 (proves A1 fix), KB
     creation succeeds and at least one source is uploaded (proves
     A3), embeddings are produced (proves B1), final report is
     persisted (proves A3 path), follow-up KB query returns results
     (proves A4).
   - Check OWUI server logs for any `TypeError`, `AttributeError`,
     `KeyError`, or 401/403 responses.

4. **Memory cap smoke test**
   - In a Python REPL, instantiate `ResearchStateManager`, call
     `get_state` 300 times with distinct ids, assert
     `len(mgr.conversation_states) == 256`.

5. **JSON-prompt smoke test**
   - Run a single cycle; assert the generated queries list parses on
     the first attempt (i.e. the per-topic fallback branch is not
     hit). Add a logger.info breadcrumb if helpful during testing,
     then remove it before commit.

---

## Out of scope (not in this plan)

- Re-architecting around per-call `OWUIClient` instances (the
  contextvar token approach in A2 is sufficient and lower-risk).
- Adding a separate embedding-model auto-detection from
  `/api/v1/models/list` — `embedding_model` is now a valve users set
  explicitly.
- Rewriting `process_search_result` from scratch — only the
  copy-paste duplication is extracted (C5); larger restructuring is
  left for a follow-up.
- The `pipeline.py` Pipelines entrypoint — per `CLAUDE.md`, that
  pathway is abandoned and not getting fixes.

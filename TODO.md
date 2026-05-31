# Deep Research — fixes needed (`deep_research/` package)

Review by Marko. Findings against the active refactored package. Each item is
`file:line — problem`. Severity-ordered. Items marked **[verified]** were
confirmed by reading the code and/or running the test suite (`pytest`); the rest
are read-and-reported by reviewers and should be confirmed before fixing.

## Critical — correctness / crashes / data loss

1. `orchestrator/phases/synthesize.py:47-48,81-83` — **[verified]** `global_citation_map`
   and `master_source_table` are snapshotted into locals *before* any section
   runs; sections mutate the live `state` dicts (`sections.py:463-464`), then
   line 81 overwrites the live citation map with the stale local and line 83
   passes the stale `master_source_table` to `generate_bibliography`. Result in
   a real run: `Generated bibliography with 0 cited entries (from 0 total
   sources)` — every report ships an empty bibliography and wrong numbering.
2. `synthesis/sections.py:453` — **[verified]** `"accessed_date": ctx.research_date`
   reads an attribute that does not exist on `RunContext` (a `slots=True`
   dataclass with no `research_date`); raises `AttributeError` whenever a
   section-level source is added. `front_back.py:56` already guards with
   `getattr(ctx, "research_date", None)`, proving the attribute is known-absent.
   Latent only because the smoke test left `section_sources` empty.
3. `synthesis/outline.py:78-112` — **[verified]** lines 78-110 build a
   similarity-ranked `outline_context` (and pay for the embedding calls), then
   line 112 unconditionally reassigns `outline_context = "### Original Research
   Outline:\n\n"`, discarding all of it. The ranking work is dead.
4. `entrypoints/owui_pipeline/pipeline.py:83-85` — the Coordinator client is
   `start()`ed inside a throwaway `asyncio.new_event_loop()` that is immediately
   closed; runs execute on a different per-call loop. This is the
   "session bound to a foreign/closed loop" hazard CLAUDE.md warns about —
   "Future attached to a different loop" / use-after-close.
5. `entrypoints/owui_pipeline/pipeline.py:98-106` — `_run_in_thread` only puts
   the sink sentinel on the success path; if `_async_run` raises, the sentinel
   is never enqueued and `_BridgeSink.__iter__`'s blocking `queue.get()` hangs
   the consumer forever.
6. `web/paywall.py:185` — `httpx.AsyncClient(verify=False, ...)` disables TLS
   certificate verification for all legacy PDF downloads; MITM exposure, and
   fabricated `X-Forwarded-For` / EZproxy cookies / fake auth tokens are sent to
   third-party hosts.
7. `entrypoints/mcp/server.py:21-28` — lazy `_coord` init has no lock; two
   concurrent tool calls can both see `_coord is None`, each build a Coordinator
   and call `start()`, leaking a client/session and racing on the global.
8. `semantics/trajectory.py:56` — the no-sample fallback hardcodes
   `TrajectoryAccumulator(384)`; `add_cycle_data` (trajectory.py:104) is later
   called with real embeddings of the model's true dimension (e.g. 768), and
   `state.py:93 self.query_sum += query_centroid` raises a numpy broadcast error
   on the 384-vs-768 shape mismatch.

## High — silent corruption / wrong results

9. `research/ranking.py:574-663` — per-topic `traj/pdv/gap` alignment scores are
   cached keyed only by `f"traj_{topic}"` etc., but their values depend on the
   trajectory/PDV/gap vector that changes every cycle; the cache persists across
   cycles, so cycle-1 scores are reused for the whole run and corrupt ranking.
10. `progress/snapshot.py:11` — **[verified]** `getattr(ctx.valves, "MAX_CYCLES", 0)`
    reads a flat attribute that does not exist (real path is
    `ctx.valves.cycles.max_cycles`), so the embed always renders "Cycle N/0".
11. `progress/embed.py:115-118` — the dedup digest is computed over HTML that
    embeds `updated_at` (`datetime.now()`) and an always-incremented `revision`,
    so the hash changes every call, `progress_embed_last_hash` never matches, and
    the embed is re-emitted unconditionally — the dedup is defeated.
12. `compression/repeated.py:93` & `compression/local_similarity.py:116` —
    `cosine_similarity([emb], [query_embedding])` is called with
    `query_embedding` that the signature allows to be `None` (forwarded unguarded
    from `search.py:240`); `None` raises inside the call, is swallowed by the
    outer bare `except`, and compression silently returns the full uncompressed
    content.
13. `web/search.py:266-328` — in `process_search_result`, when
    `content_tokens > max_tokens` but the truncated text is ≤100 chars, the
    branch calls `_register_source` (bumping `url_selected_count`, persisting)
    then falls through without returning to the bottom path that calls
    `_register_source` again — double-count and double-persist.
14. `adapter/client.py:182-202` — `_request_stream` retries a failed stream by
    re-invoking `_stream()`, which re-yields from the start; any error after
    partial content was already yielded duplicates content in the assembled
    completion.
15. `semantics/dimensions.py:94-95` — `coverage_array[i] += contribution[i] *
    (1 - current_value/2)` accumulates with no clamp, so coverage exceeds 1.0
    across cycles; downstream `coverage < 0.5` gap checks, `1.0 - cov` gap
    vectors, and the `*100` percentage display all go out of range/negative.
16. `compression/eigendecomp.py:73-74,169-211` — `n_keep` and the scoring loop
    use `len(chunks)` while embeddings / `local_coherence` are sized by
    `len(chunk_embeddings)` (smaller when any embedding is None); trailing chunks
    are silently unselectable and `ratio` is applied to an inflated count.
17. `semantics/trajectory.py:11` — `_trajectory_accumulators` is a module-global
    dict keyed by `conversation_id`, written but never evicted/cleared; grows
    unbounded for the process lifetime (unlike the 256-capped state manager).
18. `web/classify.py:64` — `owui_extraction_available` caches the reachability
    verdict in process-global `_owui_ext_cap` permanently with no invalidation;
    one transient `list_models()` failure latches `False` and disables all
    primary extraction for the entire process.
19. `web/classify.py:9` — `_owui_ext_cap` is keyed only by "web"/"doc", not by
    token/base_url; in multi-tenant deployments the first caller's verdict
    (under their credentials) is reused for all other users.
20. `research/outline_feedback.py:172-177` — indices returned in both `keep` and
    `remove` are never de-conflicted, so an item can land in both `kept_items`
    and `removed_items`, producing contradictory replacement logic.
21. `research/grouping.py:96-101` — the split-large-groups loop `enumerate`s the
    list while appending halves to it; appended halves are never re-examined, so
    a split that produces a still-oversized (>5) half is left oversized.

## Medium — robustness / hidden behavior

22. `progress/events.py:82-101` — `_flusher` sends coalesced `last_status` before
    queued `pending` MessageEvents, so a late-arriving status can reorder ahead
    of earlier streamed report content.
23. `orchestrator/phases/cycles.py:110-111` — `coverage_ratio =
    len(completed_topics)/max(len(all_topics),1)` divides topic-name completions
    by a denominator that includes every subtopic (which are ~never marked
    completed), so the ratio is structurally biased low and the early-finish
    branch rarely triggers.
24. `orchestrator/phases/cycles.py:41` — `active_outline = list(set(all_topics) -
    ...)` rebuilds the working outline through a `set`, discarding outline order
    nondeterministically before prioritization slices `[:10]`.
25. `core/errors.py:15,28,37` — `_TRANSIENT_COMPLETION_CODES = {429,502,504}` is
    misleading: the checks OR in `{500,503}`, so 500/503 are treated transient
    despite not being in the named set.
26. `adapter/client.py:191,198` — stream-retry backoff sleeps
    `1.5*(2**attempt)` after incrementing `attempt`, so the first retry waits
    ~3s (off-by-one vs `with_retry`) and there is no jitter.
27. `adapter/retry.py:31` — the terminal `raise AssertionError("unreachable")`
    is reachable if `max_retries` is negative (loop body never runs), raising an
    opaque assertion with `last_exc=None` instead of a meaningful error.
28. `semantics/eigendecomposition.py:106` — `variance_importance = eigenvalues /
    np.sum(eigenvalues)` over the retained top-N subset has no zero/negative
    guard here; `eigh`'s small negative eigenvalues can make the subset sum ~0,
    giving inf/NaN importances that poison the transformation matrix.
29. `budget/windows.py:81` — `start_char = max(0, min(start_char, len(content)-1))`
    clamps to `len-1`, returning a 1-char/empty window when the window start
    lands past the end, which `handle_repeated_content` cannot distinguish from
    real content.
30. `persistence/kb.py:138-145` — the `if knowledge is None` name-collision
    retry path in `ensure_research_kb` is unreachable: `create_kb` returns a
    validated model (raises on bad response) and never returns None.
31. `web/html_extract.py:129` — text extraction is submitted to
    `run_in_executor(None, ...)` (default loop executor) instead of
    `ctx.executor` used elsewhere, bypassing the configured `THREAD_WORKERS`
    bound.
32. `web/fetch.py:62-69` — the `POST_CLEAN_PRIMARY_OUTPUT` cleanup is wrapped in
    a bare `except Exception: pass` with no log, hiding cleanup regressions.
33. `persistence/sources.py:129` — `messages[-1].get("content")` assumes the last
    message is a dict; a malformed (e.g. string) entry raises `AttributeError`,
    unguarded, contrary to the `response_text` safety pattern.
34. `config/env.py:25-26` — `_coerce` failures are swallowed by
    `contextlib.suppress` with no log, so a typo'd numeric env var is silently
    dropped and the default used with no diagnostic.
35. `research/cycle.py:223-224` — content is sliced with
    `[: ctx.valves.web.max_result_tokens]` — a character slice using a valve
    named for *tokens*; truncation is by char count, not tokens.
36. `research/relevance.py:23,62` — `extract_topic_relevant_info` returns `[]`
    (list) on empty input but a `str` otherwise; inconsistent return type can
    break callers that branch on shape.

## Process-stable key violations (CLAUDE.md mandates `_stable_text_key`)

37. `synthesis/sections.py:43-44` — embedding cache keys use builtin
    `hash(original_query)` / `hash(subtopic)`; string `hash` is per-process
    salted, so keys don't survive across processes or match rehydrated state.
38. `synthesis/outline.py:86` — `result_embedding_{hash(url)}` — same per-process
    salt problem.
39. `semantics/similarity.py:50-51` — cache key is `hash(str(array.round(2)))`:
    salted per process and collision-prone (distinct vectors that round
    identically collide), so the similarity cache can return a wrong score.
40. `research/ranking.py:654` — cache key `hash(result_id) % 10000`: salted and
    `% 10000` invites collisions returning another result's similarity.
41. `web/paywall.py:164-168` — `hash(domain)` selects referrer/search-term
    indices; the comment claims "consistency" but string `hash` is only stable
    within a single process run.

## Config gaps

42. `config/valves.py:12` + `config/env.py` — `ModelsValves.embedding_model` is
    required for KB/embeddings but is absent from `VALVES_GROUP_MAP["models"]`,
    so `DR_MODELS_EMBEDDING_MODEL` is silently ignored; the env-only entrypoints
    (openapi_tool, mcp) cannot configure the embedding model at all.

## Dead code / misleading cruft

43. `synthesis/verify.py:133-378` — the entire `verify_citations` function is
    never called (only `verify_citation_batch` / `add_verification_note` are
    used); it also contains two indentation bugs (the "Starting verification"
    emit at `:231-236` is unreachable dead code inside the early-return block,
    and the "complete" emit at `:368-373` is mis-indented inside the per-result
    `elif` loop body). **[verified]** the function is uncalled.
44. `entrypoints/owui_function/pipe.py:89-93` — `_resolve_conversation_id` is
    dead; `pipe()` computes the id inline at line 54 and never calls it.
45. `adapter/client.py:354-370` — `get_file_content` returns
    `content_b64.encode("utf-8")` with no base64 decode despite the `_b64` name
    and `-> bytes` contract (would corrupt binary content); also unused.
46. `adapter/client.py` / `adapter/models.py:29,58,62` — `process_text`,
    `FileContentResponse`, `ChatCompletionResponse`, `WebSearchResponse` are
    defined but never used.
47. `config/constants.py:5` — `EXTRACTION_QUALITY = True` is never read; the live
    gate is `REQUIRE_EXTRACTION_QUALITY` (line 19) — dead and near-identically
    named.
48. `entrypoints/mcp/server.py:31-34` — `sink_events` accumulates every event for
    the whole run but is never read; unbounded dead accumulator.
49. `core/logging.py:8` — `setup_logger` hardcodes `setLevel(logging.DEBUG)` and
    `propagate = True` with no handler; floods DEBUG into any host with a root
    handler, untunable via config.
50. `orchestrator/coordinator.py:148` — `stream()` emits a final
    `MessageEvent(content=result.content)` with the whole report, but phases
    already stream content incrementally, risking duplicated output.

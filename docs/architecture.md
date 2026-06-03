# Architecture

## What a research run does

A run goes through nine phases:

1. **Rehydrate** — load checkpoint from `chat.deepResearch`, provision a
   per-conversation OWUI knowledge base, restore previous state.
2. **Outline feedback continuation** — if the previous turn was waiting
   for user feedback on the outline, fold the feedback into the topic
   graph.
3. **Initial queries** — detect follow-up vs fresh research, generate
   seed search queries, kick off the first cycle. If `interactive_research`
   is on and this is a fresh run, emit the proposed outline and stop here
   for one turn to let the user adjust.
4. **Outline** — produce a synthesis outline (the structure of the final
   report) from the research outline and initial results.
5. **Cycles** — the core loop. Each cycle: rank under-covered topics by
   gap vector + trajectory, generate fresh queries, fetch/process N URLs
   (REST extraction first, paywall-spoof fallback for PDFs, archive.org
   rescue on 403), filter by similarity to a learned preference direction
   vector, run quality filtering through a small LLM. Stops between
   `min_cycles` and `max_cycles` based on coverage.
6. **Compress** — stepped semantic compression of the results corpus,
   eigendecomposition-driven so chunks aligned with the research
   trajectory survive.
7. **Synthesise** — section-by-section content generation against the
   synthesis outline, with inline citations correlated to the master
   source table.
8. **Front/back matter** — titles, abstract, introduction, conclusion,
   review pass with edit application.
9. **Finalise** — assemble the full report, run citation verification,
   persist final report to the KB, export research data if enabled.

The whole run emits structured progress events (status pill, message
chunk, embed iframe) over an `EventBus` which the entrypoint translates to
OWUI's `__event_emitter__` payloads, MCP progress notifications, or — for
the OpenAPI Tool runtime — a `progress` snapshot on the polling endpoint.

---

## How a typical run looks to the user

With `interactive_research=True` (default):

1. **Turn 1 — user asks a question.** The pipe runs initial queries,
   produces a research outline, emits it as a message, and stops with a
   `waiting_for_outline_feedback` flag on conversation state. The progress
   embed shows "Awaiting outline confirmation".
2. **Turn 2 — user replies.** Empty / "looks good" / "please proceed" =
   continue as-is. Otherwise the feedback is parsed by the LLM, mapped to
   outline edits (add topic, remove topic, refine subtopic), and folded
   into the outline. Then the full research → synthesis → review pipeline
   runs to completion in this same turn.
3. **Final assistant message** contains the report. If
   `events.quiet_chat_mode=True` (default), only the final report shows
   in the assistant message; if `False`, intermediate cycle summaries are
   streamed as message chunks too.
4. **Per-conversation knowledge base** is provisioned on first turn and
   attached to the chat. Every selected source is persisted as a markdown
   file with the original URL, full extracted text, and metadata. The
   final report is also persisted, so a separate model can RAG against
   the whole research corpus.
5. **Follow-up turns** (same conversation, after a report exists) are
   detected as `post_report_user_qa` and answered against the KB instead
   of starting a fresh research run.

If `interactive_research=False`, step 1 runs straight through into
research cycles in the first turn.

---

## Package layout

```
┌─────────────────────────┐  ┌──────────────────────┐  ┌─────────────────────────────┐
│  Chat provider          │  │  Embeddings provider │  │  Open WebUI (0.9.2+)        │
│  (OpenAI-compatible)    │  │  (OpenAI-compatible) │  │  • /api/v1/retrieval/*      │
│  • /chat/completions    │  │  • /embeddings       │  │  • /api/v1/files/*          │
│  • /models              │  │                      │  │  • /api/v1/knowledge/*      │
└────────────▲────────────┘  └──────────▲───────────┘  │  • /api/v1/chats/*          │
             │                          │              └────────────▲────────────────┘
             │ httpx                    │ httpx                     │ httpx
             │ (LLMProviderClient)      │ (EmbeddingProviderClient) │ (OWUIClient)
             │                          │                           │
┌────────────┴──────────────────────────┴───────────────────────────┴──────────────┐
│                                                                                   │
│   deep_research/                                                                  │
│   ├── adapter/    LLMProviderClient, EmbeddingProviderClient, OWUIClient,        │
│   │               httpx, retries, semaphores                                      │
│   ├── core/       caches, state manager, types, errors, text utils               │
│   ├── config/     Valves, constants, env loader                                  │
│   ├── semantics/  embeddings, vocabulary, eigendecomp, dimensions,               │
│   │               trajectory, preference, similarity                             │
│   ├── budget/     token counting, context windows, packing                       │
│   ├── compression/ local-similarity, eigendecomp, repeated, stepped              │
│   ├── web/        search, classify, fetch, html/pdf extract,                     │
│   │               paywall (UA spoof + cookies, archive.org rescue)               │
│   ├── research/   query gen, ranking, relevance, outline-feedback,               │
│   │               grouping, per-query cycle driver                               │
│   ├── synthesis/  outline, sections+citations, verify, review,                   │
│   │               titles+abstract                                                │
│   ├── persistence/ chat-state checkpoints, KB ensure/attach/upload,              │
│   │                source markdown, post-report QA                               │
│   ├── orchestrator/ Coordinator + 9 phases                                       │
│   ├── progress/   event bus, snapshot, progress-embed HTML                       │
│   └── entrypoints/                                                               │
│       ├── owui_function/pipe.py        # OWUI Function shim                      │
│       ├── openapi_tool/server.py       # FastAPI JSON tool server                │
│       └── mcp/server.py                # FastMCP Streamable-HTTP                 │
│                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Every request that hits a runtime instantiates a fresh `RunContext`
(per-call, never shared). The `Coordinator` is process-shared; both
`LLMProviderClient` and `OWUIClient` HTTP sessions are process-shared. The
embedding / transformation / vocabulary caches are process-shared and
lock-protected. Two concurrent `.run()` calls for the same
`(user_id, conversation_id)` are rejected with `AlreadyRunningError` — the
same dedupe semantics as the original `pipe.py`.

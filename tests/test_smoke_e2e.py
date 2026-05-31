"""End-to-end smoke test for the deep_research engine (pytest-integrated).

No live OWUI is required: every OWUI REST endpoint is mocked with respx, and we
drive the exact coroutine the OpenAPI `/research` endpoint uses
(``Coordinator.stream``). This exercises all 9 phases end-to-end in-process, so
the synthesis / research / web / orchestrator modules are counted under
coverage (unlike the old out-of-process script).

The scenario is run once (module-scoped fixture) and individual behaviors are
asserted by separate test functions:

  (a) the user's prompt actually reaches the outbound search/query LLM call
  (b) the research KB is named from the prompt (dr-<ts>-<slug>), not dr-<ts>-research
  (c) a same-conversation follow-up answers from the KB and does NOT start a new
      search crawl (post-report mode)
  (d) turn 1 produces a substantial report (not just an outline footer)
  (e) no error-level status events are emitted
"""
import asyncio
import hashlib
import json
import os

import httpx
import pytest
import respx

from deep_research.adapter.auth import StaticToken
from deep_research.core.types import RunUser

BASE = "http://mock-owui:8080"

# Distinctive token so we can prove the prompt threaded through.
PROMPT = "Explain the mambazz architecture in detail"

LONG_TEXT = ("Mambazz is a state-space sequence model. " * 80).strip()

# Short, hermetic run configuration applied for the duration of the scenario.
SMOKE_ENV = {
    "DR_CYCLES_MIN_CYCLES": "1",
    "DR_CYCLES_MAX_CYCLES": "1",
    "DR_WEB_SEARCH_RESULTS_PER_QUERY": "1",
    "DR_WEB_SUCCESSFUL_RESULTS_PER_QUERY": "1",
    "DR_WEB_QUALITY_FILTER_ENABLED": "false",
    "DR_PERSISTENCE_INTERACTIVE_RESEARCH": "false",
    "DR_PERSISTENCE_EXPORT_RESEARCH_DATA": "false",
    "DR_EVENTS_ENABLE_PROGRESS_EMBED": "false",
    "DR_EVENTS_FLUSH_INTERVAL_MS": "10",
}

# One combined JSON blob: each phase's parser extracts only the key it needs and
# ignores the rest, so a single canned chat response satisfies every call site.
CHAT_JSON = json.dumps(
    {
        "queries": ["mambazz architecture overview", "mambazz vs transformer"],
        "outline": [
            {"topic": "Architecture", "subtopics": ["State space", "Selectivity"]},
        ],
        "completed_topics": ["Architecture"],
        "partial_topics": [],
        "irrelevant_topics": [],
        "new_topics": [],
        "analysis": "Covered the core architecture.",
        "main_title": "The Mambazz Architecture",
        "subtitle": "A Technical Overview",
        "citations": [],
        "is_relevant": True,
    }
)


def _install_mocks(router: respx.Router, state: dict) -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        state["chat_bodies"].append(request.content.decode("utf-8", "replace"))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": CHAT_JSON}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    def search_handler(request: httpx.Request) -> httpx.Response:
        state["search_calls"] += 1
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Mambazz paper",
                        "url": "https://example.com/mambazz",
                        "snippet": LONG_TEXT,
                    }
                ]
            },
        )

    def create_kb_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8", "replace"))
        name = payload.get("name", "")
        state["kb_names"].append(name)
        return httpx.Response(
            200,
            json={"id": "kb-1", "name": name, "description": payload.get("description", "")},
        )

    def embeddings_handler(request: httpx.Request) -> httpx.Response:
        # Vary the vector per input so PCA/eigendecomposition don't divide by zero.
        payload = json.loads(request.content.decode("utf-8", "replace"))
        inputs = payload.get("input") or [""]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = []
        for text in inputs:
            h = hashlib.sha256(str(text).encode("utf-8", "replace")).digest()
            vec = [((b / 255.0) - 0.5) for b in h[:8]]
            data.append({"embedding": vec})
        return httpx.Response(200, json={"data": data})

    router.post(f"{BASE}/api/chat/completions").mock(side_effect=chat_handler)
    router.post(f"{BASE}/api/embeddings").mock(side_effect=embeddings_handler)
    router.get(f"{BASE}/api/v1/models/list").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "gemma3:12b", "name": "gemma", "meta": {"context_length": 8192}}]},
        )
    )
    router.post(f"{BASE}/api/v1/retrieval/process/web/search").mock(side_effect=search_handler)
    router.post(f"{BASE}/api/v1/retrieval/process/web").mock(
        return_value=httpx.Response(200, json={"status": True, "content": LONG_TEXT})
    )
    router.post(f"{BASE}/api/v1/retrieval/process/file").mock(
        return_value=httpx.Response(200, json={"status": True, "content": LONG_TEXT})
    )
    router.post(f"{BASE}/api/v1/knowledge/create").mock(side_effect=create_kb_handler)
    router.post(url__regex=rf"{BASE}/api/v1/knowledge/.+/file/add").mock(
        return_value=httpx.Response(200, json={"status": True})
    )
    router.post(f"{BASE}/api/v1/files/").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "file-1",
                "filename": "src.md",
                "meta": {"content_type": "text/markdown", "size": len(LONG_TEXT)},
                "data": {"content": LONG_TEXT},
            },
        )
    )
    router.get(url__regex=rf"{BASE}/api/v1/files/.+").mock(
        return_value=httpx.Response(200, json={"id": "file-1", "data": {"content": LONG_TEXT}})
    )
    router.post(f"{BASE}/api/v1/retrieval/query/collection").mock(
        return_value=httpx.Response(
            200,
            json={
                "documents": [[LONG_TEXT]],
                "metadatas": [[{"source_url": "https://example.com/mambazz"}]],
                "distances": [[0.12]],
            },
        )
    )
    # vocabulary wordlist fetch (semantics/vocabulary.py) — keep it tiny & hermetic.
    wordlist = (
        "\n".join(f"word{i}" for i in range(60))
        + "\nstate\nspace\nmodel\nselective\narchitecture\ntransformer\n"
    )
    router.get("https://www.mit.edu/~ecprice/wordlist.10000").mock(
        return_value=httpx.Response(200, text=wordlist)
    )

    # chats: no chat_id is used in this harness, but mock defensively.
    router.get(url__regex=rf"{BASE}/api/v1/chats/.+").mock(return_value=httpx.Response(404))
    router.post(url__regex=rf"{BASE}/api/v1/chats/.+").mock(
        return_value=httpx.Response(200, json={"status": True})
    )


async def _consume(coord, conversation_id, prompt, token):
    last_message = ""
    events = 0
    errors: list[str] = []
    async for event in coord.stream(
        user=RunUser(id="smoke", name="Smoke"),
        conversation_id=conversation_id,
        chat_id=None,
        token=StaticToken(token),
        prompt=prompt,
        history=[],
    ):
        events += 1
        name = type(event).__name__
        if name == "MessageEvent":
            last_message = getattr(event, "content", "") or last_message
        elif name == "StatusEvent" and getattr(event, "level", "") == "error":
            errors.append(getattr(event, "description", ""))
    return last_message, events, errors


async def _run_scenario(data_dir: str) -> dict:
    from deep_research import Coordinator
    from deep_research.config.env import load_valves_from_env
    from deep_research.orchestrator.coordinator import RuntimeConfig

    valves = load_valves_from_env(prefix="DR_")
    config = RuntimeConfig(data_dir=data_dir, base_url=BASE)
    coord = Coordinator(valves=valves, config=config)
    await coord.start()

    state = {"chat_bodies": [], "search_calls": 0, "kb_names": []}
    out: dict = {}
    try:
        with respx.mock(assert_all_called=False) as router:
            _install_mocks(router, state)

            # ---- Turn 1: fresh research ----
            msg1, ev1, err1 = await _consume(coord, "smoke-conv", PROMPT, "sk-smoke")
            out["turn1_message_len"] = len(msg1)
            out["turn1_events"] = ev1
            out["searches_after_turn1"] = state["search_calls"]

            # ---- Turn 2: follow-up (same conversation) ----
            msg2, ev2, err2 = await _consume(
                coord, "smoke-conv", "What about its selectivity mechanism?", "sk-smoke"
            )
            out["turn2_message_len"] = len(msg2)
            out["searches_after_turn2"] = state["search_calls"]
    finally:
        await coord.close()

    out["chat_bodies"] = state["chat_bodies"]
    out["kb_names"] = state["kb_names"]
    out["errors"] = err1 + err2
    return out


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory) -> dict:
    """Run the two-turn research scenario once for the whole module.

    Uses a sync fixture + asyncio.run so the coordinator's httpx client and
    semaphores live entirely inside one short-lived event loop, independent of
    the per-test loop pytest-asyncio manages.
    """
    data_dir = str(tmp_path_factory.mktemp("dr_smoke"))
    saved = {k: os.environ.get(k) for k in SMOKE_ENV}
    os.environ.update(SMOKE_ENV)
    try:
        return asyncio.run(_run_scenario(data_dir))
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def test_prompt_reaches_outbound_llm(smoke_run):
    assert any("mambazz" in body for body in smoke_run["chat_bodies"])


def test_kb_named_from_prompt(smoke_run):
    assert any("mambazz" in name for name in smoke_run["kb_names"]), smoke_run["kb_names"]


def test_followup_starts_no_new_search(smoke_run):
    # Post-report mode: a same-conversation follow-up answers from the KB.
    assert smoke_run["searches_after_turn2"] == smoke_run["searches_after_turn1"]


def test_turn1_produces_substantial_report(smoke_run):
    # A bare > 0 check false-passes on the ~32-char outline footer.
    assert smoke_run["turn1_message_len"] > 500


def test_no_error_events(smoke_run):
    assert smoke_run["errors"] == []

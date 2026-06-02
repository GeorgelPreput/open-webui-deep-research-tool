"""End-to-end smoke for the embedding-throttle path under simulated 429s.

Reuses the mock harness from test_smoke_e2e.py but swaps in an embeddings
handler that 429s the first few requests (with Retry-After: 0 so the test
runs fast), then succeeds. Verifies:

  - The run completes despite initial 429s.
  - Retries actually happen (counter > 0).
  - No error-level events are emitted (degraded mode is a warning, not an error).
  - The final report is still substantial.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os

import httpx
import pytest
import respx

from deep_research.adapter.auth import StaticToken
from deep_research.core.types import RunUser
from tests.test_smoke_e2e import (
    BASE,
    EMBEDDINGS_BASE,
    LLM_BASE,
    PROMPT,
    SMOKE_ENV,
    _install_mocks,
)


def _install_mocks_with_429(router: respx.Router, state: dict, throttle_first: int) -> None:
    """Install the standard smoke mocks, then override the embeddings handler."""
    _install_mocks(router, state)

    # 0 = no delay so the test stays fast; the retry logic still observes the
    # header and clamps with jitter. The provider client sees JITTER_RATIO=0.25
    # of a 0s base, which is still 0.
    embed_calls = {"n": 0}

    def embeddings_handler(request: httpx.Request) -> httpx.Response:
        embed_calls["n"] += 1
        if embed_calls["n"] <= throttle_first:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"error_msg": "Over limit of 10k tokens per minute"},
            )
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

    # respx records the most recently registered route for the URL pattern;
    # we override the embeddings mock installed by _install_mocks.
    router.post(f"{EMBEDDINGS_BASE}/embeddings").mock(side_effect=embeddings_handler)
    state["embed_calls"] = embed_calls


async def _consume(coord, conversation_id, prompt, token):
    collected: list = []

    async def sink(ev) -> None:
        collected.append(ev)

    report = await coord.run(
        user=RunUser(id="smoke", name="Smoke"),
        conversation_id=conversation_id,
        chat_id=None,
        token=StaticToken(token),
        prompt=prompt,
        history=[],
        sink=sink,
    )
    last_message = report.content or ""
    errors: list[str] = []
    statuses: list[str] = []
    for ev in collected:
        if type(ev).__name__ == "StatusEvent":
            desc = getattr(ev, "description", "")
            level = getattr(ev, "level", "")
            statuses.append(f"[{level}] {desc}")
            if level == "error":
                errors.append(desc)
    return last_message, len(collected), errors, statuses


async def _run_scenario(data_dir: str, throttle_first: int) -> dict:
    from deep_research import Coordinator
    from deep_research.config.env import load_valves_from_env
    from deep_research.orchestrator.coordinator import RuntimeConfig

    valves = load_valves_from_env(prefix="DR_")
    config = RuntimeConfig(
        data_dir=data_dir,
        base_url=BASE,
        llm_base_url=LLM_BASE,
        llm_api_key="sk-smoke",
        embeddings_base_url=EMBEDDINGS_BASE,
        embeddings_api_key="sk-emb-smoke",
    )
    coord = Coordinator(valves=valves, config=config)
    await coord.start()

    state = {"chat_bodies": [], "search_calls": 0, "kb_names": []}
    out: dict = {}
    try:
        with respx.mock(assert_all_called=False) as router:
            _install_mocks_with_429(router, state, throttle_first=throttle_first)
            msg, events, errors, statuses = await _consume(
                coord, "smoke-throttle", PROMPT, "sk-smoke"
            )
            out["message_len"] = len(msg)
            out["events"] = events
            out["errors"] = errors
            out["statuses"] = statuses
            out["embed_calls"] = state["embed_calls"]["n"]
    finally:
        await coord.close()
    return out


@pytest.fixture(scope="module")
def smoke_throttle_run(tmp_path_factory) -> dict:
    data_dir = str(tmp_path_factory.mktemp("dr_smoke_throttle"))
    # Need DR_EMBEDDINGS_THROTTLE_MAX_RETRIES large enough to survive the first
    # few 429s without exhausting; default is already 5.
    saved = {k: os.environ.get(k) for k in SMOKE_ENV}
    os.environ.update(SMOKE_ENV)
    try:
        # 3 initial 429s, then 200s.
        return asyncio.run(_run_scenario(data_dir, throttle_first=3))
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def test_run_survives_initial_429s(smoke_throttle_run):
    """The run completes and emits no error events, despite initial 429s."""
    assert smoke_throttle_run["errors"] == [], smoke_throttle_run["errors"]
    assert smoke_throttle_run["message_len"] > 500


def test_diagnostics_line_shows_429_counter(smoke_throttle_run):
    diag_lines = [
        s for s in smoke_throttle_run["statuses"]
        if "Throttle diagnostics" in s
    ]
    assert diag_lines, "Expected a 'Throttle diagnostics' status event"
    line = diag_lines[-1]
    # The embeddings throttle should have observed at least one 429.
    assert "http_429=" in line
    assert "embeddings:" in line
    assert "llm:" in line


def test_embed_endpoint_was_retried(smoke_throttle_run):
    # If retries didn't happen, the run would have collapsed; total embedding
    # POSTs must exceed the number that were 429'd.
    assert smoke_throttle_run["embed_calls"] > 3

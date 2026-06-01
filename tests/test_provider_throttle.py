"""Throttle integration for the embedding and LLM provider clients."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from deep_research.adapter.client import AdapterError
from deep_research.adapter.llm_provider import (
    EmbeddingProviderClient,
    LLMProviderClient,
)
from deep_research.adapter.throttle import HttpThrottle


def _emb_client(
    throttle: HttpThrottle, *, batch_max_inputs: int = 0, max_retries: int = 0
) -> EmbeddingProviderClient:
    return EmbeddingProviderClient(
        base_url="http://mock-emb:9091",
        api_key="sk-emb",
        timeout_seconds=5,
        max_retries=max_retries,
        embedding_semaphore=asyncio.Semaphore(8),
        throttle=throttle,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
        batch_max_inputs=batch_max_inputs,
    )


def _llm_client(throttle: HttpThrottle, *, max_retries: int = 0) -> LLMProviderClient:
    return LLMProviderClient(
        base_url="http://mock-llm:9090",
        api_key="sk-llm",
        timeout_seconds=5,
        max_retries=max_retries,
        llm_semaphore=asyncio.Semaphore(4),
        throttle=throttle,
        base_delay_seconds=0.0,
        max_delay_seconds=0.0,
    )


@pytest.mark.asyncio
@respx.mock
async def test_embedding_429_then_success_records_one_retry_no_degraded():
    """A retried 429 that ultimately succeeds: 1 retry, 0 exhausted, not degraded."""
    route = respx.post("http://mock-emb:9091/embeddings")
    route.side_effect = [
        httpx.Response(429, json={"err": "rate limited"}),
        httpx.Response(200, json={"data": [{"embedding": [0.5]}]}),
    ]
    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    c = _emb_client(t, max_retries=3)
    await c.start()
    try:
        out = await c.embeddings("m", ["x"])
        assert out == [[0.5]]
    finally:
        await c.close()
    snap = t.snapshot()
    assert snap.attempts == 2
    assert snap.successes == 1
    assert snap.retries == 1
    assert snap.http_429 == 1
    assert snap.exhausted_429 == 0
    assert snap.degraded_now is False


@pytest.mark.asyncio
@respx.mock
async def test_embedding_persistent_429_trips_degraded():
    respx.post("http://mock-emb:9091/embeddings").mock(
        return_value=httpx.Response(429, json={"err": "rl"})
    )
    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    c = _emb_client(t, max_retries=2)
    await c.start()
    try:
        with pytest.raises(AdapterError):
            await c.embeddings("m", ["x"])
    finally:
        await c.close()
    snap = t.snapshot()
    assert snap.exhausted_429 == 1
    assert snap.degraded_now is True


@pytest.mark.asyncio
@respx.mock
async def test_llm_and_embedding_throttles_are_independent():
    """A 429 on embeddings must not degrade the LLM throttle."""
    respx.post("http://mock-emb:9091/embeddings").mock(
        return_value=httpx.Response(429, json={"err": "rl"})
    )
    respx.post("http://mock-llm:9090/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    emb_t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    llm_t = HttpThrottle(label="llm", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    emb = _emb_client(emb_t, max_retries=1)
    llm = _llm_client(llm_t, max_retries=1)
    await emb.start()
    await llm.start()
    try:
        with pytest.raises(AdapterError):
            await emb.embeddings("m", ["x"])
        resp = await llm.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert resp["choices"][0]["message"]["content"] == "ok"
    finally:
        await emb.close()
        await llm.close()
    assert emb_t.degraded is True
    assert llm_t.degraded is False
    assert llm_t.snapshot().successes == 1


@pytest.mark.asyncio
@respx.mock
async def test_embedding_batch_max_inputs_chunks_requests():
    """``batch_max_inputs=2`` should split a 5-element input into 3 POSTs."""
    captured: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        captured.append(list(body.get("input") or []))
        # Echo one zero-vector per input.
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.0]} for _ in body["input"]]},
        )

    respx.post("http://mock-emb:9091/embeddings").mock(side_effect=handler)
    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    c = _emb_client(t, batch_max_inputs=2)
    await c.start()
    try:
        out = await c.embeddings("m", ["a", "b", "c", "d", "e"])
    finally:
        await c.close()
    assert len(out) == 5
    assert captured == [["a", "b"], ["c", "d"], ["e"]]
    assert t.snapshot().attempts == 3
    assert t.snapshot().successes == 3


@pytest.mark.asyncio
@respx.mock
async def test_retry_after_header_honoured_via_provider_client(monkeypatch):
    """Provider client routes Retry-After through with_retry."""
    route = respx.post("http://mock-emb:9091/embeddings")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "3"}, json={"err": "rl"}),
        httpx.Response(200, json={"data": [{"embedding": [0.1]}]}),
    ]
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=10.0)
    c = EmbeddingProviderClient(
        base_url="http://mock-emb:9091",
        api_key="sk",
        timeout_seconds=5,
        max_retries=3,
        embedding_semaphore=asyncio.Semaphore(4),
        throttle=t,
        base_delay_seconds=0.01,  # would otherwise sleep ~10ms
        max_delay_seconds=10.0,
    )
    await c.start()
    try:
        await c.embeddings("m", ["x"])
    finally:
        await c.close()
    # The provider's jitter is JITTER_RATIO=0.25; the actual sleep is in
    # [3 * 0.75, 3 * 1.25] = [2.25, 3.75].
    assert len(sleeps) == 1
    assert 2.25 - 1e-6 <= sleeps[0] <= 3.75 + 1e-6

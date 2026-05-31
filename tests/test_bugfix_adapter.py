"""Regression tests for adapter-layer bug fixes.

Covers:
  BUG 25 - transient-code classification is a single source of truth.
  BUG 27 - with_retry does not raise a bogus AssertionError on max_retries < 0.
  BUG 14 - streaming retry does not re-yield already-emitted content.
  BUG 26 - streaming retry backoff is bounded (and retries pre-chunk failures).
"""
import asyncio
import json

import httpx
import pytest
import respx

from deep_research.adapter.auth import StaticToken
from deep_research.adapter.client import OWUIClient
from deep_research.core.errors import (
    _TRANSIENT_COMPLETION_CODES,
    classify_transient_completion_error,
)
from deep_research.adapter.retry import with_retry


# --- BUG 25: classify_transient_completion_error -----------------------------

def test_transient_constant_is_single_source_of_truth():
    # The named constant must contain every code the classifier treats as
    # transient (previously 500/503 were OR-ed in separately).
    assert {429, 500, 502, 503, 504}.issubset(_TRANSIENT_COMPLETION_CODES)


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_http_status_error_transient_codes(code):
    req = httpx.Request("GET", "http://x")
    resp = httpx.Response(code, request=req)
    err = httpx.HTTPStatusError("boom", request=req, response=resp)
    assert classify_transient_completion_error(err) == f"http_status={code}"


@pytest.mark.parametrize("code", [400, 401, 404, 422])
def test_http_status_error_non_transient_codes(code):
    req = httpx.Request("GET", "http://x")
    resp = httpx.Response(code, request=req)
    err = httpx.HTTPStatusError("boom", request=req, response=resp)
    assert classify_transient_completion_error(err) is None


def test_httpx_connect_error_is_transient():
    reason = classify_transient_completion_error(httpx.ConnectError("refused"))
    assert reason == "httpx_type=ConnectError"


def test_duck_typed_status_attribute():
    class FakeErr(Exception):
        status = 503

    class FakeErr400(Exception):
        status = 400

    assert classify_transient_completion_error(FakeErr()) == "http_status=503"
    assert classify_transient_completion_error(FakeErr400()) is None


def test_fallback_phrase_matching():
    reason = classify_transient_completion_error(Exception("Server disconnected mid-stream"))
    assert reason is not None and "fallback_phrase" in reason


# --- BUG 27 + general retry: with_retry -------------------------------------

class _Transient(Exception):
    status = 503


class _Fatal(Exception):
    status = 400


@pytest.mark.asyncio
async def test_with_retry_returns_on_first_success():
    calls = {"n": 0}

    async def ok():
        calls["n"] += 1
        return 42

    assert await with_retry(ok, max_retries=3, base_delay=0) == 42
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_with_retry_retries_transient_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Transient()
        return "ok"

    assert await with_retry(flaky, max_retries=3, base_delay=0) == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_non_transient_raises_immediately():
    calls = {"n": 0}

    async def fatal():
        calls["n"] += 1
        raise _Fatal()

    with pytest.raises(_Fatal):
        await with_retry(fatal, max_retries=3, base_delay=0)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_with_retry_negative_max_retries_raises_real_error_not_assertion():
    # BUG 27: range(max_retries+1) was empty for negative input, falling through
    # to `raise AssertionError("unreachable")`. The clamp must run >=1 attempt
    # and surface the real exception.
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise _Transient()

    with pytest.raises(_Transient):
        await with_retry(boom, max_retries=-1, base_delay=0)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_with_retry_exhausts_then_raises():
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise _Transient()

    with pytest.raises(_Transient):
        await with_retry(boom, max_retries=2, base_delay=0)
    assert calls["n"] == 3  # initial + 2 retries


# --- BUG 14 / 26: streaming retry -------------------------------------------

def _make_client(max_retries: int = 1) -> OWUIClient:
    return OWUIClient(
        base_url="http://mock-owui:8080",
        token_provider=StaticToken("mock-token"),
        timeout_seconds=5,
        max_retries=max_retries,
        llm_semaphore=asyncio.Semaphore(4),
        embedding_semaphore=asyncio.Semaphore(8),
        search_semaphore=asyncio.Semaphore(2),
        fetch_semaphore=asyncio.Semaphore(4),
    )


def _sse(*chunks: str) -> str:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]})
        for c in chunks
    ]
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


@pytest.mark.asyncio
@respx.mock
async def test_streaming_assembles_chunks_without_duplication():
    respx.post("http://mock-owui:8080/api/chat/completions").respond(
        200, text=_sse("Hello", " ", "world")
    )
    client = _make_client()
    await client.start()
    try:
        resp = await client.chat_completions("m", [{"role": "user", "content": "hi"}], stream=True)
        assert resp["choices"][0]["message"]["content"] == "Hello world"
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_streaming_retries_pre_chunk_failure(monkeypatch):
    # First attempt fails before any chunk (503); retry must succeed and the
    # content must appear exactly once. Patch sleep so the bounded backoff is fast.
    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr("deep_research.adapter.client.asyncio.sleep", _no_sleep)

    route = respx.post("http://mock-owui:8080/api/chat/completions")
    route.side_effect = [
        httpx.Response(503, text="server error"),
        httpx.Response(200, text=_sse("only", " once")),
    ]
    client = _make_client(max_retries=2)
    await client.start()
    try:
        resp = await client.chat_completions("m", [{"role": "user", "content": "hi"}], stream=True)
        assert resp["choices"][0]["message"]["content"] == "only once"
        assert route.call_count == 2
    finally:
        await client.close()

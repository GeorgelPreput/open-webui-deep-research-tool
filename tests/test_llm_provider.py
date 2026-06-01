import asyncio

import httpx
import pytest
import respx

from deep_research.adapter.client import AdapterError
from deep_research.adapter.llm_provider import LLMProviderClient


def _make_client(
    base_url: str = "http://mock-llm:9090",
    api_key: str = "sk-test",
    chat_path: str = "/chat/completions",
    max_retries: int = 1,
) -> LLMProviderClient:
    return LLMProviderClient(
        base_url=base_url,
        api_key=api_key,
        chat_path=chat_path,
        timeout_seconds=5,
        max_retries=max_retries,
        llm_semaphore=asyncio.Semaphore(4),
    )


def _chat_ok() -> dict:
    return {"choices": [{"message": {"content": "hi"}}]}


def _sse_chunks() -> str:
    return (
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        "data: [DONE]\n\n"
    )


# ---- chat_completions ----

@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_default_path():
    route = respx.post("http://mock-llm:9090/chat/completions").respond(200, json=_chat_ok())
    client = _make_client()
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert route.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_custom_path():
    default_route = respx.post("http://mock-llm:9090/chat/completions")
    custom_route = respx.post("http://mock-llm:9090/v1/chat/completions").respond(200, json=_chat_ok())
    client = _make_client(chat_path="/v1/chat/completions")
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert custom_route.call_count == 1
        assert default_route.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_uses_bearer_api_key():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json=_chat_ok())

    respx.post("http://mock-llm:9090/chat/completions").mock(side_effect=handler)
    client = _make_client(api_key="sk-mykey")
    await client.start()
    try:
        await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert captured == ["Bearer sk-mykey"]
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_4xx_raises_no_fallback():
    primary = respx.post("http://mock-llm:9090/chat/completions").respond(400, json={"detail": "bad"})
    client = _make_client()
    await client.start()
    try:
        with pytest.raises(AdapterError) as exc:
            await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert exc.value.status == 400
        assert primary.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_5xx_retries():
    route = respx.post("http://mock-llm:9090/chat/completions")
    route.side_effect = [
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json=_chat_ok()),
    ]
    client = _make_client(max_retries=3)
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert route.call_count == 2
    finally:
        await client.close()


# ---- stream_chat_completions ----

@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_completions():
    respx.post("http://mock-llm:9090/chat/completions").respond(
        200, text=_sse_chunks(), headers={"content-type": "text/event-stream"}
    )
    client = _make_client()
    await client.start()
    try:
        chunks: list[str] = []
        async for delta in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]):
            chunks.append(delta)
        assert "".join(chunks) == "hello world"
    finally:
        await client.close()


# ---- list_models ----

@pytest.mark.asyncio
@respx.mock
async def test_list_models_normalizes_context_window():
    respx.get("http://mock-llm:9090/models").respond(
        200,
        json={
            "data": [
                {"id": "m1", "name": "M1", "meta": {"context_length": 16384}},
                {"id": "m2", "name": "M2", "details": {"num_ctx": 8192}},
                {"id": "m3", "name": "M3"},
            ]
        },
    )
    client = _make_client()
    await client.start()
    try:
        models = await client.list_models()
        by_id = {m.id: m for m in models}
        assert by_id["m1"].context_window == 16384
        assert by_id["m2"].context_window == 8192
        assert by_id["m3"].context_window is None
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_list_models_path_is_slash_models():
    # Confirm the path is exactly /models (not /api/v1/models/list)
    owui_path = respx.get("http://mock-llm:9090/api/v1/models/list")
    correct_path = respx.get("http://mock-llm:9090/models").respond(
        200, json={"data": [{"id": "x", "name": "x"}]}
    )
    client = _make_client()
    await client.start()
    try:
        await client.list_models()
        assert correct_path.call_count == 1
        assert owui_path.call_count == 0
    finally:
        await client.close()

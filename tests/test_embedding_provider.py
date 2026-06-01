import asyncio

import httpx
import pytest
import respx

from deep_research.adapter.client import AdapterError
from deep_research.adapter.llm_provider import EmbeddingProviderClient, LLMProviderClient


def _make_client(
    base_url: str = "http://mock-emb:9091",
    api_key: str = "sk-emb",
    embeddings_path: str = "/embeddings",
    max_retries: int = 1,
) -> EmbeddingProviderClient:
    return EmbeddingProviderClient(
        base_url=base_url,
        api_key=api_key,
        embeddings_path=embeddings_path,
        timeout_seconds=5,
        max_retries=max_retries,
        embedding_semaphore=asyncio.Semaphore(8),
    )


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_default_path():
    route = respx.post("http://mock-emb:9091/embeddings").respond(
        200,
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]},
    )
    client = _make_client()
    await client.start()
    try:
        vectors = await client.embeddings("emb-model", ["text1", "text2"])
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert route.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_custom_path():
    default_route = respx.post("http://mock-emb:9091/embeddings")
    custom_route = respx.post("http://mock-emb:9091/v1/embeddings").respond(
        200, json={"data": [{"embedding": [1.0]}]}
    )
    client = _make_client(embeddings_path="/v1/embeddings")
    await client.start()
    try:
        vectors = await client.embeddings("m", ["t"])
        assert vectors == [[1.0]]
        assert custom_route.call_count == 1
        assert default_route.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_uses_bearer_api_key():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"data": [{"embedding": [0.0]}]})

    respx.post("http://mock-emb:9091/embeddings").mock(side_effect=handler)
    client = _make_client(api_key="sk-emb-key")
    await client.start()
    try:
        await client.embeddings("m", ["t"])
        assert captured == ["Bearer sk-emb-key"]
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_4xx_raises_no_fallback():
    route = respx.post("http://mock-emb:9091/embeddings").respond(
        401, json={"detail": "unauthorized"}
    )
    client = _make_client()
    await client.start()
    try:
        with pytest.raises(AdapterError) as exc:
            await client.embeddings("m", ["t"])
        assert exc.value.status == 401
        assert route.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_5xx_retries():
    route = respx.post("http://mock-emb:9091/embeddings")
    route.side_effect = [
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json={"data": [{"embedding": [0.5]}]}),
    ]
    client = _make_client(max_retries=3)
    await client.start()
    try:
        vectors = await client.embeddings("m", ["t"])
        assert vectors == [[0.5]]
        assert route.call_count == 2
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_distinct_from_llm_key():
    """Chat and embeddings use their own bearer tokens even when pointed at
    different mock bases. This is the core behavior that motivates the split."""
    chat_keys: list[str] = []
    emb_keys: list[str] = []

    def chat_handler(request: httpx.Request) -> httpx.Response:
        chat_keys.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    def emb_handler(request: httpx.Request) -> httpx.Response:
        emb_keys.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"data": [{"embedding": [0.0]}]})

    respx.post("http://mock-llm:9090/chat/completions").mock(side_effect=chat_handler)
    respx.post("http://mock-emb:9091/embeddings").mock(side_effect=emb_handler)

    llm = LLMProviderClient(
        base_url="http://mock-llm:9090",
        api_key="sk-chat",
        timeout_seconds=5,
        max_retries=1,
        llm_semaphore=asyncio.Semaphore(4),
    )
    emb = _make_client(api_key="sk-emb-only")

    await llm.start()
    await emb.start()
    try:
        await llm.chat_completions("m", [{"role": "user", "content": "hi"}])
        await emb.embeddings("m", ["t"])
        assert chat_keys == ["Bearer sk-chat"]
        assert emb_keys == ["Bearer sk-emb-only"]
    finally:
        await llm.close()
        await emb.close()

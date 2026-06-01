import asyncio

import httpx
import pytest
import respx

from deep_research.adapter.auth import StaticToken
from deep_research.adapter.client import OWUIClient, OWUIClientError


def _make_client(
    max_retries: int = 1,
    chat_completions_path: str = "/api/chat/completions",
    chat_completions_fallback_path: str = "",
) -> OWUIClient:
    return OWUIClient(
        base_url="http://mock-owui:8080",
        token_provider=StaticToken("mock-token"),
        timeout_seconds=5,
        max_retries=max_retries,
        llm_semaphore=asyncio.Semaphore(4),
        embedding_semaphore=asyncio.Semaphore(8),
        search_semaphore=asyncio.Semaphore(2),
        fetch_semaphore=asyncio.Semaphore(4),
        chat_completions_path=chat_completions_path,
        chat_completions_fallback_path=chat_completions_fallback_path,
    )


_NONETYPE_400 = {"detail": "'NoneType' object has no attribute 'startswith'"}


def _chat_ok() -> dict:
    return {"choices": [{"message": {"content": "hi"}}]}


def _sse_chunks() -> str:
    return (
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        "data: [DONE]\n\n"
    )


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_happy_path():
    respx.post("http://mock-owui:8080/api/embeddings").respond(
        200,
        json={"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]},
    )
    client = _make_client()
    await client.start()
    try:
        vectors = await client.embeddings("test-model", ["text1", "text2"])
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_get_chat_returns_none_on_404():
    respx.get("http://mock-owui:8080/api/v1/chats/missing").respond(404)
    client = _make_client()
    await client.start()
    try:
        result = await client.get_chat("missing")
        assert result is None
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_process_web_url_top_level_content():
    respx.post("http://mock-owui:8080/api/v1/retrieval/process/web").respond(
        200, json={"status": True, "content": "extracted text"}
    )
    client = _make_client()
    await client.start()
    try:
        resp = await client.process_web_url("https://example.com")
        assert resp.status is True
        assert resp.content == "extracted text"
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_query_collection_lists_of_lists():
    respx.post("http://mock-owui:8080/api/v1/retrieval/query/collection").respond(
        200,
        json={
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"a": 1}, {"b": 2}]],
        },
    )
    client = _make_client()
    await client.start()
    try:
        resp = await client.query_collection(["coll"], "query", k=2)
        assert resp.documents == [["doc1", "doc2"]]
        assert resp.distances == [[0.1, 0.2]]
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_list_models_normalizes_context_window():
    respx.get("http://mock-owui:8080/api/v1/models/list").respond(
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
async def test_retry_on_5xx_eventually_succeeds():
    route = respx.post("http://mock-owui:8080/api/embeddings")
    route.side_effect = [
        httpx.Response(503, text="server error"),
        httpx.Response(200, json={"data": [{"embedding": [1.0]}]}),
    ]
    client = _make_client(max_retries=3)
    await client.start()
    try:
        vectors = await client.embeddings("m", ["t"])
        assert vectors == [[1.0]]
        assert route.call_count == 2
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_default_path():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        200, json=_chat_ok()
    )
    client = _make_client()
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert primary.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_override_path():
    primary = respx.post("http://mock-owui:8080/api/chat/completions")
    override = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, json=_chat_ok()
    )
    client = _make_client(chat_completions_path="/openai/chat/completions")
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert override.call_count == 1
        assert primary.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_fallback_on_400_signature():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, json=_chat_ok()
    )
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert primary.call_count == 1
        assert fallback.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_fallback_on_404():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        404, text="not found"
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, json=_chat_ok()
    )
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        result = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert result == _chat_ok()
        assert primary.call_count == 1
        assert fallback.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_no_fallback_when_disabled():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions")
    client = _make_client()  # no fallback configured
    await client.start()
    try:
        with pytest.raises(OWUIClientError) as exc:
            await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert exc.value.status == 400
        assert primary.call_count == 1
        assert fallback.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_no_fallback_on_unrelated_400():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json={"detail": "model 'x' not found"}
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions")
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        with pytest.raises(OWUIClientError) as exc:
            await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert exc.value.status == 400
        assert primary.call_count == 1
        assert fallback.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_path_locked_after_fallback():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, json=_chat_ok()
    )
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert primary.call_count == 1
        assert fallback.call_count == 2
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_no_fallback_loop():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        with pytest.raises(OWUIClientError):
            await client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert primary.call_count == 1
        assert fallback.call_count == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_completions_override_path():
    primary = respx.post("http://mock-owui:8080/api/chat/completions")
    override = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, text=_sse_chunks(), headers={"content-type": "text/event-stream"}
    )
    client = _make_client(chat_completions_path="/openai/chat/completions")
    await client.start()
    try:
        chunks: list[str] = []
        async for delta in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]):
            chunks.append(delta)
        assert "".join(chunks) == "hello world"
        assert override.call_count == 1
        assert primary.call_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@respx.mock
async def test_stream_chat_completions_fallback_before_first_chunk():
    primary = respx.post("http://mock-owui:8080/api/chat/completions").respond(
        400, json=_NONETYPE_400
    )
    fallback = respx.post("http://mock-owui:8080/openai/chat/completions").respond(
        200, text=_sse_chunks(), headers={"content-type": "text/event-stream"}
    )
    client = _make_client(chat_completions_fallback_path="/openai/chat/completions")
    await client.start()
    try:
        chunks: list[str] = []
        async for delta in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]):
            chunks.append(delta)
        assert "".join(chunks) == "hello world"
        assert primary.call_count == 1
        assert fallback.call_count == 1
    finally:
        await client.close()

import asyncio

import httpx
import pytest
import respx

from deep_research.adapter.auth import StaticToken
from deep_research.adapter.client import OWUIClient


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

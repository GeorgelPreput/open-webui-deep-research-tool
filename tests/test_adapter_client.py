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
        search_semaphore=asyncio.Semaphore(2),
        fetch_semaphore=asyncio.Semaphore(4),
    )


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
async def test_retry_on_5xx_eventually_succeeds():
    route = respx.post("http://mock-owui:8080/api/v1/retrieval/process/file")
    route.side_effect = [
        httpx.Response(503, text="server error"),
        httpx.Response(
            200,
            json={"status": True, "collection_name": "c", "filename": "f.md", "content": "ok"},
        ),
    ]
    client = _make_client(max_retries=3)
    await client.start()
    try:
        result = await client.process_file("file-1")
        assert result.content == "ok"
        assert route.call_count == 2
    finally:
        await client.close()

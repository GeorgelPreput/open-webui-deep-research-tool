"""Tests for the synchronous `POST /research` endpoint.

Patches the module-level `_coord` with a fake coordinator that returns a
canned `Report`, so we can drive the endpoint without touching the real
research engine. Assertions cover:

  - OpenAPI schema shape (operationId, description, response_model)
  - 200 happy path returning the structured response
  - Citations derived from `report.bibliography` and enriched from
    `report.sources`
  - 409 mapping when the coordinator raises `AlreadyRunningError`
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deep_research.core.types import Report
from deep_research.entrypoints.openapi_tool import server as srv
from deep_research.orchestrator.coordinator import AlreadyRunningError


class _FakeCoord:
    def __init__(self, report: Report | None = None, raises: Exception | None = None):
        self._report = report
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs):  # noqa: D401 - test double
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        # Invoke the sink so we know the endpoint passed something callable.
        sink = kwargs.get("sink")
        if sink is not None:
            await sink(_StubStatus("planning"))
        return self._report


class _StubStatus:
    def __init__(self, description: str, level: str = "info", done: bool = False):
        self.description = description
        self.level = level
        self.done = done


@pytest.fixture
def canned_report() -> Report:
    return Report(
        content="# Topic\n\nKey finding [1] backed by [2].",
        title="Topic",
        sources={
            "https://a.example/paper": {
                "id": "S1",
                "title": "Paper A",
                "content_preview": "An influential paper about the topic.",
                "source_type": "academic",
            },
            "https://b.example/blog": {
                "id": "S2",
                "title": "Blog B",
                "content_preview": "  A blog post.  ",
                "source_type": "blog",
            },
        },
        bibliography=[
            {"id": 1, "title": "Paper A", "url": "https://a.example/paper"},
            {"id": 2, "title": "Blog B", "url": "https://b.example/blog"},
        ],
        token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        report_file_id=None,
        conversation_id="conv_test",
    )


@pytest.fixture
def client_factory(monkeypatch):
    """Yields a function that builds a TestClient backed by a fake coordinator."""
    def _build(fake: _FakeCoord) -> TestClient:
        monkeypatch.setattr(srv, "_coord", fake)
        # Reset job registry between tests so they're independent.
        srv._jobs.clear()
        return TestClient(srv.app)
    return _build


def test_openapi_schema_has_research_operation(client_factory, canned_report):
    client = client_factory(_FakeCoord(report=canned_report))
    schema = client.get("/openapi.json").json()
    op = schema["paths"]["/research"]["post"]
    assert op["operationId"] == "research"
    assert op["description"]  # non-empty, what the LLM reads
    assert "research" in op["tags"]

    # Response schema must list our fields, not be the empty {} the issue called out.
    response_schema_ref = op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    component = response_schema_ref.rsplit("/", 1)[-1]
    research_resp = schema["components"]["schemas"][component]
    props = research_resp["properties"]
    for required_field in ("report", "title", "citations", "conversation_id", "metadata"):
        assert required_field in props, f"missing {required_field} in ResearchResponse schema"


def test_research_sync_happy_path(client_factory, canned_report):
    fake = _FakeCoord(report=canned_report)
    client = client_factory(fake)
    resp = client.post(
        "/research",
        json={"prompt": "What is the topic?"},
        headers={"Authorization": "Bearer tok"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["report"].startswith("# Topic")
    assert body["title"] == "Topic"
    assert body["conversation_id"] == "conv_test"
    assert [c["id"] for c in body["citations"]] == [1, 2]
    assert body["citations"][0]["url"] == "https://a.example/paper"
    assert body["citations"][0]["title"] == "Paper A"
    # snippet is enriched from sources[url].content_preview, trimmed.
    assert body["citations"][1]["snippet"] == "A blog post."
    assert body["metadata"]["token_usage"]["total_tokens"] == 150
    # The coordinator received the prompt and a sink callable.
    assert fake.calls and fake.calls[0]["prompt"] == "What is the topic?"
    assert asyncio.iscoroutinefunction(fake.calls[0]["sink"])


def test_research_sync_already_running_returns_409(client_factory, canned_report):
    fake = _FakeCoord(raises=AlreadyRunningError("Research already running for conversation conv_x"))
    client = client_factory(fake)
    resp = client.post(
        "/research",
        json={"prompt": "x", "conversation_id": "conv_x"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "already_running"
    assert "conv_x" in detail["message"]


def test_research_sync_503_when_coordinator_unset(monkeypatch):
    monkeypatch.setattr(srv, "_coord", None)
    client = TestClient(srv.app)
    resp = client.post("/research", json={"prompt": "anything"})
    assert resp.status_code == 503


def test_research_sync_empty_bibliography_yields_empty_citations(client_factory):
    report = Report(
        content="No citations here.",
        title="Plain",
        sources={},
        bibliography=[],
        conversation_id="conv_empty",
    )
    client = client_factory(_FakeCoord(report=report))
    resp = client.post("/research", json={"prompt": "trivial"})
    assert resp.status_code == 200
    assert resp.json()["citations"] == []

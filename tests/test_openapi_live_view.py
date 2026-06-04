"""Tests for the /live_view/{id} and /live_view/{id}/status endpoints."""
from __future__ import annotations

import hashlib
import pathlib
import secrets
from typing import Any

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.entrypoints.openapi_tool import server as srv
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
)


class _StubRunner:
    def __init__(self) -> None:
        self.public_base_url = ""
        self._snapshots: dict[str, dict[str, Any]] = {}

    def get_snapshot(self, job_id: str) -> dict[str, Any]:
        return self._snapshots.get(job_id, {})

    def set_snapshot(self, job_id: str, snap: dict[str, Any]) -> None:
        self._snapshots[job_id] = snap


@pytest_asyncio.fixture
async def app_with_store(tmp_path: pathlib.Path):
    """Set up a minimal FastAPI app reusing server.py's routes but with a
    real JobStore and a stub runner — no Coordinator required."""
    store = JobStore(tmp_path / "jobs.sqlite")
    await store.start()
    runner = _StubRunner()

    app = FastAPI()
    # Mount the route handlers from server.py against this minimal app.
    for route in srv.app.routes:
        if getattr(route, "path", "").startswith(("/live_view", "/research_jobs", "/health")):
            app.router.routes.append(route)
    app.state.job_store = store
    app.state.runner = runner

    try:
        yield app, store, runner
    finally:
        await store.close()


def _create_record(store_async, **overrides):
    """Helper that returns the unhashed token + a populated record."""
    token = secrets.token_urlsafe(16)
    record = JobRecord(
        job_id=overrides.get("job_id", "jv-1"),
        user_id="u",
        user_name="U",
        conversation_id="conv-jv-1",
        chat_id="chat-jv-1",
        target_message_id="msg-jv-1",
        phase=overrides.get("phase", JobPhase.RESEARCHING),
        prompt="What is X?",
        history_json="[]",
        revision=overrides.get("revision", 3),
        view_token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
    )
    return token, record


async def test_live_view_returns_html_inline(app_with_store):
    app, store, runner = app_with_store
    token, record = _create_record(store)
    await store.create(record)
    runner.set_snapshot(record.job_id, {"query": "Hello"})

    client = TestClient(app)
    resp = client.get(f"/live_view/{record.job_id}", params={"token": token})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/html")
    assert resp.headers["content-disposition"] == "inline"
    assert "Access-Control-Expose-Headers" not in resp.headers  # set on real CORS middleware only
    assert "<html" in resp.text


async def test_live_view_html_contains_polling_script(app_with_store):
    app, store, runner = app_with_store
    token, record = _create_record(store, job_id="jv-poll")
    await store.create(record)

    client = TestClient(app)
    resp = client.get(f"/live_view/{record.job_id}", params={"token": token})
    assert resp.status_code == 200
    assert 'id="dr-bootstrap"' in resp.text
    assert "/live_view/jv-poll/status" in resp.text
    # nonce-protected script tags
    assert 'nonce="' in resp.text


async def test_live_view_wrong_token_returns_403(app_with_store):
    app, store, _ = app_with_store
    token, record = _create_record(store, job_id="jv-403")
    await store.create(record)

    client = TestClient(app)
    resp = client.get(f"/live_view/{record.job_id}", params={"token": "wrong"})
    assert resp.status_code == 403


async def test_live_view_unknown_job_returns_404(app_with_store):
    app, _, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/live_view/does-not-exist", params={"token": "whatever"})
    assert resp.status_code == 404


async def test_live_view_status_returns_snapshot_json(app_with_store):
    app, store, runner = app_with_store
    token, record = _create_record(store, job_id="jv-s", revision=7)
    await store.create(record)
    runner.set_snapshot(record.job_id, {"completed_topics": ["t1"]})

    client = TestClient(app)
    resp = client.get(
        f"/live_view/{record.job_id}/status",
        params={"token": token},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "jv-s"
    assert body["revision"] == 7
    assert body["completed"] is False
    assert body["progress"]["completed_topics"] == ["t1"]


async def test_live_view_status_204_when_no_change(app_with_store):
    app, store, _ = app_with_store
    token, record = _create_record(store, job_id="jv-204", revision=5)
    await store.create(record)

    client = TestClient(app)
    resp = client.get(
        f"/live_view/{record.job_id}/status",
        params={"token": token, "since_version": 5},
    )
    assert resp.status_code == 204
    assert resp.content == b""


async def test_live_view_status_200_when_revision_advanced(app_with_store):
    app, store, _ = app_with_store
    token, record = _create_record(store, job_id="jv-bump", revision=5)
    await store.create(record)

    client = TestClient(app)
    resp = client.get(
        f"/live_view/{record.job_id}/status",
        params={"token": token, "since_version": 4},
    )
    assert resp.status_code == 200
    assert resp.json()["revision"] == 5


async def test_live_view_status_token_mismatch_returns_403(app_with_store):
    app, store, _ = app_with_store
    token, record = _create_record(store, job_id="jv-st-403")
    await store.create(record)

    client = TestClient(app)
    resp = client.get(
        f"/live_view/{record.job_id}/status",
        params={"token": "nope"},
    )
    assert resp.status_code == 403


async def test_live_view_status_unknown_job_returns_404(app_with_store):
    app, _, _ = app_with_store
    client = TestClient(app)
    resp = client.get(
        "/live_view/nope/status",
        params={"token": "t"},
    )
    assert resp.status_code == 404


async def test_live_view_status_completed_flag(app_with_store):
    app, store, _ = app_with_store
    token, record = _create_record(store, job_id="jv-done", phase=JobPhase.COMPLETED)
    await store.create(record)

    client = TestClient(app)
    resp = client.get(
        f"/live_view/{record.job_id}/status",
        params={"token": token},
    )
    assert resp.status_code == 200
    assert resp.json()["completed"] is True

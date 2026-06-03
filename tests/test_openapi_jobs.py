"""Tests for the v2 OpenAPI Tool Server endpoint surface.

These mount the route handlers from ``server.py`` on a fresh ``FastAPI``
app with a real ``JobStore`` and a stub runner, so we can drive each
endpoint without spinning up the actual ``Coordinator``.
"""
from __future__ import annotations

import asyncio
import hashlib
import pathlib
import secrets
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.entrypoints.openapi_tool import server as srv
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
    _now_iso,
)


class _StubRunner:
    """Captures calls instead of executing a real engine."""

    def __init__(self) -> None:
        self.public_base_url = ""
        self.start_calls: list[dict[str, Any]] = []
        self.feedback_calls: list[tuple[str, str]] = []
        self.cancel_calls: list[str] = []
        self._snapshots: dict[str, dict[str, Any]] = {}

    async def start_job(self, record, *, view_token, owui_user_token):
        self.start_calls.append({
            "job_id": record.job_id,
            "view_token": view_token,
            "owui_user_token": owui_user_token,
            "chat_id": record.chat_id,
            "target_message_id": record.target_message_id,
        })

    async def submit_feedback(self, job_id, selection):
        self.feedback_calls.append((job_id, selection))

    async def cancel(self, job_id, *, timeout=30.0):
        self.cancel_calls.append(job_id)

    def get_snapshot(self, job_id):
        return self._snapshots.get(job_id, {})


@pytest_asyncio.fixture
async def app_with_state(tmp_path: pathlib.Path):
    store = JobStore(tmp_path / "jobs.sqlite")
    await store.start()
    runner = _StubRunner()

    app = FastAPI()
    for route in srv.app.routes:
        if getattr(route, "path", "").startswith(("/research_jobs", "/live_view", "/health")):
            app.router.routes.append(route)
    app.state.job_store = store
    app.state.runner = runner
    try:
        yield app, store, runner
    finally:
        await store.close()


@pytest.fixture
def client(app_with_state) -> TestClient:
    app, _, _ = app_with_state
    return TestClient(app)


def test_start_research_job_captures_forwarded_headers(app_with_state, client):
    _, store, runner = app_with_state

    resp = client.post(
        "/research_jobs",
        json={"prompt": "What is X?"},
        headers={
            "X-OpenWebUI-Chat-Id": "chat-fwd-1",
            "X-OpenWebUI-Message-Id": "msg-fwd-1",
            "Authorization": "Bearer user-token",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job_id"]
    assert body["status"] == "running"
    assert body["next_action"] == "await_user_selection"
    assert "user_facing_instruction" in body
    assert "/continue" in body["user_facing_instruction"]

    # Runner observed the forwarded headers
    assert runner.start_calls
    call = runner.start_calls[0]
    assert call["chat_id"] == "chat-fwd-1"
    assert call["target_message_id"] == "msg-fwd-1"
    assert call["owui_user_token"] == "user-token"


def test_start_research_job_returns_user_facing_instruction(client):
    resp = client.post("/research_jobs", json={"prompt": "Q"})
    assert resp.status_code == 200
    text = resp.json()["user_facing_instruction"]
    # The verbatim wording the LLM is told to emit must be self-explanatory
    for fragment in ("/k", "/r", "/continue", "live progress"):
        assert fragment in text


async def test_409_when_active_job_for_chat_exists(app_with_state):
    app, store, _ = app_with_state

    # Pre-create an active job for chat-foo
    existing = JobRecord(
        job_id="active-1",
        user_id="u",
        user_name="U",
        conversation_id="conv-active",
        chat_id="chat-foo",
        target_message_id="msg",
        phase=JobPhase.RESEARCHING,
        prompt="prompt",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )
    await store.create(existing)

    client = TestClient(app)
    resp = client.post(
        "/research_jobs",
        json={"prompt": "second"},
        headers={"X-OpenWebUI-Chat-Id": "chat-foo"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "already_running"
    assert "active-1" in detail["message"]


async def test_409_does_not_fire_when_existing_job_terminal(app_with_state):
    app, store, _ = app_with_state

    completed = JobRecord(
        job_id="done-1",
        user_id="u",
        user_name="U",
        conversation_id="conv-done",
        chat_id="chat-bar",
        target_message_id=None,
        phase=JobPhase.COMPLETED,
        prompt="prompt",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
        completed_at=_now_iso(),
    )
    await store.create(completed)

    client = TestClient(app)
    resp = client.post(
        "/research_jobs",
        json={"prompt": "next"},
        headers={"X-OpenWebUI-Chat-Id": "chat-bar"},
    )
    assert resp.status_code == 200


async def test_submit_feedback_404_unknown(app_with_state):
    app, _, _ = app_with_state
    client = TestClient(app)
    resp = client.post(
        "/research_jobs/nope/feedback",
        json={"selection": "/continue"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "unknown_job"


async def test_submit_feedback_409_when_not_awaiting(app_with_state):
    app, store, _ = app_with_state
    rec = JobRecord(
        job_id="not-awaiting",
        user_id="u",
        user_name="U",
        conversation_id="conv-x",
        chat_id="chat-x",
        target_message_id="msg",
        phase=JobPhase.RESEARCHING,
        prompt="p",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.post(
        f"/research_jobs/{rec.job_id}/feedback",
        json={"selection": "/k 1"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "not_awaiting_feedback"


async def test_submit_feedback_rebinds_target_message(app_with_state):
    app, store, runner = app_with_state

    rec = JobRecord(
        job_id="fb-rebind",
        user_id="u",
        user_name="U",
        conversation_id="conv-fb",
        chat_id="chat-fb",
        target_message_id="msg-old",
        phase=JobPhase.AWAITING_OUTLINE_FEEDBACK,
        prompt="p",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.post(
        f"/research_jobs/{rec.job_id}/feedback",
        json={"selection": "/k 1"},
        headers={"X-OpenWebUI-Message-Id": "msg-new"},
    )
    assert resp.status_code == 200
    refreshed = await store.get(rec.job_id)
    assert refreshed.target_message_id == "msg-new"
    assert runner.feedback_calls == [(rec.job_id, "/k 1")]


async def test_get_research_job_returns_completed_report(app_with_state):
    app, store, _ = app_with_state
    rec = JobRecord(
        job_id="get-done",
        user_id="u",
        user_name="U",
        conversation_id="conv",
        chat_id=None,
        target_message_id=None,
        phase=JobPhase.COMPLETED,
        prompt="p",
        history_json="[]",
        revision=4,
        view_token_hash="0" * 64,
        report_markdown="# Report",
        completed_at=_now_iso(),
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.get(f"/research_jobs/{rec.job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "completed"
    assert body["report_markdown"] == "# Report"
    assert body["error"] is None


async def test_get_research_job_returns_error_text_when_failed(app_with_state):
    app, store, _ = app_with_state
    rec = JobRecord(
        job_id="get-failed",
        user_id="u",
        user_name="U",
        conversation_id="conv",
        chat_id=None,
        target_message_id=None,
        phase=JobPhase.FAILED,
        prompt="p",
        history_json="[]",
        revision=2,
        view_token_hash="0" * 64,
        error_text="upstream 500",
        completed_at=_now_iso(),
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.get(f"/research_jobs/{rec.job_id}")
    assert resp.status_code == 200
    assert resp.json()["error"] == "upstream 500"


async def test_get_research_job_unknown_404(app_with_state):
    app, _, _ = app_with_state
    client = TestClient(app)
    resp = client.get("/research_jobs/nope")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "unknown_job"


async def test_cancel_research_job_404_for_unknown(app_with_state):
    app, _, _ = app_with_state
    client = TestClient(app)
    resp = client.post("/research_jobs/nope/cancel")
    assert resp.status_code == 404


async def test_cancel_research_job_returns_already_terminal_for_completed(
    app_with_state,
):
    app, store, _ = app_with_state
    rec = JobRecord(
        job_id="cx-term",
        user_id="u",
        user_name="U",
        conversation_id="conv",
        chat_id=None,
        target_message_id=None,
        phase=JobPhase.COMPLETED,
        prompt="p",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
        completed_at=_now_iso(),
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.post(f"/research_jobs/{rec.job_id}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "already_terminal"
    assert body["job_id"] == rec.job_id


async def test_cancel_research_job_routes_to_runner(app_with_state):
    app, store, runner = app_with_state
    rec = JobRecord(
        job_id="cx-go",
        user_id="u",
        user_name="U",
        conversation_id="conv",
        chat_id=None,
        target_message_id=None,
        phase=JobPhase.RESEARCHING,
        prompt="p",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )
    await store.create(rec)

    client = TestClient(app)
    resp = client.post(f"/research_jobs/{rec.job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancel_requested"
    assert runner.cancel_calls == [rec.job_id]


def test_openapi_schema_lists_v2_operations(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/research_jobs" in paths
    assert "/research_jobs/{job_id}" in paths
    assert "/research_jobs/{job_id}/feedback" in paths
    assert "/research_jobs/{job_id}/cancel" in paths
    assert "/live_view/{job_id}" in paths
    assert "/live_view/{job_id}/status" in paths
    # The old /research endpoint is gone
    assert "/research" not in paths

    start_op = paths["/research_jobs"]["post"]
    assert start_op["operationId"] == "start_research_job"
    assert "verbatim" in start_op["description"].lower()

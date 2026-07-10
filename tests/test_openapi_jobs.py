"""Tests for the v2 OpenAPI Tool Server endpoint surface.

These mount the route handlers from ``server.py`` on a fresh ``FastAPI``
app with a real ``JobStore`` and a stub runner, so we can drive each
endpoint without spinning up the actual ``Coordinator``.
"""
from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from deep_research.entrypoints.openapi_tool import server as srv
from deep_research.entrypoints.openapi_tool.config_audit import ConfigWarning
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
    _now_iso,
)
from deep_research.entrypoints.openapi_tool.runner import JobRunner

from ._runner_helpers import FakeCoord


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
    app.state.config_warnings = []
    app.state.config_warnings_lock = asyncio.Lock()
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

    # view_token is surfaced to the caller and matches what the runner saw
    assert body["view_token"]
    assert isinstance(body["view_token"], str)
    assert body["view_token"] == call["view_token"]


def test_start_research_job_returns_user_facing_instruction(client):
    resp = client.post("/research_jobs", json={"prompt": "Q"})
    assert resp.status_code == 200
    text = resp.json()["user_facing_instruction"]
    # The verbatim wording the LLM is told to emit must be self-explanatory
    for fragment in ("/k", "/r", "/continue", "live progress"):
        assert fragment in text


def test_user_facing_instruction_default_equals_example():
    """Both the field default and the json_schema_extra example resolve
    through the single USER_FACING_INSTRUCTION constant (Group 14). A future
    edit to one site without the other would desync them — catch it here."""
    from deep_research.entrypoints.openapi_tool.schemas import (
        USER_FACING_INSTRUCTION,
        StartResearchResponse,
    )

    assert (
        StartResearchResponse.model_fields["user_facing_instruction"].default
        == USER_FACING_INSTRUCTION
    )
    example = StartResearchResponse.model_config["json_schema_extra"]["example"]
    assert example["user_facing_instruction"] == USER_FACING_INSTRUCTION


def test_start_research_job_returns_view_token(app_with_state, client):
    _, _, runner = app_with_state
    resp = client.post("/research_jobs", json={"prompt": "Q"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["view_token"]
    assert isinstance(token, str)
    # secrets.token_urlsafe(32) yields ~43 url-safe characters
    assert len(token) >= 40
    assert all(c.isalnum() or c in "-_" for c in token)
    assert runner.start_calls
    assert runner.start_calls[0]["view_token"] == token


async def test_409_when_active_job_for_chat_exists(app_with_state, tmp_path):
    """Drive the 409 path through the real ``JobRunner`` so the sqlite
    UNIQUE-partial-index → ``ActiveJobExistsError`` → 409 flow is
    exercised end-to-end. The stub runner used by other tests doesn't
    call ``store.create`` and so cannot trigger the index."""
    app, store, _ = app_with_state

    # Swap in a real runner against the same JobStore. _FakeCoord
    # supplies an empty Coordinator surface; we never actually drive
    # a job through it, we just need start_job to attempt the
    # ``store.create`` and surface the IntegrityError.
    coord = FakeCoord()
    real_runner = JobRunner(coord=coord, store=store, outbox=None, public_base_url="")
    app.state.runner = real_runner

    # Pre-create an active job for chat-foo (this is the row the
    # second insert will collide with under the UNIQUE partial index).
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

    try:
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
        # Exactly one row for chat-foo (the pre-existing one); the
        # rejected insert did not leak.
        only = await store.find_active_by_chat("chat-foo")
        assert only is not None and only.job_id == "active-1"
    finally:
        await real_runner.shutdown()


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


async def test_start_research_job_409_for_local_chat_id(app_with_state):
    app, _, runner = app_with_state
    client = TestClient(app)
    resp = client.post(
        "/research_jobs",
        json={"prompt": "summarise X"},
        headers={"X-OpenWebUI-Chat-Id": "local:abc123"},
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "unsaved_chat_unsupported"
    assert "ephemeral" in detail["message"].lower()
    # No JobRecord was created, no runner call made.
    assert runner.start_calls == []


async def test_start_research_job_allows_persisted_chat_id(app_with_state):
    app, _, _ = app_with_state
    client = TestClient(app)
    resp = client.post(
        "/research_jobs",
        json={"prompt": "summarise X"},
        headers={"X-OpenWebUI-Chat-Id": "11111111-2222-3333-4444-555555555555"},
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

    start_resp_schema = schema["components"]["schemas"]["StartResearchResponse"]
    assert "view_token" in start_resp_schema["properties"]
    assert "view_token" in start_resp_schema["required"]


def test_health_returns_empty_config_warnings_when_clean(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["config_warnings"] == []
    # No OutboxWorker attached → outbox field is null so callers can
    # distinguish "writeback disabled" from "writeback on, zero rows".
    assert body["outbox"] is None


async def test_health_surfaces_outbox_counts_when_writeback_enabled(
    app_with_state, client, tmp_path: pathlib.Path
):
    """When ``app.state.outbox`` is set, ``/health`` returns the live
    ``count_by_status()`` mapping. Confirms the wiring works end-to-end
    without going through the lifespan handler.
    """
    from deep_research.entrypoints.openapi_tool.outbox import OutboxWorker

    class _NoopClient:
        async def post_message_event(self, *args, **kwargs):
            return None

    app, _, _ = app_with_state
    worker = OutboxWorker(
        db_path=tmp_path / "outbox.sqlite",
        owui_client=_NoopClient(),
    )
    await worker.start(spawn_loop=False)
    app.state.outbox = worker
    try:
        # Empty table: count_by_status returns {} → /health surfaces {}
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["outbox"] == {}

        # Enqueue + drain one row → status='delivered' count surfaces.
        await worker.enqueue(
            outbox_id="hb-1",
            job_id="job-h",
            chat_id="c",
            message_id="m",
            event_type="status",
            payload={"description": "ok", "done": False},
            dedupe_key="job-h:m:1",
        )
        await worker.drain_once()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["outbox"] == {"delivered": 1}
    finally:
        await worker.stop()
        app.state.outbox = None


def test_health_returns_config_warnings_when_present(app_with_state, client):
    app, _, _ = app_with_state
    app.state.config_warnings = [
        ConfigWarning(
            code="MISSING_PUBLIC_BASE_URL",
            severity="info",
            message="public base URL unset",
            remediation="set DR_OPENAPI_PUBLIC_BASE_URL",
        ),
        ConfigWarning(
            code="OWUI_API_KEY_NOT_ADMIN",
            severity="warning",
            message="key is not admin",
            remediation="use an admin key",
        ),
    ]
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    codes = [w["code"] for w in body["config_warnings"]]
    assert codes == ["MISSING_PUBLIC_BASE_URL", "OWUI_API_KEY_NOT_ADMIN"]
    # Severity is preserved
    by_code = {w["code"]: w for w in body["config_warnings"]}
    assert by_code["MISSING_PUBLIC_BASE_URL"]["severity"] == "info"
    assert by_code["OWUI_API_KEY_NOT_ADMIN"]["severity"] == "warning"


def test_headers_not_forwarded_appends_warning_once(app_with_state, client):
    """Authenticated request without X-OpenWebUI-Chat-Id appends
    OWUI_HEADERS_NOT_FORWARDED to app.state.config_warnings; a second
    identical call does NOT duplicate it (one-shot per process)."""
    app, _, _ = app_with_state

    resp1 = client.post(
        "/research_jobs",
        json={"prompt": "Q1"},
        headers={"Authorization": "Bearer user-token"},
    )
    assert resp1.status_code == 200

    codes = [w.code for w in app.state.config_warnings]
    assert codes == ["OWUI_HEADERS_NOT_FORWARDED"]

    resp2 = client.post(
        "/research_jobs",
        json={"prompt": "Q2"},
        headers={"Authorization": "Bearer user-token"},
    )
    assert resp2.status_code == 200

    # Still exactly one — the one-shot flag prevents duplicates
    codes_after = [w.code for w in app.state.config_warnings]
    assert codes_after == ["OWUI_HEADERS_NOT_FORWARDED"]


def test_headers_not_forwarded_silent_when_chat_id_present(app_with_state, client):
    """When the OWUI headers DO arrive, no warning is appended."""
    app, _, _ = app_with_state

    resp = client.post(
        "/research_jobs",
        json={"prompt": "Q"},
        headers={
            "Authorization": "Bearer user-token",
            "X-OpenWebUI-Chat-Id": "chat-1",
            "X-OpenWebUI-Message-Id": "msg-1",
        },
    )
    assert resp.status_code == 200
    assert app.state.config_warnings == []


def test_headers_not_forwarded_fires_without_bearer(app_with_state, client):
    """Detector gates on chat-id absence ALONE: a request with no bearer and
    no X-OpenWebUI-Chat-Id still trips OWUI_HEADERS_NOT_FORWARDED. This closes
    the blind spot where an OWUI tool server configured auth:none with header
    forwarding off would silently disable writeback with no operator signal."""
    app, _, _ = app_with_state

    resp = client.post("/research_jobs", json={"prompt": "Q"})
    assert resp.status_code == 200
    codes = [w.code for w in app.state.config_warnings]
    assert codes == ["OWUI_HEADERS_NOT_FORWARDED"]


def test_headers_not_forwarded_holds_lock_during_append(app_with_state, client):
    """The runtime warning append happens inside config_warnings_lock; the
    `append` call sits between `__aenter__` and `__aexit__` so a concurrent
    handler couldn't observe the half-written state."""
    app, _, _ = app_with_state
    events: list[str] = []
    real_lock = app.state.config_warnings_lock

    class _RecordingLock:
        async def __aenter__(self):
            events.append("lock_acquired")
            await real_lock.__aenter__()
            return self

        async def __aexit__(self, *exc):
            events.append("lock_released")
            return await real_lock.__aexit__(*exc)

    class _RecordingList(list):
        def append(self, item):
            events.append("append")
            return super().append(item)

    app.state.config_warnings_lock = _RecordingLock()
    app.state.config_warnings = _RecordingList()

    resp = client.post(
        "/research_jobs",
        json={"prompt": "Q"},
        headers={"Authorization": "Bearer user-token"},
    )
    assert resp.status_code == 200
    assert events == ["lock_acquired", "append", "lock_released"]

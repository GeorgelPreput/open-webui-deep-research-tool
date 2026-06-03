"""End-to-end writeback test: JobRunner + OutboxWorker → OWUI ``/event``.

Drives a job through the start-job → outline gate → submit-feedback →
completion sequence with a fake Coordinator and a fake OWUIClient on the
writeback path. Asserts the *full* sequence of persisted event types
posted to OWUI, in order:

  1. ``embeds`` (bootstrap iframe on the first tool-call message)
  2. ``status`` events emitted during phase 1 (initial queries)
  3. ``replace`` (the topic list from the outline gate, as a MessageEvent)
  4. ``embeds`` (bootstrap iframe on the SECOND tool-call message)
  5. ``status`` / ``source`` during phase 2 (research → finalize)
  6. ``replace`` (final report)
  7. ``embeds`` (clear)

Also asserts:
  - Long-name aliases (``chat:message:embeds`` etc.) never appear; only
    the short names the ``/event`` endpoint persists.
  - The runner skips writeback when ``chat_id`` is None.
  - The runner skips writeback when ``chat_id`` starts with ``local:``.
"""
from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest
import pytest_asyncio

from deep_research.adapter.client import PERSISTED_EVENT_TYPES
from deep_research.core.types import Report
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
)
from deep_research.entrypoints.openapi_tool.outbox import OutboxWorker
from deep_research.entrypoints.openapi_tool.runner import JobRunner
from deep_research.progress.events import (
    CitationEvent,
    EmbedEvent,
    MessageEvent,
    StatusEvent,
)


class _FakeStateManager:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    def get_state(self, cid: str) -> dict[str, Any]:
        return self.states.setdefault(cid, {})

    def set_waiting(self, cid: str, *, outline=None) -> None:
        st = self.get_state(cid)
        st["waiting_for_outline_feedback"] = True
        st["outline_feedback_data"] = {
            "outline_items": outline or [{"topic": "x"}],
        }


class _FakeCoord:
    """Coordinator double — invokes the supplied side-effect on .run()."""

    def __init__(self) -> None:
        self.state_manager = _FakeStateManager()
        self.run_calls: list[dict[str, Any]] = []
        self.on_run: Any = None  # async callable(kwargs) -> Report

    async def run(self, **kwargs):
        self.run_calls.append(kwargs)
        if self.on_run is not None:
            return await self.on_run(kwargs)
        return Report(content="", conversation_id=kwargs.get("conversation_id"))


class _FakeOWUIClient:
    """Captures /event posts so we can assert the writeback sequence."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post_message_event(
        self, chat_id: str, message_id: str, event_type: str, data: dict
    ) -> None:
        self.calls.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "event_type": event_type,
            "data": data,
        })


def _make_record(
    *,
    job_id: str = "wb-1",
    chat_id: str | None = "chat-1",
    target_message_id: str | None = "msg-1",
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        user_id="u",
        user_name="U",
        conversation_id=f"conv_{job_id}",
        chat_id=chat_id,
        target_message_id=target_message_id,
        phase=JobPhase.QUEUED,
        prompt="research the thing",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )


@pytest_asyncio.fixture
async def store(tmp_path: pathlib.Path):
    s = JobStore(tmp_path / "jobs.sqlite")
    await s.start()
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def client():
    return _FakeOWUIClient()


@pytest_asyncio.fixture
async def outbox(tmp_path: pathlib.Path, client: _FakeOWUIClient):
    # spawn_loop=False so we drive drains explicitly between phases
    w = OutboxWorker(
        db_path=tmp_path / "jobs.sqlite",  # same DB as the JobStore
        owui_client=client,
        poll_interval_s=0.05,
    )
    await w.start(spawn_loop=False)
    try:
        yield w
    finally:
        await w.stop()


@pytest_asyncio.fixture
async def coord():
    return _FakeCoord()


@pytest_asyncio.fixture
async def runner(coord: _FakeCoord, store: JobStore, outbox: OutboxWorker):
    r = JobRunner(
        coord=coord,
        store=store,
        outbox=outbox,
        public_base_url="https://tool.example.com",
    )
    yield r
    await r.shutdown()


async def _wait_task(runner: JobRunner, job_id: str) -> None:
    t = runner._tasks.get(job_id)
    if t is not None:
        await t


async def test_writeback_sequence_through_full_lifecycle(
    runner, coord, outbox, client, store
):
    record = _make_record()
    await store.create(record)

    # Phase 1: initial run emits a status + topic-list MessageEvent, then
    # pauses at the outline gate.
    async def first_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="generating initial queries"))
        await sink(MessageEvent(content="### Research Outline\n**1. Topic A**\n"))
        coord.state_manager.set_waiting(kwargs["conversation_id"])
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = first_run
    await runner.start_job(record, view_token="vtok", owui_user_token="utok")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    # Phase 2: feedback resumes, emits a citation, then returns a final report.
    async def second_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="researching"))
        await sink(
            CitationEvent(
                url="https://example.com/a",
                title="A",
                snippet="An interesting source.",
            )
        )
        return Report(
            content="# Final report\n\nDetailed findings here.",
            conversation_id=kwargs["conversation_id"],
        )

    coord.on_run = second_run
    # Rebind the target_message to a NEW message (the LLM's second tool call)
    await store.rebind_target_message(record.job_id, "msg-2")
    await runner.submit_feedback(record.job_id, "/k 1")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    types = [c["event_type"] for c in client.calls]

    # All event types are short-name (persisted) variants
    assert all(t in PERSISTED_EVENT_TYPES for t in types), types

    # The first emitted writeback is the bootstrap iframe
    assert types[0] == "embeds"
    # Message goes to first tool-call message
    first_msg_calls = [c for c in client.calls if c["message_id"] == "msg-1"]
    second_msg_calls = [c for c in client.calls if c["message_id"] == "msg-2"]
    assert first_msg_calls, "expected at least one writeback to msg-1"
    assert second_msg_calls, "expected writebacks to the rebound msg-2"

    # Topic-list replace appears (from MessageEvent)
    assert any(
        c["event_type"] == "replace"
        and "Research Outline" in c["data"].get("content", "")
        for c in first_msg_calls
    )

    # Status event landed
    assert any(c["event_type"] == "status" for c in first_msg_calls)

    # Second-message: bootstrap iframe + status + citation + final replace + clear
    second_types = [c["event_type"] for c in second_msg_calls]
    assert second_types[0] == "embeds"  # bootstrap on msg-2
    assert "status" in second_types
    assert "source" in second_types  # CitationEvent
    # Final replace + clear embeds happen at the end
    assert second_types[-2] == "replace"
    assert "Final report" in second_msg_calls[-2]["data"]["content"]
    assert second_types[-1] == "embeds"
    assert second_msg_calls[-1]["data"]["embeds"] == []  # iframe cleared


async def test_writeback_skipped_when_chat_id_none(
    runner, coord, outbox, client, store
):
    record = _make_record(chat_id=None, target_message_id=None)
    await store.create(record)

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="a status"))
        return Report(content="done", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    assert client.calls == []  # nothing posted — no writeback binding


async def test_writeback_skipped_when_chat_id_is_local(
    runner, coord, outbox, client, store
):
    record = _make_record(chat_id="local:abc123", target_message_id="m")
    await store.create(record)

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="a status"))
        return Report(content="done", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    assert client.calls == []  # OWUI local: chats don't persist the event


async def test_short_event_names_only(runner, coord, outbox, client, store):
    """Belt-and-braces: walk the outbox call sequence and confirm no
    long-name aliases (``chat:message`` etc.) ever made it through."""
    record = _make_record(job_id="short-names")
    await store.create(record)

    async def first_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="s1"))
        await sink(EmbedEvent(html="<div>engine embed</div>"))
        coord.state_manager.set_waiting(kwargs["conversation_id"])
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = first_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    for c in client.calls:
        assert ":" not in c["event_type"], (
            f"long-name alias leaked into writeback: {c['event_type']!r}"
        )
        assert c["event_type"] in PERSISTED_EVENT_TYPES

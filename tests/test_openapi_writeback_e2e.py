"""End-to-end writeback test: JobRunner + OutboxWorker → OWUI ``/event``.

Drives a job through the start-job → outline gate → submit-feedback →
completion sequence with a fake Coordinator and a fake OWUIClient on the
writeback path. Asserts the *full* sequence of persisted event types
posted to OWUI, in order:

  1. ``status`` events emitted during phase 1 (initial queries)
  2. ``replace`` (the topic list from the outline gate, as a MessageEvent)
  3. ``embeds`` (bootstrap iframe on the SECOND tool-call message — the
     preliminary phase no longer bootstraps an iframe)
  4. ``status`` / ``source`` during phase 2 (research → finalize)
  5. ``status`` (terminal "Research complete" pill with ``done=True``)
  6. ``replace`` (final report)

No trailing ``embeds: []`` clear: the iframe's last snapshot is
preserved in the message for user reference.

Also asserts:
  - Long-name aliases (``chat:message:embeds`` etc.) never appear; only
    the short names the ``/event`` endpoint persists.
  - The runner skips writeback when ``chat_id`` is None.
  - The runner skips writeback when ``chat_id`` starts with ``local:``.
"""
from __future__ import annotations

import asyncio
import contextlib
import pathlib
from typing import Any

import pytest_asyncio

from deep_research.adapter.client import PERSISTED_EVENT_TYPES
from deep_research.core.cancellation import CancellationToken
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
        with contextlib.suppress(asyncio.CancelledError):
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

    # Message goes to first tool-call message
    first_msg_calls = [c for c in client.calls if c["message_id"] == "msg-1"]
    second_msg_calls = [c for c in client.calls if c["message_id"] == "msg-2"]
    assert first_msg_calls, "expected at least one writeback to msg-1"
    assert second_msg_calls, "expected writebacks to the rebound msg-2"

    # No bootstrap iframe on msg-1: the preliminary phase doesn't post
    # one. The first writeback to msg-1 is the topic-list replace (or a
    # preceding status event).
    first_types = [c["event_type"] for c in first_msg_calls]
    assert "embeds" not in first_types, first_types

    # Topic-list replace appears (from MessageEvent)
    assert any(
        c["event_type"] == "replace"
        and "Research Outline" in c["data"].get("content", "")
        for c in first_msg_calls
    )

    # Status event landed
    assert any(c["event_type"] == "status" for c in first_msg_calls)

    # Second-message: bootstrap iframe + status + citation + terminal
    # status + final replace (NO trailing embeds:[] clear).
    second_types = [c["event_type"] for c in second_msg_calls]
    assert second_types[0] == "embeds"  # bootstrap on msg-2
    assert "status" in second_types
    assert "source" in second_types  # CitationEvent
    # Terminal sequence: status(done=True, "Research complete") then replace(report).
    assert second_types[-2] == "status"
    assert second_msg_calls[-2]["data"]["description"] == "Research complete"
    assert second_msg_calls[-2]["data"]["done"] is True
    assert second_types[-1] == "replace"
    assert "Final report" in second_msg_calls[-1]["data"]["content"]
    # The iframe is no longer cleared at the end.
    assert not any(
        c["event_type"] == "embeds" and c["data"].get("embeds") == []
        for c in client.calls
    ), "terminal writeback must NOT enqueue an embeds:[] clear"


async def test_citation_event_maps_to_source_row(
    runner, coord, outbox, client, store
):
    """A CitationEvent emitted through the runner sink lands as a `source`
    outbox row whose `data` matches CitationEvent.to_dict()["data"]. Pinned
    per-key (not whole-dict) so future additions to the event payload don't
    force unrelated test churn.
    """
    record = _make_record(job_id="cite-map")
    await store.create(record)

    citation = CitationEvent(
        url="https://example.com/a",
        title="Example Title",
        snippet="A short excerpt.",
    )

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(citation)
        return Report(content="ok", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    source_calls = [c for c in client.calls if c["event_type"] == "source"]
    assert len(source_calls) == 1
    data = source_calls[0]["data"]
    # Per-key assertions — survive future field additions to CitationEvent.
    assert data["type"] == "external"
    assert data["source"] == {"type": "external", "name": "Example Title"}
    assert data["document"] == ["A short excerpt."]
    assert data["metadata"] == [{"source": "https://example.com/a"}]


async def test_multiple_citations_with_same_url_deduplicate(
    runner, coord, outbox, client, store
):
    """Two CitationEvents with the same URL but different snippets produce
    exactly one outbox row reaching OWUI — the first emission's snippet
    survives (`INSERT OR IGNORE` semantics against the UNIQUE dedupe_key
    constraint). Pins URL-as-canonical-identity + first-wins; a future change
    to overwrite-semantics would break this test deliberately.
    """
    record = _make_record(job_id="cite-dedup")
    await store.create(record)

    first = CitationEvent(
        url="https://example.com/dup",
        title="First",
        snippet="First snippet.",
    )
    second = CitationEvent(
        url="https://example.com/dup",
        title="Second",
        snippet="Second snippet.",
    )

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(first)
        await sink(second)
        return Report(content="ok", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    source_calls = [c for c in client.calls if c["event_type"] == "source"]
    assert len(source_calls) == 1, (
        f"expected first-wins dedupe; got {len(source_calls)} source rows"
    )
    # First-wins: the surviving row carries the first emission's title/snippet.
    data = source_calls[0]["data"]
    assert data["source"]["name"] == "First"
    assert data["document"] == ["First snippet."]


async def test_runner_emits_source_not_citation(
    runner, coord, outbox, client, store
):
    """`PERSISTED_EVENT_TYPES` accepts both `source` and `citation` as
    historical OWUI aliases (see deep_research/adapter/client.py:22-24).
    Phase 2 settled on emitting `source` from
    `_event_to_outbox` (runner.py:488). Pin that choice so an accidental
    switch to `citation` would break the test.
    """
    record = _make_record(job_id="cite-name")
    await store.create(record)

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(CitationEvent(url="https://example.com/x", title="X"))
        return Report(content="ok", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    citation_event_types = [
        c["event_type"] for c in client.calls
        if c["event_type"] in ("source", "citation")
    ]
    assert citation_event_types == ["source"], citation_event_types


async def test_runner_does_not_filter_empty_url_citation(
    runner, coord, outbox, client, store
):
    """The runner's `_event_to_outbox` does NOT filter CitationEvents with
    empty `url`. The emit-side gatekeeper at
    `deep_research/orchestrator/phases/finalize.py:83-85` is the sole filter
    in normal operation (`if not url: continue`). This test pins the
    pass-through behaviour so a defensive filter is not added to the runner
    by mistake — keeping responsibility in one place avoids the
    "filtered twice / filtered nowhere" ambiguity.
    """
    record = _make_record(job_id="cite-empty-url")
    await store.create(record)

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(CitationEvent(url="", title="No URL", snippet="x"))
        return Report(content="ok", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    source_calls = [c for c in client.calls if c["event_type"] == "source"]
    assert len(source_calls) == 1, (
        "Runner must pass CitationEvent through even with empty url; "
        "finalize.py is the sole gatekeeper"
    )
    # The empty url survives end-to-end into the OWUI payload.
    assert source_calls[0]["data"]["metadata"] == [{"source": ""}]


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


async def test_running_phase_cancel_posts_status_and_replace(
    runner, coord, outbox, client, store
):
    """A mid-phase cancellation posts a 'Cancelled by user' status + a
    replace carrying the cancellation notice. No embeds:[] clear."""
    record = _make_record(job_id="cx-mid", chat_id="chat-c", target_message_id="msg-c")
    await store.create(record)

    released = asyncio.Event()

    async def on_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="working"))
        cancel = kwargs.get("cancellation_token")
        while not (isinstance(cancel, CancellationToken) and cancel.is_cancelled()):
            await asyncio.sleep(0.01)
        released.set()
        raise asyncio.CancelledError("cancelled by user")

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await asyncio.sleep(0.05)
    await runner.cancel(record.job_id, timeout=2.0)
    assert released.is_set()
    await _wait_task(runner, record.job_id)
    await outbox.drain_once(limit=64)

    refreshed = await store.get(record.job_id)
    assert refreshed.phase == JobPhase.CANCELLED

    types = [c["event_type"] for c in client.calls]
    # Terminal sequence: status(done=True, "Cancelled by user") then replace(notice).
    assert types[-2] == "status"
    assert client.calls[-2]["data"]["description"] == "Cancelled by user"
    assert client.calls[-2]["data"]["done"] is True
    assert types[-1] == "replace"
    assert "cancelled" in client.calls[-1]["data"]["content"].lower()
    # No embeds:[] clear.
    assert not any(
        c["event_type"] == "embeds" and c["data"].get("embeds") == []
        for c in client.calls
    )


async def test_gate_cancel_posts_status_and_replace(
    runner, coord, outbox, client, store
):
    """A cancellation issued while the engine is paused at the outline
    gate (task already done) still posts the terminal writeback. The
    CancelledError handler isn't reached; runner.cancel posts inline."""
    record = _make_record(job_id="cx-gate", chat_id="chat-g", target_message_id="msg-g")
    await store.create(record)

    async def on_run(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"])
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = on_run
    await runner.start_job(record, view_token="vt", owui_user_token="ut")
    await _wait_task(runner, record.job_id)
    # Engine is now paused at the outline gate; the task is done.
    assert runner._tasks[record.job_id].done()

    await runner.cancel(record.job_id, timeout=2.0)
    await outbox.drain_once(limit=64)

    refreshed = await store.get(record.job_id)
    assert refreshed.phase == JobPhase.CANCELLED
    assert refreshed.completed_at is not None

    types = [c["event_type"] for c in client.calls]
    assert types[-2] == "status"
    assert client.calls[-2]["data"]["description"] == "Cancelled by user"
    assert client.calls[-2]["data"]["done"] is True
    assert types[-1] == "replace"
    assert "cancelled" in client.calls[-1]["data"]["content"].lower()
    assert not any(
        c["event_type"] == "embeds" and c["data"].get("embeds") == []
        for c in client.calls
    )


async def test_cancel_writeback_skipped_for_local_chat_and_none(
    runner, coord, outbox, client, store
):
    """Cancellation paths must respect the same writeback skip rules as
    in-flight events: ``chat_id`` None or ``local:`` → no writeback."""
    for case_id, chat_id, target_message_id in (
        ("cx-none", None, None),
        ("cx-local", "local:abc", "msg-x"),
    ):
        client.calls.clear()
        record = _make_record(
            job_id=case_id, chat_id=chat_id, target_message_id=target_message_id
        )
        await store.create(record)

        async def on_run(kwargs):
            cancel = kwargs.get("cancellation_token")
            while not (isinstance(cancel, CancellationToken) and cancel.is_cancelled()):
                await asyncio.sleep(0.01)
            raise asyncio.CancelledError("cancelled")

        coord.on_run = on_run
        await runner.start_job(record, view_token="vt", owui_user_token="ut")
        await asyncio.sleep(0.05)
        await runner.cancel(record.job_id, timeout=2.0)
        await _wait_task(runner, record.job_id)
        await outbox.drain_once(limit=64)

        assert client.calls == [], (case_id, client.calls)


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

"""Tests for the JobRunner — spawn, suspend/resume, cancel, shutdown."""
from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest
import pytest_asyncio

from deep_research.core.cancellation import CancellationToken
from deep_research.core.types import Report
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
)
from deep_research.entrypoints.openapi_tool.runner import JobRunner


class _FakeStateManager:
    """Imitates ResearchStateManager's surface used by the runner."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    def get_state(self, cid: str) -> dict[str, Any]:
        return self.states.setdefault(cid, {})

    def set_waiting(self, cid: str, waiting: bool, outline=None) -> None:
        st = self.get_state(cid)
        st["waiting_for_outline_feedback"] = waiting
        if waiting:
            st["outline_feedback_data"] = {"outline_items": outline or [{"topic": "t"}]}


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


def _make_record(job_id: str = "j1", phase: JobPhase = JobPhase.QUEUED) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        user_id="u",
        user_name="U",
        conversation_id=f"conv_{job_id}",
        chat_id=None,
        target_message_id=None,
        phase=phase,
        prompt="prompt",
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
async def coord():
    return _FakeCoord()


@pytest_asyncio.fixture
async def runner(coord: _FakeCoord, store: JobStore):
    r = JobRunner(coord=coord, store=store, outbox=None, public_base_url="http://t/")
    yield r
    await r.shutdown()


async def test_start_job_creates_task_and_invokes_coordinator(runner, coord, store):
    record = _make_record("start-1")
    await store.create(record)

    async def _on_run(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"], True, outline=[{"topic": "x"}])
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")

    task = runner._tasks[record.job_id]
    await task

    refreshed = await store.get(record.job_id)
    assert refreshed.phase == JobPhase.AWAITING_OUTLINE_FEEDBACK
    assert coord.run_calls and coord.run_calls[0]["prompt"] == "prompt"
    # Outline JSON populated from the engine state.
    assert refreshed.outline_json is not None


async def test_start_job_marks_completed_when_engine_finishes(runner, coord, store):
    record = _make_record("start-2")
    await store.create(record)

    async def _on_run(kwargs):
        # No `waiting_for_outline_feedback` set → full completion path
        return Report(content="final report", title="Done", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await runner._tasks[record.job_id]

    refreshed = await store.get(record.job_id)
    assert refreshed.phase == JobPhase.COMPLETED
    assert refreshed.report_markdown == "final report"
    assert refreshed.completed_at is not None


async def test_start_job_failure_marks_failed(runner, coord, store):
    record = _make_record("start-3")
    await store.create(record)

    async def _on_run(kwargs):
        raise RuntimeError("kaboom")

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await runner._tasks[record.job_id]

    refreshed = await store.get(record.job_id)
    assert refreshed.phase == JobPhase.FAILED
    assert "kaboom" in (refreshed.error_text or "")


async def test_submit_feedback_triggers_second_run(runner, coord, store):
    record = _make_record("fb-1")
    await store.create(record)

    async def _first(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _first
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await runner._tasks[record.job_id]

    # Engine is paused at outline-feedback gate; submit a selection.
    second_done = asyncio.Event()

    async def _second(kwargs):
        second_done.set()
        return Report(content="post-feedback report", conversation_id=kwargs["conversation_id"])

    coord.on_run = _second
    await runner.submit_feedback("fb-1", "/k 1,3")
    await asyncio.wait_for(second_done.wait(), timeout=2.0)
    await runner._tasks["fb-1"]

    refreshed = await store.get("fb-1")
    assert refreshed.phase == JobPhase.COMPLETED
    assert refreshed.report_markdown == "post-feedback report"
    # Second run got the user's selection as the prompt.
    assert coord.run_calls[1]["prompt"] == "/k 1,3"


async def test_submit_feedback_rejects_if_not_awaiting(runner, coord, store):
    record = _make_record("fb-no")
    await store.create(record)

    async def _on_run(kwargs):
        # Engine returns without setting the waiting flag → completion
        return Report(content="done", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await runner._tasks[record.job_id]

    with pytest.raises(RuntimeError):
        await runner.submit_feedback("fb-no", "/continue")


async def test_cancel_marks_cancelled(runner, coord, store):
    record = _make_record("cx-1")
    await store.create(record)

    released = asyncio.Event()

    async def _on_run(kwargs):
        cancel = kwargs.get("cancellation_token")
        # Wait until cancelled, then bail like the engine would
        while not (isinstance(cancel, CancellationToken) and cancel.is_cancelled()):
            await asyncio.sleep(0.01)
        released.set()
        raise asyncio.CancelledError("cancelled by user")

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await asyncio.sleep(0.05)  # let the task start
    await runner.cancel("cx-1", timeout=2.0)
    assert released.is_set()

    refreshed = await store.get("cx-1")
    assert refreshed.phase == JobPhase.CANCELLED
    assert refreshed.completed_at is not None


async def test_cancel_at_gate_marks_cancelled(runner, coord, store):
    """Cancellation while the engine is paused at the outline gate (the
    task is already .done()) must still land the job in CANCELLED. The
    CancelledError handler doesn't fire here; runner.cancel updates the
    phase inline."""
    record = _make_record("cx-gate")
    await store.create(record)

    async def _on_run(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await runner._tasks[record.job_id]
    assert runner._tasks[record.job_id].done()

    await runner.cancel("cx-gate", timeout=2.0)

    refreshed = await store.get("cx-gate")
    assert refreshed.phase == JobPhase.CANCELLED
    assert refreshed.completed_at is not None


async def test_shutdown_cancels_all_active_jobs(coord, store):
    runner = JobRunner(coord=coord, store=store, outbox=None, public_base_url="")
    rec_a = _make_record("sd-a")
    rec_b = _make_record("sd-b")
    await store.create(rec_a)
    await store.create(rec_b)

    async def _on_run(kwargs):
        cancel = kwargs.get("cancellation_token")
        while not (isinstance(cancel, CancellationToken) and cancel.is_cancelled()):
            await asyncio.sleep(0.01)
        raise asyncio.CancelledError("shutdown")

    coord.on_run = _on_run
    await runner.start_job(rec_a, view_token="va", owui_user_token="tok")
    await runner.start_job(rec_b, view_token="vb", owui_user_token="tok")
    await asyncio.sleep(0.05)
    await runner.shutdown(timeout=2.0)

    for jid in ("sd-a", "sd-b"):
        ref = await store.get(jid)
        assert ref.phase == JobPhase.CANCELLED


async def test_get_snapshot_carries_status_event(runner, coord, store):
    from deep_research.progress.events import StatusEvent

    record = _make_record("snap-1")
    await store.create(record)

    arrived = asyncio.Event()

    async def _on_run(kwargs):
        sink = kwargs["sink"]
        await sink(StatusEvent(description="phase 1", level="info"))
        arrived.set()
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await asyncio.wait_for(arrived.wait(), timeout=2.0)
    await runner._tasks["snap-1"]

    snap = runner.get_snapshot("snap-1")
    assert snap.get("latest_status") == "phase 1"

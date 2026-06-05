"""Tests for the JobRunner — spawn, suspend/resume, cancel, shutdown."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _restore_deep_research_log_propagation():
    """`deep_research/config/logging.py::configure_logging` sets
    `logging.getLogger("deep_research").propagate = False`. When
    `tests/test_config_logging.py` runs earlier in the suite that state
    leaks forward, and caplog (which listens on the root logger) no
    longer receives records from `deep_research.*` loggers. Restore
    propagation around each test so caplog-based assertions stay
    reliable regardless of suite ordering."""
    dr_logger = logging.getLogger("deep_research")
    saved = dr_logger.propagate
    dr_logger.propagate = True
    try:
        yield
    finally:
        dr_logger.propagate = saved

from deep_research.core.cancellation import CancellationToken  # noqa: E402
from deep_research.core.types import Report  # noqa: E402
from deep_research.entrypoints.openapi_tool.jobs import (  # noqa: E402
    JobPhase,
    JobRecord,
    JobStore,
)
from deep_research.entrypoints.openapi_tool.runner import (  # noqa: E402
    ActiveJobExistsError,
    FeedbackCancelledError,
    JobRunner,
)

from ._runner_helpers import FakeCoord as _FakeCoord  # noqa: E402,F401  (alias kept for fixture)


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

    entered = asyncio.Event()  # signalled when engine reaches wait-loop
    released = asyncio.Event()  # signalled when engine bails on cancel

    async def _on_run(kwargs):
        cancel = kwargs.get("cancellation_token")
        entered.set()
        while not (isinstance(cancel, CancellationToken) and cancel.is_cancelled()):
            await asyncio.sleep(0.01)
        released.set()
        raise asyncio.CancelledError("cancelled by user")

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    # Wait for the engine to actually enter the wait-loop — replaces
    # the prior `asyncio.sleep(0.05)` timing primitive.
    await asyncio.wait_for(entered.wait(), timeout=2.0)
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


class _RaisingOutbox:
    """Outbox stub whose enqueue always raises — exercises the sink's
    `_event_to_outbox` defensive-logging branch."""

    def __init__(self) -> None:
        self.calls = 0

    async def enqueue(self, **kwargs):
        self.calls += 1
        raise RuntimeError("boom")


def _make_bound_record(job_id: str) -> JobRecord:
    """A record with chat_id + target_message_id populated so the runner
    does not short-circuit at `_writeback_target`."""
    return JobRecord(
        job_id=job_id,
        user_id="u",
        user_name="U",
        conversation_id=f"conv_{job_id}",
        chat_id="chat-1",
        target_message_id="msg-1",
        phase=JobPhase.QUEUED,
        prompt="prompt",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )


async def test_make_sink_logs_outbox_enqueue_failure(coord, store, caplog):
    """A failing outbox.enqueue must not crash the engine, but must
    leave a logger.exception line tagged with the job id and event
    class."""
    import logging

    from deep_research.progress.events import StatusEvent

    outbox = _RaisingOutbox()
    runner_with_outbox = JobRunner(
        coord=coord, store=store, outbox=outbox, public_base_url=""
    )
    record = _make_bound_record("sink-outbox")

    arrived = asyncio.Event()

    async def _on_run(kwargs):
        await kwargs["sink"](StatusEvent(description="hello", level="info"))
        arrived.set()
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run

    with caplog.at_level(logging.ERROR, logger="deep_research.entrypoints.openapi.runner"):
        await runner_with_outbox.start_job(
            record, view_token="vt", owui_user_token="tok"
        )
        await asyncio.wait_for(arrived.wait(), timeout=2.0)
        await runner_with_outbox._tasks["sink-outbox"]

    # The outbox actually got called (so the suppress branch fired).
    assert outbox.calls >= 1
    # And the failure was logged with the job id and event class.
    matching = [
        r for r in caplog.records
        if r.name == "deep_research.entrypoints.openapi.runner"
        and "writeback enqueue failed" in r.getMessage()
        and "sink-outbox" in r.getMessage()
        and "StatusEvent" in r.getMessage()
    ]
    assert matching, [r.getMessage() for r in caplog.records]
    # Traceback is attached (logger.exception captures sys.exc_info()).
    assert matching[0].exc_info is not None

    await runner_with_outbox.shutdown()


async def test_embed_event_bumps_revision_and_merges_snapshot(runner, coord, store):
    """An engine-emitted EmbedEvent must bump record.revision (so the
    self-polling live-view iframe reloads and the outbox dedupe key
    stays unique per refresh) and must merge its source snapshot dict
    into the runner's in-memory snapshot cache (so the writeback-
    disabled path can re-render the iframe with topic/cycle data)."""
    from deep_research.progress.events import EmbedEvent

    record = _make_record("embed-rev")

    arrived = asyncio.Event()
    payload = {
        "cycle": 2,
        "max_cycles": 5,
        "completed_topics": ["a", "b"],
        "remaining_topics": ["c"],
        "results_tokens": 1234,
        "total_tokens": 1500,
    }

    async def _on_run(kwargs):
        sink = kwargs["sink"]
        await sink(EmbedEvent(html="<div>cycle 1</div>", snapshot=payload))
        await sink(EmbedEvent(html="<div>cycle 2</div>", snapshot={**payload, "cycle": 3}))
        arrived.set()
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="tok")
    await asyncio.wait_for(arrived.wait(), timeout=2.0)
    await runner._tasks["embed-rev"]

    refreshed = await store.get("embed-rev")
    assert refreshed is not None
    # Two EmbedEvents → revision incremented twice (plus once for the
    # AWAITING_OUTLINE_FEEDBACK phase transition the fake coord triggers
    # via set_waiting → at least 2). We assert the floor, not exact, so
    # this stays robust to unrelated future phase bumps.
    assert refreshed.revision >= 2

    snap = runner.get_snapshot("embed-rev")
    assert snap.get("cycle") == 3
    assert snap.get("completed_topics") == ["a", "b"]
    assert snap.get("remaining_topics") == ["c"]
    assert snap.get("results_tokens") == 1234
    assert snap.get("has_embed") is True


async def test_make_sink_logs_snapshot_update_failure(runner, coord, store, caplog):
    """Monkey-patching `_update_snapshot` to raise must not crash the
    engine; the failure must leave a logger.exception line."""
    import logging

    from deep_research.progress.events import StatusEvent

    record = _make_record("sink-snap")

    def _boom(job_id, event):  # noqa: ARG001
        raise RuntimeError("snap-boom")

    runner._update_snapshot = _boom  # type: ignore[method-assign]

    arrived = asyncio.Event()

    async def _on_run(kwargs):
        await kwargs["sink"](StatusEvent(description="hi", level="info"))
        arrived.set()
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run

    with caplog.at_level(logging.ERROR, logger="deep_research.entrypoints.openapi.runner"):
        await runner.start_job(record, view_token="vt", owui_user_token="tok")
        await asyncio.wait_for(arrived.wait(), timeout=2.0)
        await runner._tasks["sink-snap"]

    matching = [
        r for r in caplog.records
        if r.name == "deep_research.entrypoints.openapi.runner"
        and "snapshot update failed" in r.getMessage()
        and "sink-snap" in r.getMessage()
        and "StatusEvent" in r.getMessage()
    ]
    assert matching, [r.getMessage() for r in caplog.records]
    assert matching[0].exc_info is not None


async def test_failed_phase_store_update_error_still_marks_snapshot(
    runner, coord, store, caplog
):
    """If the FAILED-phase store.update raises (sqlite IO error), the
    runner must (a) not propagate the sqlite error out of the task,
    (b) still mark the in-memory snapshot FAILED, and (c) log both
    the original engine failure and the store-update failure."""
    import logging

    record = _make_record("fail-store")

    original_update = store.update

    async def _failing_update(job_id, **fields):
        if fields.get("phase") == JobPhase.FAILED:
            raise RuntimeError("sqlite-boom")
        return await original_update(job_id, **fields)

    store.update = _failing_update  # type: ignore[method-assign]

    async def _on_run(kwargs):  # noqa: ARG001
        raise RuntimeError("kaboom")

    coord.on_run = _on_run

    with caplog.at_level(logging.ERROR, logger="deep_research.entrypoints.openapi.runner"):
        await runner.start_job(record, view_token="vt", owui_user_token="tok")
        # Task should complete normally — sqlite error is swallowed (logged).
        await runner._tasks["fail-store"]

    # Snapshot reflects FAILED even though the DB write itself raised.
    snap = runner.get_snapshot("fail-store")
    assert snap.get("phase") == JobPhase.FAILED.value

    messages = [r.getMessage() for r in caplog.records if r.name ==
                "deep_research.entrypoints.openapi.runner"]
    # Outer logger.exception captured the original engine failure.
    assert any("initial run failed" in m and "fail-store" in m for m in messages), messages
    # Inner logger.exception captured the store-update failure.
    assert any(
        "FAILED-phase store update raised" in m and "fail-store" in m
        for m in messages
    ), messages


def test_deserialise_history_logs_on_bad_blob(caplog):
    """`_deserialise_history` returns [] on a corrupted blob, logs a
    warning with PII-safe metadata only (length + exception class +
    str(exc)), and never includes any fragment of the input string."""
    import logging

    bad = "not json{"

    with caplog.at_level(logging.WARNING, logger="deep_research.entrypoints.openapi.runner"):
        result = JobRunner._deserialise_history(bad)

    assert result == []

    matching = [
        r for r in caplog.records
        if r.name == "deep_research.entrypoints.openapi.runner"
        and "Failed to decode history_json" in r.getMessage()
    ]
    assert matching, [r.getMessage() for r in caplog.records]

    formatted = matching[0].getMessage()
    assert f"length={len(bad)}" in formatted
    assert "error_class=JSONDecodeError" in formatted

    # PII-safety invariant: no fragment of the raw input appears in the log.
    # Use a 4-char prefix as a conservative content fingerprint.
    assert bad[:4] not in formatted, formatted
    assert bad not in formatted, formatted


# --------------------------------------------------------------- per-job locking
#
# Tests below pin the per-job-lock invariant introduced for TODO Group 2:
# every lifecycle transition (start / feedback / cancel) for a given
# job_id is serialised by `JobRunner._job_locks[job_id]`. The sqlite
# UNIQUE partial index on (chat_id) WHERE phase NOT IN terminal is the
# defence-in-depth backstop.


def _make_record_for_chat(job_id: str, chat_id: str) -> JobRecord:
    """A record with a populated chat_id so the UNIQUE partial index
    can fire on duplicates."""
    return JobRecord(
        job_id=job_id,
        user_id="u",
        user_name="U",
        conversation_id=f"conv_{job_id}",
        chat_id=chat_id,
        target_message_id=None,
        phase=JobPhase.QUEUED,
        prompt="prompt",
        history_json="[]",
        revision=0,
        view_token_hash="0" * 64,
    )


async def test_concurrent_start_same_chat_one_wins(runner, coord, store):
    """Two start_job calls for the same chat_id, scheduled in parallel.
    Exactly one succeeds; the other raises ActiveJobExistsError. The
    sqlite UNIQUE partial index is the source of truth."""
    chat_id = "chat-conc"
    rec_a = _make_record_for_chat("conc-a", chat_id)
    rec_b = _make_record_for_chat("conc-b", chat_id)

    async def _slow_run(kwargs):
        # Hold the engine open long enough that both start_job calls
        # observe each other's effects (or don't).
        cancel = kwargs.get("cancellation_token")
        for _ in range(50):
            if isinstance(cancel, CancellationToken) and cancel.is_cancelled():
                break
            await asyncio.sleep(0.01)
        return Report(content="x", conversation_id=kwargs["conversation_id"])

    coord.on_run = _slow_run

    results = await asyncio.gather(
        runner.start_job(rec_a, view_token="va", owui_user_token="t"),
        runner.start_job(rec_b, view_token="vb", owui_user_token="t"),
        return_exceptions=True,
    )
    # Exactly one ActiveJobExistsError; the other returned None.
    errs = [r for r in results if isinstance(r, ActiveJobExistsError)]
    successes = [r for r in results if r is None]
    assert len(errs) == 1, results
    assert len(successes) == 1, results

    # Exactly one row exists in the store with this chat_id.
    only = await store.find_active_by_chat(chat_id)
    assert only is not None
    assert only.job_id in ("conc-a", "conc-b")


async def test_concurrent_submit_feedback_one_wins(runner, coord, store):
    """Two submit_feedback calls in parallel for a paused job. Exactly
    one transitions the engine to RESEARCHING; the other gets a
    RuntimeError because the lock-protected phase check sees the
    transition."""
    record = _make_record("fb-conc")

    async def _first(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _first
    await runner.start_job(record, view_token="vt", owui_user_token="t")
    await runner._tasks[record.job_id]

    second_calls = 0

    async def _second(kwargs):
        nonlocal second_calls
        second_calls += 1
        # Hold long enough that the second submit_feedback can run.
        await asyncio.sleep(0.05)
        return Report(content="done", conversation_id=kwargs["conversation_id"])

    coord.on_run = _second
    results = await asyncio.gather(
        runner.submit_feedback("fb-conc", "/k 1"),
        runner.submit_feedback("fb-conc", "/k 2"),
        return_exceptions=True,
    )
    # Drain whichever task got spawned.
    if "fb-conc" in runner._tasks:
        with contextlib.suppress(Exception):
            await runner._tasks["fb-conc"]

    errs = [r for r in results if isinstance(r, RuntimeError)]
    successes = [r for r in results if r is None]
    assert len(errs) == 1, results
    assert len(successes) == 1, results
    # Only one second-phase Coordinator.run.
    assert second_calls == 1


async def test_cancel_during_natural_completion_lands_cancelled(runner, coord, store):
    """The `_cancel_requested` early-check in the success branch makes
    a cancel arriving during natural completion still land CANCELLED."""
    record = _make_record("cx-natural")

    entered_run = asyncio.Event()

    async def _on_run(kwargs):
        entered_run.set()
        # Long enough that cancel can land before this returns.
        await asyncio.sleep(0.1)
        # No `waiting_for_outline_feedback` → success branch will fire.
        return Report(content="report body", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="t")
    await asyncio.wait_for(entered_run.wait(), timeout=1.0)
    # While Coordinator.run is still mid-sleep, request cancel.
    await runner.cancel("cx-natural", timeout=2.0)
    # Drain the task — it should have returned early on the
    # `_cancel_requested` check.
    if "cx-natural" in runner._tasks:
        with contextlib.suppress(Exception):
            await runner._tasks["cx-natural"]

    refreshed = await store.get("cx-natural")
    assert refreshed.phase == JobPhase.CANCELLED
    # The success-path report was NOT written to the DB.
    assert refreshed.report_markdown is None or refreshed.report_markdown == ""
    assert refreshed.completed_at is not None


async def test_submit_feedback_then_cancel_serialise(runner, coord, store):
    """submit_feedback and cancel running in parallel: the per-job lock
    serialises them. Either feedback wins (and the subsequent cancel
    lands CANCELLED via Phase C) OR cancel wins (and feedback's Phase
    C raises FeedbackCancelledError). Either way, terminal state =
    CANCELLED."""
    record = _make_record("fb-vs-cx")

    async def _first(kwargs):
        coord.state_manager.set_waiting(kwargs["conversation_id"], True)
        return Report(content="", conversation_id=kwargs["conversation_id"])

    coord.on_run = _first
    await runner.start_job(record, view_token="vt", owui_user_token="t")
    await runner._tasks[record.job_id]

    async def _second(kwargs):
        cancel = kwargs.get("cancellation_token")
        for _ in range(50):
            if isinstance(cancel, CancellationToken) and cancel.is_cancelled():
                break
            await asyncio.sleep(0.01)
        raise asyncio.CancelledError("test cancelled")

    coord.on_run = _second
    results = await asyncio.gather(
        runner.submit_feedback("fb-vs-cx", "/k 1"),
        runner.cancel("fb-vs-cx", timeout=2.0),
        return_exceptions=True,
    )
    if "fb-vs-cx" in runner._tasks:
        with contextlib.suppress(Exception):
            await runner._tasks["fb-vs-cx"]

    refreshed = await store.get("fb-vs-cx")
    assert refreshed.phase == JobPhase.CANCELLED, (refreshed.phase, results)
    # The feedback caller either succeeded (None) or raised
    # FeedbackCancelledError. Anything else is a regression.
    fb_result = results[0]
    assert fb_result is None or isinstance(fb_result, FeedbackCancelledError), fb_result


async def test_job_state_gc_after_terminal_phase(runner, coord, store):
    """Per-job state dicts are GC'd after the task ends in a terminal
    phase. `_job_locks` is intentionally NOT GC'd."""
    record = _make_record("gc-1")

    async def _on_run(kwargs):
        return Report(content="done", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="t")
    await runner._tasks[record.job_id]
    # The done-callback schedules cleanup via call_soon; give it two
    # event-loop iterations to fire.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert "gc-1" not in runner._tasks
    assert "gc-1" not in runner._snapshots
    assert "gc-1" not in runner._cancellation_tokens
    assert "gc-1" not in runner._owui_tokens
    assert "gc-1" not in runner._view_tokens
    assert "gc-1" not in runner._status_dedupe_counter
    assert "gc-1" not in runner._cancel_requested
    # Lock is intentionally NOT GC'd (see _lock_for docstring).
    assert "gc-1" in runner._job_locks


async def test_job_state_gc_does_not_fire_for_non_terminal_task_end(runner, coord, store):
    """If a task ends without setting a terminal snapshot phase
    (legitimate cancel-vs-success race outcome where the early
    `_cancel_requested` check returned), the GC must NOT drop state.
    cancel()'s own `call_soon` is the trigger."""
    record = _make_record("gc-deferred")

    # Manually mark the job_id as cancel-requested before the engine
    # finishes so the success branch returns early.
    entered = asyncio.Event()

    async def _on_run(kwargs):
        entered.set()
        await asyncio.sleep(0.05)
        return Report(content="x", conversation_id=kwargs["conversation_id"])

    coord.on_run = _on_run
    await runner.start_job(record, view_token="vt", owui_user_token="t")
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    runner._cancel_requested.add("gc-deferred")
    await runner._tasks[record.job_id]
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # The task ended in non-terminal snapshot phase
    # (BOOTSTRAPPING — the engine's last write). The GC saw the
    # non-terminal phase and did NOT drop state.
    snap = runner._snapshots.get("gc-deferred", {})
    assert snap.get("phase") == JobPhase.BOOTSTRAPPING.value
    assert "gc-deferred" in runner._tasks  # still present

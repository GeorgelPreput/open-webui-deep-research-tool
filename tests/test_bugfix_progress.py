"""Regression tests for progress-layer bug fixes.

Covers:
  BUG 22 - queued MessageEvents are flushed before the coalesced StatusEvent
           (status no longer jumps ahead of earlier report content).
  BUG 10 - the progress snapshot reads the real valve path
           (ctx.valves.cycles.max_cycles), not a non-existent flat MAX_CYCLES.
"""
import asyncio

import pytest

from deep_research.progress.events import EventBus, MessageEvent, StatusEvent
from deep_research.progress.snapshot import build_progress_snapshot


# --- BUG 22: flush order ----------------------------------------------------

@pytest.mark.asyncio
async def test_messages_flushed_before_status():
    sink = []

    async def collect(event):
        sink.append(event)

    bus = EventBus(collect, flush_interval_ms=50)
    await bus.start()
    try:
        await bus.emit(MessageEvent(content="report body part"))
        await bus.emit(StatusEvent(description="working..."))
        await asyncio.sleep(0.25)
    finally:
        await bus.stop()

    msg_idx = next(i for i, e in enumerate(sink) if isinstance(e, MessageEvent))
    status_idx = next(i for i, e in enumerate(sink) if isinstance(e, StatusEvent))
    assert msg_idx < status_idx


@pytest.mark.asyncio
async def test_final_drain_emits_message_before_status():
    # No sleep: everything is drained at stop(); the final-drain order must also
    # put the queued message ahead of the trailing status.
    sink = []

    async def collect(event):
        sink.append(event)

    bus = EventBus(collect, flush_interval_ms=10_000)
    await bus.start()
    await bus.emit(MessageEvent(content="final body"))
    await bus.emit(StatusEvent(description="done", done=True))
    await bus.stop()

    msg_idx = next(i for i, e in enumerate(sink) if isinstance(e, MessageEvent))
    status_idx = next(i for i, e in enumerate(sink) if isinstance(e, StatusEvent))
    assert msg_idx < status_idx


# --- BUG 10: snapshot max_cycles --------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_reads_nested_max_cycles(run_context):
    run_context.valves.cycles.max_cycles = 7
    snap = build_progress_snapshot(run_context)
    assert snap["max_cycles"] == 7


@pytest.mark.asyncio
async def test_snapshot_default_max_cycles_nonzero(run_context):
    snap = build_progress_snapshot(run_context)
    assert snap["max_cycles"] == run_context.valves.cycles.max_cycles
    assert snap["max_cycles"] > 0

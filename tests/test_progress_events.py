import asyncio

import pytest

from deep_research.progress.events import (
    EmbedEvent,
    EventBus,
    MessageEvent,
    StatusEvent,
)


@pytest.mark.asyncio
async def test_status_events_coalesce_to_latest():
    sink_calls = []

    async def sink(event):
        sink_calls.append(event)

    bus = EventBus(sink, flush_interval_ms=50)
    await bus.start()
    try:
        for i in range(10):
            await bus.emit(StatusEvent(description=f"step {i}"))
        # Give the flusher more than one interval to drain
        await asyncio.sleep(0.25)
    finally:
        await bus.stop()

    statuses = [e for e in sink_calls if isinstance(e, StatusEvent)]
    assert statuses, "expected at least one status flush"
    # Last status received should be the most recent
    assert statuses[-1].description == "step 9"
    # 10 emits should NOT produce 10 sink calls (coalescing)
    assert len(statuses) < 10


@pytest.mark.asyncio
async def test_message_events_all_delivered():
    sink_calls = []

    async def sink(event):
        sink_calls.append(event)

    bus = EventBus(sink, flush_interval_ms=50)
    await bus.start()
    try:
        await bus.emit(MessageEvent(content="part 1"))
        await bus.emit(MessageEvent(content="part 2"))
        await bus.emit(MessageEvent(content="part 3"))
    finally:
        await bus.stop()

    messages = [e for e in sink_calls if isinstance(e, MessageEvent)]
    assert [m.content for m in messages] == ["part 1", "part 2", "part 3"]


@pytest.mark.asyncio
async def test_embed_replaced_by_latest_within_window():
    sink_calls = []

    async def sink(event):
        sink_calls.append(event)

    bus = EventBus(sink, flush_interval_ms=300)
    await bus.start()
    try:
        await bus.emit(EmbedEvent(html="<div>v1</div>"))
        await bus.emit(EmbedEvent(html="<div>v2</div>"))
        await bus.emit(EmbedEvent(html="<div>v3</div>"))
        await asyncio.sleep(0.4)
    finally:
        await bus.stop()

    embeds = [e for e in sink_calls if isinstance(e, EmbedEvent)]
    # Latest-wins coalescing keeps at most v2 + v3 (current-latest semantics)
    assert any(e.html == "<div>v3</div>" for e in embeds)


@pytest.mark.asyncio
async def test_critical_bypasses_buffer():
    sink_calls = []

    async def sink(event):
        sink_calls.append(event)

    bus = EventBus(sink, flush_interval_ms=10_000)  # long flush interval
    await bus.start()
    try:
        await bus.emit_critical(StatusEvent(description="urgent!", level="error", done=True))
        # No sleep — critical must have flushed synchronously
        assert any(
            isinstance(e, StatusEvent) and e.description == "urgent!"
            for e in sink_calls
        )
    finally:
        await bus.stop()


@pytest.mark.asyncio
async def test_event_to_dict_emitter_shapes():
    assert StatusEvent(description="x", done=True).to_dict() == {
        "type": "status",
        "data": {"status": "complete", "description": "x", "done": True},
    }
    assert MessageEvent(content="hi").to_dict() == {
        "type": "message",
        "data": {"content": "hi"},
    }
    embed = EmbedEvent(html="<div/>", title="t").to_dict()
    assert embed["type"] == "embeds"
    assert embed["data"]["embeds"][0]["html"] == "<div/>"

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


def _snapshot() -> dict:
    return {
        "query": "What is X?",
        "cycle": 1,
        "max_cycles": 5,
        "revision": 3,
        "updated_at": "2026-01-01T00:00:00",
        "all_topics": [],
        "completed_topics": [],
        "partial_topics": [],
        "new_topics": [],
        "irrelevant_topics": [],
        "remaining_topics": [],
        "results_tokens": 0,
        "synthesis_tokens": 0,
        "total_tokens": 0,
    }


def test_render_progress_embed_push_only_omits_poll_script():
    from deep_research.progress.embed import render_progress_embed_html

    html = render_progress_embed_html(_snapshot())
    assert "dr-bootstrap" not in html
    assert "Content-Security-Policy" in html
    assert "reportHeight" in html  # height-reporting script kept


def test_render_progress_embed_with_poll_url_emits_polling_script():
    from deep_research.progress.embed import render_progress_embed_html

    html = render_progress_embed_html(
        _snapshot(),
        poll_url="http://example.test/live_view/job-1/status",
        view_token="vt-xyz",
    )
    assert 'id="dr-bootstrap"' in html
    assert "live_view/job-1/status" in html
    assert "vt-xyz" in html
    # Bootstrap JSON includes the current revision as since_version baseline;
    # JSON keys live in an HTML data attribute so quotes are entity-escaped.
    assert "&quot;since_version&quot;: 3" in html
    # nonce attribute is applied to inline scripts
    assert 'nonce="' in html


def test_render_progress_embed_requires_view_token_when_polling():
    from deep_research.progress.embed import render_progress_embed_html

    with pytest.raises(ValueError):
        render_progress_embed_html(_snapshot(), poll_url="http://x/status")

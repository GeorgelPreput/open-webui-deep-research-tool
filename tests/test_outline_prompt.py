"""Pin the outline-prompt rendering and the renderer↔parser contract.

The prompt and the slash-command parser at
``deep_research.research.outline_feedback.process_outline_feedback_continuation``
share a flat-items contract. If the rendered numbering ever drifts from
the parser's indexing, the user's ``/k 1,3,5`` reply silently picks the
wrong items. These tests pin both sides.
"""
import asyncio

import pytest

from deep_research.research.outline_feedback import (
    process_outline_feedback_continuation,
    render_outline_prompt,
)

SAMPLE_OUTLINE = [
    {"topic": "Architecture", "subtopics": ["State space", "Selectivity"]},
    {"topic": "Performance", "subtopics": ["Latency", "Throughput"]},
]


def test_prompt_advertises_all_slash_commands():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    for fragment in ("/k", "/keep", "/r", "/remove", "/continue", "/c"):
        assert fragment in text, fragment


def test_prompt_advertises_range_syntax():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    assert "5-7" in text


def test_prompt_advertises_natural_language_fallback():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    assert "natural language" in text.lower()


def test_prompt_contains_pause_marker():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    assert "pause" in text.lower()
    assert "await your response" in text.lower()


def test_flat_items_topic_then_subtopic_order():
    _, flat_items = render_outline_prompt(SAMPLE_OUTLINE)
    assert flat_items == [
        "Architecture",
        "State space",
        "Selectivity",
        "Performance",
        "Latency",
        "Throughput",
    ]


def test_rendered_numbering_matches_flat_items_indexing():
    text, flat_items = render_outline_prompt(SAMPLE_OUTLINE)
    for idx, item in enumerate(flat_items, start=1):
        assert f"{idx}. {item}" in text, (idx, item)


def test_topic_rendered_in_bold():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    assert "**1. Architecture**" in text
    assert "**4. Performance**" in text


def test_subtopic_rendered_indented_no_bold():
    text, _ = render_outline_prompt(SAMPLE_OUTLINE)
    assert "   2. State space" in text
    assert "   3. Selectivity" in text


def test_empty_outline_renders_help_block_only():
    text, flat_items = render_outline_prompt([])
    assert flat_items == []
    assert "/continue" in text
    assert "Research Outline" in text


def test_outline_without_subtopics_handled():
    text, flat_items = render_outline_prompt([{"topic": "Solo"}])
    assert flat_items == ["Solo"]
    assert "**1. Solo**" in text


def test_missing_topic_key_renders_empty_string():
    text, flat_items = render_outline_prompt([{"subtopics": ["only sub"]}])
    assert flat_items == ["", "only sub"]
    assert "**1. **" in text
    assert "   2. only sub" in text


class _FakeStateManager:
    def __init__(self) -> None:
        self._state: dict[str, dict] = {}

    def get_state(self, cid: str) -> dict:
        return self._state.setdefault(cid, {})

    def update_state(self, cid: str, key: str, value) -> None:
        self._state.setdefault(cid, {})[key] = value


class _RecordingEvents:
    """Stub events sink for tests that run the parser past its
    short-circuits and into the body, where _emit_message →
    ctx.events.emit is called."""

    def __init__(self) -> None:
        self.emitted: list = []

    async def emit(self, event) -> None:
        self.emitted.append(event)


class _FakeCtx:
    """Minimal ctx surface for parser tests. ``cancellation_token``
    defaults to None so the parser's ``getattr(ctx,
    "cancellation_token", None)`` reads a real value rather than
    depending on attribute absence; tests that want a real token
    assign it explicitly. ``events`` is a recording stub so
    slash-command tests that proceed past the short-circuit can be
    exercised without standing up the full coordinator."""

    def __init__(self) -> None:
        self.state = _FakeStateManager()
        self.conversation_id = "conv-test"
        self.cancellation_token = None
        self.events = _RecordingEvents()


@pytest.mark.parametrize("cmd", ["/q", "/quit", "/Q", "  /quit  "])
def test_parser_raises_cancelled_on_slash_q(cmd):
    """The parser short-circuits /q and /quit (case-insensitive, with
    surrounding whitespace) to asyncio.CancelledError so the engine's
    CancelledError handler in JobRunner picks it up and posts the
    terminal cancellation writeback."""
    ctx = _FakeCtx()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_outline_feedback_continuation(ctx, cmd))


def test_parser_signals_cancel_token_on_slash_q():
    """The parser must signal ctx.cancellation_token BEFORE raising
    CancelledError. Any post-unwind is_cancelled() reader (the
    runner's gate-cancel branch, the engine's raise_if_cancelled)
    observes the cancel consistently."""
    from deep_research.core.cancellation import CancellationToken

    ctx = _FakeCtx()
    ctx.cancellation_token = CancellationToken()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_outline_feedback_continuation(ctx, "/q"))
    assert ctx.cancellation_token.is_cancelled() is True


def test_parser_signals_cancel_token_on_slash_quit():
    """Same as above but for the long form /quit."""
    from deep_research.core.cancellation import CancellationToken

    ctx = _FakeCtx()
    ctx.cancellation_token = CancellationToken()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_outline_feedback_continuation(ctx, "/quit"))
    assert ctx.cancellation_token.is_cancelled() is True


def test_parser_handles_missing_cancel_token_on_slash_q():
    """ctx.cancellation_token is None on the Function-runtime ctx (no
    cancel surface). The parser must still raise CancelledError
    without an AttributeError from `None.cancel()`. The `getattr` form
    in outline_feedback also tolerates a ctx that has no
    cancellation_token attribute at all; this test pins the
    None-explicit branch."""
    ctx = _FakeCtx()
    assert ctx.cancellation_token is None
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_outline_feedback_continuation(ctx, "/q"))


@pytest.mark.parametrize("cmd", [
    "/k 1,2,3",
    "/K 1,2,3",
    "/keep 1,2,3",
    "/Keep 1,2,3",
    "/KEEP 1,2,3",
])
def test_parser_keep_command_case_insensitive(cmd):
    """All variants of /k and /keep must yield identical results.
    Pins the re.IGNORECASE flag on both the slash_keep_patterns match
    and the items-extraction re.sub. Uses 'keep all' so the PDV
    call's fast-path fires (no embeddings stack needed)."""
    ctx = _FakeCtx()
    state = ctx.state.get_state(ctx.conversation_id)
    state["outline_feedback_data"] = {
        "flat_items": ["topic-a", "topic-b", "topic-c"],
    }
    result = asyncio.run(
        process_outline_feedback_continuation(ctx, cmd)
    )
    assert result["kept_items"] == ["topic-a", "topic-b", "topic-c"]
    assert result["removed_items"] == []


@pytest.mark.parametrize("cmd", [
    "/r 1,2,3",
    "/R 1,2,3",
    "/remove 1,2,3",
    "/Remove 1,2,3",
    "/REMOVE 1,2,3",
])
def test_parser_remove_command_case_insensitive(cmd):
    """Symmetric: all variants of /r and /remove must mark every item
    for removal. 'remove all' triggers the PDV fast-path too."""
    ctx = _FakeCtx()
    state = ctx.state.get_state(ctx.conversation_id)
    state["outline_feedback_data"] = {
        "flat_items": ["topic-a", "topic-b", "topic-c"],
    }
    result = asyncio.run(
        process_outline_feedback_continuation(ctx, cmd)
    )
    assert result["kept_items"] == []
    assert result["removed_items"] == ["topic-a", "topic-b", "topic-c"]

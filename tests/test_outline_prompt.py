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
    _rebuild_outline_from_kept,
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


# --- slash_command flag propagation ----------------------------------
#
# The flag drives the replacement-topic short-circuit in
# `continue_research_after_feedback`: True → trim the outline to the
# user's pick and proceed; False → invoke the LLM/embedding-heavy
# replacement-topic pipeline. The downstream gate inverts behaviour, so
# the parser's contract on this flag is load-bearing.


def test_slash_keep_sets_slash_command_true():
    """``/k`` keeping all items hits the PDV fast-path (no embeddings
    stack required); just pins the flag for the slash-command branch."""
    ctx = _FakeCtx()
    state = ctx.state.get_state(ctx.conversation_id)
    state["outline_feedback_data"] = {
        "flat_items": ["topic-a", "topic-b", "topic-c"],
    }
    result = asyncio.run(
        process_outline_feedback_continuation(ctx, "/k 1,2,3")
    )
    assert result["slash_command"] is True


def test_slash_remove_sets_slash_command_true():
    """``/r`` removing all items hits the PDV fast-path symmetrically."""
    ctx = _FakeCtx()
    state = ctx.state.get_state(ctx.conversation_id)
    state["outline_feedback_data"] = {
        "flat_items": ["topic-a", "topic-b", "topic-c"],
    }
    result = asyncio.run(
        process_outline_feedback_continuation(ctx, "/r 1,2,3")
    )
    assert result["slash_command"] is True


@pytest.mark.parametrize("cmd", ["continue", "/continue", "/c", ""])
def test_continue_short_circuit_sets_slash_command_true(cmd):
    """``/continue`` / empty input is also a structured user signal,
    not natural language. Flag is True so the downstream gate's
    interpretation stays consistent (the gate also short-circuits when
    removed_items is empty, so this flag value is the safer of the
    two)."""
    ctx = _FakeCtx()
    state = ctx.state.get_state(ctx.conversation_id)
    state["outline_feedback_data"] = {
        "flat_items": ["topic-a", "topic-b"],
    }
    result = asyncio.run(
        process_outline_feedback_continuation(ctx, cmd)
    )
    assert result["slash_command"] is True


# --- _rebuild_outline_from_kept --------------------------------------
#
# Pure transformation used by the slash-command short-circuit (and
# eligible for reuse by the natural-language path). Pin the
# topic/subtopic hierarchy behaviour so a future refactor of the
# natural-language path can call this helper safely.


def test_rebuild_keeps_topic_and_kept_subtopics():
    outline = [
        {"topic": "Architecture", "subtopics": ["State space", "Selectivity"]},
        {"topic": "Performance", "subtopics": ["Latency", "Throughput"]},
    ]
    kept = ["Architecture", "State space", "Latency"]
    new_outline, new_all = _rebuild_outline_from_kept(outline, kept)
    # Architecture is kept (+ its kept subtopic).
    # Performance is dropped as a main topic but Latency is kept;
    # the helper restores Performance as a parent for Latency.
    assert new_outline == [
        {"topic": "Architecture", "subtopics": ["State space"]},
        {"topic": "Performance", "subtopics": ["Latency"]},
    ]
    assert new_all == ["Architecture", "State space", "Performance", "Latency"]


def test_rebuild_drops_topics_with_no_kept_subtopics():
    outline = [
        {"topic": "Architecture", "subtopics": ["A1", "A2"]},
        {"topic": "Performance", "subtopics": ["P1"]},
    ]
    kept = ["A1"]
    new_outline, new_all = _rebuild_outline_from_kept(outline, kept)
    assert new_outline == [{"topic": "Architecture", "subtopics": ["A1"]}]
    assert new_all == ["Architecture", "A1"]


def test_rebuild_keeps_solo_topic_with_no_subtopics():
    outline = [{"topic": "Solo", "subtopics": []}]
    new_outline, new_all = _rebuild_outline_from_kept(outline, ["Solo"])
    assert new_outline == [{"topic": "Solo", "subtopics": []}]
    assert new_all == ["Solo"]


def test_rebuild_empty_when_nothing_kept():
    outline = [{"topic": "Architecture", "subtopics": ["A1"]}]
    new_outline, new_all = _rebuild_outline_from_kept(outline, [])
    assert new_outline == []
    assert new_all == []


# --- continue_research_after_feedback slash-command short-circuit ----


def test_continue_research_skips_replacement_on_slash_command(monkeypatch):
    """The slash-command branch must NOT invoke replacement-topic
    generation, grouping, query gen, refinement, or research. Mock
    each and assert ``assert_not_called`` so a future regression
    (e.g. someone removing the ``if slash_command:`` gate) trips
    immediately."""
    from unittest.mock import AsyncMock

    from deep_research.research import outline_feedback as of

    gen_replacement = AsyncMock()
    monkeypatch.setattr(of, "generate_replacement_topics", gen_replacement)
    # Inside _finalize_trimmed_outline:
    monkeypatch.setattr(
        of, "get_embedding", AsyncMock(return_value=[0.1, 0.2, 0.3])
    )
    monkeypatch.setattr(
        of, "initialize_research_dimensions", AsyncMock(return_value=None)
    )

    outline_items = [
        {"topic": "Architecture", "subtopics": ["State space", "Selectivity"]},
        {"topic": "Performance", "subtopics": ["Latency"]},
    ]
    all_topics = ["Architecture", "State space", "Selectivity", "Performance", "Latency"]
    feedback_result = {
        "kept_items": ["Architecture", "State space"],
        "removed_items": ["Selectivity", "Performance", "Latency"],
        "kept_indices": [0, 1],
        "removed_indices": [2, 3, 4],
        "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        "slash_command": True,
    }

    ctx = _FakeCtx()
    new_outline, new_all, new_emb = asyncio.run(
        of.continue_research_after_feedback(
            ctx,
            feedback_result,
            user_message="probe",
            outline_items=outline_items,
            all_topics=all_topics,
            outline_embedding=[0.0, 0.0, 0.0],
        )
    )

    gen_replacement.assert_not_called()
    assert new_outline == [
        {"topic": "Architecture", "subtopics": ["State space"]},
    ]
    assert new_all == ["Architecture", "State space"]
    assert new_emb == [0.1, 0.2, 0.3]
    # Waiting flag is cleared so the engine proceeds to main research.
    assert (
        ctx.state.get_state(ctx.conversation_id).get("waiting_for_outline_feedback")
        is False
    )


def test_continue_research_falls_back_when_pick_leaves_nothing(monkeypatch):
    """Defensive: ``/r`` removing every item must not crash; keep the
    original outline and proceed."""
    from unittest.mock import AsyncMock

    from deep_research.research import outline_feedback as of

    gen_replacement = AsyncMock()
    monkeypatch.setattr(of, "generate_replacement_topics", gen_replacement)
    monkeypatch.setattr(
        of, "get_embedding", AsyncMock(return_value=[0.1])
    )
    monkeypatch.setattr(
        of, "initialize_research_dimensions", AsyncMock(return_value=None)
    )

    outline_items = [{"topic": "Architecture", "subtopics": ["A1"]}]
    feedback_result = {
        "kept_items": [],
        "removed_items": ["Architecture", "A1"],
        "kept_indices": [],
        "removed_indices": [0, 1],
        "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        "slash_command": True,
    }

    ctx = _FakeCtx()
    new_outline, _, _ = asyncio.run(
        of.continue_research_after_feedback(
            ctx,
            feedback_result,
            user_message="probe",
            outline_items=outline_items,
            all_topics=["Architecture", "A1"],
            outline_embedding=[0.0],
        )
    )
    gen_replacement.assert_not_called()
    assert new_outline == outline_items

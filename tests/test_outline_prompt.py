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


class _FakeCtx:
    """Minimal ctx surface for the parser short-circuit case — the cancel
    branch runs before any of the heavier ctx attributes are touched."""

    def __init__(self) -> None:
        self.state = _FakeStateManager()
        self.conversation_id = "conv-test"


@pytest.mark.parametrize("cmd", ["/q", "/quit", "/Q", "  /quit  "])
def test_parser_raises_cancelled_on_slash_q(cmd):
    """The parser short-circuits /q and /quit (case-insensitive, with
    surrounding whitespace) to asyncio.CancelledError so the engine's
    CancelledError handler in JobRunner picks it up and posts the
    terminal cancellation writeback."""
    ctx = _FakeCtx()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_outline_feedback_continuation(ctx, cmd))

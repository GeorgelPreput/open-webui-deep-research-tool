import pytest

from deep_research.core.text import (
    chunk_text,
    escape_html,
    materialize_embedding,
    snapshot_embedding,
    stable_text_key,
)


def test_stable_text_key_deterministic():
    assert stable_text_key("hello") == stable_text_key("hello")


def test_stable_text_key_unique_per_input():
    assert stable_text_key("hello") != stable_text_key("Hello")
    assert stable_text_key("a") != stable_text_key("b")


def test_stable_text_key_handles_none():
    # Should not raise on None
    assert stable_text_key(None) == stable_text_key("")


def test_chunk_text_level_0_returns_whole():
    chunks = chunk_text("hello world. another sentence.", chunk_level=0)
    assert chunks == ["hello world. another sentence."]


def test_chunk_text_level_2_sentences():
    chunks = chunk_text(
        "First sentence. Second sentence! Third? Fourth.", chunk_level=2
    )
    assert len(chunks) == 4
    assert chunks[0] == "First sentence."


def test_chunk_text_level_3_paragraphs():
    text = "Line one.\nLine two.\nLine three."
    chunks = chunk_text(text, chunk_level=3)
    assert chunks == ["Line one.", "Line two.", "Line three."]


def test_chunk_text_empty_skipped():
    chunks = chunk_text("\n\n  \n", chunk_level=2)
    assert chunks == []


def test_snapshot_then_materialize_roundtrip():
    snap = snapshot_embedding([1.0, 2.0, 3.0])
    out = materialize_embedding(snap)
    assert out == pytest.approx([1.0, 2.0, 3.0])


def test_snapshot_none_or_empty_returns_none():
    assert snapshot_embedding(None) is None
    assert snapshot_embedding([]) is None


def test_escape_html_basic():
    assert "&lt;" in escape_html("<script>")
    assert "&amp;" in escape_html("a & b")

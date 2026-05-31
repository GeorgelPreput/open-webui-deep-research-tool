"""Regression test for BUG 29.

extract_token_window must not return a 1-character sliver when the requested
start lands past the end of the content; it falls back to the final
window-sized slice instead, which the caller can use as real content.
"""
import pytest

from deep_research.budget.windows import extract_token_window


@pytest.mark.asyncio
async def test_window_past_end_returns_meaningful_tail(run_context):
    content = "Hello world. " * 60  # a few hundred tokens
    result = await extract_token_window(
        run_context, content, start_token=100_000, window_size=50
    )
    # The old behaviour clamped start to len-1 and returned a 1-char sliver;
    # the fix returns a proper window-sized tail.
    assert len(result) > 10
    assert result.strip()


@pytest.mark.asyncio
async def test_window_in_range_returns_prefix(run_context):
    content = "alpha beta gamma delta. " * 60
    result = await extract_token_window(
        run_context, content, start_token=0, window_size=20
    )
    assert len(result) > 0
    # A window starting at token 0 is a (possibly sentence-trimmed) prefix.
    assert content.startswith(result)


@pytest.mark.asyncio
async def test_window_empty_content(run_context):
    result = await extract_token_window(run_context, "", start_token=0, window_size=10)
    assert result == ""

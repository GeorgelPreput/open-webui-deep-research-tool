"""Coverage tests for budget/tokens.py token counting helpers."""
import pytest

from deep_research.budget.tokens import count_message_tokens, count_tokens


@pytest.mark.asyncio
async def test_count_tokens_empty_is_zero():
    assert await count_tokens(None, "") == 0


@pytest.mark.asyncio
async def test_count_tokens_nonempty_is_positive():
    n = await count_tokens(None, "The quick brown fox jumps over the lazy dog.")
    assert n > 0


@pytest.mark.asyncio
async def test_count_tokens_longer_text_more_tokens():
    short = await count_tokens(None, "hello")
    long = await count_tokens(None, "hello " * 50)
    assert long > short


@pytest.mark.asyncio
async def test_count_message_tokens_includes_overhead():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi there"},
    ]
    total = await count_message_tokens(None, messages)
    # priming (3) + 2 * per-message overhead (10) at minimum.
    assert total >= 3 + 2 * 10


@pytest.mark.asyncio
async def test_count_message_tokens_handles_list_content():
    # OpenAI multimodal-style content: list of parts.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
    ]
    total = await count_message_tokens(None, messages)
    assert total > 0


@pytest.mark.asyncio
async def test_count_message_tokens_empty_list():
    # Only the priming overhead.
    assert await count_message_tokens(None, []) == 3

import logging
from typing import Any

import tiktoken

from deep_research.core.types import RunContext

logger = logging.getLogger(__name__)


async def count_tokens(ctx: RunContext, text: str) -> int:
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens with tiktoken: {e}")
        return int(len(text.split()) * 1.3)


async def count_message_tokens(ctx: RunContext, messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for an OpenAI-style messages list.

    Adds a small per-message framing overhead plus a priming overhead for
    the assistant turn, matching what tiktoken-only counting misses.
    """
    per_message_overhead = 10
    priming_overhead = 3

    total = priming_overhead
    for msg in messages:
        role_tokens = await count_tokens(ctx, msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        content_tokens = await count_tokens(ctx, content)
        total += role_tokens + content_tokens + per_message_overhead
    return total

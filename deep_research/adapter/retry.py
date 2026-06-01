import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from deep_research.core.errors import classify_transient_completion_error

T = TypeVar("T")

logger = logging.getLogger("deep_research.adapter.retry")


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    label: str = "request",
) -> T:
    max_retries = max(0, max_retries)
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except Exception as e:
            last_exc = e
            reason = classify_transient_completion_error(e)
            if reason is None:
                raise
            if attempt >= max_retries:
                logger.warning(
                    "Retry %s: exhausted after %d attempts reason=%s exc=%s",
                    label,
                    attempt + 1,
                    reason,
                    type(e).__name__,
                )
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                "Retry %s: attempt=%d/%d delay=%.2fs reason=%s exc=%s",
                label,
                attempt + 1,
                max_retries,
                delay,
                reason,
                type(e).__name__,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable") from last_exc

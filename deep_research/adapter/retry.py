import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from deep_research.core.errors import (
    classify_transient_completion_error,
    extract_retry_after_seconds,
)

T = TypeVar("T")

logger = logging.getLogger("deep_research.adapter.retry")


# Optional callback signature: (exc, attempt, transient_reason) -> None.
# Used by the provider clients to surface 429s and retries to their
# HttpThrottle's counters. Kept Optional so the OWUI client retry sites that
# don't have a throttle stay one-line.
RetryObserver = Callable[[BaseException, int, str], None]


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: float = 0.0,
    respect_retry_after: bool = True,
    label: str = "request",
    on_transient: RetryObserver | None = None,
    on_exhausted: RetryObserver | None = None,
) -> T:
    """Retry a transient HTTP/coroutine failure with backoff.

    - ``base_delay``/``max_delay``: exponential window. Delay for attempt N
      is ``base_delay * 2**N``, clamped to ``max_delay``.
    - ``jitter``: multiplicative spread, ``delay *= U(1-j, 1+j)``. Set to 0
      to keep the legacy (no-jitter) behaviour used by the OWUI client.
    - ``respect_retry_after``: when True and the exception carries a
      Retry-After header (httpx HTTPStatusError or AdapterError with
      headers), that value replaces the exponential delay (still clamped
      to ``max_delay``).
    - ``on_transient`` fires on every transient failure that will be retried;
      ``on_exhausted`` fires once when the final retry is given up. Both
      receive ``(exc, attempt_index, transient_reason)``.
    """
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
                if on_exhausted is not None:
                    try:
                        on_exhausted(e, attempt, reason)
                    except Exception:
                        logger.exception("on_exhausted callback failed")
                raise
            delay = min(base_delay * (2**attempt), max_delay)
            if respect_retry_after:
                retry_after = extract_retry_after_seconds(e)
                if retry_after is not None and retry_after > 0:
                    delay = min(retry_after, max_delay)
            if jitter > 0:
                spread = max(0.0, min(1.0, jitter))
                delay = max(0.0, delay * random.uniform(1.0 - spread, 1.0 + spread))
            logger.warning(
                "Retry %s: attempt=%d/%d delay=%.2fs reason=%s exc=%s",
                label,
                attempt + 1,
                max_retries,
                delay,
                reason,
                type(e).__name__,
            )
            if on_transient is not None:
                try:
                    on_transient(e, attempt, reason)
                except Exception:
                    logger.exception("on_transient callback failed")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable") from last_exc

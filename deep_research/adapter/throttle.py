"""Per-client HTTP throttle and diagnostics.

Provider-agnostic: one instance lives inside each provider client
(``LLMProviderClient`` and ``EmbeddingProviderClient``). The throttle adds a
token bucket and a min-interval gate on top of the existing semaphore, plus
counters that feed degraded-mode signalling and end-of-run diagnostics.

The non-tunable constants below are pinned defaults rather than env-tunable
valves. Operators have not asked for these knobs, and the derived
threshold/cooldown values scale automatically with the user-tunable
``max_delay_seconds`` valve.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger("deep_research.adapter.throttle")


# ---- Non-tunable constants -------------------------------------------------

# Multiplicative jitter applied to computed backoff delays: delay *= U(1-j, 1+j).
# Spreads retries from concurrent callers so they don't burst the upstream in
# lockstep after the same 429.
JITTER_RATIO: float = 0.25

# When True, an upstream Retry-After header replaces the computed exponential
# backoff (clamped to ``max_delay_seconds``). Honoured for every 429-class
# transient error.
RESPECT_RETRY_AFTER: bool = True

# When True, the throttle latches degraded mode after a full retry exhaustion
# on a 429. Degraded mode is consumed by embedding-heavy consumer paths to
# scale down opportunistic work without aborting the run.
DEGRADE_ON_429: bool = True

# Exhausted-retry 429 events that must accumulate before degraded mode latches.
# One full exhaustion is already strong evidence we're over quota — there's no
# value in waiting for multiple cycles.
DEGRADE_CONSECUTIVE_429_THRESHOLD: int = 1


def derive_degrade_cooldown_seconds(max_delay_seconds: float) -> float:
    """Cooldown derived from the user-tunable backoff ceiling.

    After two worst-case backoff windows of clean traffic, the quota is
    treated as recovered. Floor at 60s so very-low-latency providers don't
    bounce in and out of degraded mode.
    """
    return max(60.0, 2.0 * float(max_delay_seconds))


# ---- Diagnostics types -----------------------------------------------------


@dataclass
class ThrottleStats:
    """Snapshot of a throttle's counters for end-of-run diagnostics."""

    label: str
    attempts: int = 0
    successes: int = 0
    retries: int = 0
    http_429: int = 0
    exhausted_429: int = 0
    skipped: int = 0
    degraded_activations: int = 0
    degraded_now: bool = False

    def format_line(self) -> str:
        return (
            f"{self.label}: attempts={self.attempts} successes={self.successes} "
            f"retries={self.retries} http_429={self.http_429} "
            f"exhausted_429={self.exhausted_429} skipped={self.skipped} "
            f"degraded={'yes' if self.degraded_now else 'no'}"
        )


@dataclass
class ThrottleDiagnostics:
    """Read-only view of a throttle exposed via ``RunContext``.

    Consumers should never grab the throttle object directly; instead they
    read ``ctx.embeddings_diagnostics.degraded`` (and snapshot for logging).
    """

    _throttle: HttpThrottle = field(repr=False)

    @property
    def label(self) -> str:
        return self._throttle.label

    @property
    def degraded(self) -> bool:
        return self._throttle.degraded

    def snapshot(self) -> ThrottleStats:
        return self._throttle.snapshot()

    def record_skipped(self) -> None:
        """Count a consumer-side skip (e.g. degraded-mode bail) for diagnostics."""
        self._throttle.record_skipped()


# ---- The throttle ----------------------------------------------------------


class HttpThrottle:
    """Token-bucket + min-interval gate + 429 counters for one HTTP client.

    ``acquire()`` must be awaited before each dispatched HTTP call. The bucket
    and gate compose with (do not replace) the client's existing semaphore.
    Counter updates are explicit (``record_*``) so the calling layer can
    distinguish e.g. exhausted-retry 429s (which trip degraded mode) from
    one-off 429s that succeeded on retry.
    """

    def __init__(
        self,
        *,
        label: str,
        max_rps: float,
        min_interval_ms: int,
        max_delay_seconds: float,
    ) -> None:
        self.label = label
        self._max_rps = max(0.0, float(max_rps))
        # Bucket capacity is at least 1 token; for >1 RPS it scales with the
        # rate so brief bursts can drain the bucket before queueing kicks in.
        self._capacity = max(1.0, math.ceil(self._max_rps)) if self._max_rps > 0 else 0.0
        self._tokens = self._capacity
        self._min_interval = max(0, int(min_interval_ms)) / 1000.0
        self._last_dispatch: float = 0.0
        self._refill_at: float = time.monotonic()
        self._lock = asyncio.Lock()

        self._degrade_threshold = DEGRADE_CONSECUTIVE_429_THRESHOLD
        self._degrade_cooldown = derive_degrade_cooldown_seconds(max_delay_seconds)

        # Counters
        self._attempts = 0
        self._successes = 0
        self._retries = 0
        self._http_429 = 0
        self._exhausted_429 = 0
        self._skipped = 0
        self._consecutive_exhausted_429 = 0
        self._degrade_activations = 0
        self._degraded_until: float = 0.0  # monotonic; 0 means not degraded
        self._last_success_at: float = 0.0

    # ---- gate ----

    async def acquire(self) -> None:
        if self._max_rps <= 0 and self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            # Token bucket refill
            if self._max_rps > 0:
                elapsed = now - self._refill_at
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity, self._tokens + elapsed * self._max_rps
                    )
                    self._refill_at = now
                if self._tokens < 1.0:
                    needed = 1.0 - self._tokens
                    wait = needed / self._max_rps
                else:
                    wait = 0.0
            else:
                wait = 0.0
            # Min-interval gate
            if self._min_interval > 0 and self._last_dispatch > 0:
                gap = now + wait - self._last_dispatch
                if gap < self._min_interval:
                    wait += self._min_interval - gap
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
                if self._max_rps > 0:
                    elapsed = now - self._refill_at
                    self._tokens = min(
                        self._capacity, self._tokens + elapsed * self._max_rps
                    )
                    self._refill_at = now
            if self._max_rps > 0:
                self._tokens -= 1.0
            self._last_dispatch = now

    # ---- counters ----

    def record_attempt(self) -> None:
        self._attempts += 1

    def record_success(self) -> None:
        self._successes += 1
        self._consecutive_exhausted_429 = 0
        self._last_success_at = time.monotonic()

    def record_retry(self) -> None:
        self._retries += 1

    def record_429(self, *, exhausted: bool) -> None:
        self._http_429 += 1
        if exhausted:
            self._exhausted_429 += 1
            self._consecutive_exhausted_429 += 1
            if (
                DEGRADE_ON_429
                and self._consecutive_exhausted_429 >= self._degrade_threshold
            ):
                was_degraded = self.degraded
                self._degraded_until = time.monotonic() + self._degrade_cooldown
                if not was_degraded:
                    self._degrade_activations += 1
                    logger.warning(
                        "Throttle %s: degraded mode activated "
                        "(consecutive exhausted 429s=%d, cooldown=%.1fs)",
                        self.label,
                        self._consecutive_exhausted_429,
                        self._degrade_cooldown,
                    )

    def record_skipped(self) -> None:
        self._skipped += 1

    # ---- state ----

    @property
    def degraded(self) -> bool:
        if not DEGRADE_ON_429:
            return False
        if self._degraded_until <= 0:
            return False
        if time.monotonic() >= self._degraded_until and self._last_success_at >= (
            self._degraded_until - self._degrade_cooldown
        ):
            # Cooldown elapsed and we've seen successful traffic in that window
            self._degraded_until = 0.0
            self._consecutive_exhausted_429 = 0
            logger.info("Throttle %s: degraded mode cleared", self.label)
            return False
        return True

    def snapshot(self) -> ThrottleStats:
        return ThrottleStats(
            label=self.label,
            attempts=self._attempts,
            successes=self._successes,
            retries=self._retries,
            http_429=self._http_429,
            exhausted_429=self._exhausted_429,
            skipped=self._skipped,
            degraded_activations=self._degrade_activations,
            degraded_now=self.degraded,
        )

    def diagnostics(self) -> ThrottleDiagnostics:
        return ThrottleDiagnostics(_throttle=self)

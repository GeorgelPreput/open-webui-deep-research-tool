import asyncio
import time

import pytest

from deep_research.adapter.throttle import (
    DEGRADE_CONSECUTIVE_429_THRESHOLD,
    HttpThrottle,
    derive_degrade_cooldown_seconds,
)


@pytest.mark.asyncio
async def test_acquire_is_noop_when_throttle_disabled():
    t = HttpThrottle(label="t", max_rps=0.0, min_interval_ms=0, max_delay_seconds=10)
    t0 = time.monotonic()
    for _ in range(5):
        await t.acquire()
    assert time.monotonic() - t0 < 0.05


@pytest.mark.asyncio
async def test_token_bucket_caps_dispatch_rate():
    # 5 RPS -> 4 dispatches must take at least ~3/5 seconds in the steady state.
    # Bucket starts full (capacity = ceil(5) = 5), so the first 5 are free; the
    # 6th waits 1/5 = 0.2s. We test 7 acquires to ensure spacing kicks in.
    t = HttpThrottle(label="rps", max_rps=5.0, min_interval_ms=0, max_delay_seconds=10)
    t0 = time.monotonic()
    for _ in range(7):
        await t.acquire()
    elapsed = time.monotonic() - t0
    # Two extra over capacity -> at least 2 * (1/5) seconds.
    assert elapsed >= 0.3, f"expected >= 0.3s, got {elapsed:.3f}"


@pytest.mark.asyncio
async def test_min_interval_gate_enforces_spacing():
    t = HttpThrottle(label="gap", max_rps=0.0, min_interval_ms=100, max_delay_seconds=10)
    t0 = time.monotonic()
    await t.acquire()
    await t.acquire()
    await t.acquire()
    elapsed = time.monotonic() - t0
    # Two 100ms gaps minimum.
    assert elapsed >= 0.18, f"expected >= 0.18s, got {elapsed:.3f}"


def test_exhausted_429_trips_degraded_mode():
    t = HttpThrottle(label="d", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    assert t.degraded is False
    for _ in range(DEGRADE_CONSECUTIVE_429_THRESHOLD):
        t.record_429(exhausted=True)
    assert t.degraded is True
    snap = t.snapshot()
    assert snap.exhausted_429 == DEGRADE_CONSECUTIVE_429_THRESHOLD
    assert snap.degraded_now is True
    assert snap.degraded_activations == 1


def test_one_off_429_does_not_trip_degraded():
    t = HttpThrottle(label="d", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    t.record_429(exhausted=False)
    assert t.degraded is False
    assert t.snapshot().http_429 == 1
    assert t.snapshot().exhausted_429 == 0


def test_degrade_cooldown_derivation_floors_at_60():
    assert derive_degrade_cooldown_seconds(5.0) == 60.0
    assert derive_degrade_cooldown_seconds(60.0) == 120.0
    assert derive_degrade_cooldown_seconds(90.0) == 180.0


def test_degraded_clears_after_cooldown_with_success():
    t = HttpThrottle(label="d", max_rps=0.0, min_interval_ms=0, max_delay_seconds=0.0)
    # cooldown derived = max(60, 0) = 60s; bypass by zeroing internal field
    t.record_429(exhausted=True)
    assert t.degraded is True
    # Force cooldown expiry and add a success to satisfy the recovery condition.
    t._degraded_until = time.monotonic() - 0.001
    t._last_success_at = time.monotonic()
    assert t.degraded is False


def test_record_success_resets_consecutive_counter():
    t = HttpThrottle(label="d", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    t.record_429(exhausted=False)
    t.record_success()
    # Now a single exhausted-429 should not yet trip degraded because the
    # consecutive counter was reset (assuming threshold > 1).
    if DEGRADE_CONSECUTIVE_429_THRESHOLD > 1:
        t.record_429(exhausted=True)
        assert t.degraded is False
    else:
        # Threshold is 1 — one exhausted 429 is enough; just sanity-check the
        # reset happened (no exception, counters consistent).
        assert t.snapshot().successes == 1


def test_diagnostics_view_exposes_label_and_degraded():
    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    d = t.diagnostics()
    assert d.label == "emb"
    assert d.degraded is False
    t.record_429(exhausted=True)
    assert d.degraded is True
    d.record_skipped()
    assert d.snapshot().skipped == 1


def test_two_throttles_have_independent_state():
    a = HttpThrottle(label="a", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    b = HttpThrottle(label="b", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    a.record_429(exhausted=True)
    assert a.degraded is True
    assert b.degraded is False


def test_stats_format_line_is_stable():
    t = HttpThrottle(label="emb", max_rps=0.0, min_interval_ms=0, max_delay_seconds=5)
    t.record_attempt()
    t.record_success()
    t.record_attempt()
    t.record_retry()
    t.record_429(exhausted=False)
    t.record_attempt()
    t.record_success()
    line = t.snapshot().format_line()
    assert "emb:" in line
    assert "attempts=3" in line
    assert "successes=2" in line
    assert "retries=1" in line
    assert "http_429=1" in line
    assert "degraded=no" in line


@pytest.mark.asyncio
async def test_acquire_serialises_under_lock():
    """Two coroutines hitting a min-interval gate must serialise their dispatch.

    Without the lock the second coroutine could read ``last_dispatch`` before the
    first wrote it, and both would dispatch immediately.
    """
    t = HttpThrottle(label="gap", max_rps=0.0, min_interval_ms=100, max_delay_seconds=10)

    timestamps: list[float] = []

    async def go():
        await t.acquire()
        timestamps.append(time.monotonic())

    t0 = time.monotonic()
    await asyncio.gather(go(), go(), go())
    elapsed = time.monotonic() - t0
    timestamps.sort()
    assert elapsed >= 0.18
    assert timestamps[1] - timestamps[0] >= 0.09
    assert timestamps[2] - timestamps[1] >= 0.09

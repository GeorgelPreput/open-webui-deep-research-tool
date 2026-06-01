"""Tests for the augmented with_retry: Retry-After honoured, jitter applied."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from deep_research.adapter.client import AdapterError
from deep_research.adapter.retry import with_retry


async def _failer(exc_factory, succeed_after: int) -> Any:
    counter = {"n": 0}

    async def call():
        counter["n"] += 1
        if counter["n"] <= succeed_after:
            raise exc_factory(counter["n"])
        return "ok"

    return call, counter


def _httpstatus(status: int, headers: dict[str, str] | None = None) -> Exception:
    req = httpx.Request("POST", "http://x/")
    resp = httpx.Response(status, headers=headers or {}, request=req)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured(monkeypatch):
    """A Retry-After header on a 429 replaces the exponential delay."""
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    counter = {"n": 0}

    async def call():
        counter["n"] += 1
        if counter["n"] == 1:
            raise AdapterError("rate limited", status=429, headers={"Retry-After": "7"})
        return "ok"

    result = await with_retry(
        call,
        max_retries=3,
        base_delay=0.5,
        max_delay=60.0,
        jitter=0.0,
        respect_retry_after=True,
        label="t",
    )
    assert result == "ok"
    assert len(sleeps) == 1
    assert abs(sleeps[0] - 7.0) < 1e-6


@pytest.mark.asyncio
async def test_retry_after_clamped_to_max_delay(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    counter = {"n": 0}

    async def call():
        counter["n"] += 1
        if counter["n"] == 1:
            raise AdapterError("rl", status=429, headers={"Retry-After": "9999"})
        return "ok"

    await with_retry(
        call,
        max_retries=3,
        base_delay=0.5,
        max_delay=30.0,
        jitter=0.0,
        label="t",
    )
    assert sleeps == [30.0]


@pytest.mark.asyncio
async def test_jitter_spreads_delay(monkeypatch):
    """With jitter > 0 the actual sleep falls within the U(1-j, 1+j) band."""
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def call():
        # Always fails — we just want to observe sleep values.
        raise _httpstatus(503)

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(
            call,
            max_retries=4,
            base_delay=1.0,
            max_delay=10.0,
            jitter=0.25,
            label="t",
        )
    # Exponential nominal sequence: 1, 2, 4, 8 (with last clamped at 10).
    # Jitter band of 0.25 means actual values fall in:
    #   [0.75, 1.25], [1.5, 2.5], [3.0, 5.0], [6.0, 10.0]
    expected_min = [0.75, 1.5, 3.0, 6.0]
    expected_max = [1.25, 2.5, 5.0, 10.0]
    for s, lo, hi in zip(sleeps, expected_min, expected_max, strict=False):
        assert lo - 1e-6 <= s <= hi + 1e-6, f"sleep={s} outside [{lo}, {hi}]"


@pytest.mark.asyncio
async def test_non_transient_does_not_retry(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    counter = {"n": 0}

    async def call():
        counter["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await with_retry(call, max_retries=3, label="t")
    assert counter["n"] == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_observer_callbacks_fire(monkeypatch):
    async def fake_sleep(d: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    transients: list[tuple[str, int]] = []
    exhausted: list[tuple[str, int]] = []

    def on_transient(exc, attempt, reason):
        transients.append((reason, attempt))

    def on_exhausted(exc, attempt, reason):
        exhausted.append((reason, attempt))

    async def call():
        raise _httpstatus(429)

    with pytest.raises(httpx.HTTPStatusError):
        await with_retry(
            call,
            max_retries=2,
            base_delay=0.01,
            max_delay=0.1,
            jitter=0.0,
            label="t",
            on_transient=on_transient,
            on_exhausted=on_exhausted,
        )
    assert len(transients) == 2
    assert len(exhausted) == 1
    assert transients[0][0].startswith("http_status=429")
    assert exhausted[0][0].startswith("http_status=429")

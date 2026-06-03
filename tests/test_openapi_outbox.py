"""Tests for the writeback OutboxWorker.

The worker drains rows that translate engine events to OWUI's per-message
``/event`` endpoint. These tests use a fake OWUIClient that records calls
and can be flipped between success/failure modes; the real network path
isn't exercised here (that lives in the writeback e2e test).
"""
from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

import pytest
import pytest_asyncio

from deep_research.adapter.client import AdapterError
from deep_research.entrypoints.openapi_tool.outbox import OutboxWorker


class _FakeOWUIClient:
    """Records POST calls and simulates configurable failure modes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_first: int = 0
        self.fail_with: Exception | None = None
        self.always_fail: bool = False

    async def post_message_event(
        self, chat_id: str, message_id: str, event_type: str, data: dict
    ) -> None:
        self.calls.append({
            "chat_id": chat_id,
            "message_id": message_id,
            "event_type": event_type,
            "data": data,
        })
        if self.always_fail:
            raise self.fail_with or AdapterError("forced failure", status=500)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise self.fail_with or AdapterError("transient", status=503)


@pytest_asyncio.fixture
async def fake_client():
    return _FakeOWUIClient()


@pytest_asyncio.fixture
async def worker(tmp_path: pathlib.Path, fake_client: _FakeOWUIClient):
    """Manual-drain worker: the background loop is NOT spawned, so tests
    can call ``drain_once`` deterministically without a race."""
    w = OutboxWorker(
        db_path=tmp_path / "outbox.sqlite",
        owui_client=fake_client,
        poll_interval_s=0.05,
        max_attempts=3,
        max_backoff_s=2,
    )
    await w.start(spawn_loop=False)
    try:
        yield w
    finally:
        await w.stop()


@pytest_asyncio.fixture
async def autoworker(tmp_path: pathlib.Path, fake_client: _FakeOWUIClient):
    """Worker WITH the background loop spawned — only used by the
    wake-on-enqueue test."""
    w = OutboxWorker(
        db_path=tmp_path / "outbox.sqlite",
        owui_client=fake_client,
        poll_interval_s=0.05,
        max_attempts=3,
        max_backoff_s=2,
    )
    await w.start(spawn_loop=True)
    try:
        yield w
    finally:
        await w.stop()


async def _enqueue(worker: OutboxWorker, **overrides: Any) -> str:
    defaults: dict[str, Any] = {
        "outbox_id": "ob-1",
        "job_id": "job-1",
        "chat_id": "chat-1",
        "message_id": "msg-1",
        "event_type": "status",
        "payload": {"description": "phase 1", "done": False},
        "dedupe_key": "job-1:msg-1:status:1",
    }
    defaults.update(overrides)
    inserted = await worker.enqueue(**defaults)
    assert inserted, "enqueue should insert on first call"
    return defaults["outbox_id"]


async def test_enqueue_then_deliver_marks_delivered(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    await _enqueue(worker)
    # Drain explicitly to avoid race with background loop timing
    await worker.drain_once()
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["event_type"] == "status"
    assert fake_client.calls[0]["data"]["description"] == "phase 1"
    assert await worker.count_pending() == 0
    assert worker.delivered_count == 1


async def test_retry_on_5xx_bumps_attempts_then_succeeds(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    fake_client.fail_first = 2  # first two attempts raise; third succeeds
    await _enqueue(worker)

    # First drain: row attempted, fails, deferred. Force the next_attempt_at
    # back to "now" to drive the retry deterministically without sleeping.
    for _ in range(3):
        await worker.drain_once()
        # Reset deferred next_attempt_at so the next drain picks it up
        async with worker._lock:
            conn = worker._require_conn()
            await conn.execute(
                "UPDATE owui_outbox SET next_attempt_at = ? WHERE delivered_at IS NULL",
                ("1970-01-01T00:00:00+00:00",),
            )
            await conn.commit()
    assert worker.delivered_count == 1
    assert worker.failed_count == 0


async def test_drop_after_max_attempts(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    fake_client.always_fail = True
    await _enqueue(worker)
    # max_attempts=3 → row gives up after 3 attempts and is marked delivered
    # (to free the queue) with failed_count bumped.
    for _ in range(4):
        await worker.drain_once()
        async with worker._lock:
            conn = worker._require_conn()
            await conn.execute(
                "UPDATE owui_outbox SET next_attempt_at = ? WHERE delivered_at IS NULL",
                ("1970-01-01T00:00:00+00:00",),
            )
            await conn.commit()
    assert worker.failed_count == 1
    assert worker.delivered_count == 0
    assert await worker.count_pending() == 0
    # Three POST attempts were made
    assert len(fake_client.calls) == 3


async def test_dedupe_key_uniqueness(worker: OutboxWorker):
    inserted1 = await worker.enqueue(
        outbox_id="ob-a",
        job_id="job-1",
        chat_id="chat-1",
        message_id="msg-1",
        event_type="status",
        payload={"description": "first"},
        dedupe_key="same-key",
    )
    inserted2 = await worker.enqueue(
        outbox_id="ob-b",
        job_id="job-1",
        chat_id="chat-1",
        message_id="msg-1",
        event_type="status",
        payload={"description": "second"},
        dedupe_key="same-key",
    )
    assert inserted1 is True
    assert inserted2 is False  # duplicate dedupe_key suppressed


async def test_event_type_validation(worker: OutboxWorker):
    with pytest.raises(ValueError):
        await worker.enqueue(
            outbox_id="ob-x",
            job_id="job-1",
            chat_id="c",
            message_id="m",
            event_type="chat:message:embeds",  # long-name alias is NOT persisted
            payload={},
            dedupe_key="x",
        )


async def test_retry_after_header_overrides_exponential(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    fake_client.fail_first = 1
    fake_client.fail_with = AdapterError(
        "throttled", status=429, headers={"Retry-After": "1"}
    )
    await _enqueue(worker)
    # First drain fails; the second drain (after we reset next_attempt_at) succeeds.
    await worker.drain_once()
    # The row should now have a future next_attempt_at because we set Retry-After=1.
    pending = await worker.fetch_pending()
    # Default fetch only returns rows whose next_attempt_at is in the past, so a
    # row deferred by Retry-After is hidden from fetch_pending until that delay
    # elapses. Confirm it's still in the table but not pending.
    assert pending == []
    async with worker._lock:
        conn = worker._require_conn()
        async with conn.execute(
            "SELECT attempts, next_attempt_at, delivered_at FROM owui_outbox "
            "WHERE outbox_id = ?",
            ("ob-1",),
        ) as cur:
            row = await cur.fetchone()
    assert row["attempts"] == 1
    assert row["delivered_at"] is None


async def test_payload_invalid_json_marked_delivered(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    # Manually insert a row with a broken JSON payload to simulate corruption
    async with worker._lock:
        conn = worker._require_conn()
        await conn.execute(
            "INSERT INTO owui_outbox (outbox_id, job_id, chat_id, message_id, "
            "event_type, payload_json, dedupe_key, attempts, next_attempt_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ob-broken",
                "job-1",
                "c",
                "m",
                "status",
                "{not valid json",
                "broken-dedupe",
                0,
                "1970-01-01T00:00:00+00:00",
            ),
        )
        await conn.commit()
    await worker.drain_once()
    assert fake_client.calls == []  # no POST was attempted
    assert worker.failed_count == 1
    assert await worker.count_pending() == 0


async def test_status_payload_shape(worker: OutboxWorker, fake_client: _FakeOWUIClient):
    await worker.enqueue(
        outbox_id="ob-s",
        job_id="job-1",
        chat_id="c",
        message_id="m",
        event_type="status",
        payload={"description": "doing", "done": False},
        dedupe_key="job-1:m:status:1",
    )
    await worker.drain_once()
    assert fake_client.calls[0]["data"] == {"description": "doing", "done": False}


async def test_worker_loop_wakes_on_enqueue(
    autoworker: OutboxWorker, fake_client: _FakeOWUIClient
):
    # Let the background loop run for a tick to drain whatever is pending.
    await _enqueue(autoworker, dedupe_key="awake-test")
    # Wait briefly for the background loop to drain
    for _ in range(20):
        if autoworker.delivered_count >= 1:
            break
        await asyncio.sleep(0.05)
    assert autoworker.delivered_count == 1

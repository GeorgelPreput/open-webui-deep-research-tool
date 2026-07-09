"""Tests for the writeback OutboxWorker.

The worker drains rows that translate engine events to OWUI's per-message
``/event`` endpoint. These tests use a fake OWUIClient that records calls
and can be flipped between success/failure modes; the real network path
isn't exercised here (that lives in the writeback e2e test).
"""
from __future__ import annotations

import asyncio
import pathlib
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import pytest_asyncio

from deep_research.adapter.client import AdapterError
from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
)
from deep_research.entrypoints.openapi_tool.outbox import (
    OutboxRow,
    OutboxStatus,
    OutboxWorker,
)
from deep_research.entrypoints.openapi_tool.runner import JobRunner
from deep_research.progress.events import CitationEvent


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
    async with worker._lock:
        conn = worker._require_conn()
        async with conn.execute(
            "SELECT status, last_error FROM owui_outbox WHERE outbox_id = ?",
            ("ob-1",),
        ) as cur:
            row = await cur.fetchone()
    assert row["status"] == "abandoned"
    assert row["last_error"] is not None


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
    # worker fixture: max_backoff_s=2, max_retry_after_s defaults to 600. Use
    # a Retry-After value larger than max_backoff_s so the test proves the
    # cap on Retry-After was removed (when it was clamped to max_backoff_s
    # the resulting delta would have been ~2s, not ~5s).
    fake_client.fail_first = 1
    fake_client.fail_with = AdapterError(
        "throttled", status=429, headers={"Retry-After": "5"}
    )
    await _enqueue(worker)
    before = datetime.now(UTC)
    await worker.drain_once()
    # Row deferred; not pending until Retry-After elapses.
    assert await worker.fetch_pending() == []
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
    next_at = datetime.fromisoformat(row["next_attempt_at"])
    delta = (next_at - before).total_seconds()
    # 1s slack for worker's own elapsed time + ISO-second truncation.
    assert delta >= 4.0, f"Retry-After=5 clamped: delta={delta:.2f}s"


async def test_retry_after_value_capped_at_max_retry_after_s(
    tmp_path: pathlib.Path, fake_client: _FakeOWUIClient
):
    """A misbehaving upstream returning Retry-After: 86400 is clamped to the
    operator-tunable ceiling, not allowed to stall the queue."""
    w = OutboxWorker(
        db_path=tmp_path / "outbox.sqlite",
        owui_client=fake_client,
        poll_interval_s=0.05,
        max_attempts=3,
        max_backoff_s=2,
        max_retry_after_s=2,
    )
    await w.start(spawn_loop=False)
    try:
        fake_client.fail_first = 1
        fake_client.fail_with = AdapterError(
            "throttled", status=429, headers={"Retry-After": "86400"}
        )
        await _enqueue(w)
        before = datetime.now(UTC)
        await w.drain_once()
        async with w._lock:
            conn = w._require_conn()
            async with conn.execute(
                "SELECT next_attempt_at FROM owui_outbox WHERE outbox_id = ?",
                ("ob-1",),
            ) as cur:
                row = await cur.fetchone()
        next_at = datetime.fromisoformat(row["next_attempt_at"])
        delta = (next_at - before).total_seconds()
        assert delta <= 3.0, f"max_retry_after_s ceiling not honoured: {delta:.2f}s"
    finally:
        await w.stop()


async def test_retry_after_honoured_for_httpx_http_status_error(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    """Step 2's broader exception-type branch: Retry-After must be read from
    httpx.HTTPStatusError too, not only AdapterError."""
    request = httpx.Request("POST", "http://owui.test/event")
    response = httpx.Response(
        429, headers={"Retry-After": "5"}, request=request
    )
    fake_client.fail_first = 1
    fake_client.fail_with = httpx.HTTPStatusError(
        "throttled", request=request, response=response
    )
    await _enqueue(worker)
    before = datetime.now(UTC)
    await worker.drain_once()
    async with worker._lock:
        conn = worker._require_conn()
        async with conn.execute(
            "SELECT next_attempt_at FROM owui_outbox WHERE outbox_id = ?",
            ("ob-1",),
        ) as cur:
            row = await cur.fetchone()
    next_at = datetime.fromisoformat(row["next_attempt_at"])
    delta = (next_at - before).total_seconds()
    assert delta >= 4.0, f"httpx HTTPStatusError Retry-After ignored: delta={delta:.2f}s"


async def test_payload_invalid_json_marked_abandoned(
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
    async with worker._lock:
        conn = worker._require_conn()
        async with conn.execute(
            "SELECT status, last_error FROM owui_outbox WHERE outbox_id = ?",
            ("ob-broken",),
        ) as cur:
            row = await cur.fetchone()
    assert row["status"] == "abandoned"
    assert row["last_error"] is not None


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


async def test_source_payload_round_trip(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    """A 'source' row preserves its nested payload field-for-field through
    the outbox to the writeback client. The CitationEvent → outbox mapping
    builds payloads with this shape (type/source/document/metadata stacked
    inside `data`); a regression that flattened or reshaped it would still
    pass the existing status-shaped tests, so we exercise it explicitly here.
    """
    payload = {
        "type": "external",
        "source": {"type": "external", "name": "Example Title"},
        "document": ["A short excerpt from the source."],
        "metadata": [{"source": "https://example.com/a"}],
    }
    await worker.enqueue(
        outbox_id="ob-src",
        job_id="job-1",
        chat_id="c",
        message_id="m",
        event_type="source",
        payload=payload,
        dedupe_key="job-1:m:source:https://example.com/a",
    )
    await worker.drain_once()
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["event_type"] == "source"
    assert call["data"] == payload


async def test_citation_event_mapped_to_source_row(
    worker: OutboxWorker,
    fake_client: _FakeOWUIClient,
    tmp_path: pathlib.Path,
):
    """The CitationEvent → outbox mapping in ``JobRunner._event_to_outbox``
    sets ``event_type='source'`` (NOT ``'citation'`` — both are valid OWUI
    aliases; the runner pins on ``'source'``), routes to the record's
    chat/message, and builds ``data`` from ``CitationEvent.to_dict()['data']``.

    Complements ``test_source_payload_round_trip`` (which proves only the
    queue echoes a hand-built payload) by pinning the *mapping* itself: a
    regression in ``CitationEvent.to_dict`` or the runner's CitationEvent
    branch would surface here even if the queue round-trip still passed.
    The JobStore uses a sibling sqlite file so it never contends with the
    worker's ``outbox.sqlite``.
    """
    store = JobStore(tmp_path / "jobs.sqlite")
    await store.start()
    try:
        record = JobRecord(
            job_id="ct-map-1",
            user_id="u",
            user_name="U",
            conversation_id="conv-ct",
            chat_id="chat-ct",
            target_message_id="msg-ct",
            phase=JobPhase.RESEARCHING,
            prompt="p",
            history_json="[]",
            revision=0,
            view_token_hash="0" * 64,
        )
        await store.create(record)

        # No Coordinator needed — we call _event_to_outbox directly.
        runner = JobRunner(
            coord=None,  # type: ignore[arg-type]
            store=store,
            outbox=worker,
            public_base_url="",
        )

        event = CitationEvent(
            url="https://example.com/ct",
            title="CT Title",
            snippet="CT excerpt.",
        )
        await runner._event_to_outbox(record.job_id, event)
        await worker.drain_once()

        assert len(fake_client.calls) == 1
        call = fake_client.calls[0]
        assert call["event_type"] == "source"
        assert call["chat_id"] == "chat-ct"
        assert call["message_id"] == "msg-ct"
        assert call["data"] == {
            "type": "external",
            "source": {"type": "external", "name": "CT Title"},
            "document": ["CT excerpt."],
            "metadata": [{"source": "https://example.com/ct"}],
        }
    finally:
        await store.close()


async def test_count_by_status_distinguishes_outcomes(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    """Delivered and abandoned rows are individually countable.

    Drain ordering is deterministic: enqueue+drain `ob-ok` first while
    `always_fail=False`, then flip `always_fail=True` and exhaust `ob-bad`.
    """
    await worker.enqueue(
        outbox_id="ob-ok",
        job_id="job-1",
        chat_id="c",
        message_id="m",
        event_type="status",
        payload={"description": "ok", "done": False},
        dedupe_key="job-1:m:ok",
    )
    await worker.drain_once()
    assert worker.delivered_count == 1

    fake_client.always_fail = True
    await worker.enqueue(
        outbox_id="ob-bad",
        job_id="job-1",
        chat_id="c",
        message_id="m",
        event_type="status",
        payload={"description": "bad", "done": False},
        dedupe_key="job-1:m:bad",
    )
    # max_attempts=3 (set in worker fixture); drain 3× resetting
    # next_attempt_at between iterations to land ob-bad as abandoned.
    for _ in range(4):
        await worker.drain_once()
        async with worker._lock:
            conn = worker._require_conn()
            await conn.execute(
                "UPDATE owui_outbox SET next_attempt_at = ? "
                "WHERE delivered_at IS NULL",
                ("1970-01-01T00:00:00+00:00",),
            )
            await conn.commit()

    counts = await worker.count_by_status()
    assert counts.get("delivered", 0) == 1
    assert counts.get("abandoned", 0) == 1
    # No pending or retrying rows should remain.
    assert counts.get("pending", 0) == 0
    assert counts.get("retrying", 0) == 0


async def test_from_db_row_works_without_row_factory(
    worker: OutboxWorker, fake_client: _FakeOWUIClient
):
    """A caller using a plain tuple-returning cursor can still build an
    OutboxRow; pins the foreign-caller-safety contract for from_db_row.
    """
    await _enqueue(worker)
    async with worker._lock:
        conn = worker._require_conn()
        original_factory = conn.row_factory
        conn.row_factory = None
        try:
            async with conn.execute(
                "SELECT outbox_id, job_id, chat_id, message_id, event_type, "
                "payload_json, dedupe_key, attempts, next_attempt_at, "
                "delivered_at, status, last_error FROM owui_outbox "
                "WHERE outbox_id = ?",
                ("ob-1",),
            ) as cur:
                tuple_row = await cur.fetchone()
        finally:
            conn.row_factory = original_factory
    parsed = OutboxRow.from_db_row(tuple_row)
    assert parsed.outbox_id == "ob-1"
    assert parsed.status == OutboxStatus.PENDING
    assert parsed.attempts == 0


async def test_migration_back_fills_status_on_existing_rows(
    tmp_path: pathlib.Path, fake_client: _FakeOWUIClient
):
    """An older DB without status/last_error columns gets back-filled on
    start(): a row with delivered_at set becomes 'delivered'; a row with
    delivered_at NULL stays 'pending' (its default).
    """
    import aiosqlite as _aiosqlite

    db_path = tmp_path / "legacy.sqlite"
    legacy_schema = """
    CREATE TABLE owui_outbox (
        outbox_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        chat_id TEXT NOT NULL,
        message_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        dedupe_key TEXT NOT NULL UNIQUE,
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TEXT NOT NULL,
        delivered_at TEXT
    );
    """
    async with _aiosqlite.connect(db_path) as conn:
        await conn.executescript(legacy_schema)
        await conn.execute(
            "INSERT INTO owui_outbox (outbox_id, job_id, chat_id, message_id, "
            "event_type, payload_json, dedupe_key, attempts, next_attempt_at, "
            "delivered_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "legacy-delivered",
                "job-1",
                "c",
                "m",
                "status",
                "{}",
                "dk-1",
                1,
                "1970-01-01T00:00:00+00:00",
                "1970-01-01T00:00:01+00:00",
            ),
        )
        await conn.execute(
            "INSERT INTO owui_outbox (outbox_id, job_id, chat_id, message_id, "
            "event_type, payload_json, dedupe_key, attempts, next_attempt_at, "
            "delivered_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "legacy-pending",
                "job-1",
                "c",
                "m",
                "status",
                "{}",
                "dk-2",
                0,
                "1970-01-01T00:00:00+00:00",
                None,
            ),
        )
        await conn.commit()

    w = OutboxWorker(db_path=db_path, owui_client=fake_client)
    await w.start(spawn_loop=False)
    try:
        counts = await w.count_by_status()
    finally:
        await w.stop()
    assert counts.get("delivered", 0) == 1
    assert counts.get("pending", 0) == 1

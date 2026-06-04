"""Durable writeback queue for OWUI per-message ``/event`` posts.

The OpenAPI Tool Server lands chat content (topic list, citations, final
report, iframe) into the assistant message by POSTing to OWUI's
``/api/v1/chats/{chat_id}/messages/{message_id}/event`` endpoint with
one of the documented short event types. Unlike chat-update endpoints,
``/event`` admin-bypasses chat ownership — so a server-side admin token
can write to chats owned by other users. This is the channel that makes
the Phase 2 "topic list lands in chat content directly" UX work.

Doing those POSTs in the engine's sink would block the event loop on
every emit and tie engine state to OWUI's availability. Instead, the
runner translates each engine event to an ``OutboxRow``, persists it
to sqlite, and the worker loop drains the table on its own cadence with
exponential backoff and retry-after honoring. Engine progress and OWUI
writeback are fully decoupled.

The table lives in the same sqlite file as ``research_jobs`` so a
single ``DR_DATA_DIR`` volume gives durability for both.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiosqlite

from deep_research.adapter.client import PERSISTED_EVENT_TYPES, AdapterError
from deep_research.core.errors import extract_retry_after_seconds

if TYPE_CHECKING:
    from deep_research.adapter.client import OWUIClient

logger = logging.getLogger("deep_research.entrypoints.openapi.outbox")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso_from(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


@dataclass
class OutboxRow:
    outbox_id: str
    job_id: str
    chat_id: str
    message_id: str
    event_type: str
    payload_json: str
    dedupe_key: str
    attempts: int = 0
    next_attempt_at: str = field(default_factory=_now_iso)
    delivered_at: str | None = None

    @classmethod
    def from_db_row(cls, row: aiosqlite.Row) -> OutboxRow:
        return cls(
            outbox_id=row["outbox_id"],
            job_id=row["job_id"],
            chat_id=row["chat_id"],
            message_id=row["message_id"],
            event_type=row["event_type"],
            payload_json=row["payload_json"],
            dedupe_key=row["dedupe_key"],
            attempts=row["attempts"],
            next_attempt_at=row["next_attempt_at"],
            delivered_at=row["delivered_at"],
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS owui_outbox (
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
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON owui_outbox(delivered_at, next_attempt_at);
"""

_INSERT_SQL = """
INSERT INTO owui_outbox (
    outbox_id, job_id, chat_id, message_id, event_type, payload_json,
    dedupe_key, attempts, next_attempt_at, delivered_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_IGNORE_SQL = (
    "INSERT OR IGNORE INTO owui_outbox ("
    "outbox_id, job_id, chat_id, message_id, event_type, payload_json,"
    " dedupe_key, attempts, next_attempt_at, delivered_at"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class OutboxWorker:
    """Background worker draining outbox rows to OWUI's ``/event`` endpoint.

    One worker per process; the worker takes a writeback OWUIClient at
    construction time (instantiated with a static admin token) and a
    sqlite path. ``start()`` opens the connection, creates the table,
    and spawns the drain loop. ``stop()`` cancels the loop and closes
    the connection.

    Retry policy: exponential backoff base 1s, capped at
    ``max_backoff_s``. ``Retry-After`` from an upstream
    :class:`AdapterError` replaces the exponential delay. After
    ``max_attempts`` we log an error and mark the row delivered (giving
    up), so a permanently broken target message can't deadlock the
    queue.
    """

    def __init__(
        self,
        *,
        db_path: pathlib.Path,
        owui_client: OWUIClient,
        poll_interval_s: float = 0.25,
        max_attempts: int = 10,
        max_backoff_s: int = 60,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._db_path = db_path
        self._client = owui_client
        self._poll_interval = max(0.05, float(poll_interval_s))
        self._max_attempts = max(1, int(max_attempts))
        self._max_backoff = max(1, int(max_backoff_s))
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._stopped = False
        self._delivered_count = 0
        self._failed_count = 0

    @property
    def delivered_count(self) -> int:
        """Test/diagnostic accessor — successful POST count this process."""
        return self._delivered_count

    @property
    def failed_count(self) -> int:
        """Test/diagnostic accessor — rows that exhausted max_attempts."""
        return self._failed_count

    async def start(self, *, spawn_loop: bool = True) -> None:
        """Open the sqlite connection and (by default) spawn the drain loop.

        ``spawn_loop=False`` opens the DB but does not start the background
        worker — tests that exercise ``drain_once`` deterministically use
        this mode to avoid racing with the loop.
        """
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute(
            f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}"
        )
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()
        self._stopped = False
        if spawn_loop:
            self._task = asyncio.create_task(self._worker_loop())
        logger.info(
            "OutboxWorker started db=%s loop=%s",
            self._db_path,
            "spawned" if spawn_loop else "manual",
        )

    async def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
        logger.info(
            "OutboxWorker stopped delivered=%d failed=%d",
            self._delivered_count,
            self._failed_count,
        )

    async def enqueue(
        self,
        *,
        outbox_id: str,
        job_id: str,
        chat_id: str,
        message_id: str,
        event_type: str,
        payload: dict[str, Any],
        dedupe_key: str,
    ) -> bool:
        """Insert a row idempotently. Returns True if inserted, False if
        the dedupe_key was already present.

        ``event_type`` is validated up-front because OWUI silently drops
        long-name aliases; catching that here is faster than chasing a
        ghost POST that succeeded but didn't persist.
        """
        if event_type not in PERSISTED_EVENT_TYPES:
            raise ValueError(
                f"event_type {event_type!r} is not persisted by OWUI; "
                f"allowed: {sorted(PERSISTED_EVENT_TYPES)}"
            )
        payload_json = json.dumps(payload, ensure_ascii=False)
        async with self._lock:
            conn = self._require_conn()
            cursor = await conn.execute(
                _INSERT_IGNORE_SQL,
                (
                    outbox_id,
                    job_id,
                    chat_id,
                    message_id,
                    event_type,
                    payload_json,
                    dedupe_key,
                    0,
                    _now_iso(),
                    None,
                ),
            )
            await conn.commit()
            inserted = cursor.rowcount == 1
        if inserted:
            self._wake.set()
        return inserted

    async def fetch_pending(self, limit: int = 32) -> list[OutboxRow]:
        """Return undelivered rows whose ``next_attempt_at`` is in the past.

        Secondary sort is sqlite's ``rowid`` (monotonic per insert) so
        rows enqueued in the same second still deliver in insertion
        order. UUIDs as ``outbox_id`` make that secondary key useless
        for ordering.

        Public for tests; the worker loop uses it internally.
        """
        async with self._lock:
            conn = self._require_conn()
            sql = (
                "SELECT * FROM owui_outbox "
                "WHERE delivered_at IS NULL AND next_attempt_at <= ? "
                "ORDER BY next_attempt_at ASC, rowid ASC "
                "LIMIT ?"
            )
            async with conn.execute(sql, (_now_iso(), limit)) as cur:
                rows = await cur.fetchall()
        return [OutboxRow.from_db_row(r) for r in rows]

    async def count_pending(self) -> int:
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(
                "SELECT COUNT(*) AS c FROM owui_outbox WHERE delivered_at IS NULL"
            ) as cur:
                row = await cur.fetchone()
        return int(row["c"]) if row is not None else 0

    async def drain_once(self, *, limit: int = 32) -> int:
        """Process up to ``limit`` pending rows. Returns the number of
        rows attempted. Public for tests; the worker loop uses it.
        """
        rows = await self.fetch_pending(limit=limit)
        for row in rows:
            await self._deliver(row)
        return len(rows)

    # -------------------------------------------------------------- internals

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("OutboxWorker.start() was not called")
        return self._conn

    async def _worker_loop(self) -> None:
        while not self._stopped:
            try:
                attempted = await self.drain_once()
            except Exception:
                logger.exception("Outbox drain failed")
                attempted = 0
            if attempted == 0:
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self._poll_interval
                    )

    async def _deliver(self, row: OutboxRow) -> None:
        try:
            payload = json.loads(row.payload_json)
        except (TypeError, ValueError):
            logger.error(
                "Outbox row %s has invalid payload JSON; marking delivered to skip",
                row.outbox_id,
            )
            await self._mark_delivered(row.outbox_id)
            self._failed_count += 1
            return

        try:
            await self._client.post_message_event(
                row.chat_id, row.message_id, row.event_type, payload
            )
            await self._mark_delivered(row.outbox_id)
            self._delivered_count += 1
            logger.debug(
                "Outbox delivered job=%s msg=%s type=%s attempt=%d",
                row.job_id,
                row.message_id,
                row.event_type,
                row.attempts + 1,
            )
        except Exception as exc:
            new_attempts = row.attempts + 1
            if new_attempts >= self._max_attempts:
                logger.error(
                    "Outbox row %s gave up after %d attempts (job=%s type=%s): %s",
                    row.outbox_id,
                    new_attempts,
                    row.job_id,
                    row.event_type,
                    exc,
                )
                await self._mark_delivered(row.outbox_id)
                self._failed_count += 1
                return
            delay = self._compute_backoff(exc, new_attempts)
            next_at = _iso_from(_now() + timedelta(seconds=delay))
            logger.warning(
                "Outbox row %s deferred attempt=%d/%d delay=%.2fs reason=%s",
                row.outbox_id,
                new_attempts,
                self._max_attempts,
                delay,
                type(exc).__name__,
            )
            async with self._lock:
                conn = self._require_conn()
                await conn.execute(
                    "UPDATE owui_outbox SET attempts = ?, next_attempt_at = ? "
                    "WHERE outbox_id = ?",
                    (new_attempts, next_at, row.outbox_id),
                )
                await conn.commit()

    async def _mark_delivered(self, outbox_id: str) -> None:
        async with self._lock:
            conn = self._require_conn()
            await conn.execute(
                "UPDATE owui_outbox SET delivered_at = ? WHERE outbox_id = ?",
                (_now_iso(), outbox_id),
            )
            await conn.commit()

    def _compute_backoff(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, AdapterError):
            retry_after = extract_retry_after_seconds(exc)
            if retry_after is not None and retry_after > 0:
                return float(min(retry_after, self._max_backoff))
        # Exponential: 1s, 2s, 4s, 8s, ..., capped at max_backoff_s.
        delay = float(min(2 ** (attempt - 1), self._max_backoff))
        return delay

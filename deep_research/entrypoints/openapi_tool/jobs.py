"""Durable job store for the OpenAPI Tool Server runtime.

`JobStore` persists a `JobRecord` per /research_jobs invocation so the
server can survive a restart-mid-run (the record can be marked
FAILED on next boot) and so the live-view iframe can locate a job by
id without holding any process-local state.

The store is sqlite-backed via `aiosqlite`; all writes serialise
through an `asyncio.Lock` to keep `revision` bumps monotonic and to
match the single-writer pattern sqlite likes. Reads also take the
lock for simplicity — we expect O(10) concurrent jobs per server, not
O(10K).
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import aiosqlite

logger = logging.getLogger("deep_research.entrypoints.openapi.jobs")


class JobPhase(StrEnum):
    QUEUED = "queued"
    BOOTSTRAPPING = "bootstrapping"
    OUTLINING = "outlining"
    AWAITING_OUTLINE_FEEDBACK = "awaiting_outline_feedback"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_PHASES: frozenset[JobPhase] = frozenset(
    {JobPhase.COMPLETED, JobPhase.FAILED, JobPhase.CANCELLED}
)

# Pre-sorted phase values for SQL `IN (...)` callers. `frozenset` iteration
# order is unspecified and varies between processes, so any site that bakes
# the order into a SQL string (placeholder layout, parameter tuple) must use
# this constant rather than iterating `TERMINAL_PHASES` directly. Membership
# tests (`x in TERMINAL_PHASES`) are order-independent and keep using the set.
_TERMINAL_PHASE_VALUES: tuple[str, ...] = tuple(
    sorted(p.value for p in TERMINAL_PHASES)
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class JobRecord:
    job_id: str
    user_id: str
    user_name: str
    conversation_id: str
    chat_id: str | None
    target_message_id: str | None
    phase: JobPhase
    prompt: str
    history_json: str
    revision: int
    view_token_hash: str
    outline_json: str | None = None
    selection_json: str | None = None
    report_markdown: str | None = None
    error_text: str | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None

    def to_db_row(self) -> tuple[Any, ...]:
        return (
            self.job_id,
            self.user_id,
            self.user_name,
            self.conversation_id,
            self.chat_id,
            self.target_message_id,
            self.phase.value,
            self.prompt,
            self.history_json,
            self.revision,
            self.view_token_hash,
            self.outline_json,
            self.selection_json,
            self.report_markdown,
            self.error_text,
            self.created_at,
            self.updated_at,
            self.completed_at,
        )

    @classmethod
    def from_db_row(cls, row: aiosqlite.Row) -> JobRecord:
        return cls(
            job_id=row["job_id"],
            user_id=row["user_id"],
            user_name=row["user_name"],
            conversation_id=row["conversation_id"],
            chat_id=row["chat_id"],
            target_message_id=row["target_message_id"],
            phase=JobPhase(row["phase"]),
            prompt=row["prompt"],
            history_json=row["history_json"],
            revision=row["revision"],
            view_token_hash=row["view_token_hash"],
            outline_json=row["outline_json"],
            selection_json=row["selection_json"],
            report_markdown=row["report_markdown"],
            error_text=row["error_text"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def model_dump(self) -> dict[str, Any]:
        out = asdict(self)
        out["phase"] = self.phase.value
        return out


# Base schema — table + non-unique indexes. Created unconditionally on
# every start (CREATE ... IF NOT EXISTS). Must complete before
# _PRE_MIGRATION_DEDUPE_SQL runs so the table exists on a fresh
# database, and before _UNIQUE_INDEX_SQL runs so the dedupe query has
# rows to operate on.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    chat_id TEXT,
    target_message_id TEXT,
    phase TEXT NOT NULL,
    prompt TEXT NOT NULL,
    history_json TEXT NOT NULL DEFAULT '[]',
    revision INTEGER NOT NULL DEFAULT 0,
    view_token_hash TEXT NOT NULL,
    outline_json TEXT,
    selection_json TEXT,
    report_markdown TEXT,
    error_text TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_chat_active ON research_jobs(chat_id, phase);
CREATE INDEX IF NOT EXISTS idx_jobs_completed_at ON research_jobs(completed_at);
"""

# Pre-migration cleanup: any existing duplicate non-terminal rows for
# the same chat_id would block the UNIQUE partial index below. Keep
# the most recent row per chat (ORDER BY created_at DESC) and mark
# older duplicates FAILED so the index can be created. Safe to run on
# every start — no-op when no duplicates exist.
_PRE_MIGRATION_DEDUPE_SQL = """
UPDATE research_jobs
SET phase = 'failed',
    error_text = COALESCE(error_text, 'superseded_by_concurrent_start_pre_migration'),
    completed_at = COALESCE(completed_at, ?),
    updated_at = ?
WHERE job_id IN (
    SELECT job_id FROM (
        SELECT job_id,
               ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) AS rn
        FROM research_jobs
        WHERE chat_id IS NOT NULL
          AND phase NOT IN ('completed', 'failed', 'cancelled')
    )
    WHERE rn > 1
)
"""

# Enforces "one active job per chat" at the sqlite level. Defence in
# depth against the in-process per-job lock in JobRunner. Runs AFTER
# _PRE_MIGRATION_DEDUPE_SQL so existing duplicates have been resolved.
_UNIQUE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_chat_active_unique
    ON research_jobs(chat_id)
    WHERE chat_id IS NOT NULL
      AND phase NOT IN ('completed', 'failed', 'cancelled')
"""

_INSERT_SQL = """
INSERT INTO research_jobs (
    job_id, user_id, user_name, conversation_id, chat_id, target_message_id,
    phase, prompt, history_json, revision, view_token_hash,
    outline_json, selection_json, report_markdown, error_text,
    created_at, updated_at, completed_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATABLE_FIELDS = {
    "phase",
    "target_message_id",
    "outline_json",
    "selection_json",
    "report_markdown",
    "error_text",
    "completed_at",
    "prompt",
    "history_json",
}


class JobStore:
    def __init__(self, db_path: pathlib.Path, *, busy_timeout_ms: int = 5000) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._conn is not None:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(
                self._db_path, timeout=self._busy_timeout_ms / 1000.0
            )
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.executescript(_SCHEMA)
            now = _now_iso()
            await self._conn.execute(_PRE_MIGRATION_DEDUPE_SQL, (now, now))
            await self._conn.execute(_UNIQUE_INDEX_SQL)
            await self._conn.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("JobStore.start() was not called")
        return self._conn

    async def create(self, record: JobRecord) -> None:
        async with self._lock:
            conn = self._require_conn()
            await conn.execute(_INSERT_SQL, record.to_db_row())
            await conn.commit()

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        return JobRecord.from_db_row(row)

    async def update(self, job_id: str, **fields: Any) -> JobRecord:
        """Bump revision automatically and persist updates. Returns the new record.

        Phase values may be passed as either the JobPhase enum or its
        string value. updated_at is overwritten with the current time.
        """
        if not fields:
            current = await self.get(job_id)
            if current is None:
                raise KeyError(job_id)
            return current

        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unknown updatable fields: {sorted(unknown)}")

        if "phase" in fields and isinstance(fields["phase"], JobPhase):
            fields["phase"] = fields["phase"].value

        async with self._lock:
            conn = self._require_conn()
            assignments = [f"{k} = ?" for k in fields]
            assignments.append("revision = revision + 1")
            assignments.append("updated_at = ?")
            params: list[Any] = list(fields.values())
            params.append(_now_iso())
            params.append(job_id)
            sql = (
                f"UPDATE research_jobs SET {', '.join(assignments)} "
                f"WHERE job_id = ?"
            )
            await conn.execute(sql, params)
            await conn.commit()
            async with conn.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobRecord.from_db_row(row)

    async def rebind_target_message(self, job_id: str, new_message_id: str) -> JobRecord:
        """Repoint a job at a new tool-call message and bump revision."""
        return await self.update(job_id, target_message_id=new_message_id)

    async def bump_revision(self, job_id: str) -> JobRecord:
        """Bump revision + updated_at without touching other fields.

        Called from the runner's event sink on each engine-emitted
        EmbedEvent so the live-view iframe's self-polling loop sees a
        revision change and reloads, and so the engine-embeds outbox
        dedupe key (which embeds the revision) stays unique per refresh.
        """
        async with self._lock:
            conn = self._require_conn()
            await conn.execute(
                "UPDATE research_jobs SET revision = revision + 1, "
                "updated_at = ? WHERE job_id = ?",
                (_now_iso(), job_id),
            )
            await conn.commit()
            async with conn.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobRecord.from_db_row(row)

    async def find_active_by_chat(self, chat_id: str) -> JobRecord | None:
        """Return the most-recent non-terminal job for a chat, if any."""
        terminal_values = _TERMINAL_PHASE_VALUES
        placeholders = ",".join("?" for _ in terminal_values)
        sql = (
            "SELECT * FROM research_jobs "
            f"WHERE chat_id = ? AND phase NOT IN ({placeholders}) "
            "ORDER BY created_at DESC LIMIT 1"
        )
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(sql, (chat_id, *terminal_values)) as cur:
                row = await cur.fetchone()
        return JobRecord.from_db_row(row) if row is not None else None

    async def list_expired(
        self, completed_retention_s: float, failed_retention_s: float
    ) -> list[str]:
        """Return job_ids whose terminal phase is past its retention window.

        Times are measured against `completed_at`. Cancelled jobs share
        the failed bucket (both treated as undesirable outcomes).
        """
        now = datetime.now(UTC)
        completed_cutoff = (
            now - _seconds_to_timedelta(completed_retention_s)
        ).isoformat(timespec="seconds")
        failed_cutoff = (
            now - _seconds_to_timedelta(failed_retention_s)
        ).isoformat(timespec="seconds")
        sql = (
            "SELECT job_id FROM research_jobs WHERE completed_at IS NOT NULL "
            "AND ("
            "(phase = ? AND completed_at < ?) OR "
            "(phase IN (?, ?) AND completed_at < ?)"
            ")"
        )
        params = (
            JobPhase.COMPLETED.value,
            completed_cutoff,
            JobPhase.FAILED.value,
            JobPhase.CANCELLED.value,
            failed_cutoff,
        )
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
        return [r["job_id"] for r in rows]

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            conn = self._require_conn()
            await conn.execute("DELETE FROM research_jobs WHERE job_id = ?", (job_id,))
            await conn.commit()


def _seconds_to_timedelta(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=float(seconds))


def history_to_json(history: list[Any]) -> str:
    """Best-effort serialise a list of history messages (dict or pydantic)."""
    out: list[dict[str, Any]] = []
    for m in history:
        if hasattr(m, "model_dump"):
            out.append(m.model_dump())
        elif isinstance(m, dict):
            out.append(m)
        else:
            out.append({"role": "user", "content": str(m)})
    return json.dumps(out)

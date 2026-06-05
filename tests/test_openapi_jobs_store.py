"""Tests for the durable JobStore."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from deep_research.entrypoints.openapi_tool.jobs import (
    TERMINAL_PHASES,
    JobPhase,
    JobRecord,
    JobStore,
    _now_iso,
    history_to_json,
)


def _make_record(job_id: str, **overrides) -> JobRecord:
    defaults = {
        "job_id": job_id,
        "user_id": "u1",
        "user_name": "User One",
        "conversation_id": f"conv_{job_id}",
        "chat_id": "chat-abc",
        "target_message_id": "msg-1",
        "phase": JobPhase.QUEUED,
        "prompt": "Summarise post-quantum cryptography in 2026",
        "history_json": "[]",
        "revision": 0,
        "view_token_hash": "0" * 64,
    }
    defaults.update(overrides)
    return JobRecord(**defaults)


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    s = JobStore(tmp_path / "jobs.sqlite")
    await s.start()
    try:
        yield s
    finally:
        await s.close()


async def test_create_and_get_roundtrip(store: JobStore):
    record = _make_record("job-1")
    await store.create(record)

    got = await store.get("job-1")
    assert got is not None
    assert got.job_id == "job-1"
    assert got.phase == JobPhase.QUEUED
    assert got.prompt == record.prompt
    assert got.revision == 0


async def test_get_unknown_returns_none(store: JobStore):
    assert await store.get("nope") is None


async def test_update_bumps_revision(store: JobStore):
    await store.create(_make_record("job-2"))
    updated = await store.update("job-2", phase=JobPhase.OUTLINING)
    assert updated.phase == JobPhase.OUTLINING
    assert updated.revision == 1

    again = await store.update("job-2", report_markdown="hello")
    assert again.revision == 2
    assert again.report_markdown == "hello"


async def test_update_accepts_enum_or_string_phase(store: JobStore):
    await store.create(_make_record("job-3"))
    updated = await store.update("job-3", phase="researching")
    assert updated.phase == JobPhase.RESEARCHING


async def test_update_rejects_unknown_fields(store: JobStore):
    await store.create(_make_record("job-4"))
    with pytest.raises(ValueError):
        await store.update("job-4", revision=99)  # revision is auto-bumped


async def test_rebind_target_message(store: JobStore):
    await store.create(_make_record("job-5", target_message_id="msg-old"))
    updated = await store.rebind_target_message("job-5", "msg-new")
    assert updated.target_message_id == "msg-new"
    assert updated.revision == 1


async def test_find_active_by_chat_returns_none_when_terminal(store: JobStore):
    chat_id = "chat-XYZ"
    rec = _make_record("job-6", chat_id=chat_id, phase=JobPhase.QUEUED)
    await store.create(rec)
    assert (await store.find_active_by_chat(chat_id)).job_id == "job-6"

    await store.update("job-6", phase=JobPhase.COMPLETED, completed_at=_now_iso())
    assert await store.find_active_by_chat(chat_id) is None


async def test_find_active_by_chat_handles_cancelled_and_failed(store: JobStore):
    chat_id = "chat-T"
    await store.create(_make_record("job-cancelled", chat_id=chat_id))
    await store.update("job-cancelled", phase=JobPhase.CANCELLED, completed_at=_now_iso())
    await store.create(_make_record("job-failed", chat_id=chat_id))
    await store.update("job-failed", phase=JobPhase.FAILED, completed_at=_now_iso())
    assert await store.find_active_by_chat(chat_id) is None


async def test_list_expired_respects_phase_retention(store: JobStore, monkeypatch):
    # Two completed jobs, one fresh, one old. completed_retention_s=60.
    now = datetime.now(UTC)
    fresh = (now - timedelta(seconds=5)).isoformat(timespec="seconds")
    old = (now - timedelta(seconds=3600)).isoformat(timespec="seconds")

    await store.create(_make_record("fresh-completed"))
    await store.update(
        "fresh-completed", phase=JobPhase.COMPLETED, completed_at=fresh
    )
    await store.create(_make_record("old-completed"))
    await store.update(
        "old-completed", phase=JobPhase.COMPLETED, completed_at=old
    )

    expired = await store.list_expired(
        completed_retention_s=60.0, failed_retention_s=60.0
    )
    assert "old-completed" in expired
    assert "fresh-completed" not in expired


async def test_list_expired_uses_failed_retention_for_failed_and_cancelled(
    store: JobStore,
):
    now = datetime.now(UTC)
    old = (now - timedelta(seconds=120)).isoformat(timespec="seconds")
    await store.create(_make_record("old-failed"))
    await store.update("old-failed", phase=JobPhase.FAILED, completed_at=old)
    await store.create(_make_record("old-cancelled"))
    await store.update("old-cancelled", phase=JobPhase.CANCELLED, completed_at=old)

    # 60s failed retention → both expired
    expired = await store.list_expired(
        completed_retention_s=10000.0, failed_retention_s=60.0
    )
    assert set(expired) == {"old-failed", "old-cancelled"}


async def test_concurrent_writes_serialise(store: JobStore):
    """All 20 concurrent updates land and revision counts every one."""
    await store.create(_make_record("job-conc"))

    async def bump():
        await store.update("job-conc", error_text="bump")

    await asyncio.gather(*(bump() for _ in range(20)))
    final = await store.get("job-conc")
    assert final is not None
    assert final.revision == 20
    assert final.error_text == "bump"


async def test_delete_removes_record(store: JobStore):
    await store.create(_make_record("job-d"))
    assert await store.get("job-d") is not None
    await store.delete("job-d")
    assert await store.get("job-d") is None


def test_history_to_json_handles_dicts_and_pydantic():
    class _Stub:
        def __init__(self, role, content):
            self._payload = {"role": role, "content": content}

        def model_dump(self):
            return self._payload

    raw = history_to_json([
        {"role": "user", "content": "hi"},
        _Stub("assistant", "hello"),
    ])
    import json as _json
    decoded = _json.loads(raw)
    assert decoded == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_terminal_phases_constant():
    assert JobPhase.COMPLETED in TERMINAL_PHASES
    assert JobPhase.FAILED in TERMINAL_PHASES
    assert JobPhase.CANCELLED in TERMINAL_PHASES
    assert JobPhase.RESEARCHING not in TERMINAL_PHASES


# -------------------------------------------------------- UNIQUE partial index


async def test_unique_partial_index_rejects_duplicate_active_chat(store: JobStore):
    """The UNIQUE partial index on (chat_id) WHERE phase NOT IN terminal
    blocks a second non-terminal row from being inserted for the same
    chat. After the first row goes terminal, another non-terminal row
    for the same chat can be inserted."""
    import aiosqlite as _aiosqlite

    chat_id = "chat-unique"
    await store.create(_make_record("u1", chat_id=chat_id, phase=JobPhase.RESEARCHING))

    with pytest.raises(_aiosqlite.IntegrityError):
        await store.create(_make_record("u2", chat_id=chat_id, phase=JobPhase.QUEUED))

    # Mark the first one terminal; a third insert for the same chat now succeeds.
    await store.update("u1", phase=JobPhase.COMPLETED, completed_at=_now_iso())
    await store.create(_make_record("u3", chat_id=chat_id, phase=JobPhase.QUEUED))
    active = await store.find_active_by_chat(chat_id)
    assert active is not None and active.job_id == "u3"


async def test_unique_partial_index_allows_multiple_null_chat(store: JobStore):
    """NULL chat_id is excluded from the partial index — multiple
    non-terminal rows with NULL chat_id are allowed (out-of-band
    callers)."""
    await store.create(_make_record("n1", chat_id=None, phase=JobPhase.QUEUED))
    await store.create(_make_record("n2", chat_id=None, phase=JobPhase.QUEUED))
    # Both rows survive.
    assert await store.get("n1") is not None
    assert await store.get("n2") is not None


async def test_pre_migration_resolves_duplicate_active_rows(tmp_path):
    """Open a sqlite file with raw aiosqlite, force two duplicate
    non-terminal rows for the same chat in (bypassing the UNIQUE
    index), close, re-open via JobStore.start. The pre-migration query
    should mark the older duplicate FAILED so the index can be created
    cleanly."""
    import aiosqlite as _aiosqlite

    db_path = tmp_path / "premig.sqlite"
    # First pass: open via JobStore so the schema is built.
    store = JobStore(db_path)
    await store.start()
    await store.close()

    # Drop the UNIQUE index, then insert two duplicate non-terminal
    # rows for the same chat with distinct created_at values.
    conn = await _aiosqlite.connect(db_path)
    conn.row_factory = _aiosqlite.Row
    await conn.execute("DROP INDEX IF EXISTS idx_jobs_chat_active_unique")
    older = "2026-01-01T00:00:00+00:00"
    newer = "2026-06-01T00:00:00+00:00"
    insert_sql = (
        "INSERT INTO research_jobs ("
        "job_id, user_id, user_name, conversation_id, chat_id, "
        "target_message_id, phase, prompt, history_json, revision, "
        "view_token_hash, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    common = ("u", "U", "conv", "chat-dup", None, "queued", "p", "[]", 0, "0" * 64)
    await conn.execute(insert_sql, ("dup-old", *common, older, older))
    await conn.execute(insert_sql, ("dup-new", *common, newer, newer))
    await conn.commit()
    await conn.close()

    # Re-open via JobStore — pre-migration runs, then UNIQUE index is
    # re-created.
    store2 = JobStore(db_path)
    await store2.start()
    try:
        old = await store2.get("dup-old")
        new = await store2.get("dup-new")
        assert old is not None and new is not None
        # The older duplicate was marked FAILED with the documented
        # error_text token.
        assert old.phase == JobPhase.FAILED
        assert "pre_migration" in (old.error_text or "")
        # The newer duplicate is preserved as the active one.
        assert new.phase == JobPhase.QUEUED
        # Only one active row remains.
        active = await store2.find_active_by_chat("chat-dup")
        assert active is not None and active.job_id == "dup-new"
    finally:
        await store2.close()


async def test_start_is_idempotent_under_concurrent_calls(tmp_path):
    """Two parallel start() calls on the same JobStore both return
    cleanly without double-connecting."""
    s = JobStore(tmp_path / "idem.sqlite")
    try:
        results = await asyncio.gather(s.start(), s.start(), return_exceptions=True)
        for r in results:
            assert r is None, r
        # Connection is live.
        assert s._conn is not None
        await s.create(_make_record("idem-1", chat_id=None))
        assert await s.get("idem-1") is not None
    finally:
        await s.close()


async def test_close_is_safe_during_start(tmp_path, monkeypatch):
    """close() racing start() doesn't leave the connection in an
    inconsistent state. Patch aiosqlite.connect to delay so we can
    fire close() while start is mid-connect."""
    import aiosqlite as _aiosqlite

    original_connect = _aiosqlite.connect
    started = asyncio.Event()
    proceed = asyncio.Event()

    def slow_connect(*args, **kwargs):
        async def _delayed():
            started.set()
            await proceed.wait()
            return await original_connect(*args, **kwargs)

        # aiosqlite.connect returns a Connection awaitable; wrap it.
        return _delayed()

    monkeypatch.setattr(_aiosqlite, "connect", slow_connect)

    s = JobStore(tmp_path / "race.sqlite")
    start_task = asyncio.create_task(s.start())
    await started.wait()
    # While start() is awaiting connect, fire close(). close() takes
    # the same self._lock as start(), so it will queue until start
    # releases.
    close_task = asyncio.create_task(s.close())
    # Let start finish.
    proceed.set()
    await asyncio.gather(start_task, close_task, return_exceptions=True)

    # Post-race state: serialised by the lock so neither task leaves
    # garbage state. We don't pin which order won (depends on event
    # loop scheduling); just verify close completed without raising.
    assert s._conn is None

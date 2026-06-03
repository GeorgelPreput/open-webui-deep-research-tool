"""Tests for the durable JobStore."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from deep_research.entrypoints.openapi_tool.jobs import (
    JobPhase,
    JobRecord,
    JobStore,
    TERMINAL_PHASES,
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
    now = datetime.now(timezone.utc)
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
    now = datetime.now(timezone.utc)
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

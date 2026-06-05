"""Unit tests for the startup configuration audit.

Drives :func:`audit_writeback_configuration` directly with a fake
writeback-client stub — no FastAPI mounting, no real HTTP. Each test
exercises exactly one warning code so a regression points clearly at
the broken check.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from deep_research.adapter.client import AdapterError
from deep_research.config.valves import Valves
from deep_research.entrypoints.openapi_tool.config_audit import (
    ConfigWarning,
    audit_writeback_configuration,
)


class _FakeWritebackClient:
    """Stand-in for OWUIClient with a controllable get_session_user()."""

    def __init__(self, *, response: dict | None = None, exc: Exception | None = None) -> None:
        self._response = response if response is not None else {"role": "admin"}
        self._exc = exc
        self.calls = 0

    async def get_session_user(self) -> dict:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


def _valves(*, writeback_enabled: bool = True) -> Valves:
    v = Valves()
    v.jobs.writeback_enabled = writeback_enabled
    return v


def _codes(warnings: list[ConfigWarning]) -> set[str]:
    return {w.code for w in warnings}


@pytest.mark.asyncio
async def test_audit_no_probe_warnings_when_writeback_disabled() -> None:
    """writeback_enabled=False suppresses every probe-related warning."""
    warnings = await audit_writeback_configuration(
        _valves(writeback_enabled=False),
        env={"DR_OPENAPI_PUBLIC_BASE_URL": "https://x"},
        writeback_client=None,
    )
    codes = _codes(warnings)
    assert "OWUI_API_KEY_NOT_ADMIN" not in codes
    assert "OWUI_API_KEY_PROBE_FAILED" not in codes


@pytest.mark.asyncio
async def test_audit_flags_missing_public_base_url() -> None:
    """env without DR_OPENAPI_PUBLIC_BASE_URL → MISSING_PUBLIC_BASE_URL (severity=info)."""
    warnings = await audit_writeback_configuration(
        _valves(),
        env={"DR_OWUI_API_KEY": "sk-admin"},
        writeback_client=_FakeWritebackClient(),
    )
    codes = _codes(warnings)
    assert "MISSING_PUBLIC_BASE_URL" in codes
    by_code = {w.code: w for w in warnings}
    assert by_code["MISSING_PUBLIC_BASE_URL"].severity == "info"


@pytest.mark.asyncio
async def test_audit_passes_when_all_set() -> None:
    """All env vars set + probe returns role=admin → no warnings."""
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-admin",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://research.example.com",
        },
        writeback_client=_FakeWritebackClient(response={"role": "admin"}),
    )
    assert warnings == []


@pytest.mark.asyncio
async def test_audit_flags_non_admin_token() -> None:
    """Probe returns role=user → OWUI_API_KEY_NOT_ADMIN."""
    client = _FakeWritebackClient(response={"role": "user"})
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-not-admin",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        writeback_client=client,
    )
    assert "OWUI_API_KEY_NOT_ADMIN" in _codes(warnings)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_audit_flags_probe_failure() -> None:
    """Probe raises → OWUI_API_KEY_PROBE_FAILED (not propagated)."""
    client = _FakeWritebackClient(exc=RuntimeError("OWUI unreachable"))
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-something",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        writeback_client=client,
    )
    assert "OWUI_API_KEY_PROBE_FAILED" in _codes(warnings)


@pytest.mark.asyncio
async def test_audit_skips_probe_when_writeback_client_none() -> None:
    """writeback_client=None → no probe codes."""
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-something",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        writeback_client=None,
    )
    codes = _codes(warnings)
    assert "OWUI_API_KEY_NOT_ADMIN" not in codes
    assert "OWUI_API_KEY_PROBE_FAILED" not in codes


@pytest.mark.asyncio
async def test_audit_maps_adapter_error_to_probe_failed() -> None:
    """get_session_user raises AdapterError (e.g. non-dict response) →
    OWUI_API_KEY_PROBE_FAILED, not the misleading OWUI_API_KEY_NOT_ADMIN."""
    client = _FakeWritebackClient(
        exc=AdapterError("non-dict body", status=200)
    )
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-x",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        writeback_client=client,
    )
    codes = _codes(warnings)
    assert "OWUI_API_KEY_PROBE_FAILED" in codes
    assert "OWUI_API_KEY_NOT_ADMIN" not in codes


@pytest.mark.asyncio
async def test_run_audit_with_timeout_returns_probe_failed_when_audit_hangs(
    monkeypatch,
) -> None:
    """If audit_writeback_configuration sleeps past the timeout budget,
    _run_audit_with_timeout returns a single OWUI_API_KEY_PROBE_FAILED
    warning and elapsed time stays well under the wallclock sleep."""
    from deep_research.entrypoints.openapi_tool import server as srv

    async def _hanging_audit(valves, env, writeback_client):
        await asyncio.sleep(30)
        return []

    monkeypatch.setattr(srv, "audit_writeback_configuration", _hanging_audit)
    t0 = time.monotonic()
    warnings = await srv._run_audit_with_timeout(
        _valves(), env={}, writeback_client=None, timeout_s=0.1
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    assert _codes(warnings) == {"OWUI_API_KEY_PROBE_FAILED"}


def test_maybe_floor_warning_emits_info_for_short_interval() -> None:
    from deep_research.entrypoints.openapi_tool import server as srv
    valves = _valves()
    valves.jobs.cleanup_interval_s = 10
    warning = srv._maybe_floor_warning(valves)
    assert warning is not None
    assert warning.code == "CLEANUP_INTERVAL_FLOORED"
    assert warning.severity == "info"


def test_maybe_floor_warning_returns_none_at_or_above_60() -> None:
    from deep_research.entrypoints.openapi_tool import server as srv
    valves = _valves()
    valves.jobs.cleanup_interval_s = 60
    assert srv._maybe_floor_warning(valves) is None
    valves.jobs.cleanup_interval_s = 3600
    assert srv._maybe_floor_warning(valves) is None

"""Unit tests for the startup configuration audit.

Drives :func:`audit_writeback_configuration` directly with fake valve
and coord stubs — no FastAPI mounting, no real HTTP. Each test exercises
exactly one warning code so a regression points clearly at the broken
check.
"""
from __future__ import annotations

from typing import Any

import pytest

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


class _FakeCoord:
    def __init__(self, writeback_client: Any) -> None:
        self.writeback_client = writeback_client


def _valves(*, writeback_enabled: bool = True) -> Valves:
    v = Valves()
    v.jobs.writeback_enabled = writeback_enabled
    return v


def _codes(warnings: list[ConfigWarning]) -> set[str]:
    return {w.code for w in warnings}


@pytest.mark.asyncio
async def test_audit_flags_missing_owui_api_key() -> None:
    """env without DR_OWUI_API_KEY + writeback_enabled=True → MISSING_OWUI_API_KEY."""
    coord = _FakeCoord(writeback_client=None)
    warnings = await audit_writeback_configuration(
        _valves(),
        env={"DR_OPENAPI_PUBLIC_BASE_URL": "https://x"},
        coord=coord,
    )
    assert "MISSING_OWUI_API_KEY" in _codes(warnings)


@pytest.mark.asyncio
async def test_audit_silent_when_writeback_disabled_and_key_unset() -> None:
    """writeback_enabled=False suppresses the missing-key warning even when key is unset."""
    coord = _FakeCoord(writeback_client=None)
    warnings = await audit_writeback_configuration(
        _valves(writeback_enabled=False),
        env={"DR_OPENAPI_PUBLIC_BASE_URL": "https://x"},
        coord=coord,
    )
    assert "MISSING_OWUI_API_KEY" not in _codes(warnings)


@pytest.mark.asyncio
async def test_audit_flags_missing_public_base_url() -> None:
    """env without DR_OPENAPI_PUBLIC_BASE_URL → MISSING_PUBLIC_BASE_URL (severity=info)."""
    coord = _FakeCoord(writeback_client=_FakeWritebackClient())
    warnings = await audit_writeback_configuration(
        _valves(),
        env={"DR_OWUI_API_KEY": "sk-admin"},
        coord=coord,
    )
    codes = _codes(warnings)
    assert "MISSING_PUBLIC_BASE_URL" in codes
    by_code = {w.code: w for w in warnings}
    assert by_code["MISSING_PUBLIC_BASE_URL"].severity == "info"


@pytest.mark.asyncio
async def test_audit_passes_when_all_set() -> None:
    """All env vars set + probe returns role=admin → no warnings."""
    coord = _FakeCoord(writeback_client=_FakeWritebackClient(response={"role": "admin"}))
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-admin",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://research.example.com",
        },
        coord=coord,
    )
    assert warnings == []


@pytest.mark.asyncio
async def test_audit_flags_non_admin_token() -> None:
    """Probe returns role=user → OWUI_API_KEY_NOT_ADMIN."""
    client = _FakeWritebackClient(response={"role": "user"})
    coord = _FakeCoord(writeback_client=client)
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-not-admin",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        coord=coord,
    )
    assert "OWUI_API_KEY_NOT_ADMIN" in _codes(warnings)
    assert client.calls == 1


@pytest.mark.asyncio
async def test_audit_flags_probe_failure() -> None:
    """Probe raises → OWUI_API_KEY_PROBE_FAILED (not propagated)."""
    client = _FakeWritebackClient(exc=RuntimeError("OWUI unreachable"))
    coord = _FakeCoord(writeback_client=client)
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-something",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        coord=coord,
    )
    assert "OWUI_API_KEY_PROBE_FAILED" in _codes(warnings)


@pytest.mark.asyncio
async def test_audit_skips_probe_when_writeback_client_none() -> None:
    """Key set but coord.writeback_client is None (defensive path) → no probe."""
    coord = _FakeCoord(writeback_client=None)
    warnings = await audit_writeback_configuration(
        _valves(),
        env={
            "DR_OWUI_API_KEY": "sk-something",
            "DR_OPENAPI_PUBLIC_BASE_URL": "https://x",
        },
        coord=coord,
    )
    # No probe codes appear; the MISSING_OWUI_API_KEY check goes through
    # the `elif` branch so it doesn't fire either when the key IS set.
    codes = _codes(warnings)
    assert "OWUI_API_KEY_NOT_ADMIN" not in codes
    assert "OWUI_API_KEY_PROBE_FAILED" not in codes
    assert "MISSING_OWUI_API_KEY" not in codes

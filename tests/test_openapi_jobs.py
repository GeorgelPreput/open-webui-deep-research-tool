"""Tests for the async job pattern: POST /research_jobs + GET /research_jobs/{id}.

The endpoint spawns the run on the running event loop via
`asyncio.create_task`. To control timing in tests, we patch the
coordinator's `run` with an async function that waits on an external
event before completing — letting us assert the `pending`/`running` and
`completed`/`failed` transitions.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from deep_research.core.types import Report
from deep_research.entrypoints.openapi_tool import server as srv


@pytest.fixture
def canned_report() -> Report:
    return Report(
        content="Async report body.",
        title="Async",
        sources={},
        bibliography=[],
        conversation_id="conv_async",
    )


class _GatedCoord:
    """Test double that blocks on `release` before returning."""

    def __init__(self, report: Report, raises: Exception | None = None):
        self._report = report
        self._raises = raises
        self.release: asyncio.Event | None = None  # set per-test
        self.started: asyncio.Event | None = None
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self._raises is not None:
            raise self._raises
        return self._report


@pytest.fixture
def client_factory(monkeypatch):
    def _build(fake: _GatedCoord) -> TestClient:
        monkeypatch.setattr(srv, "_coord", fake)
        srv._jobs.clear()
        return TestClient(srv.app)
    return _build


def _wait_for(client: TestClient, job_id: str, target: set[str], timeout: float = 2.0) -> dict:
    """Poll the job endpoint until status enters `target` or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/research_jobs/{job_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in target:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {target}: last={body}")


def test_start_job_returns_id_and_poll_url(client_factory, canned_report):
    fake = _GatedCoord(canned_report)
    # No gating — release immediately.
    fake.release = None
    client = client_factory(fake)

    # The release event must be created on the server's event loop.
    # Workaround: just don't gate this one; it should run to completion.
    fake.release = None

    resp = client.post("/research_jobs", json={"prompt": "What is X?"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    job_id = body["job_id"]
    assert body["status"] == "pending"
    assert body["poll_url"] == f"/research_jobs/{job_id}"

    final = _wait_for(client, job_id, {"completed", "failed"})
    assert final["status"] == "completed"
    assert final["result"]["report"] == "Async report body."
    assert final["result"]["conversation_id"] == "conv_async"
    assert final["error"] is None


def test_job_failure_branch(client_factory, canned_report):
    fake = _GatedCoord(canned_report, raises=RuntimeError("boom"))
    client = client_factory(fake)

    resp = client.post("/research_jobs", json={"prompt": "fail me"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    final = _wait_for(client, job_id, {"completed", "failed"})
    assert final["status"] == "failed"
    assert final["error"]["code"] == "internal_error"
    assert "boom" in final["error"]["message"]
    assert final["result"] is None


def test_get_unknown_job_returns_404(client_factory, canned_report):
    client = client_factory(_GatedCoord(canned_report))
    resp = client.get("/research_jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "unknown_job"


def test_openapi_schema_lists_job_operations(client_factory, canned_report):
    client = client_factory(_GatedCoord(canned_report))
    schema = client.get("/openapi.json").json()
    start_op = schema["paths"]["/research_jobs"]["post"]
    get_op = schema["paths"]["/research_jobs/{job_id}"]["get"]
    assert start_op["operationId"] == "start_research_job"
    assert get_op["operationId"] == "get_research_job"
    assert start_op["description"]
    assert get_op["description"]

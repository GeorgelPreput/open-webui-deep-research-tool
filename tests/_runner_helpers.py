"""Shared fakes for tests that drive a real ``JobRunner`` against a
fake ``Coordinator``. Lives in tests/ rather than under the package
itself so production imports don't pick it up.
"""
from __future__ import annotations

from typing import Any

from deep_research.core.types import Report
from deep_research.entrypoints.openapi_tool.jobs import JobPhase, JobRecord


class FakeStateManager:
    """Imitates ``ResearchStateManager``'s surface used by ``JobRunner``."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    def get_state(self, cid: str) -> dict[str, Any]:
        return self.states.setdefault(cid, {})

    def set_waiting(self, cid: str, waiting: bool = True, outline=None) -> None:
        st = self.get_state(cid)
        st["waiting_for_outline_feedback"] = waiting
        if waiting:
            st["outline_feedback_data"] = {
                "outline_items": outline or [{"topic": "t"}],
            }


class FakeCoord:
    """Coordinator double — invokes the supplied side-effect on ``.run()``.

    Tests assign ``coord.on_run`` to an async callable taking the
    kwargs dict and returning a Report (or raising). When ``on_run``
    is None the runner gets an empty Report — the no-op path.
    """

    def __init__(self) -> None:
        self.state_manager = FakeStateManager()
        self.run_calls: list[dict[str, Any]] = []
        self.on_run: Any = None

    async def run(self, **kwargs):
        self.run_calls.append(kwargs)
        if self.on_run is not None:
            return await self.on_run(kwargs)
        return Report(content="", conversation_id=kwargs.get("conversation_id"))


def make_record(job_id: str, **overrides) -> JobRecord:
    """Minimal JobRecord factory used by tests that drive the real
    runner. ``chat_id`` defaults to None (no UNIQUE-index collision)."""
    defaults: dict[str, Any] = {
        "job_id": job_id,
        "user_id": "u",
        "user_name": "U",
        "conversation_id": f"conv_{job_id}",
        "chat_id": None,
        "target_message_id": None,
        "phase": JobPhase.QUEUED,
        "prompt": "prompt",
        "history_json": "[]",
        "revision": 0,
        "view_token_hash": "0" * 64,
    }
    defaults.update(overrides)
    return JobRecord(**defaults)

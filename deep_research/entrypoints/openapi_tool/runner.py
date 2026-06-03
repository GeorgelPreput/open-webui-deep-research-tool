"""Per-job orchestration for the OpenAPI Tool Server.

The OWUI Function runtime calls ``Coordinator.run`` once per user
message; concurrent users get concurrent ``asyncio.Task`` instances
because OWUI is itself the async server. The OpenAPI runtime doesn't
have that luxury: a single REST handler must return immediately while
the engine runs in the background.

`JobRunner` provides that decoupling:

  - ``start_job(record, view_token, owui_user_token)`` spawns an
    ``asyncio.Task`` that runs ``Coordinator.run`` once and either
    pauses at the outline-feedback gate (``AWAITING_OUTLINE_FEEDBACK``)
    or completes end-to-end (``COMPLETED``).
  - ``submit_feedback(job_id, selection)`` spawns a second task that
    runs ``Coordinator.run`` *again* on the same ``conversation_id``;
    the engine's ``run_outline_feedback`` phase consumes the new
    prompt as the user's reply and resumes from there.

This matches how the Function path works today — the engine's
suspend/resume model is built on ``ResearchStateManager`` keyed by
``conversation_id``; no new primitive is needed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any

from deep_research.adapter.auth import StaticToken
from deep_research.core.cancellation import CancellationToken
from deep_research.core.types import ChatMessage, RunUser
from deep_research.progress.events import EmbedEvent, MessageEvent, StatusEvent

from .jobs import JobPhase, JobRecord, JobStore, TERMINAL_PHASES, _now_iso

if TYPE_CHECKING:
    from deep_research.orchestrator.coordinator import Coordinator

logger = logging.getLogger("deep_research.entrypoints.openapi.runner")


class JobRunner:
    """Spawn, track, and cancel research jobs.

    Lifetime is owned by ``server.py``'s lifespan handler: one
    runner per process.
    """

    def __init__(
        self,
        *,
        coord: "Coordinator",
        store: JobStore,
        outbox: Any | None = None,  # Phase 2: OutboxWorker
        public_base_url: str = "",
    ) -> None:
        self._coord = coord
        self._store = store
        self._outbox = outbox
        self.public_base_url = public_base_url.rstrip("/")
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancellation_tokens: dict[str, CancellationToken] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._owui_tokens: dict[str, str] = {}
        self._view_tokens: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ public

    async def start_job(
        self,
        record: JobRecord,
        *,
        view_token: str,
        owui_user_token: str,
    ) -> None:
        cancel_token = CancellationToken()
        async with self._lock:
            self._cancellation_tokens[record.job_id] = cancel_token
            self._owui_tokens[record.job_id] = owui_user_token
            self._view_tokens[record.job_id] = view_token
            self._snapshots[record.job_id] = {
                "phase": JobPhase.QUEUED.value,
                "prompt": record.prompt,
                "cycle": 0,
                "max_cycles": 0,
                "completed_topics": [],
                "partial_topics": [],
                "new_topics": [],
                "irrelevant_topics": [],
                "remaining_topics": [],
                "all_topics": [],
                "results_tokens": 0,
                "synthesis_tokens": 0,
                "total_tokens": 0,
            }
            task = asyncio.create_task(
                self._run_initial(record, cancel_token=cancel_token, owui_user_token=owui_user_token)
            )
            self._tasks[record.job_id] = task

    async def submit_feedback(self, job_id: str, selection: str) -> None:
        """Resume a paused engine with the user's outline-feedback reply.

        Waits for the start task to finish before spawning the feedback
        task; otherwise the second ``Coordinator.run`` would race the
        first into the inflight-dedupe lock.
        """
        async with self._lock:
            first_task = self._tasks.get(job_id)
            cancel_token = self._cancellation_tokens.get(job_id)
            owui_user_token = self._owui_tokens.get(job_id, "")

        if first_task is not None and not first_task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await first_task

        record = await self._store.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record.phase != JobPhase.AWAITING_OUTLINE_FEEDBACK:
            raise RuntimeError(
                f"Job {job_id} is in phase {record.phase.value}, not awaiting feedback"
            )

        if cancel_token is None or cancel_token.is_cancelled():
            cancel_token = CancellationToken()
        async with self._lock:
            self._cancellation_tokens[job_id] = cancel_token
            task = asyncio.create_task(
                self._run_feedback(
                    record,
                    selection=selection,
                    cancel_token=cancel_token,
                    owui_user_token=owui_user_token,
                )
            )
            self._tasks[job_id] = task

    async def cancel(self, job_id: str, *, timeout: float = 30.0) -> None:
        """Signal cancellation and wait briefly for the task to unwind."""
        async with self._lock:
            token = self._cancellation_tokens.get(job_id)
            task = self._tasks.get(job_id)
        if token is not None:
            token.cancel()
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=timeout)

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Signal every active job and wait for unwind, best-effort."""
        async with self._lock:
            tokens = list(self._cancellation_tokens.values())
            tasks = list(self._tasks.values())
        for t in tokens:
            t.cancel()
        for task in tasks:
            if task.done():
                continue
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=timeout)

    def get_snapshot(self, job_id: str) -> dict[str, Any]:
        return dict(self._snapshots.get(job_id, {}))

    def get_view_token(self, job_id: str) -> str | None:
        return self._view_tokens.get(job_id)

    # ----------------------------------------------------------------- internal

    async def _run_initial(
        self,
        record: JobRecord,
        *,
        cancel_token: CancellationToken,
        owui_user_token: str,
    ) -> None:
        try:
            await self._store.update(record.job_id, phase=JobPhase.BOOTSTRAPPING)
            self._mark_phase(record.job_id, JobPhase.BOOTSTRAPPING)

            sink = self._make_sink(record.job_id)
            user = RunUser(id=record.user_id, name=record.user_name)
            history = self._deserialise_history(record.history_json)

            report = await self._coord.run(
                user=user,
                conversation_id=record.conversation_id,
                chat_id=record.chat_id,
                token=StaticToken(owui_user_token),
                prompt=record.prompt,
                history=history,
                sink=sink,
                cancellation_token=cancel_token,
                target_message_id=record.target_message_id,
            )

            conv_state = self._coord.state_manager.get_state(record.conversation_id)
            if conv_state.get("waiting_for_outline_feedback"):
                outline_items = (conv_state.get("outline_feedback_data") or {}).get(
                    "outline_items"
                )
                await self._store.update(
                    record.job_id,
                    phase=JobPhase.AWAITING_OUTLINE_FEEDBACK,
                    outline_json=json.dumps(outline_items) if outline_items else None,
                )
                self._mark_phase(record.job_id, JobPhase.AWAITING_OUTLINE_FEEDBACK)
            else:
                await self._store.update(
                    record.job_id,
                    phase=JobPhase.COMPLETED,
                    report_markdown=report.content,
                    completed_at=_now_iso(),
                )
                self._mark_phase(record.job_id, JobPhase.COMPLETED)
        except asyncio.CancelledError:
            await self._store.update(
                record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
            )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            raise
        except Exception as exc:
            logger.exception("Job %s initial run failed", record.job_id)
            await self._store.update(
                record.job_id,
                phase=JobPhase.FAILED,
                error_text=str(exc) or exc.__class__.__name__,
                completed_at=_now_iso(),
            )
            self._mark_phase(record.job_id, JobPhase.FAILED)

    async def _run_feedback(
        self,
        record: JobRecord,
        *,
        selection: str,
        cancel_token: CancellationToken,
        owui_user_token: str,
    ) -> None:
        try:
            await self._store.update(
                record.job_id,
                phase=JobPhase.RESEARCHING,
                selection_json=json.dumps(selection),
            )
            self._mark_phase(record.job_id, JobPhase.RESEARCHING)

            sink = self._make_sink(record.job_id)
            user = RunUser(id=record.user_id, name=record.user_name)
            # History is unused on resume: per-conversation state lives in
            # the Coordinator's process-shared ResearchStateManager.
            report = await self._coord.run(
                user=user,
                conversation_id=record.conversation_id,
                chat_id=record.chat_id,
                token=StaticToken(owui_user_token),
                prompt=selection,
                history=[],
                sink=sink,
                cancellation_token=cancel_token,
                target_message_id=record.target_message_id,
            )
            await self._store.update(
                record.job_id,
                phase=JobPhase.COMPLETED,
                report_markdown=report.content,
                completed_at=_now_iso(),
            )
            self._mark_phase(record.job_id, JobPhase.COMPLETED)
        except asyncio.CancelledError:
            await self._store.update(
                record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
            )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            raise
        except Exception as exc:
            logger.exception("Job %s feedback run failed", record.job_id)
            await self._store.update(
                record.job_id,
                phase=JobPhase.FAILED,
                error_text=str(exc) or exc.__class__.__name__,
                completed_at=_now_iso(),
            )
            self._mark_phase(record.job_id, JobPhase.FAILED)

    def _make_sink(self, job_id: str):
        async def sink(event: Any) -> None:
            self._update_snapshot(job_id, event)
            if self._outbox is not None:
                with contextlib.suppress(Exception):
                    await self._event_to_outbox(job_id, event)

        return sink

    def _update_snapshot(self, job_id: str, event: Any) -> None:
        snap = self._snapshots.setdefault(job_id, {})
        if isinstance(event, StatusEvent):
            snap["latest_status"] = event.description
            snap["status_level"] = event.level
            snap["done"] = event.done
        elif isinstance(event, MessageEvent):
            snap["latest_message_preview"] = event.content[:200]
        elif isinstance(event, EmbedEvent):
            # The iframe polls /live_view/{id}/status for a JSON snapshot
            # and reloads itself on revision bump; the EmbedEvent HTML
            # itself is consumed by the OWUI writeback path (Phase 2).
            snap["has_embed"] = True

    async def _event_to_outbox(self, job_id: str, event: Any) -> None:
        """Phase 2 wiring point — translate events to OutboxWorker rows."""
        # Placeholder: filled in P2.4. Returning early keeps Phase 1
        # behaviour identical regardless of whether outbox is set.
        return None

    def _mark_phase(self, job_id: str, phase: JobPhase) -> None:
        snap = self._snapshots.setdefault(job_id, {})
        snap["phase"] = phase.value
        snap["completed"] = phase in TERMINAL_PHASES

    @staticmethod
    def _deserialise_history(history_json: str | None) -> list[ChatMessage]:
        if not history_json:
            return []
        try:
            items = json.loads(history_json)
        except (TypeError, ValueError):
            return []
        out: list[ChatMessage] = []
        for item in items:
            if isinstance(item, dict) and "role" in item and "content" in item:
                out.append(ChatMessage(role=item["role"], content=item["content"]))
        return out

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
import uuid
from typing import TYPE_CHECKING, Any

from deep_research.adapter.auth import StaticToken
from deep_research.core.cancellation import CancellationToken
from deep_research.core.types import ChatMessage, RunUser
from deep_research.progress.embed import render_progress_embed_html
from deep_research.progress.events import (
    CitationEvent,
    EmbedEvent,
    MessageEvent,
    StatusEvent,
)

from .jobs import JobPhase, JobRecord, JobStore, TERMINAL_PHASES, _now_iso

if TYPE_CHECKING:
    from deep_research.entrypoints.openapi_tool.outbox import OutboxWorker
    from deep_research.orchestrator.coordinator import Coordinator

logger = logging.getLogger("deep_research.entrypoints.openapi.runner")


def _writeback_target(record: JobRecord) -> tuple[str, str] | None:
    """Return (chat_id, target_message_id) if the record is bindable for
    OWUI writeback, otherwise None.

    Local-only chats (``local:`` prefix) are OWUI ephemeral conversations
    that don't persist; the per-message ``/event`` endpoint accepts the
    POST but the event is dropped. Records without forwarded headers
    (``chat_id`` or ``target_message_id`` missing) are also unbindable.
    """
    if record.chat_id is None or record.target_message_id is None:
        return None
    if record.chat_id.startswith("local:"):
        return None
    return (record.chat_id, record.target_message_id)


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
        outbox: "OutboxWorker | None" = None,
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
        self._status_dedupe_counter: dict[str, int] = {}
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
        # No bootstrap embed in the preliminary phase: the only useful
        # content the engine produces before the outline gate is the topic
        # list (delivered as a MessageEvent → replace). The iframe is
        # bootstrapped on submit_feedback, which is when the engine moves
        # into research and the live snapshot becomes meaningful.
        async with self._lock:
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

        if cancel_token is None:
            cancel_token = CancellationToken()
        view_token = self._view_tokens.get(job_id, "")
        if view_token:
            await self._enqueue_bootstrap_embed(record, view_token, marker="feedback")
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
        """Signal cancellation and ensure the job lands in CANCELLED.

        Two paths:
          - Task is running: signal token + wait for unwind. The
            ``CancelledError`` handler in ``_run_initial`` / ``_run_feedback``
            updates the phase and posts the terminal writeback.
          - Task is done (gate cancellation: engine paused at outline
            feedback, task already returned): the handler won't fire;
            update the phase and post the writeback inline.
        """
        async with self._lock:
            token = self._cancellation_tokens.get(job_id)
            task = self._tasks.get(job_id)
        if token is not None:
            token.cancel()
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=timeout)
            return

        record = await self._store.get(job_id)
        if record is None or record.phase in TERMINAL_PHASES:
            return
        await self._store.update(
            job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
        )
        self._mark_phase(job_id, JobPhase.CANCELLED)
        await self._enqueue_terminal_writeback(
            record,
            content=(
                "_Research cancelled by user._\n\n"
                "The run was stopped before completion; no report was generated."
            ),
            status_description="Cancelled by user",
            status_level="warning",
            dedupe_suffix="cancelled",
        )

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
                # Path used by post-report QA: the first Coordinator.run
                # returns a final answer immediately (no outline gate).
                # Land it in the same tool-call message so the LLM doesn't
                # need to retrieve the answer text.
                await self._enqueue_terminal_writeback(
                    record,
                    content=report.content,
                    status_description="Research complete",
                    status_level="info",
                    dedupe_suffix="final",
                )
        except asyncio.CancelledError:
            await self._store.update(
                record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
            )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            await self._enqueue_terminal_writeback(
                record,
                content=(
                    "_Research cancelled by user._\n\n"
                    "The run was stopped before completion; no report was generated."
                ),
                status_description="Cancelled by user",
                status_level="warning",
                dedupe_suffix="cancelled",
            )
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
            # Phase 2: writeback the final report as message content. The
            # user sees the report in the tool-call message without the
            # LLM having to retrieve it; the iframe's last snapshot is
            # preserved alongside.
            await self._enqueue_terminal_writeback(
                record,
                content=report.content,
                status_description="Research complete",
                status_level="info",
                dedupe_suffix="final",
            )
        except asyncio.CancelledError:
            await self._store.update(
                record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
            )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            await self._enqueue_terminal_writeback(
                record,
                content=(
                    "_Research cancelled by user._\n\n"
                    "The run was stopped before completion; no report was generated."
                ),
                status_description="Cancelled by user",
                status_level="warning",
                dedupe_suffix="cancelled",
            )
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
        """Translate an engine event into an OutboxWorker row.

        Maps:
          - StatusEvent   → ``status``  (in-flight progress pills)
          - MessageEvent  → ``replace`` (topic list at the gate, etc.)
          - EmbedEvent    → ``embeds``  (live iframe HTML refresh)
          - CitationEvent → ``source``  (side-panel citations)

        Skips silently when the record has no writeback binding
        (``chat_id`` / ``target_message_id`` missing or ``local:`` chat).
        """
        if self._outbox is None:
            return
        record = await self._store.get(job_id)
        if record is None:
            return
        target = _writeback_target(record)
        if target is None:
            return
        chat_id, message_id = target

        if isinstance(event, StatusEvent):
            seq = self._status_dedupe_counter.get(job_id, 0) + 1
            self._status_dedupe_counter[job_id] = seq
            await self._outbox.enqueue(
                outbox_id=str(uuid.uuid4()),
                job_id=job_id,
                chat_id=chat_id,
                message_id=message_id,
                event_type="status",
                payload={
                    "description": event.description,
                    "done": event.done,
                },
                dedupe_key=f"{job_id}:{message_id}:status:{seq}",
            )
        elif isinstance(event, MessageEvent):
            await self._outbox.enqueue(
                outbox_id=str(uuid.uuid4()),
                job_id=job_id,
                chat_id=chat_id,
                message_id=message_id,
                event_type="replace",
                payload={"content": event.content},
                dedupe_key=(
                    f"{job_id}:{message_id}:replace:"
                    f"msg:{record.revision}:{hash(event.content) & 0xFFFFFFFF}"
                ),
            )
        elif isinstance(event, EmbedEvent):
            await self._outbox.enqueue(
                outbox_id=str(uuid.uuid4()),
                job_id=job_id,
                chat_id=chat_id,
                message_id=message_id,
                event_type="embeds",
                payload={"embeds": [event.html], "replace": True},
                dedupe_key=(
                    f"{job_id}:{message_id}:embeds:"
                    f"engine:{record.revision}"
                ),
            )
        elif isinstance(event, CitationEvent):
            payload = event.to_dict()["data"]
            await self._outbox.enqueue(
                outbox_id=str(uuid.uuid4()),
                job_id=job_id,
                chat_id=chat_id,
                message_id=message_id,
                event_type="source",
                payload=payload,
                dedupe_key=f"{job_id}:{message_id}:source:{event.url}",
            )

    async def _enqueue_bootstrap_embed(
        self,
        record: JobRecord,
        view_token: str,
        *,
        marker: str,
    ) -> None:
        """Post the iframe HTML to the (current) target message once.

        ``marker`` distinguishes the start-of-job bootstrap (``bootstrap``)
        from the post-feedback re-attach (``feedback``), so the dedupe
        key stays unique across the two phases of a single job.
        """
        if self._outbox is None:
            return
        target = _writeback_target(record)
        if target is None:
            return
        chat_id, message_id = target
        snapshot = self._snapshots.get(record.job_id, {}) or {}
        snapshot = {**snapshot, "query": record.prompt}
        status_url = (
            f"{self.public_base_url}/live_view/{record.job_id}/status"
            if self.public_base_url
            else ""
        )
        iframe_html = render_progress_embed_html(
            snapshot,
            poll_url=status_url or None,
            view_token=view_token if status_url else None,
        )
        await self._outbox.enqueue(
            outbox_id=str(uuid.uuid4()),
            job_id=record.job_id,
            chat_id=chat_id,
            message_id=message_id,
            event_type="embeds",
            payload={"embeds": [iframe_html], "replace": True},
            dedupe_key=f"{record.job_id}:{message_id}:embeds:{marker}",
        )

    async def _enqueue_terminal_writeback(
        self,
        record: JobRecord,
        *,
        content: str,
        status_description: str,
        status_level: str,
        dedupe_suffix: str,
    ) -> None:
        """Post the final status pill + message content to the assistant message.

        Shared by success (final report) and cancellation paths. Does NOT
        enqueue an ``embeds: []`` clear — the iframe's last state (topic
        list / progress dashboard) is preserved in the message for user
        reference. Re-reads the record to pick up the rebound
        ``target_message_id``.

        ``status_level`` is captured for telemetry/future use; the OWUI
        ``status`` payload schema only carries ``description`` + ``done``
        today.
        """
        if self._outbox is None:
            return
        refreshed = await self._store.get(record.job_id)
        if refreshed is None:
            return
        target = _writeback_target(refreshed)
        if target is None:
            return
        chat_id, message_id = target

        seq = self._status_dedupe_counter.get(refreshed.job_id, 0) + 1
        self._status_dedupe_counter[refreshed.job_id] = seq
        await self._outbox.enqueue(
            outbox_id=str(uuid.uuid4()),
            job_id=refreshed.job_id,
            chat_id=chat_id,
            message_id=message_id,
            event_type="status",
            payload={"description": status_description, "done": True},
            dedupe_key=f"{refreshed.job_id}:{message_id}:status:{seq}:{dedupe_suffix}",
        )
        await self._outbox.enqueue(
            outbox_id=str(uuid.uuid4()),
            job_id=refreshed.job_id,
            chat_id=chat_id,
            message_id=message_id,
            event_type="replace",
            payload={"content": content},
            dedupe_key=f"{refreshed.job_id}:{message_id}:replace:{dedupe_suffix}",
        )

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

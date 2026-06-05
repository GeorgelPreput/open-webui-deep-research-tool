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
from typing import TYPE_CHECKING, Any, Literal

import aiosqlite

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

from .jobs import TERMINAL_PHASES, JobPhase, JobRecord, JobStore, _now_iso

if TYPE_CHECKING:
    from deep_research.entrypoints.openapi_tool.outbox import OutboxWorker
    from deep_research.orchestrator.coordinator import Coordinator

logger = logging.getLogger("deep_research.entrypoints.openapi.runner")


# User-visible body for a terminal cancellation. Single source of truth;
# the cancel-running-task, cancel-at-gate, and run-failed-by-cancel paths
# all post the same content.
TERMINAL_CANCEL_CONTENT = (
    "_Research cancelled by user._\n\n"
    "The run was stopped before completion; no report was generated."
)

# Short status-pill descriptions for the two terminal kinds.
TERMINAL_CANCEL_DESCRIPTION = "Cancelled by user"
TERMINAL_COMPLETE_DESCRIPTION = "Research complete"

# Dedupe namespace shared by the success-terminal and cancel-terminal
# writebacks. Both terminal kinds emit ``replace`` and ``status`` against
# the same ``{job_id}:{message_id}:{event_type}:terminal`` key, so
# ``INSERT OR IGNORE`` lets whichever lands first survive; the second
# is dropped. Closes the cancel-during-final-enqueue race where
# ``replace:final`` and ``replace:cancelled`` previously both inserted
# and drained in rowid order, producing a mismatched status pill + body.
TERMINAL_DEDUPE_SUFFIX = "terminal"


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


class ActiveJobExistsError(RuntimeError):
    """A concurrent start tried to create a second active job for the
    same chat. Maps to HTTP 409 ``already_running``.
    """

    def __init__(self, chat_id: str | None) -> None:
        super().__init__(f"active job exists for chat {chat_id!r}")
        self.chat_id = chat_id


class FeedbackCancelledError(RuntimeError):
    """Raised by ``JobRunner.submit_feedback`` when its Phase C
    re-validation finds the phase moved out of
    ``AWAITING_OUTLINE_FEEDBACK`` (typically by a concurrent
    ``cancel`` during Phase B's unlocked task wait). Distinct type so
    the server handler can map this to HTTP 409
    ``cancelled_during_feedback`` rather than the generic
    ``not_awaiting_feedback`` it raises for the wrong-phase case at
    Phase A entry.
    """


class JobRunner:
    """Spawn, track, and cancel research jobs.

    Lifetime is owned by ``server.py``'s lifespan handler: one
    runner per process.
    """

    def __init__(
        self,
        *,
        coord: Coordinator,
        store: JobStore,
        outbox: OutboxWorker | None = None,
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
        # Two-tier locking: `_registry_lock` serialises mutations of
        # `_job_locks` (and is the cross-job lock used by shutdown).
        # Each `_job_locks[job_id]` serialises lifecycle transitions
        # (start / feedback / cancel) for that specific job. See
        # `_lock_for` for the GC-policy invariant.
        self._registry_lock = asyncio.Lock()
        self._job_locks: dict[str, asyncio.Lock] = {}
        # Cancel-intent flag set in `cancel()`'s Phase A. Read by
        # `_run_initial`/`_run_feedback`'s success branches just
        # before they would write COMPLETED to the store — if the
        # flag is set, the task returns without setting any terminal
        # phase, leaving the cancellation writeback to `cancel()`'s
        # Phase C. Prevents the cancel-vs-natural-completion race
        # from swallowing the user's cancel intent.
        self._cancel_requested: set[str] = set()

    # ------------------------------------------------------------------ helpers

    async def _lock_for(self, job_id: str) -> asyncio.Lock:
        """Return (and lazily create) the per-job lock for ``job_id``.

        Per-job locks serialise every lifecycle transition for that
        job (start / feedback / cancel) so two concurrent callers
        for the same job_id cannot interleave their phase-check +
        state-mutation steps. The lock is held for the duration of
        the transition, with one exception: ``submit_feedback`` and
        ``cancel`` follow a two-phase shape that releases the lock
        for the engine-task wait and re-acquires it for the
        finalisation. Holding the lock across an indefinite engine
        wait would block concurrent cancel.

        **Locks are NOT GC'd on terminal phase.** A coroutine that
        has already received this lock reference and is waiting to
        acquire it could otherwise race a newcomer that creates a
        fresh lock for the same job_id — the two would think they're
        serialised but actually hold different lock objects. Server
        handlers short-circuit calls for terminal jobs at the
        handler layer, so the lock-leak rate is bounded by observed
        active job_ids.
        """
        async with self._registry_lock:
            lock = self._job_locks.get(job_id)
            if lock is None:
                lock = asyncio.Lock()
                self._job_locks[job_id] = lock
            return lock

    def _maybe_drop_job_state(self, job_id: str) -> None:
        """Drop per-job state dicts if the snapshot phase is terminal.

        Called via ``call_soon`` from a task's done-callback and from
        ``cancel()``'s Phase C cleanup-scheduling block.

        A task that ends in **non-terminal** snapshot phase is a
        *legitimate* outcome — specifically the cancel-vs-success
        race where the success branch returned early on
        ``_cancel_requested`` without writing COMPLETED. The GC must
        NOT fire on that observation; ``cancel()``'s own
        ``call_soon(_maybe_drop_job_state, ...)`` is what triggers
        cleanup once Phase C lands CANCELLED.

        Idempotent — both paths may schedule cleanup; later calls
        find the dicts already drained.
        """
        snap = self._snapshots.get(job_id, {})
        phase_value = snap.get("phase")
        if phase_value not in (
            JobPhase.COMPLETED.value,
            JobPhase.FAILED.value,
            JobPhase.CANCELLED.value,
        ):
            return
        self._tasks.pop(job_id, None)
        self._cancellation_tokens.pop(job_id, None)
        self._owui_tokens.pop(job_id, None)
        self._view_tokens.pop(job_id, None)
        self._snapshots.pop(job_id, None)
        self._status_dedupe_counter.pop(job_id, None)
        self._cancel_requested.discard(job_id)
        # _job_locks intentionally NOT dropped — see `_lock_for`
        # docstring for the rationale.

    def _attach_cleanup_callback(
        self, job_id: str, task: asyncio.Task
    ) -> None:
        """Schedule ``_maybe_drop_job_state`` when the task ends.

        ``_run_initial`` / ``_run_feedback`` call ``_mark_phase`` for
        their terminal state before returning or re-raising, so the
        snapshot phase reflects the final state by the time the
        done-callback fires. The actual GC is deferred via
        ``call_soon`` to run after the holder has fully unwound.
        """

        def _cb(t: asyncio.Task) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.call_soon(self._maybe_drop_job_state, job_id)

        task.add_done_callback(_cb)

    def _schedule_cleanup(self, job_id: str) -> None:
        """Schedule ``_maybe_drop_job_state`` from a non-task context.

        Used by ``cancel()``'s gate-cancel branch, which lands
        CANCELLED inline (no task transition) and therefore won't
        trigger any done-callback.
        """
        # No running loop happens only during interpreter shutdown,
        # at which point cleanup is moot.
        with contextlib.suppress(RuntimeError):
            asyncio.get_running_loop().call_soon(
                self._maybe_drop_job_state, job_id
            )

    # ------------------------------------------------------------------ public

    async def start_job(
        self,
        record: JobRecord,
        *,
        view_token: str,
        owui_user_token: str,
    ) -> None:
        """Insert the JobRecord and spawn the initial engine task.

        Idempotent against concurrent same-chat starts via the sqlite
        UNIQUE partial index on (chat_id) WHERE phase NOT IN terminal.
        A concurrent caller racing on the same chat sees
        ``aiosqlite.IntegrityError`` from ``store.create``; the
        runner translates that to :class:`ActiveJobExistsError` for
        the server layer to map to HTTP 409.

        No bootstrap embed in the preliminary phase: the only useful
        content the engine produces before the outline gate is the
        topic list (delivered as a MessageEvent → replace). The
        iframe is bootstrapped on ``submit_feedback``.
        """
        lock = await self._lock_for(record.job_id)
        cancel_token = CancellationToken()
        async with lock:
            try:
                await self._store.create(record)
            except aiosqlite.IntegrityError as exc:
                raise ActiveJobExistsError(record.chat_id) from exc
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
                self._run_initial(
                    record,
                    cancel_token=cancel_token,
                    owui_user_token=owui_user_token,
                )
            )
            self._tasks[record.job_id] = task
            self._attach_cleanup_callback(record.job_id, task)

    async def submit_feedback(self, job_id: str, selection: str) -> None:
        """Resume a paused engine with the user's outline-feedback reply.

        Two-phase locking:

          - Phase A (under per-job lock): snapshot the start task ref
            and the per-job tokens; validate ``record.phase ==
            AWAITING_OUTLINE_FEEDBACK``; release the lock.
          - Phase B (lock released): ``await first_task`` if it is
            not already ``.done()``. The lock is released so a
            concurrent ``cancel`` can enter and signal the
            cancellation token to unblock a hung engine.
          - Phase C (re-acquire per-job lock): re-validate phase
            (a cancel during Phase B may have moved it); enqueue
            the bootstrap iframe; spawn the feedback task and store
            its reference under the lock.

        Raises:
          - :class:`KeyError` for an unknown ``job_id``.
          - :class:`RuntimeError` from Phase A if the job is not in
            ``AWAITING_OUTLINE_FEEDBACK``.
          - :class:`FeedbackCancelledError` from Phase C if a
            concurrent cancel moved the phase during Phase B.
        """
        lock = await self._lock_for(job_id)

        # Phase A: validate + snapshot
        async with lock:
            first_task = self._tasks.get(job_id)
            cancel_token = self._cancellation_tokens.get(job_id)
            owui_user_token = self._owui_tokens.get(job_id, "")
            view_token = self._view_tokens.get(job_id, "")

            record = await self._store.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.phase != JobPhase.AWAITING_OUTLINE_FEEDBACK:
                raise RuntimeError(
                    f"Job {job_id} is in phase {record.phase.value}, "
                    f"not awaiting feedback"
                )

        # Phase B: outside lock — let concurrent cancel reach the
        # cancellation token while we wait for the start task to fully
        # exit. In normal flow the start task is already `.done()` by
        # the time we get here (the engine returned when it set
        # waiting_for_outline_feedback), so the await is a no-op.
        if first_task is not None and not first_task.done():
            logger.warning(
                "Job %s: submit_feedback found non-done start task; "
                "awaiting before spawning feedback task",
                job_id,
            )
            try:
                await first_task
            except asyncio.CancelledError:
                logger.info(
                    "Job %s start task cancelled during feedback wait",
                    job_id,
                )
            except Exception:
                logger.exception(
                    "Job %s start task raised during feedback wait",
                    job_id,
                )

        # Phase C: re-validate + spawn
        async with lock:
            refreshed = await self._store.get(job_id)
            if refreshed is None:
                raise KeyError(job_id)
            if refreshed.phase != JobPhase.AWAITING_OUTLINE_FEEDBACK:
                # Typical cause: a cancel arrived during Phase B and
                # moved the phase to CANCELLED.
                raise FeedbackCancelledError(
                    f"Job {job_id} cancelled during feedback wait "
                    f"(phase={refreshed.phase.value})"
                )

            if cancel_token is None:
                cancel_token = CancellationToken()
            if view_token:
                await self._enqueue_bootstrap_embed(
                    refreshed, view_token, marker="feedback"
                )
            else:
                # State-dict inconsistency: record is non-terminal but
                # the in-memory token slot is empty. Should not happen
                # unless cleanup ran prematurely.
                logger.warning(
                    "Job %s: view_token missing during submit_feedback; "
                    "bootstrap iframe skipped",
                    job_id,
                )
            self._cancellation_tokens[job_id] = cancel_token
            task = asyncio.create_task(
                self._run_feedback(
                    refreshed,
                    selection=selection,
                    cancel_token=cancel_token,
                    owui_user_token=owui_user_token,
                )
            )
            self._tasks[job_id] = task
            self._attach_cleanup_callback(job_id, task)

    async def cancel(self, job_id: str, *, timeout: float = 30.0) -> None:
        """Signal cancellation and ensure the job lands in CANCELLED.

        Three-phase shape:

          - Phase A (under per-job lock): set
            ``_cancel_requested[job_id]``; snapshot token+task; read
            the record. If already terminal, bail.
          - Phase B (lock released): signal the cancellation token;
            ``wait_for(asyncio.shield(task), timeout=...)``. The
            shield keeps ``wait_for``'s timeout from cancelling the
            task; only the cancellation token is allowed to do that.
          - Phase C (re-acquire per-job lock): if the task is still
            running (wait_for timed out), defer to the engine's
            terminal-state path to avoid duplicate writebacks.
            Otherwise re-read the record. If CANCELLED already, the
            task's own except handler did the work. If COMPLETED /
            FAILED from a natural-completion race, log + override to
            CANCELLED per user intent. Otherwise force CANCELLED and
            post the cancellation writeback.
        """
        lock = await self._lock_for(job_id)
        schedule_cleanup = False

        # Phase A
        async with lock:
            self._cancel_requested.add(job_id)
            token = self._cancellation_tokens.get(job_id)
            task = self._tasks.get(job_id)
            record_before = await self._store.get(job_id)
            if record_before is None:
                self._cancel_requested.discard(job_id)
                return
            if record_before.phase in TERMINAL_PHASES:
                self._cancel_requested.discard(job_id)
                return

        # Phase B
        if token is not None:
            token.cancel()
        if task is not None and not task.done():
            with contextlib.suppress(
                asyncio.TimeoutError, asyncio.CancelledError, Exception
            ):
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=timeout
                )

        # Phase C
        async with lock:
            if task is not None and not task.done():
                # wait_for timed out; the engine is still running.
                # Defer to its terminal-state path; posting now would
                # race the task's own CancelledError handler and
                # produce duplicate writebacks.
                logger.warning(
                    "Cancel(%s): wait_for timed out after %.1fs; "
                    "engine still running; deferring to engine's "
                    "terminal state",
                    job_id, timeout,
                )
                self._cancel_requested.discard(job_id)
                return

            refreshed = await self._store.get(job_id)
            if refreshed is None:
                self._cancel_requested.discard(job_id)
                schedule_cleanup = True
            elif refreshed.phase == JobPhase.CANCELLED:
                # Task's CancelledError handler already did the work.
                self._cancel_requested.discard(job_id)
                schedule_cleanup = True
            else:
                if refreshed.phase in TERMINAL_PHASES:
                    logger.info(
                        "Job %s completed during cancel wait; "
                        "overriding to CANCELLED per user intent",
                        job_id,
                    )
                try:
                    await self._store.update(
                        job_id,
                        phase=JobPhase.CANCELLED,
                        completed_at=_now_iso(),
                    )
                except Exception:
                    logger.exception(
                        "Job %s CANCELLED-phase store update raised; "
                        "in-memory snapshot will still be marked "
                        "CANCELLED",
                        job_id,
                    )
                self._mark_phase(job_id, JobPhase.CANCELLED)
                await self._enqueue_terminal_writeback(
                    refreshed,
                    kind="cancelled",
                    content=TERMINAL_CANCEL_CONTENT,
                )
                self._cancel_requested.discard(job_id)
                schedule_cleanup = True

        # Schedule cleanup outside the lock. Idempotent against the
        # task's own done-callback (both paths use `.pop(..., None)` /
        # `.discard`).
        if schedule_cleanup:
            self._schedule_cleanup(job_id)

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Signal every active job and wait for unwind, best-effort."""
        async with self._registry_lock:
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
                if record.job_id in self._cancel_requested:
                    # Cancel was requested between Coordinator.run
                    # resolving and us reaching this branch. Skip the
                    # COMPLETED store-update + writeback; cancel()'s
                    # Phase C will land CANCELLED. Returning without
                    # setting a terminal snapshot phase here is
                    # deliberate — `_maybe_drop_job_state` no-ops on
                    # non-terminal snapshot phases, and cancel()
                    # schedules its own cleanup.
                    return
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
                    kind="final",
                    content=report.content,
                )
        except asyncio.CancelledError:
            try:
                await self._store.update(
                    record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
                )
            except Exception:
                logger.exception(
                    "Job %s CANCELLED-phase store update raised; "
                    "in-memory snapshot will still be marked CANCELLED",
                    record.job_id,
                )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            await self._enqueue_terminal_writeback(
                record,
                kind="cancelled",
                content=TERMINAL_CANCEL_CONTENT,
            )
            raise
        except Exception as exc:
            logger.exception("Job %s initial run failed", record.job_id)
            try:
                await self._store.update(
                    record.job_id,
                    phase=JobPhase.FAILED,
                    error_text=str(exc) or exc.__class__.__name__,
                    completed_at=_now_iso(),
                )
            except Exception:
                logger.exception(
                    "Job %s FAILED-phase store update raised; "
                    "in-memory snapshot will still be marked FAILED",
                    record.job_id,
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
            if record.job_id in self._cancel_requested:
                # Cancel was requested between Coordinator.run
                # resolving and us reaching this branch. Skip the
                # COMPLETED store-update + writeback; cancel()'s
                # Phase C will land CANCELLED. See _run_initial for
                # the matching invariant.
                return
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
                kind="final",
                content=report.content,
            )
        except asyncio.CancelledError:
            try:
                await self._store.update(
                    record.job_id, phase=JobPhase.CANCELLED, completed_at=_now_iso()
                )
            except Exception:
                logger.exception(
                    "Job %s CANCELLED-phase store update raised; "
                    "in-memory snapshot will still be marked CANCELLED",
                    record.job_id,
                )
            self._mark_phase(record.job_id, JobPhase.CANCELLED)
            await self._enqueue_terminal_writeback(
                record,
                kind="cancelled",
                content=TERMINAL_CANCEL_CONTENT,
            )
            raise
        except Exception as exc:
            logger.exception("Job %s feedback run failed", record.job_id)
            try:
                await self._store.update(
                    record.job_id,
                    phase=JobPhase.FAILED,
                    error_text=str(exc) or exc.__class__.__name__,
                    completed_at=_now_iso(),
                )
            except Exception:
                logger.exception(
                    "Job %s FAILED-phase store update raised; "
                    "in-memory snapshot will still be marked FAILED",
                    record.job_id,
                )
            self._mark_phase(record.job_id, JobPhase.FAILED)

    def _make_sink(self, job_id: str):
        async def sink(event: Any) -> None:
            try:
                self._update_snapshot(job_id, event)
            except Exception:
                logger.exception(
                    "Job %s snapshot update failed for event %s",
                    job_id,
                    type(event).__name__,
                )
            if self._outbox is not None:
                try:
                    await self._event_to_outbox(job_id, event)
                except Exception:
                    logger.exception(
                        "Job %s writeback enqueue failed for event %s; "
                        "subsequent writebacks for this job may also fail",
                        job_id,
                        type(event).__name__,
                    )

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
        kind: Literal["final", "cancelled"],
        content: str,
    ) -> None:
        """Post the terminal status pill + message content to the
        assistant message.

        Shared by the success path (final report) and the cancellation
        paths (running-task cancel, gate cancel,
        cancel-overrides-natural-completion). The ``kind`` argument
        picks the status-pill description; both kinds share the
        ``terminal`` dedupe suffix so that a cancel landing during a
        successful enqueue cannot produce a mismatched status/body
        pair — ``INSERT OR IGNORE`` lets the first row of each
        event-type win.

        Does NOT enqueue an ``embeds: []`` clear — the iframe's last
        state is preserved in the message for user reference. Re-reads
        the record to pick up the rebound ``target_message_id``.
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

        description = (
            TERMINAL_COMPLETE_DESCRIPTION
            if kind == "final"
            else TERMINAL_CANCEL_DESCRIPTION
        )
        await self._outbox.enqueue(
            outbox_id=str(uuid.uuid4()),
            job_id=refreshed.job_id,
            chat_id=chat_id,
            message_id=message_id,
            event_type="status",
            payload={"description": description, "done": True},
            dedupe_key=(
                f"{refreshed.job_id}:{message_id}:status:{TERMINAL_DEDUPE_SUFFIX}"
            ),
        )
        await self._outbox.enqueue(
            outbox_id=str(uuid.uuid4()),
            job_id=refreshed.job_id,
            chat_id=chat_id,
            message_id=message_id,
            event_type="replace",
            payload={"content": content},
            dedupe_key=(
                f"{refreshed.job_id}:{message_id}:replace:{TERMINAL_DEDUPE_SUFFIX}"
            ),
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
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Failed to decode history_json (length=%d, error_class=%s, detail=%s); "
                "dropping conversation history for this run",
                len(history_json),
                type(exc).__name__,
                exc,
            )
            return []
        out: list[ChatMessage] = []
        for item in items:
            if isinstance(item, dict) and "role" in item and "content" in item:
                out.append(ChatMessage(role=item["role"], content=item["content"]))
        return out

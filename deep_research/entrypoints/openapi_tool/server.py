"""OpenAPI Tool Server entrypoint (v2).

Six endpoints, plus /health:

  POST /research_jobs                       — start a job
  POST /research_jobs/{job_id}/feedback     — submit outline selection
  GET  /research_jobs/{job_id}              — JSON status snapshot
  POST /research_jobs/{job_id}/cancel       — request cancellation
  GET  /live_view/{job_id}                  — HTML iframe (view-token auth)
  GET  /live_view/{job_id}/status           — JSON snapshot for polling

The synchronous ``POST /research`` endpoint and the in-memory ``_jobs``
dict from v1 are gone. Job state lives in a durable sqlite store
(:mod:`.jobs`), so a server restart no longer drops in-flight runs
(jobs are marked FAILED on the next status read).

Live progress goes to the iframe via short-lived view tokens (sha256
hashed at rest). Phase 2 adds an outbox-driven OWUI ``/event`` writeback
so chat content (topic list, final report) lands directly in the
assistant message.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import os
import pathlib
import secrets
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from deep_research import Coordinator
from deep_research.config.env import load_valves_from_env
from deep_research.config.logging import configure_logging
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.embed import render_progress_embed_html

from .config_audit import ConfigWarning, audit_writeback_configuration
from .jobs import (
    TERMINAL_PHASES,
    JobPhase,
    JobRecord,
    JobStore,
    history_to_json,
)
from .outbox import OutboxWorker
from .runner import ActiveJobExistsError, FeedbackCancelledError, JobRunner
from .schemas import (
    CancelResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    JobStatusResponse,
    LiveViewSnapshot,
    StartResearchRequest,
    StartResearchResponse,
)

if TYPE_CHECKING:
    from deep_research.adapter.client import OWUIClient

logger = logging.getLogger("deep_research.entrypoints.openapi")


_AUDIT_TIMEOUT_S = 30.0


async def _run_audit_with_timeout(
    valves: Any,
    env: Mapping[str, str],
    writeback_client: OWUIClient | None,
    *,
    timeout_s: float = _AUDIT_TIMEOUT_S,
) -> list[ConfigWarning]:
    """Run the startup audit with an upper time bound.

    A briefly-unreachable OWUI at boot would otherwise hang the lifespan
    on the admin probe's HTTP timeout. On `asyncio.TimeoutError` we
    synthesise an `OWUI_API_KEY_PROBE_FAILED` warning — the same code
    the audit's `except Exception` branch emits — so callers see one
    stable shape for "probe couldn't run" regardless of cause.
    """
    try:
        return await asyncio.wait_for(
            audit_writeback_configuration(valves, env, writeback_client),
            timeout=timeout_s,
        )
    except TimeoutError:
        logger.warning(
            "Config audit: admin probe to OWUI timed out after %.1fs; "
            "starting server with OWUI_API_KEY_PROBE_FAILED warning.",
            timeout_s,
        )
        return [
            ConfigWarning(
                code="OWUI_API_KEY_PROBE_FAILED",
                severity="warning",
                message=(
                    f"Probe to GET /api/v1/auths/ did not return within "
                    f"{timeout_s:.0f}s at startup. OWUI may be unreachable; "
                    "writeback POSTs may fail at runtime."
                ),
                remediation=(
                    "Verify the OWUI base URL is reachable from this pod. "
                    "Once OWUI is reachable, restart the tool server to "
                    "re-run the probe."
                ),
            )
        ]


def _maybe_floor_warning(valves: Any) -> ConfigWarning | None:
    """Synthesise the `CLEANUP_INTERVAL_FLOORED` warning when the operator
    set `DR_JOBS_CLEANUP_INTERVAL_S` below the 60s floor.

    The retention loop floors the runtime interval at 60s; this function
    surfaces that decision in `/health` so operators don't have to read
    source to discover their value was overridden.
    """
    configured = int(valves.jobs.cleanup_interval_s)
    if configured >= 60:
        return None
    return ConfigWarning(
        code="CLEANUP_INTERVAL_FLOORED",
        severity="info",
        message=(
            f"DR_JOBS_CLEANUP_INTERVAL_S={configured}s is below the 60s "
            "minimum and was floored to 60s. The retention sweep runs "
            "every 60s."
        ),
        remediation=(
            "Set DR_JOBS_CLEANUP_INTERVAL_S to 60 or higher. If you "
            "need faster cleanup, lower completed_retention_s or "
            "failed_retention_s instead."
        ),
    )


START_DESCRIPTION = (
    "Run a research investigation. Returns immediately with a `job_id` "
    "while the engine runs in the background.\n\n"
    "**Critical workflow rule:** When you call this tool, your assistant "
    "message has a live progress iframe attached to it that the user can "
    "see in real time. You MUST emit the `user_facing_instruction` string "
    "returned in the response **verbatim** as part of your reply. Do NOT "
    "paraphrase, summarise, or replace it with your own explanation. The "
    "user needs to see the exact slash-command instructions to drive the "
    "next step.\n\n"
    "When the user replies with their selection (e.g. `/k 1,3,5`, "
    "`/r 8,9,10`, or `/continue`), call `submit_research_feedback` with "
    "the user's verbatim reply.\n\n"
    "If this tool returns HTTP 409 with code `unsaved_chat_unsupported`, "
    "tell the user that this chat is ephemeral and ask them to send any "
    "brief message (even \"hi\") so OWUI persists the chat, then ask "
    "them to re-run the research. Do not retry the tool call until the "
    "user replies — the chat must be saved before the tool can stream "
    "progress into it."
)

FEEDBACK_DESCRIPTION = (
    "Submit the user's outline-feedback reply. Forward the user's reply "
    "verbatim (e.g. `/k 1,3,5`, `/keep 1, 3, 5`, `/r 2,4,8-10`, "
    "`/continue`, or freeform text). Returns immediately while the engine "
    "resumes; the user sees research progress, citations, and the final "
    "report stream into their chat message directly.\n\n"
    "**After a 200 response, do NOT call this tool again, do NOT call "
    "`get_research_job`, and do NOT speculate about the engine's state.** "
    "The engine is now running in the background; progress streams "
    "directly into the user's chat and live-progress iframe — you cannot "
    "see it from here, and the absence of an immediate visible change "
    "does NOT mean the run has stalled or that the chat is ephemeral. "
    "If the user's next message expresses impatience (\"is it running?\", "
    "\"please continue\", \"hello?\"), simply acknowledge that the "
    "research is in progress and the results will appear when the engine "
    "finishes. Only call `cancel_research_job` if the user explicitly "
    "asks to stop (`/q`, `/quit`, or unambiguous natural-language cancel)."
)

GET_DESCRIPTION = (
    "Fetch the current snapshot of a research job. While running, the "
    "`progress` field carries in-flight phase/cycle/topic info. Once "
    "`phase` is `completed`, `report_markdown` holds the full report. On "
    "`failed`, `error` carries the error description."
)

CANCEL_DESCRIPTION = (
    "Request cancellation of a research job. Returns immediately; the "
    "engine bails at the next phase boundary (typically within seconds "
    "to a minute depending on the active phase). Cancellation is "
    "**irreversible** — there is no undo and no resume; the user must "
    "start a new research job from scratch.\n\n"
    "**When to call:** unambiguous slash commands `/q` or `/quit`. "
    "Forward these immediately, without a confirmation question.\n\n"
    "**When NOT to call:** ambiguous natural-language phrases ("
    "\"stop\", \"cancel\", \"never mind\", \"wait\", \"hold on\") may "
    "express frustration, a question about the run, or a request to "
    "narrow scope — they are not unambiguous cancellation requests. "
    "Before calling this tool on natural language, ask the user one "
    "short clarifying question (\"Cancel the research entirely, or "
    "would you like to narrow the scope?\") and only call cancel if "
    "they confirm. False-positive cancels destroy in-flight work and "
    "are user-hostile."
)

LIVE_VIEW_DESCRIPTION = (
    "Render the progress iframe HTML. Authenticated via the per-job view "
    "token query parameter."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    valves = load_valves_from_env(prefix="DR_")
    configure_logging(valves)

    if valves.jobs.writeback_enabled and not os.environ.get("DR_OWUI_API_KEY"):
        raise RuntimeError(
            "DR_OWUI_API_KEY is required when DR_JOBS_WRITEBACK_ENABLED is "
            "true (default). The OpenAPI Tool Server uses this token to post "
            "writeback events to OWUI on behalf of arbitrary users; the "
            "token must have OWUI's admin role. Set DR_OWUI_API_KEY to an "
            "admin API key, or set DR_JOBS_WRITEBACK_ENABLED=false to "
            "disable writeback."
        )

    config = RuntimeConfig(
        data_dir=os.environ.get("DR_DATA_DIR", "/data/deep_research"),
        base_url=os.environ.get("DR_OWUI_BASE_URL", "http://localhost:8080"),
        llm_base_url=valves.llm.base_url,
        llm_api_key=valves.llm.api_key,
        llm_chat_path=valves.llm.chat_path,
        embeddings_base_url=valves.embeddings.base_url,
        embeddings_api_key=valves.embeddings.api_key,
        embeddings_path=valves.embeddings.embeddings_path,
    )

    logger.info(
        "OpenAPI server startup: owui_base=%s llm_base=%s llm_chat=%s "
        "llm_key=%s embeddings_base=%s embeddings_path=%s embeddings_key=%s "
        "data_dir=%s research_model=%s synthesis_model=%s embedding_model=%s "
        "owui_key=%s",
        config.base_url,
        config.llm_base_url,
        config.llm_chat_path,
        "set" if config.llm_api_key else "unset",
        config.embeddings_base_url,
        config.embeddings_path,
        "set" if config.embeddings_api_key else "unset",
        config.data_dir,
        valves.models.research_model,
        valves.models.synthesis_model,
        valves.models.embedding_model,
        "set" if os.environ.get("DR_OWUI_API_KEY") else "unset",
    )

    coord = Coordinator(valves=valves, config=config)
    writeback_token = os.environ.get("DR_OWUI_API_KEY", "")
    await coord.start(writeback_token=writeback_token or None)
    app.state.coord = coord
    app.state.valves = valves

    app.state.config_warnings = await _run_audit_with_timeout(
        valves, os.environ, coord.writeback_client
    )
    app.state.config_warnings_lock = asyncio.Lock()
    for w in app.state.config_warnings:
        logger.warning(
            "Config audit: [%s] %s — %s", w.code, w.message, w.remediation
        )

    floor_warning = _maybe_floor_warning(valves)
    if floor_warning is not None:
        async with app.state.config_warnings_lock:
            app.state.config_warnings.append(floor_warning)
        logger.warning(
            "Config audit: [%s] %s — %s",
            floor_warning.code, floor_warning.message, floor_warning.remediation,
        )

    db_path = pathlib.Path(config.data_dir) / "jobs.sqlite"
    store = JobStore(db_path, busy_timeout_ms=valves.jobs.sqlite_busy_timeout_ms)
    await store.start()
    app.state.job_store = store

    outbox: OutboxWorker | None = None
    writeback_client = coord.writeback_client
    if valves.jobs.writeback_enabled and writeback_client is not None:
        outbox = OutboxWorker(
            db_path=db_path,
            owui_client=writeback_client,
            poll_interval_s=max(0.05, valves.jobs.outbox_poll_interval_ms / 1000.0),
            max_attempts=valves.jobs.outbox_max_attempts,
            max_backoff_s=valves.jobs.outbox_max_backoff_s,
            max_retry_after_s=valves.jobs.outbox_max_retry_after_s,
            busy_timeout_ms=valves.jobs.sqlite_busy_timeout_ms,
        )
        await outbox.start()
        logger.info(
            "OutboxWorker started (writeback enabled); poll_interval_ms=%d "
            "max_attempts=%d max_backoff_s=%d max_retry_after_s=%d",
            valves.jobs.outbox_poll_interval_ms,
            valves.jobs.outbox_max_attempts,
            valves.jobs.outbox_max_backoff_s,
            valves.jobs.outbox_max_retry_after_s,
        )
    app.state.outbox = outbox

    app.state.runner = JobRunner(
        coord=coord,
        store=store,
        outbox=outbox,
        public_base_url=os.environ.get("DR_OPENAPI_PUBLIC_BASE_URL", ""),
    )

    app.state.retention_task = asyncio.create_task(
        _retention_loop(store, valves.jobs)
    )

    try:
        yield
    finally:
        app.state.retention_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await app.state.retention_task
        await app.state.runner.shutdown()
        if app.state.outbox is not None:
            await app.state.outbox.stop()
        await app.state.job_store.close()
        await app.state.coord.close()


async def _retention_loop(store: JobStore, jobs_valves: Any) -> None:
    interval = max(60, int(jobs_valves.cleanup_interval_s))
    while True:
        await asyncio.sleep(interval)
        try:
            expired = await store.list_expired(
                completed_retention_s=jobs_valves.completed_retention_s,
                failed_retention_s=jobs_valves.failed_retention_s,
            )
            for job_id in expired:
                await store.delete(job_id)
            if expired:
                logger.info("Job retention swept %d expired records", len(expired))
        except Exception:
            logger.exception("Retention sweep failed")


app = FastAPI(
    title="Deep Research Tool",
    description=(
        "OpenAPI Tool Server for the Deep Research engine. Exposes a "
        "two-call job workflow (start → user reply → submit feedback → "
        "background completion) plus a self-polling live-progress iframe."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# OWUI mounts the live-view iframe sandboxed `allow-scripts` *without*
# `allow-same-origin`, so the iframe runs with an opaque origin and must use a
# wildcard CORS policy to call back to this server. The poll request explicitly
# uses `credentials: 'omit'`, so `allow_credentials=False` matches the actual
# call pattern and keeps the CORS posture spec-conformant.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nosemgrep: python.fastapi.security.wildcard-cors.wildcard-cors
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


security = HTTPBearer(auto_error=False)
_BearerCreds = Annotated[HTTPAuthorizationCredentials | None, Depends(security)]


def _api_token(creds: _BearerCreds) -> str:
    if creds is not None and creds.credentials:
        return creds.credentials
    return os.environ.get("DR_OWUI_API_KEY", "")


_ApiToken = Annotated[str, Depends(_api_token)]


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(code=code, message=message).model_dump(),
    )


def _progress_dict_for(record: JobRecord, runner: JobRunner) -> dict[str, Any]:
    """Reconstruct an iframe-ready snapshot dict from the persisted record
    plus the runner's in-memory cache.

    The engine emits richer in-memory snapshots; the cached snapshot has
    the most recent token/topic data the runner has seen for this job.
    Falls back to a minimal dict for jobs that haven't started emitting
    progress yet (queued/bootstrapping).
    """
    cached = runner.get_snapshot(record.job_id) or {}
    out: dict[str, Any] = {
        "phase": record.phase.value,
        "query": record.prompt,
        "revision": record.revision,
        "updated_at": record.updated_at,
        "cycle": cached.get("cycle", 0),
        "max_cycles": cached.get("max_cycles", 0),
        "completed_topics": cached.get("completed_topics", []),
        "partial_topics": cached.get("partial_topics", []),
        "new_topics": cached.get("new_topics", []),
        "irrelevant_topics": cached.get("irrelevant_topics", []),
        "remaining_topics": cached.get("remaining_topics", []),
        "all_topics": cached.get("all_topics", []),
        "results_tokens": cached.get("results_tokens", 0),
        "synthesis_tokens": cached.get("synthesis_tokens", 0),
        "total_tokens": cached.get("total_tokens", 0),
        "latest_status": cached.get("latest_status", ""),
    }
    return out


def _public_base_for(request: Request) -> str:
    configured = getattr(request.app.state.runner, "public_base_url", "")
    if configured:
        return configured
    return str(request.base_url).rstrip("/")


# ----------------------------------------------------------- endpoints


@app.post(
    "/research_jobs",
    operation_id="start_research_job",
    summary="Start a research job. Returns immediately with a job_id.",
    description=START_DESCRIPTION,
    response_model=StartResearchResponse,
    responses={
        409: {
            "model": ErrorResponse,
            "description": (
                "Job cannot be created in this chat. Codes: "
                "`already_running` (an active job exists for this chat) "
                "or `unsaved_chat_unsupported` (the chat is an OWUI "
                "ephemeral `local:` chat)."
            ),
        },
        503: {
            "model": ErrorResponse,
            "description": "Coordinator not yet initialised.",
        },
    },
    tags=["research"],
)
async def start_research_job(
    req: StartResearchRequest,
    request: Request,
    token: _ApiToken,
) -> StartResearchResponse:
    chat_id = request.headers.get("X-OpenWebUI-Chat-Id")
    message_id = request.headers.get("X-OpenWebUI-Message-Id")

    # chat_id handling is intentionally asymmetric across three input states:
    #   - "local:..."  → hard-reject 409 (below): OWUI's /event endpoint
    #     silently drops events posted to ephemeral chats, so a multi-minute
    #     run whose progress/report can never reach the user is bad value.
    #     Fail fast and have the LLM ask the user to persist the chat first.
    #   - None / ""    → soft-degrade (the `if not chat_id` block further
    #     down): the job runs, writeback skips (runner._writeback_target
    #     returns None), and OWUI_HEADERS_NOT_FORWARDED surfaces on /health.
    #     This is a deployment-config issue (ENABLE_FORWARD_USER_INFO_HEADERS
    #     off), not a per-call user error.
    #   - real UUID    → proceed normally.
    # See CLAUDE.md "`local:` chats are refused at `start_research_job`" and
    # "Config warning code taxonomy". Do NOT symmetrise None into a 409 —
    # it would break the headers-off OWUI deployment story.
    if chat_id and chat_id.startswith("local:"):
        raise _error(
            409,
            "unsaved_chat_unsupported",
            (
                "This chat is ephemeral (its ID starts with 'local:'). "
                "Deep Research can't post progress or the final report "
                "to an unsaved chat. Send any brief message in this "
                "chat first so OWUI persists it, then re-run the "
                "research."
            ),
        )

    # Detector gates on chat-id absence ALONE, not on bearer presence: the
    # signal for "OWUI isn't forwarding user-info headers" is the missing
    # X-OpenWebUI-Chat-Id, and OWUI's inbound auth type (none/bearer/session)
    # is an orthogonal knob. Gating on credentials would blind us to the
    # OWUI-with-auth:none-tool-server-plus-forwarding-off deployment (writeback
    # silently dies, operator never told). The cost is a benign one-shot false
    # positive if a non-OWUI caller starts a job without the header.
    if not chat_id:
        async with request.app.state.config_warnings_lock:
            if not getattr(request.app.state, "_forward_headers_warned", False):
                request.app.state._forward_headers_warned = True
                warning = ConfigWarning(
                    code="OWUI_HEADERS_NOT_FORWARDED",
                    severity="warning",
                    message=(
                        "A request arrived without X-OpenWebUI-Chat-Id. If it "
                        "came from OWUI, user-info headers are not being "
                        "forwarded and writeback is silently disabled."
                    ),
                    remediation=(
                        "On the OWUI container, set "
                        "ENABLE_FORWARD_USER_INFO_HEADERS=true and restart."
                    ),
                )
                existing = getattr(request.app.state, "config_warnings", None)
                if existing is None:
                    existing = []
                    request.app.state.config_warnings = existing
                existing.append(warning)
                logger.warning(
                    "Config audit (runtime): [%s] %s — %s",
                    warning.code, warning.message, warning.remediation,
                )

    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    job_id = str(uuid4())
    view_token = secrets.token_urlsafe(32)
    view_token_hash = hashlib.sha256(view_token.encode("utf-8")).hexdigest()

    record = JobRecord(
        job_id=job_id,
        user_id=req.user_id,
        user_name=req.user_name,
        conversation_id=chat_id or f"openapi_{job_id}",
        chat_id=chat_id,
        target_message_id=message_id,
        phase=JobPhase.QUEUED,
        prompt=req.prompt,
        history_json=history_to_json(req.history),
        revision=0,
        view_token_hash=view_token_hash,
    )
    try:
        await runner.start_job(
            record, view_token=view_token, owui_user_token=token
        )
    except ActiveJobExistsError:
        # The sqlite UNIQUE partial index rejected the insert at
        # JobRunner.start_job. Re-fetch the winning row to give the
        # LLM a useful job_id in the 409 message.
        active = (
            await store.find_active_by_chat(chat_id) if chat_id else None
        )
        if active is not None:
            msg = f"Active job exists for this chat: {active.job_id}"
        else:
            msg = "Active job exists for this chat."
        raise _error(409, "already_running", msg) from None

    return StartResearchResponse(
        job_id=job_id,
        status="running",
        next_action="await_user_selection",
        view_token=view_token,
    )


@app.post(
    "/research_jobs/{job_id}/feedback",
    operation_id="submit_research_feedback",
    summary="Submit the user's outline-feedback reply.",
    description=FEEDBACK_DESCRIPTION,
    response_model=FeedbackResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown job_id."},
        409: {
            "model": ErrorResponse,
            "description": "Job is not waiting for outline feedback.",
        },
    },
    tags=["research"],
)
async def submit_research_feedback(
    job_id: str,
    req: FeedbackRequest,
    request: Request,
) -> FeedbackResponse:
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise _error(404, "unknown_job", job_id)
    if record.phase != JobPhase.AWAITING_OUTLINE_FEEDBACK:
        raise _error(
            409,
            "not_awaiting_feedback",
            f"phase={record.phase.value}",
        )

    new_message_id = request.headers.get("X-OpenWebUI-Message-Id")
    if new_message_id and new_message_id != record.target_message_id:
        await store.rebind_target_message(job_id, new_message_id)

    try:
        await runner.submit_feedback(job_id, req.selection)
    except FeedbackCancelledError as exc:
        # Race-window backstop: a concurrent cancel moved the phase
        # while the runner was awaiting the start task outside the
        # per-job lock. Distinct code so consumers can tell this
        # from the common "client called feedback on a job that was
        # never paused" case the handler fast-path serves.
        raise _error(409, "cancelled_during_feedback", str(exc)) from None
    except RuntimeError as exc:
        # Defence in depth: the handler's pre-check above normally
        # catches the wrong-phase case before reaching here, but a
        # concurrent transition between the check and the runner
        # call would surface as RuntimeError from the runner's own
        # Phase A check.
        raise _error(409, "not_awaiting_feedback", str(exc)) from None
    except KeyError:
        # Race: the record was deleted between the handler's
        # `store.get` and the runner's own re-read.
        raise _error(404, "unknown_job", job_id) from None

    return FeedbackResponse(
        job_id=job_id,
        status="running",
        next_phase="researching",
    )


@app.get(
    "/research_jobs/{job_id}",
    operation_id="get_research_job",
    summary="Get the snapshot of a research job.",
    description=GET_DESCRIPTION,
    response_model=JobStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown job_id."},
    },
    tags=["research"],
)
async def get_research_job(job_id: str, request: Request) -> JobStatusResponse:
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise _error(404, "unknown_job", job_id)

    return JobStatusResponse(
        job_id=record.job_id,
        phase=record.phase.value,
        revision=record.revision,
        progress=_progress_dict_for(record, runner),
        report_markdown=(
            record.report_markdown
            if record.phase == JobPhase.COMPLETED
            else None
        ),
        error=record.error_text if record.phase == JobPhase.FAILED else None,
    )


@app.post(
    "/research_jobs/{job_id}/cancel",
    operation_id="cancel_research_job",
    summary="Request cancellation of a research job.",
    description=CANCEL_DESCRIPTION,
    response_model=CancelResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown job_id."},
    },
    tags=["research"],
)
async def cancel_research_job(job_id: str, request: Request) -> CancelResponse:
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise _error(404, "unknown_job", job_id)
    if record.phase in TERMINAL_PHASES:
        return CancelResponse(job_id=job_id, status="already_terminal")

    await runner.cancel(job_id)
    return CancelResponse(job_id=job_id, status="cancel_requested")


@app.get(
    "/live_view/{job_id}",
    summary="Render the progress iframe HTML.",
    description=LIVE_VIEW_DESCRIPTION,
    response_class=HTMLResponse,
    tags=["live_view"],
)
async def live_view(
    job_id: str,
    request: Request,
    token: Annotated[str, Query(description="Per-job view token from start_research_job response.")],
):
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404)
    if not hmac.compare_digest(
        hashlib.sha256(token.encode("utf-8")).hexdigest(),
        record.view_token_hash,
    ):
        raise HTTPException(status_code=403)

    snapshot = _progress_dict_for(record, runner)
    public_base = _public_base_for(request)
    status_url = f"{public_base}/live_view/{job_id}/status"
    html = render_progress_embed_html(snapshot, poll_url=status_url, view_token=token)
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": "inline"},
    )


@app.get(
    "/live_view/{job_id}/status",
    summary="JSON snapshot of a job for the iframe's polling loop.",
    description=(
        "Returns a `LiveViewSnapshot` JSON body with the current job state. "
        "If the client passes `since_version=N` and the job's current "
        "revision equals N (no change since the last poll), the endpoint "
        "returns `204 No Content` with an empty body so the polling iframe "
        "can skip a re-render. The 204 branch is only taken when "
        "`since_version` is supplied; omitting it always yields a 200 with "
        "the snapshot."
    ),
    response_model=LiveViewSnapshot,
    responses={
        204: {
            "description": (
                "No change since `since_version`. Empty body. The iframe's "
                "polling script treats this as a no-op and schedules the "
                "next poll."
            ),
        },
        403: {"description": "View token does not match the job's stored hash."},
        404: {"description": "Unknown job_id."},
    },
    tags=["live_view"],
)
async def live_view_status(
    job_id: str,
    request: Request,
    token: Annotated[str, Query()],
    since_version: Annotated[int | None, Query(ge=0)] = None,
) -> Response:
    """Poll a job's snapshot for the live-view iframe.

    Returns a 200 with a `LiveViewSnapshot` JSON body by default. If the
    caller supplies `since_version=N` and the job's current revision equals
    N, returns a bare 204 with an empty body — the iframe's polling script
    reads this as "no change, keep polling." 403/404 use FastAPI's default
    ``{"detail": ...}`` shape (bare ``HTTPException``), not ``ErrorResponse``.
    """
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404)
    if not hmac.compare_digest(
        hashlib.sha256(token.encode("utf-8")).hexdigest(),
        record.view_token_hash,
    ):
        raise HTTPException(status_code=403)

    if since_version is not None and record.revision == since_version:
        return Response(status_code=204)

    payload = LiveViewSnapshot(
        job_id=job_id,
        phase=record.phase.value,
        revision=record.revision,
        progress=_progress_dict_for(record, runner),
        completed=record.phase in TERMINAL_PHASES,
    )
    return JSONResponse(payload.model_dump())


@app.get("/health", tags=["health"])
async def health(request: Request) -> dict:
    warnings = getattr(request.app.state, "config_warnings", [])
    outbox: OutboxWorker | None = getattr(request.app.state, "outbox", None)
    outbox_counts: dict[str, int] | None = None
    if outbox is not None:
        outbox_counts = await outbox.count_by_status()
    return {
        "status": "ok",
        "config_warnings": [
            {
                "code": w.code,
                "severity": w.severity,
                "message": w.message,
                "remediation": w.remediation,
            }
            for w in warnings
        ],
        "outbox": outbox_counts,
    }

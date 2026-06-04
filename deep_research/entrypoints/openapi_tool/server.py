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
import json
import logging
import os
import pathlib
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    status,
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
    JobPhase,
    JobRecord,
    JobStore,
    TERMINAL_PHASES,
    _now_iso,
    history_to_json,
)
from .outbox import OutboxWorker
from .runner import JobRunner
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

logger = logging.getLogger("deep_research.entrypoints.openapi")


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
    "the user's verbatim reply."
)

FEEDBACK_DESCRIPTION = (
    "Submit the user's outline-feedback reply. Forward the user's reply "
    "verbatim (e.g. `/k 1,3,5`, `/keep 1, 3, 5`, `/r 2,4,8-10`, "
    "`/continue`, or freeform text). Returns immediately while the engine "
    "resumes; the user sees research progress, citations, and the final "
    "report stream into their chat message directly."
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
    "to a minute depending on the active phase).\n\n"
    "Call this tool when the user types `/q` or `/quit`, or makes any "
    "natural-language cancellation request (\"stop\", \"cancel\", "
    "\"never mind\")."
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

    app.state.config_warnings = await audit_writeback_configuration(
        valves, os.environ, coord
    )
    for w in app.state.config_warnings:
        logger.warning(
            "Config audit: [%s] %s — %s", w.code, w.message, w.remediation
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
            busy_timeout_ms=valves.jobs.sqlite_busy_timeout_ms,
        )
        await outbox.start()
        logger.info(
            "OutboxWorker started (writeback enabled); poll_interval_ms=%d max_attempts=%d max_backoff_s=%d",
            valves.jobs.outbox_poll_interval_ms,
            valves.jobs.outbox_max_attempts,
            valves.jobs.outbox_max_backoff_s,
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
            "description": "An active research job already exists for this chat.",
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
    creds: _BearerCreds,
) -> StartResearchResponse:
    chat_id = request.headers.get("X-OpenWebUI-Chat-Id")
    message_id = request.headers.get("X-OpenWebUI-Message-Id")

    if (
        creds is not None
        and creds.credentials
        and not chat_id
        and not getattr(request.app.state, "_forward_headers_warned", False)
    ):
        request.app.state._forward_headers_warned = True
        warning = ConfigWarning(
            code="OWUI_HEADERS_NOT_FORWARDED",
            severity="warning",
            message=(
                "An authenticated request arrived without "
                "X-OpenWebUI-Chat-Id. OWUI is not forwarding user-info "
                "headers; writeback is silently disabled."
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

    if chat_id:
        existing = await store.find_active_by_chat(chat_id)
        if existing is not None and existing.phase not in TERMINAL_PHASES:
            raise _error(
                409,
                "already_running",
                f"Active job exists for this chat: {existing.job_id}",
            )

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
    await store.create(record)
    await runner.start_job(record, view_token=view_token, owui_user_token=token)

    return StartResearchResponse(
        job_id=job_id,
        status="running",
        next_action="await_user_selection",
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

    await runner.submit_feedback(job_id, req.selection)
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
    if hashlib.sha256(token.encode("utf-8")).hexdigest() != record.view_token_hash:
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
    response_model=LiveViewSnapshot,
    tags=["live_view"],
)
async def live_view_status(
    job_id: str,
    request: Request,
    token: Annotated[str, Query()],
    since_version: Annotated[int | None, Query(ge=0)] = None,
) -> Response:
    store: JobStore = request.app.state.job_store
    runner: JobRunner = request.app.state.runner

    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404)
    if hashlib.sha256(token.encode("utf-8")).hexdigest() != record.view_token_hash:
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
    }

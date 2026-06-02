import asyncio
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from deep_research import Coordinator, Report
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.config.logging import (
    configure_logging,
    reset_log_context,
    set_log_context,
)
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import AlreadyRunningError, RuntimeConfig
from deep_research.progress.events import Event, StatusEvent

from .schemas import (
    Citation,
    HistoryMessage,
    JobProgress,
    ResearchErrorResponse,
    ResearchJobAccepted,
    ResearchJobStatus,
    ResearchMetadata,
    ResearchRequest,
    ResearchResponse,
)

logger = logging.getLogger("deep_research.entrypoints.openapi")

RESEARCH_DESCRIPTION = (
    "Run a multi-cycle deep research investigation and return the final"
    " report as structured JSON.\n\n"
    "Call this when the user's question benefits from grounded, cited"
    " research across the open web (technical comparisons, market or"
    " literature scans, multi-source explainers). Do not call it for"
    " trivial factual questions you can answer directly.\n\n"
    "The call is synchronous and may take several minutes for non-trivial"
    " prompts. For long-running queries that may exceed the tool-server"
    " timeout, prefer `start_research_job` + `get_research_job` instead.\n\n"
    "The response includes a markdown `report` with inline `[N]` citation"
    " markers that index into the `citations` array; surface the markdown"
    " to the user verbatim so the markers remain meaningful."
)

JOB_START_DESCRIPTION = (
    "Start a deep-research run asynchronously and return a `job_id`"
    " immediately. Use this for prompts that may exceed the tool-call"
    " timeout. Poll `get_research_job` with the returned `job_id` until"
    " its `status` is `completed` or `failed`."
)

JOB_GET_DESCRIPTION = (
    "Fetch the current status of a research job started via"
    " `start_research_job`. While running, returns a `progress` snapshot."
    " Once `status` is `completed`, the full report is available in"
    " `result` with the same shape as the synchronous `/research`"
    " response."
)

JOB_RETENTION_S = float(os.environ.get("DR_JOB_RETENTION_S", "3600"))

app = FastAPI(
    title="Deep Research Tool",
    description=(
        "OpenAPI tool server for the Deep Research engine. Exposes a"
        " synchronous `POST /research` endpoint plus an async job pattern"
        " (`POST /research_jobs`, `GET /research_jobs/{id}`) suitable for"
        " consumption by Open WebUI as a tool server."
    ),
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)
_coord: Coordinator | None = None


@dataclass
class _JobRecord:
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    result: ResearchResponse | None = None
    error: ResearchErrorResponse | None = None
    progress: JobProgress = field(default_factory=JobProgress)
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None
    task: asyncio.Task | None = None


_jobs: dict[str, _JobRecord] = {}
_jobs_lock = asyncio.Lock()


# --------------------------- helpers ---------------------------------


def _get_coord() -> Coordinator:
    if _coord is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Coordinator not initialised",
        )
    return _coord


_BearerCreds = Annotated[HTTPAuthorizationCredentials | None, Depends(security)]


def _api_token(creds: _BearerCreds) -> str:
    """Return the bearer token to forward to OWUI for this call.

    Falls back to `DR_OWUI_API_KEY` so operators can pin a server-side
    token rather than expecting the model to know it.
    """
    if creds is not None and creds.credentials:
        return creds.credentials
    return os.environ.get("DR_OWUI_API_KEY", "")


_ApiToken = Annotated[str, Depends(_api_token)]


def _to_history(items: Iterable[HistoryMessage]) -> list[ChatMessage]:
    return [ChatMessage(role=m.role, content=m.content) for m in items]


def _report_to_citations(report: Report) -> list[Citation]:
    """Translate `report.bibliography` (BibliographyEntry list) into the
    Citation response model, enriching with snippet/title from
    `report.sources` (= master_source_table) when available."""
    sources: dict[str, Any] = report.sources or {}
    out: list[Citation] = []
    for entry in report.bibliography:
        url = entry.get("url", "")
        title = entry.get("title", "") or sources.get(url, {}).get("title", "") or url
        snippet = None
        if url in sources:
            preview = sources[url].get("content_preview")
            if isinstance(preview, str) and preview.strip():
                snippet = preview.strip()[:500]
        out.append(
            Citation(
                id=int(entry.get("id", 0)),
                url=url,
                title=title,
                snippet=snippet,
            )
        )
    return out


def _build_response(report: Report, conversation_id: str, elapsed_s: float) -> ResearchResponse:
    return ResearchResponse(
        report=report.content,
        title=report.title,
        citations=_report_to_citations(report),
        conversation_id=report.conversation_id or conversation_id,
        metadata=ResearchMetadata(
            token_usage={k: int(v) for k, v in (report.token_usage or {}).items() if isinstance(v, int | float)},
            elapsed_s=elapsed_s,
            report_file_id=report.report_file_id,
        ),
    )


async def _gc_old_jobs() -> None:
    if JOB_RETENTION_S <= 0:
        return
    cutoff = time.monotonic() - JOB_RETENTION_S
    async with _jobs_lock:
        dead = [
            jid for jid, rec in _jobs.items()
            if rec.completed_at is not None and rec.completed_at < cutoff
        ]
        for jid in dead:
            _jobs.pop(jid, None)


# --------------------------- lifecycle -------------------------------


@app.on_event("startup")
async def _startup() -> None:
    global _coord
    valves = load_valves_from_env(prefix="DR_")
    configure_logging(valves)
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
    # Log presence-only for API keys: avoids any portion of the credential
    # reaching log archives. Operators see "set"/"unset" — enough to confirm
    # bootstrapping picked up the env vars.
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
    _coord = Coordinator(valves=valves, config=config)
    await _coord.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    async with _jobs_lock:
        pending = [rec.task for rec in _jobs.values() if rec.task is not None and not rec.task.done()]
    for task in pending:
        task.cancel()
    if _coord is not None:
        await _coord.close()


# --------------------------- endpoints -------------------------------


@app.post(
    "/research",
    operation_id="research",
    summary="Run a deep research investigation and return the final report",
    description=RESEARCH_DESCRIPTION,
    response_model=ResearchResponse,
    responses={
        409: {"model": ResearchErrorResponse, "description": "A run for this conversation is already in flight."},
        503: {"model": ResearchErrorResponse, "description": "The research coordinator is not yet initialised."},
        500: {"model": ResearchErrorResponse, "description": "Unhandled error inside the research engine."},
    },
    tags=["research"],
)
async def research(req: ResearchRequest, token: _ApiToken) -> ResearchResponse:
    coord = _get_coord()
    conv_id = req.conversation_id or f"api_{uuid4()}"
    req_id = str(uuid4())

    log_handle = set_log_context(
        conversation_id=conv_id,
        chat_id=req.chat_id or "-",
        request_id=req_id,
    )
    logger.info(
        "POST /research accepted: conversation_id=%s chat_id=%s prompt_chars=%d",
        conv_id,
        req.chat_id or "-",
        len(req.prompt or ""),
    )

    started = time.monotonic()
    try:
        async def _drop(_ev: Event) -> None:
            return None

        report = await coord.run(
            user=RunUser(id=req.user_id, name=req.user_name),
            conversation_id=conv_id,
            chat_id=req.chat_id,
            token=StaticToken(token),
            prompt=req.prompt,
            history=_to_history(req.history),
            sink=_drop,
        )
        return _build_response(report, conv_id, time.monotonic() - started)
    except AlreadyRunningError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ResearchErrorResponse(code="already_running", message=str(e)).model_dump(),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Research run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ResearchErrorResponse(code="internal_error", message=str(e)).model_dump(),
        ) from e
    finally:
        reset_log_context(log_handle)


@app.post(
    "/research_jobs",
    operation_id="start_research_job",
    summary="Start a deep research run asynchronously",
    description=JOB_START_DESCRIPTION,
    response_model=ResearchJobAccepted,
    responses={
        503: {"model": ResearchErrorResponse, "description": "The research coordinator is not yet initialised."},
    },
    status_code=status.HTTP_202_ACCEPTED,
    tags=["research"],
)
async def start_research_job(req: ResearchRequest, token: _ApiToken) -> ResearchJobAccepted:
    coord = _get_coord()
    await _gc_old_jobs()

    job_id = str(uuid4())
    conv_id = req.conversation_id or f"api_{uuid4()}"
    record = _JobRecord(job_id=job_id)
    async with _jobs_lock:
        _jobs[job_id] = record

    async def _runner() -> None:
        record.status = "running"
        started = time.monotonic()

        async def _progress_sink(ev: Event) -> None:
            if isinstance(ev, StatusEvent):
                record.progress = JobProgress(phase=ev.level or "info", message=ev.description)

        log_handle = set_log_context(
            conversation_id=conv_id,
            chat_id=req.chat_id or "-",
            request_id=job_id,
        )
        try:
            report = await coord.run(
                user=RunUser(id=req.user_id, name=req.user_name),
                conversation_id=conv_id,
                chat_id=req.chat_id,
                token=StaticToken(token),
                prompt=req.prompt,
                history=_to_history(req.history),
                sink=_progress_sink,
            )
            record.result = _build_response(report, conv_id, time.monotonic() - started)
            record.status = "completed"
        except AlreadyRunningError as e:
            record.error = ResearchErrorResponse(code="already_running", message=str(e))
            record.status = "failed"
        except Exception as e:
            logger.exception("Research job %s failed", job_id)
            record.error = ResearchErrorResponse(code="internal_error", message=str(e))
            record.status = "failed"
        finally:
            record.completed_at = time.monotonic()
            reset_log_context(log_handle)

    record.task = asyncio.create_task(_runner())
    return ResearchJobAccepted(job_id=job_id, poll_url=f"/research_jobs/{job_id}")


@app.get(
    "/research_jobs/{job_id}",
    operation_id="get_research_job",
    summary="Get the status (and eventual result) of a research job",
    description=JOB_GET_DESCRIPTION,
    response_model=ResearchJobStatus,
    responses={
        404: {"model": ResearchErrorResponse, "description": "Unknown or expired job_id."},
    },
    tags=["research"],
)
async def get_research_job(job_id: str) -> ResearchJobStatus:
    async with _jobs_lock:
        record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ResearchErrorResponse(
                code="unknown_job",
                message=f"No such job_id: {job_id}",
            ).model_dump(),
        )
    return ResearchJobStatus(
        job_id=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        result=record.result,
        error=record.error,
        progress=record.progress if record.status in ("pending", "running") else None,
    )


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}

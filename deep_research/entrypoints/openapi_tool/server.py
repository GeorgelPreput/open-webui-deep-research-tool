import json
import logging
import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from deep_research import Coordinator
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.config.logging import (
    configure_logging,
    redact_secret,
    reset_log_context,
    set_log_context,
)
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig

configure_logging()
logger = logging.getLogger("deep_research.entrypoints.openapi")

app = FastAPI(title="Deep Research Tool")
_coord: Coordinator | None = None


class ResearchRequest(BaseModel):
    prompt: str
    user_id: str = "api_user"
    user_name: str = "API User"
    conversation_id: str | None = None
    chat_id: str | None = None
    history: list[dict] = []


@app.on_event("startup")
async def _startup() -> None:
    global _coord
    valves = load_valves_from_env(prefix="DR_")
    configure_logging(valves)
    config = RuntimeConfig(
        data_dir=os.environ.get("DR_DATA_DIR", "/data/deep_research"),
        base_url=os.environ.get("DR_OWUI_BASE_URL", "http://localhost:8080"),
        chat_completions_path=os.environ.get(
            "DR_OWUI_CHAT_COMPLETIONS_PATH", "/api/chat/completions"
        ),
        chat_completions_fallback_path=os.environ.get(
            "DR_OWUI_CHAT_COMPLETIONS_FALLBACK_PATH", ""
        ),
    )
    logger.info(
        "OpenAPI server startup: base_url=%s data_dir=%s "
        "research_model=%s synthesis_model=%s embedding_model=%s api_key=%s",
        config.base_url,
        config.data_dir,
        valves.models.research_model,
        valves.models.synthesis_model,
        valves.models.embedding_model,
        redact_secret(os.environ.get("DR_OWUI_API_KEY", "")),
    )
    _coord = Coordinator(valves=valves, config=config)
    await _coord.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _coord is not None:
        await _coord.close()


@app.post("/research")
async def research(req: ResearchRequest) -> EventSourceResponse:
    if _coord is None:
        raise HTTPException(status_code=503, detail="Coordinator not initialised")
    coord = _coord
    api_key = os.environ.get("DR_OWUI_API_KEY", "")
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

    async def event_stream():
        try:
            async for event in coord.stream(
                user=RunUser(id=req.user_id, name=req.user_name),
                conversation_id=conv_id,
                chat_id=req.chat_id,
                token=StaticToken(api_key),
                prompt=req.prompt,
                history=[
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                    for m in req.history
                ],
            ):
                yield {"event": type(event).__name__, "data": json.dumps(event.to_dict())}
        finally:
            reset_log_context(log_handle)

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

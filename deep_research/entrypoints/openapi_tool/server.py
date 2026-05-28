import json
import os
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from deep_research import Coordinator
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig

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
    config = RuntimeConfig(
        data_dir=os.environ.get("DR_DATA_DIR", "/data/deep_research")
    )
    _coord = Coordinator(valves=valves, config=config)
    await _coord.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _coord is not None:
        await _coord.close()


@app.post("/research")
async def research(req: ResearchRequest) -> EventSourceResponse:
    api_key = os.environ.get("DR_OWUI_API_KEY", "")
    conv_id = req.conversation_id or f"api_{uuid4()}"

    async def event_stream():
        async for event in _coord.stream(
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

    return EventSourceResponse(event_stream())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

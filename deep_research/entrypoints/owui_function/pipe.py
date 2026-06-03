import asyncio
import os
from typing import Any
from uuid import uuid4

from deep_research import Coordinator
from deep_research import Valves as DRValves
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.config.logging import (
    configure_logging,
    reset_log_context,
    set_log_context,
)
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.events import (
    CitationEvent,
    EmbedEvent,
    Event,
    MessageEvent,
    StatusEvent,
)

name = "Deep Research"


class Pipe:
    type = "manifold"
    name = "deep_research"

    class Valves(DRValves):
        pass

    def __init__(self) -> None:
        self.valves = Pipe.Valves()
        self._coordinator: Coordinator | None = None
        self._coord_lock = asyncio.Lock()

    def pipes(self) -> list[dict]:
        return [{"id": "deep_research", "name": "Deep Research"}]

    async def _ensure_coordinator(self) -> Coordinator:
        if self._coordinator is None:
            async with self._coord_lock:
                if self._coordinator is None:
                    configure_logging(self.valves)
                    # Prefer admin-UI valve values (self.valves.llm.*);
                    # fall back to env via a fresh load only for fields the admin
                    # UI might not have populated (rare in Function runtime).
                    env_valves = load_valves_from_env(prefix="DR_")
                    llm = self.valves.llm
                    emb = self.valves.embeddings
                    # If the admin UI left base_url blank, try the env value.
                    if not llm.base_url:
                        llm = env_valves.llm
                    if not emb.base_url:
                        emb = env_valves.embeddings
                    config = RuntimeConfig(
                        data_dir=os.environ.get("DR_DATA_DIR", "/tmp/deep_research"),
                        base_url=os.environ.get("DR_OWUI_BASE_URL", "http://localhost:8080"),
                        llm_base_url=llm.base_url,
                        llm_api_key=llm.api_key,
                        llm_chat_path=llm.chat_path,
                        embeddings_base_url=emb.base_url,
                        embeddings_api_key=emb.api_key,
                        embeddings_path=emb.embeddings_path,
                    )
                    self._coordinator = Coordinator(valves=self.valves, config=config)
                    await self._coordinator.start()
        return self._coordinator

    async def pipe(self, body: dict, __user__: dict, __request__: Any, __event_emitter__: Any, **kwargs) -> str:
        coord = await self._ensure_coordinator()

        token_str = _extract_bearer(__request__) or os.environ.get("DR_OWUI_API_KEY", "")
        token = StaticToken(token_str)

        user = RunUser(
            id=__user__.get("id", ""),
            name=__user__.get("name", ""),
            email=__user__.get("email"),
        )

        conversation_id = str(body.get("metadata", {}).get("chat_id", ""))
        chat_id = body.get("metadata", {}).get("chat_id")

        messages_raw = body.get("messages", [])
        prompt = messages_raw[-1].get("content", "") if messages_raw else ""
        history = [
            ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in messages_raw[:-1]
        ]

        log_handle = set_log_context(
            conversation_id=conversation_id or "-",
            chat_id=str(chat_id) if chat_id else "-",
            request_id=str(uuid4()),
        )

        async def sink(event: Event) -> None:
            await _translate_event(event, __event_emitter__)

        try:
            report = await coord.run(
                user=user,
                conversation_id=conversation_id,
                chat_id=chat_id,
                token=token,
                prompt=prompt,
                history=history,
                sink=sink,
            )
            return report.content
        finally:
            reset_log_context(log_handle)


def _extract_bearer(request: Any) -> str:
    try:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
    except Exception:
        pass
    return ""


async def _translate_event(event: Event, emitter: Any) -> None:
    if isinstance(event, StatusEvent):
        await emitter({
            "type": "status",
            "data": {
                "status": "in_progress" if not event.done else "complete",
                "description": event.description,
                "done": event.done,
            },
        })
    elif isinstance(event, MessageEvent):
        await emitter({
            "type": "message",
            "data": {"content": event.content},
        })
    elif isinstance(event, EmbedEvent):
        # OWUI >=0.9.5: the `replace` flag overwrites message.embeds instead of
        # appending. Without it, every progress refresh stacks another iframe on
        # reload (open-webui#23940). Older OWUI versions ignore the flag and
        # fall back to append behaviour, so this is forward-only and safe.
        embed_dict = {"name": event.title or "Research Progress", "html": event.html}
        await emitter({
            "type": "embeds",
            "data": {"embeds": [embed_dict], "replace": True},
        })
    elif isinstance(event, CitationEvent):
        await emitter(event.to_dict())

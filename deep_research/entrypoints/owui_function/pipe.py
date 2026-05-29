import asyncio
import os
from typing import Any

from deep_research import Coordinator
from deep_research import Valves as DRValves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.events import EmbedEvent, Event, MessageEvent, StatusEvent

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
                    config = RuntimeConfig(
                        data_dir=os.environ.get("DR_DATA_DIR", "/tmp/deep_research"),
                        base_url=os.environ.get("DR_OWUI_BASE_URL", "http://localhost:8080"),
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

        async def sink(event: Event) -> None:
            await _translate_event(event, __event_emitter__)

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


def _extract_bearer(request: Any) -> str:
    try:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
    except Exception:
        pass
    return ""


def _resolve_conversation_id(body: dict, request: Any) -> str:
    chat_id = body.get("metadata", {}).get("chat_id", "")
    if chat_id:
        return str(chat_id)
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

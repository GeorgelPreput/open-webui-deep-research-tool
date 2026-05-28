import asyncio
import os
import queue
import threading
import time
from collections.abc import Iterator

from deep_research import Coordinator
from deep_research import Valves as BaseValves
from deep_research.adapter.auth import StaticToken
from deep_research.core.types import ChatMessage, RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.events import Event, MessageEvent, StatusEvent


class _BridgeSink:
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=64)

    def put(self, text: str) -> None:
        self._queue.put(text)

    def done(self) -> None:
        self._queue.put(None)

    def __iter__(self) -> Iterator[str]:
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item


class _ReasoningBlock:
    def __init__(self, summary: str = "Research") -> None:
        self._parts: list[str] = []
        self._summary = summary
        self._opened_at = time.monotonic()

    def add(self, text: str) -> None:
        self._parts.append(text)

    def render(self, done: bool = False, duration: float = 0) -> str:
        body = "\n".join(self._parts)
        attrs = 'type="reasoning"'
        if done:
            attrs += f' done="true" duration="{duration:.1f}"'
        return f"\n\n<details {attrs}>\n<summary>{self._summary}</summary>\n{body}\n</details>\n\n"


class Pipeline:
    class Valves(BaseValves):
        OWUI_BASE_URL: str = "http://localhost:8080"
        OWUI_API_KEY: str = ""

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.type = "manifold"
        self._coord: Coordinator | None = None
        self._coord_lock = threading.Lock()

    def pipelines(self) -> list[dict]:
        return [{"id": "deep_research", "name": "Deep Research"}]

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        if self._coord is not None:
            await self._coord.close()

    def _ensure_coordinator(self) -> Coordinator:
        if self._coord is None:
            with self._coord_lock:
                if self._coord is None:
                    config = RuntimeConfig(
                        data_dir=os.environ.get("DR_DATA_DIR", "/tmp/deep_research")
                    )
                    self._coord = Coordinator(valves=self.valves, config=config)
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._coord.start())
                    loop.close()
        return self._coord

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict) -> Iterator[str]:
        sink = _BridgeSink()
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(sink, user_message, messages, body),
            daemon=True,
        )
        thread.start()
        return iter(sink)

    def _run_in_thread(
        self, sink: _BridgeSink, user_message: str, messages: list, body: dict
    ) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_run(sink, user_message, messages, body))
        finally:
            loop.close()

    async def _async_run(
        self, sink: _BridgeSink, user_message: str, messages: list, body: dict
    ) -> None:
        coord = self._ensure_coordinator()
        token = StaticToken(self.valves.OWUI_API_KEY)
        user = RunUser(id="pipeline_user", name="Pipeline User")

        chat_id = body.get("chat_id") if isinstance(body, dict) else None
        conversation_id = str(chat_id or "pipeline_conv")

        history = [
            ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in (messages or [])[:-1]
        ]

        reasoning = _ReasoningBlock(summary="Deep Research")

        async def event_sink(event: Event) -> None:
            if isinstance(event, StatusEvent):
                reasoning.add(f"- {event.description}")
                if time.monotonic() - reasoning._opened_at > 4.0:
                    sink.put(reasoning.render(done=False))
                    reasoning._parts = []
                    reasoning._opened_at = time.monotonic()
            elif isinstance(event, MessageEvent):
                if reasoning._parts:
                    duration = time.monotonic() - reasoning._opened_at
                    sink.put(reasoning.render(done=True, duration=duration))
                    reasoning._parts = []
                sink.put(event.content)

        result = await coord.run(
            user=user,
            conversation_id=conversation_id,
            chat_id=chat_id,
            token=token,
            prompt=user_message,
            history=history,
            sink=event_sink,
        )

        if reasoning._parts:
            sink.put(reasoning.render(done=True, duration=time.monotonic() - reasoning._opened_at))
        if result and result.content:
            sink.put(result.content)
        sink.done()

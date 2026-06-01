import asyncio
import os
import queue
import threading
import time
from collections.abc import Iterator
from uuid import uuid4

from deep_research import Coordinator
from deep_research import Valves as BaseValves
from deep_research.adapter.auth import StaticToken
from deep_research.config.logging import (
    configure_logging,
    reset_log_context,
    set_log_context,
)
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

    def pipelines(self) -> list[dict]:
        return [{"id": "deep_research", "name": "Deep Research"}]

    async def on_startup(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        # The Coordinator is built and torn down per call inside the worker
        # loop (its httpx client/semaphores are loop-bound), so there is no
        # long-lived coordinator to close here.
        pass

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
        reasoning = _ReasoningBlock(summary="Deep Research")
        log_handle: object | None = None
        try:
            configure_logging(self.valves)
            # Build + start the Coordinator on THIS (per-call worker) loop: its
            # httpx client and semaphores are bound to the loop they're created
            # on, so a shared/long-lived coordinator from another loop is unusable.
            config = RuntimeConfig(
                data_dir=os.environ.get("DR_DATA_DIR", "/tmp/deep_research"),
                base_url=os.environ.get("DR_OWUI_BASE_URL", self.valves.OWUI_BASE_URL),
                chat_completions_path=os.environ.get(
                    "DR_OWUI_CHAT_COMPLETIONS_PATH", "/api/chat/completions"
                ),
                chat_completions_fallback_path=os.environ.get(
                    "DR_OWUI_CHAT_COMPLETIONS_FALLBACK_PATH", ""
                ),
            )
            coord = Coordinator(valves=self.valves, config=config)
            await coord.start()
            try:
                token = StaticToken(self.valves.OWUI_API_KEY)
                user = RunUser(id="pipeline_user", name="Pipeline User")

                chat_id = body.get("chat_id") if isinstance(body, dict) else None
                conversation_id = str(chat_id or "pipeline_conv")

                log_handle = set_log_context(
                    conversation_id=conversation_id,
                    chat_id=str(chat_id) if chat_id else "-",
                    request_id=str(uuid4()),
                )

                history = [
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                    for m in (messages or [])[:-1]
                ]

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
            finally:
                await coord.close()
        except Exception as e:
            sink.put(f"\n\n**Research failed:** {e}\n")
        finally:
            if log_handle is not None:
                reset_log_context(log_handle)
            # Always enqueue the sentinel, or _BridgeSink.__iter__ blocks forever.
            sink.done()

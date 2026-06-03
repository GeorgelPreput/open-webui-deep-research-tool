from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StatusEvent:
    description: str
    level: str = "info"
    done: bool = False

    def to_dict(self) -> dict:
        return {"type": "status", "data": {
            "status": "in_progress" if not self.done else "complete",
            "description": self.description,
            "done": self.done,
        }}


@dataclass(frozen=True, slots=True)
class MessageEvent:
    content: str

    def to_dict(self) -> dict:
        return {"type": "message", "data": {"content": self.content}}


@dataclass(frozen=True, slots=True)
class EmbedEvent:
    html: str
    title: str | None = None

    def to_dict(self) -> dict:
        return {"type": "embeds", "data": {
            "embeds": [{"name": self.title, "html": self.html}]
        }}


@dataclass(frozen=True, slots=True)
class CitationEvent:
    """Per-source citation surfaced once at finalize time.

    Maps to OWUI's ``source`` event type, which the chat UI renders in
    the message's side-panel citation list. The Function runtime can
    convert it to the same OWUI shape; the OpenAPI runner translates it
    to an outbox row.
    """
    url: str
    title: str
    snippet: str | None = None

    def to_dict(self) -> dict:
        return {"type": "source", "data": {
            "type": "external",
            "source": {"type": "external", "name": self.title or self.url},
            "document": [self.snippet or ""],
            "metadata": [{"source": self.url}],
        }}


Event = StatusEvent | MessageEvent | EmbedEvent | CitationEvent
Sink = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self, sink: Sink, flush_interval_ms: int = 400):
        self._sink = sink
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._flush_interval = flush_interval_ms / 1000.0
        self._task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._flusher())

    async def stop(self) -> None:
        await self._queue.put(None)
        self._wake_event.set()
        if self._task is not None:
            await self._task

    async def emit(self, event: Event) -> None:
        await self._queue.put(event)
        self._wake_event.set()

    async def emit_critical(self, event: Event) -> None:
        await self._sink(event)

    async def _flusher(self) -> None:
        pending: list[Event] = []
        last_status: StatusEvent | None = None
        last_embed: EmbedEvent | None = None
        last_flush = time.monotonic()

        while True:
            try:
                event = self._queue.get_nowait()
                if event is None:
                    break
                if isinstance(event, StatusEvent):
                    last_status = event
                elif isinstance(event, EmbedEvent):
                    if last_embed is not None:
                        pending.append(last_embed)
                    last_embed = event
                else:
                    pending.append(event)
                continue
            except asyncio.QueueEmpty:
                pass

            now = time.monotonic()
            if now - last_flush >= self._flush_interval and (last_status is not None or pending or last_embed is not None):
                for ev in pending:
                    await self._sink(ev)
                pending.clear()
                if last_embed is not None:
                    await self._sink(last_embed)
                    last_embed = None
                if last_status is not None:
                    await self._sink(last_status)
                    last_status = None
                last_flush = now

            self._wake_event.clear()
            wait = max(0.0, self._flush_interval - (time.monotonic() - last_flush))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake_event.wait(), timeout=wait)

        for ev in pending:
            await self._sink(ev)
        if last_embed is not None:
            await self._sink(last_embed)
        if last_status is not None:
            await self._sink(last_status)

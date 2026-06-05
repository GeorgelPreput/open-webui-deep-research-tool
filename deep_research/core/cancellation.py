"""Cooperative-cancellation primitive for long-running research runs.

The engine's phases call ``ctx.cancellation_token.raise_if_cancelled()``
at boundaries (start of each phase, start of each cycle iteration).
Entrypoints that want to expose cancellation (the OpenAPI runtime's
``POST /research_jobs/{id}/cancel``, MCP's cancellation notifications)
create a token, pass it into ``Coordinator.run(cancellation_token=...)``,
and call ``token.cancel()`` when the cancellation source fires.

Entrypoints that don't want to expose cancellation (the Function path)
pass ``None``; the engine's checks become no-ops.
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    """A one-shot, cooperative cancellation flag.

    Backed by an ``asyncio.Event`` so awaiting code can ``await
    token.wait_cancelled()`` for early termination. Phase boundaries
    instead call the synchronous ``raise_if_cancelled()`` which raises
    ``asyncio.CancelledError`` — surfacing through the engine's normal
    exception path and triggering ``Coordinator.run`` to unwind.
    """

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = asyncio.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise asyncio.CancelledError()

    async def wait_cancelled(self) -> None:
        await self._cancelled.wait()

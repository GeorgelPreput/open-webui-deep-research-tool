import asyncio
import logging
import os
from uuid import uuid4

from fastmcp import Context, FastMCP

from deep_research import Coordinator
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.config.logging import configure_logging
from deep_research.core.cancellation import CancellationToken
from deep_research.core.types import RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.events import Event, StatusEvent

configure_logging()
logger = logging.getLogger("deep_research.entrypoints.mcp")
mcp = FastMCP("deep_research")
_coord: Coordinator | None = None
_coord_lock = asyncio.Lock()


@mcp.tool()
async def deep_research(
    prompt: str,
    ctx: Context,
    conversation_id: str | None = None,
) -> str:
    """Run a deep research investigation on `prompt`. Returns the final report.

    Engine ``StatusEvent``s are surfaced via MCP progress notifications
    (``ctx.report_progress(progress=N, message=...)``). Other event types
    are dropped — the final report is returned from the tool call itself.

    Cancellation: when the MCP request is cancelled, anyio cancels the
    enclosing task and ``asyncio.CancelledError`` unwinds the engine; the
    ``CancellationToken`` we hand to ``coord.run`` is also cancelled so
    any phase-boundary check sees the signal even if the cancellation
    arrives between awaits.
    """
    global _coord
    if _coord is None:
        async with _coord_lock:
            if _coord is None:
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
                coord = Coordinator(valves=valves, config=config)
                await coord.start()
                # Publish only after a successful start so a failed start
                # doesn't latch a half-initialized coordinator.
                _coord = coord

    api_key = os.environ.get("DR_OWUI_API_KEY", "")
    cancel_token = CancellationToken()
    progress_counter = 0

    async def sink(event: Event) -> None:
        nonlocal progress_counter
        if isinstance(event, StatusEvent):
            progress_counter += 1
            try:
                await ctx.report_progress(
                    progress=float(progress_counter),
                    message=event.description,
                )
            except Exception:
                # Progress is best-effort; never let a transport hiccup
                # poison the engine sink.
                logger.debug(
                    "report_progress failed (ignored)", exc_info=True
                )

    try:
        report = await _coord.run(
            user=RunUser(id="mcp_user", name="MCP Client"),
            conversation_id=conversation_id or f"mcp_{uuid4()}",
            chat_id=None,
            token=StaticToken(api_key),
            prompt=prompt,
            history=[],
            sink=sink,
            cancellation_token=cancel_token,
        )
        return report.content
    except asyncio.CancelledError:
        cancel_token.cancel()
        raise


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=9000)

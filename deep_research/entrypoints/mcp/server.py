import os
from uuid import uuid4

from fastmcp import FastMCP

from deep_research import Coordinator
from deep_research.adapter.auth import StaticToken
from deep_research.config.env import load_valves_from_env
from deep_research.core.types import RunUser
from deep_research.orchestrator.coordinator import RuntimeConfig
from deep_research.progress.events import Event

mcp = FastMCP("deep_research")
_coord: Coordinator | None = None


@mcp.tool()
async def deep_research(prompt: str, conversation_id: str | None = None) -> str:
    """Run a deep research investigation on `prompt`. Returns the final report."""
    global _coord
    if _coord is None:
        valves = load_valves_from_env(prefix="DR_")
        config = RuntimeConfig(
            data_dir=os.environ.get("DR_DATA_DIR", "/data/deep_research")
        )
        _coord = Coordinator(valves=valves, config=config)
        await _coord.start()

    api_key = os.environ.get("DR_OWUI_API_KEY", "")
    sink_events: list[Event] = []

    async def sink(event: Event) -> None:
        sink_events.append(event)

    report = await _coord.run(
        user=RunUser(id="mcp_user", name="MCP Client"),
        conversation_id=conversation_id or f"mcp_{uuid4()}",
        chat_id=None,
        token=StaticToken(api_key),
        prompt=prompt,
        history=[],
        sink=sink,
    )
    return report.content


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=9000)

import logging
from typing import Any

from deep_research.core.types import RunContext
from deep_research.progress.events import StatusEvent
from deep_research.synthesis.outline import generate_synthesis_outline

logger = logging.getLogger("deep_research.orchestrator.phases.outline")


async def run_outline(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    conv_state = ctx.state.get_state(ctx.conversation_id)

    research_outline = ps.get("research_outline", conv_state.get("research_state", {}).get("research_outline", []))
    completed_topics = set(conv_state.get("completed_topics", []))
    user_message = ps.get("user_message", "")
    results_history = conv_state.get("results_history", []) + (ps.get("initial_results") or [])

    await ctx.events.emit(StatusEvent(description="Generating refined outline for synthesis...", level="info", done=False))

    if not research_outline:
        logger.warning("No research outline available for synthesis outline generation")
        ps["synthesis_outline"] = []
        return ps

    synthesis_outline = await generate_synthesis_outline(
        ctx,
        research_outline,
        completed_topics,
        user_message,
        results_history,
    )

    if not synthesis_outline:
        synthesis_outline = research_outline

    ps["synthesis_outline"] = synthesis_outline
    ps["results_history"] = results_history
    return ps

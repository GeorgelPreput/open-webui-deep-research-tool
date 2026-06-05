import logging
from typing import Any

from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import initialize_research_state
from deep_research.progress.events import StatusEvent
from deep_research.research.outline_feedback import (
    continue_research_after_feedback,
    process_outline_feedback_continuation,
)

logger = logging.getLogger("deep_research.orchestrator.phases.outline_feedback")


async def run_outline_feedback(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    ctx.raise_if_cancelled()
    conv_state = ctx.state.get_state(ctx.conversation_id)
    if not conv_state.get("waiting_for_outline_feedback", False):
        return state

    feedback_data = conv_state.get("outline_feedback_data", {})
    if not feedback_data:
        conv_state["waiting_for_outline_feedback"] = False
        return state

    ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", False)

    user_message = state.get("user_message", "")
    feedback_result = await process_outline_feedback_continuation(ctx, user_message)

    original_query = feedback_data.get("original_query", "")
    outline_items = feedback_data.get("outline_items", [])

    all_topics = []
    for topic_item in outline_items:
        all_topics.append(topic_item["topic"])
        all_topics.extend(topic_item.get("subtopics", []))

    outline_text = " ".join(all_topics)
    from deep_research.semantics.embeddings import get_embedding
    outline_embedding = await get_embedding(ctx, outline_text)

    research_outline, all_topics, outline_embedding = await continue_research_after_feedback(
        ctx,
        feedback_result,
        original_query,
        outline_items,
        all_topics,
        outline_embedding,
    )

    await initialize_research_state(ctx, original_query, research_outline, all_topics, outline_embedding)
    await ctx.events.emit(StatusEvent(description="Outline feedback processed", level="info", done=False))

    state["user_message"] = original_query
    state["research_outline"] = research_outline
    state["all_topics"] = all_topics
    state["outline_embedding"] = outline_embedding
    # Signal to the coordinator that the outline gate has been resumed
    # this turn: `initial_queries` already ran on the prior turn (it's
    # what produced the gate the user just answered). Re-running it now
    # would regenerate the outline from scratch and re-arm the gate.
    state["outline_finalized"] = True
    return state

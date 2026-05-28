import logging
from typing import Any

from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import (
    checkpoint,
    get_dr_state,
    load_persisted_dr_state,
    new_dr_state,
    set_dr_state,
)
from deep_research.persistence.kb import ensure_research_kb, rehydrate_working_corpus_from_kb
from deep_research.progress.events import StatusEvent

logger = logging.getLogger("deep_research.orchestrator.phases.rehydrate")


async def run_rehydrate(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    await ctx.events.emit(StatusEvent(description="Rehydrating research state...", level="info", done=False))

    persisted_dr = await load_persisted_dr_state(ctx, ctx.chat_id)
    if persisted_dr and not get_dr_state(ctx):
        set_dr_state(ctx, persisted_dr)
        logger.info(
            f"Rehydrated deepResearch checkpoint for chat={ctx.chat_id} "
            f"mode={persisted_dr.get('mode')} status={persisted_dr.get('status')} "
            f"sources={len(persisted_dr.get('source_manifest') or {})}"
        )
        try:
            await rehydrate_working_corpus_from_kb(ctx)
        except Exception as e:
            logger.warning(f"Working corpus rehydration failed: {e}")

    dr = get_dr_state(ctx)
    if dr and dr.get("mode") == "post_report_user_qa" and dr.get("kb_id"):
        state["post_report_mode"] = True
        return state

    if dr is None:
        user_request = state.get("user_message", "")
        dr_init = new_dr_state(ctx, user_request=user_request)
        set_dr_state(ctx, dr_init)
        try:
            await ensure_research_kb(ctx, user_request)
        except Exception as e:
            logger.warning(f"Initial KB provisioning failed: {e}")
        await checkpoint(ctx)

    conv_state = ctx.state.get_state(ctx.conversation_id)
    if "master_source_table" not in conv_state:
        ctx.state.update_state(ctx.conversation_id, "master_source_table", {})
    if "memory_stats" not in conv_state:
        ctx.state.update_state(ctx.conversation_id, "memory_stats", {
            "results_tokens": 0, "section_tokens": {}, "synthesis_tokens": 0, "total_tokens": 0,
        })
    if "url_selected_count" not in conv_state:
        ctx.state.update_state(ctx.conversation_id, "url_selected_count", {})
    if "url_token_counts" not in conv_state:
        ctx.state.update_state(ctx.conversation_id, "url_token_counts", {})

    return state

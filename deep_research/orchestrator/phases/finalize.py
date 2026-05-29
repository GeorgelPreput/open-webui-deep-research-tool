import logging
from typing import Any

from deep_research.core.types import Report, RunContext
from deep_research.persistence.chat_state import checkpoint, get_dr_state
from deep_research.persistence.kb import attach_collection_to_chat, persist_final_report
from deep_research.progress.embed import refresh_progress_embed
from deep_research.progress.events import StatusEvent

logger = logging.getLogger("deep_research.orchestrator.phases.finalize")


async def run_finalize(ctx: RunContext, ps: dict[str, Any]) -> Report:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    comprehensive_answer = ps.get("comprehensive_answer", "")
    titles = ps.get("titles", {})
    user_message = ps.get("user_message", "")

    if not comprehensive_answer:
        logger.warning("No comprehensive answer to finalize")
        return Report(content="", conversation_id=ctx.conversation_id)

    memory_stats = conv_state.get("memory_stats", {})
    results_tokens = memory_stats.get("results_tokens", 0)
    synthesis_tokens = memory_stats.get("synthesis_tokens", 0)
    section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
    total_tokens = results_tokens + section_tokens_sum + synthesis_tokens
    memory_stats["total_tokens"] = total_tokens
    conv_state["memory_stats"] = memory_stats

    conv_state["research_completed"] = True

    report_title = (
        titles.get("main_title") if isinstance(titles, dict) else None
    ) or f"Deep Research Report: {user_message[:80]}"

    report_file_id: str | None = None
    try:
        report_file_id = await persist_final_report(ctx, comprehensive_answer, report_title)
        dr = get_dr_state(ctx)
        if dr and dr.get("kb_id"):
            await attach_collection_to_chat(ctx, ctx.chat_id, dr["kb_id"], dr.get("kb_name") or dr["kb_id"])
        await checkpoint(ctx)
        if report_file_id:
            logger.info(f"Final report persisted to KB file_id={report_file_id}")
    except Exception as e:
        logger.warning(f"Final-report persistence failed: {e}")

    await ctx.events.emit(StatusEvent(description="Final synthesis complete!", level="info", done=True))

    dr = get_dr_state(ctx)
    if dr:
        dr["mode"] = "post_report_user_qa"
        from deep_research.persistence.chat_state import set_dr_state
        set_dr_state(ctx, dr)

    conv_state["prev_comprehensive_summary"] = comprehensive_answer

    await refresh_progress_embed(ctx, force=True)

    cache_stats = ctx.caches.embedding.stats()
    logger.info(f"Embedding cache stats: {cache_stats}")

    if ctx.valves.persistence.export_research_data:
        try:
            await ctx.events.emit(StatusEvent(description="Exporting research data...", level="info", done=False))
        except Exception as e:
            logger.error(f"Export failed: {e}")

    await ctx.events.emit(StatusEvent(description="Deep research complete!", level="success", done=True))

    sources = conv_state.get("master_source_table", {})
    return Report(
        content=comprehensive_answer,
        title=report_title,
        sources=sources,
        token_usage=memory_stats,
        report_file_id=report_file_id,
        conversation_id=ctx.conversation_id,
    )

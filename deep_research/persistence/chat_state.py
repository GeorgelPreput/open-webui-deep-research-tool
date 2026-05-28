import logging
from datetime import datetime
from typing import Any

from deep_research.budget.tokens import count_tokens
from deep_research.core.state import TrajectoryAccumulator
from deep_research.core.types import RunContext

logger = logging.getLogger("Deep Research")

DR_STATE_VERSION = 2

async def initialize_research_state(
    ctx: RunContext,
    user_message,
    research_outline,
    all_topics,
    outline_embedding,
    initial_results=None,
):
    """Initialize or reset research state consistently across interactive and non-interactive modes"""
    state = ctx.state.get_state(ctx.conversation_id)

    # Core research state
    ctx.state.update_state(ctx.conversation_id,
        "research_state",
        {
            "research_outline": research_outline,
            "all_topics": all_topics,
            "outline_embedding": outline_embedding,
            "user_message": user_message,
        },
    )

    # Initialize memory statistics with proper structure
    memory_stats = state.get("memory_stats", {})
    if not memory_stats or not isinstance(memory_stats, dict):
        memory_stats = {
            "results_tokens": 0,
            "section_tokens": {},
            "synthesis_tokens": 0,
            "total_tokens": 0,
        }
    ctx.state.update_state(ctx.conversation_id, "memory_stats", memory_stats)

    # Update results_tokens if we have initial results
    if initial_results:
        results_tokens = 0
        for result in initial_results:
            # Get or calculate tokens for this result
            tokens = result.get("tokens", 0)
            if tokens == 0 and "content" in result:
                tokens = await count_tokens(ctx, str(result.get("content", "")))
                result["tokens"] = tokens
            results_tokens += tokens

        # Update memory stats with token count
        memory_stats["results_tokens"] = results_tokens
        ctx.state.update_state(ctx.conversation_id, "memory_stats", memory_stats)

    # Initialize tracking variables
    ctx.state.update_state(ctx.conversation_id, "topic_usage_counts", state.get("topic_usage_counts", {}))
    ctx.state.update_state(ctx.conversation_id, "completed_topics", list(state.get("completed_topics", [])))
    ctx.state.update_state(ctx.conversation_id, "irrelevant_topics", list(state.get("irrelevant_topics", [])))
    ctx.state.update_state(ctx.conversation_id, "active_outline", all_topics.copy())
    ctx.state.update_state(ctx.conversation_id, "cycle_summaries", state.get("cycle_summaries", []))

    # Progress artifact state (reset per run so the embed starts fresh)
    ctx.state.update_state(ctx.conversation_id, "partial_topics", [])
    ctx.state.update_state(ctx.conversation_id, "latest_new_topics", [])
    ctx.state.update_state(ctx.conversation_id, "latest_completed_topics", [])
    ctx.state.update_state(ctx.conversation_id, "latest_irrelevant_topics", [])
    ctx.state.update_state(ctx.conversation_id, "all_topics", all_topics.copy())
    ctx.state.update_state(ctx.conversation_id, "progress_cycle", 0)
    ctx.state.update_state(ctx.conversation_id, "progress_embed_last_hash", "")
    ctx.state.update_state(ctx.conversation_id, "progress_embed_revision", 0)
    ctx.state.update_state(ctx.conversation_id, "progress_last_updated_at", "")
    ctx.state.update_state(ctx.conversation_id, "current_cycle_queries", [])

    # Results tracking
    results_history = state.get("results_history", [])
    if initial_results:
        results_history.extend(initial_results)
    ctx.state.update_state(ctx.conversation_id, "results_history", results_history)

    # Search history
    search_history = state.get("search_history", [])
    ctx.state.update_state(ctx.conversation_id, "search_history", search_history)

    # Initialize dimension tracking
    research_dimensions = state.get("research_dimensions")
    if research_dimensions:
        ctx.state.update_state(ctx.conversation_id,
            "latest_dimension_coverage", research_dimensions["coverage"].copy()
        )

    # Source tracking
    ctx.state.update_state(ctx.conversation_id, "master_source_table", state.get("master_source_table", {}))
    ctx.state.update_state(ctx.conversation_id, "url_selected_count", state.get("url_selected_count", {}))
    ctx.state.update_state(ctx.conversation_id, "url_token_counts", state.get("url_token_counts", {}))

    # Trajectory accumulator reset
    ctx.trajectory_accumulator = TrajectoryAccumulator()

    logger.info(
        f"Research state initialized with {len(all_topics)} topics and {len(results_history)} initial results"
    )

async def update_token_counts(ctx: RunContext, new_results=None):
    """Centralized function to update token counts consistently"""
    state = ctx.state.get_state(ctx.conversation_id)
    memory_stats = state.get(
        "memory_stats",
        {
            "results_tokens": 0,
            "section_tokens": {},
            "synthesis_tokens": 0,
            "total_tokens": 0,
        },
    )

    # Update results tokens if new results provided
    if new_results:
        for result in new_results:
            tokens = result.get("tokens", 0)
            if tokens == 0 and "content" in result:
                tokens = await count_tokens(ctx, str(result.get("content", "")))
                result["tokens"] = tokens
            memory_stats["results_tokens"] += tokens

    # If no results tokens but we have results history, recalculate
    results_history = state.get("results_history", [])
    if memory_stats["results_tokens"] == 0 and results_history:
        total_tokens = 0
        for result in results_history:
            tokens = result.get("tokens", 0)
            if tokens == 0 and "content" in result:
                tokens = await count_tokens(ctx, str(result.get("content", "")))
                result["tokens"] = tokens
            total_tokens += tokens
        memory_stats["results_tokens"] = total_tokens

    # Recalculate total tokens
    section_tokens_sum = sum(memory_stats.get("section_tokens", {}).values())
    memory_stats["total_tokens"] = (
        memory_stats["results_tokens"]
        + section_tokens_sum
        + memory_stats.get("synthesis_tokens", 0)
    )

    # Update state
    ctx.state.update_state(ctx.conversation_id, "memory_stats", memory_stats)

    return memory_stats

def new_dr_state(ctx: RunContext, *, user_request: str = "") -> dict[str, Any]:
    """Build a fresh deepResearch checkpoint object."""
    now_iso = datetime.now().isoformat()
    return {
        "version": DR_STATE_VERSION,
        "mode": "research",
        "status": "discovering",
        "kb_id": None,
        "kb_name": None,
        "created_at": now_iso,
        "last_checkpoint_at": now_iso,
        "conversation_title_snapshot": "",
        "source_manifest": {},
        "report_file_id": None,
        "report_completed": False,
        "section_plan": [],
        "completed_sections": [],
        "pending_sections": [],
        "resume_cursor": {
            "phase": "init",
            "current_section": "",
            "current_url": "",
        },
        "token_usage": {
            "research_model": {"prompt": 0, "completion": 0, "total": 0},
            "report_writer": {"prompt": 0, "completion": 0, "total": 0},
            "synthesis_time_kb_qa": {"prompt": 0, "completion": 0, "total": 0},
            "grand_total": {"prompt": 0, "completion": 0, "total": 0},
        },
        "user_request_summary": user_request[:1000] if user_request else "",
        "followup_constraints_summary": "",
    }

def get_dr_state(ctx: RunContext) -> dict[str, Any] | None:
    return ctx.state.get_state(ctx.conversation_id).get("dr_state")

def set_dr_state(ctx: RunContext, dr_state: dict[str, Any]) -> None:
    dr_state["last_checkpoint_at"] = datetime.now().isoformat()
    ctx.state.update_state(ctx.conversation_id, "dr_state", dr_state)

def resolve_chat_id(ctx: RunContext, body: dict[str, Any]) -> str | None:
    """Pull the OWUI chat_id from body, nested metadata, or request.state."""
    md = body.get("metadata") or {}
    chat_id = md.get("chat_id") or body.get("chat_id")
    if not chat_id:
        pass  # request metadata lookup removed
    return chat_id

async def load_persisted_dr_state(
    ctx: RunContext, chat_id: str | None
) -> dict[str, Any] | None:
    """Read deepResearch checkpoint from the chat record."""
    if not chat_id:
        return None
    try:
        chat = await ctx.client.get_chat(chat_id)
        if not chat:
            return None
        chat_data = chat.get("chat") if isinstance(chat, dict) and "chat" in chat else chat
        if not chat_data:
            logger.warning(f"Chat {chat_id} has no chat data")
            return None
        dr = chat_data.get("deepResearch") if isinstance(chat_data, dict) else None
        return dr if isinstance(dr, dict) else None
    except Exception as e:
        logger.warning(
            f"Failed to load deepResearch checkpoint for chat {chat_id}: {e}"
        )
        return None

async def save_persisted_dr_state(
    ctx: RunContext, chat_id: str | None, dr_state: dict[str, Any]
) -> None:
    """Read-merge-write the deepResearch checkpoint into chat JSON.

    Open WebUI's update_chat_by_id REPLACES the entire chat dict, so we
    fetch the current chat, merge our branch in, and write it back.
    """
    if not chat_id:
        return
    try:
        chat = await ctx.client.get_chat(chat_id)
        if not chat:
            return
        chat_data = chat.get("chat") if isinstance(chat, dict) and "chat" in chat else chat
        if not chat_data:
            logger.warning(f"Chat {chat_id} has no chat data, skipping save")
            return
        merged = dict(chat_data or {})
        dr_state["last_checkpoint_at"] = datetime.now().isoformat()
        merged["deepResearch"] = dr_state
        await ctx.client.update_chat(chat_id, merged)
    except Exception as e:
        logger.warning(
            f"Failed to persist deepResearch checkpoint for chat {chat_id}: {e}"
        )

async def checkpoint(ctx: RunContext) -> None:
    """Persist the current in-memory dr_state to the chat record."""
    dr = get_dr_state(ctx)
    if not dr:
        return
    await save_persisted_dr_state(ctx, getattr(ctx, "chat_id", None), dr)

def record_token_usage(ctx: RunContext, bucket: str, prompt: int, completion: int) -> None:
    """Increment a named token bucket on the dr_state checkpoint."""
    dr = get_dr_state(ctx)
    if not dr:
        return
    usage = dr.setdefault(
        "token_usage",
        {
            "research_model": {"prompt": 0, "completion": 0, "total": 0},
            "report_writer": {"prompt": 0, "completion": 0, "total": 0},
            "synthesis_time_kb_qa": {"prompt": 0, "completion": 0, "total": 0},
            "grand_total": {"prompt": 0, "completion": 0, "total": 0},
        },
    )
    b = usage.setdefault(bucket, {"prompt": 0, "completion": 0, "total": 0})
    b["prompt"] = int(b.get("prompt", 0)) + max(0, int(prompt or 0))
    b["completion"] = int(b.get("completion", 0)) + max(0, int(completion or 0))
    b["total"] = b["prompt"] + b["completion"]
    gt = usage.setdefault("grand_total", {"prompt": 0, "completion": 0, "total": 0})
    gt["prompt"] = int(gt.get("prompt", 0)) + max(0, int(prompt or 0))
    gt["completion"] = int(gt.get("completion", 0)) + max(0, int(completion or 0))
    gt["total"] = gt["prompt"] + gt["completion"]
    set_dr_state(ctx, dr)

def extract_token_counts(response: dict[str, Any]) -> tuple[int, int]:
    """Best-effort extraction of (prompt_tokens, completion_tokens) from
    an OWUI completion response. Returns (0, 0) if unavailable."""
    try:
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            p = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            c = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            return p, c
    except Exception:
        pass
    return 0, 0


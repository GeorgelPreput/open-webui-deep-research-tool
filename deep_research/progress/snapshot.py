from datetime import datetime
from typing import Any

from deep_research.core.types import RunContext


def build_progress_snapshot(ctx: RunContext, cycle: int | None = None) -> dict[str, Any]:
    state = ctx.state.get_state(ctx.conversation_id)
    research_state = state.get("research_state") or {}
    user_message = research_state.get("user_message", "")
    max_cycles = getattr(ctx.valves, "MAX_CYCLES", 0)

    memory_stats = state.get("memory_stats", {}) or {}
    results_tokens = memory_stats.get("results_tokens", 0)
    synthesis_tokens = memory_stats.get("synthesis_tokens", 0)
    total_tokens = memory_stats.get("total_tokens", 0)

    completed = state.get("completed_topics", []) or []
    irrelevant = state.get("irrelevant_topics", []) or []
    partial = state.get("partial_topics", []) or []
    latest_new = state.get("latest_new_topics", []) or []
    active_outline = state.get("active_outline", []) or []
    all_topics = state.get("all_topics", []) or []

    revision = state.get("progress_embed_revision", 0) + 1
    current_cycle = cycle if cycle is not None else state.get("progress_cycle", 0)

    return {
        "query": user_message,
        "cycle": current_cycle,
        "max_cycles": max_cycles,
        "completed_topics": list(completed),
        "partial_topics": list(partial),
        "new_topics": list(latest_new),
        "irrelevant_topics": list(irrelevant),
        "remaining_topics": list(active_outline),
        "all_topics": list(all_topics),
        "results_tokens": results_tokens,
        "synthesis_tokens": synthesis_tokens,
        "total_tokens": total_tokens,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "revision": revision,
    }


def normalize_progress_categories(
    snapshot: dict[str, Any],
) -> dict[str, list[str]]:
    """Apply precedence (irrelevant > completed > partial > remaining) and stable ordering."""
    all_topics = list(snapshot.get("all_topics", []))
    new_topics = [t for t in snapshot.get("new_topics", []) if t]

    base_order: list[str] = []
    seen: set[str] = set()
    for t in all_topics + new_topics:
        if t and t not in seen:
            seen.add(t)
            base_order.append(t)

    irrelevant_set = {t for t in snapshot.get("irrelevant_topics", []) if t}
    completed_set = {
        t
        for t in snapshot.get("completed_topics", [])
        if t and t not in irrelevant_set
    }
    partial_set = {
        t
        for t in snapshot.get("partial_topics", [])
        if t and t not in irrelevant_set and t not in completed_set
    }
    remaining_set = {
        t
        for t in snapshot.get("remaining_topics", [])
        if t
        and t not in irrelevant_set
        and t not in completed_set
        and t not in partial_set
    }

    def ordered(topics: set[str]) -> list[str]:
        known = [t for t in base_order if t in topics]
        extras = [t for t in sorted(topics) if t not in set(known)]
        return known + extras

    return {
        "completed": ordered(completed_set),
        "partial": ordered(partial_set),
        "new": [t for t in new_topics if t not in irrelevant_set],
        "irrelevant": ordered(irrelevant_set),
        "remaining": ordered(remaining_set),
    }

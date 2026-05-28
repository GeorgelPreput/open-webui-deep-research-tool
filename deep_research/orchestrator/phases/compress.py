import logging
from typing import Any

from deep_research.compression.stepped import apply_stepped_compression
from deep_research.core.types import RunContext
from deep_research.progress.events import StatusEvent

logger = logging.getLogger("deep_research.orchestrator.phases.compress")


async def run_compress(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    results_history = ps.get("results_history", [])
    summary_embedding = ps.get("summary_embedding")

    if not ctx.valves.compression.stepped_synthesis_compression or len(results_history) <= 2:
        return ps

    await ctx.events.emit(StatusEvent(description="Applying stepped compression to research results...", level="info", done=False))

    total_before = 0
    for r in results_history:
        from deep_research.budget.tokens import count_tokens
        total_before += await count_tokens(ctx, r.get("content", ""))

    compressed = await apply_stepped_compression(ctx, results_history, None, summary_embedding)

    total_after = sum(r.get("tokens", 0) for r in compressed)
    if total_before > 0:
        reduction = total_before - total_after
        pct = (reduction / total_before) * 100
        logger.info(f"Compression: {total_before} -> {total_after} tokens ({pct:.1f}% reduction)")
        await ctx.events.emit(StatusEvent(
            description=f"Compressed: {total_before} -> {total_after} tokens",
            level="info", done=False,
        ))

    ps["results_history"] = compressed
    return ps

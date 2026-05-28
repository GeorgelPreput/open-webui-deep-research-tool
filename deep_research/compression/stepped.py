import logging
from typing import Any

from deep_research.budget.tokens import count_tokens
from deep_research.compression.eigendecomp import (
    compress_content_with_eigendecomposition,
)
from deep_research.config.constants import COMPRESSION_SETPOINT
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.compression.stepped")


async def apply_stepped_compression(
    ctx: RunContext,
    results_history: list[dict[str, Any]],
    query_embedding: list[float] | None,
    summary_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    if not ctx.valves.compression.stepped_synthesis_compression or len(results_history) <= 2:
        return results_history

    results = results_history.copy()

    mid_point = len(results) // 2
    older_results = results[:mid_point]
    newer_results = results[mid_point:]

    total_tokens_before = 0
    total_tokens_after = 0

    max_tokens = COMPRESSION_SETPOINT

    processed_older = []
    for result in older_results:
        content = result.get("content", "")

        original_tokens = await count_tokens(ctx, content)
        total_tokens_before += original_tokens

        if len(content) < 300:
            result["tokens"] = original_tokens
            processed_older.append(result)
            total_tokens_after += original_tokens
            continue

        compression_level = ctx.valves.compression.compression_level

        compress_ratios = {
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.5,
            6: 0.4,
            7: 0.3,
            8: 0.2,
            9: 0.15,
            10: 0.1,
        }
        ratio = compress_ratios.get(compression_level, 0.5)

        try:
            compressed = await compress_content_with_eigendecomposition(
                ctx, content, query_embedding, summary_embedding, ratio, max_tokens
            )

            compressed_tokens = await count_tokens(ctx, compressed)
            total_tokens_after += compressed_tokens

            new_result = result.copy()
            new_result["content"] = compressed
            new_result["tokens"] = compressed_tokens

            logger.info(
                f"Standard compression (older result): {original_tokens} \u2192 {compressed_tokens} tokens "
                f"({compressed_tokens / original_tokens:.1%} of original)"
            )

            processed_older.append(new_result)

        except Exception as e:
            logger.error(f"Error during standard compression: {e}")
            result["tokens"] = original_tokens
            processed_older.append(result)
            total_tokens_after += original_tokens

    processed_newer = []
    for result in newer_results:
        content = result.get("content", "")

        original_tokens = await count_tokens(ctx, content)
        total_tokens_before += original_tokens

        if len(content) < 300:
            result["tokens"] = original_tokens
            processed_newer.append(result)
            total_tokens_after += original_tokens
            continue

        compression_level = min(10, ctx.valves.compression.compression_level + 1)

        compress_ratios = {
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.5,
            6: 0.4,
            7: 0.3,
            8: 0.2,
            9: 0.15,
            10: 0.1,
        }
        ratio = compress_ratios.get(compression_level, 0.5)

        try:
            compressed = await compress_content_with_eigendecomposition(
                ctx, content, query_embedding, summary_embedding, ratio, max_tokens
            )

            compressed_tokens = await count_tokens(ctx, compressed)
            total_tokens_after += compressed_tokens

            new_result = result.copy()
            new_result["content"] = compressed
            new_result["tokens"] = compressed_tokens

            logger.info(
                f"Higher compression (newer result): {original_tokens} \u2192 {compressed_tokens} tokens "
                f"({compressed_tokens / original_tokens:.1%} of original)"
            )

            processed_newer.append(new_result)

        except Exception as e:
            logger.error(f"Error during higher compression: {e}")
            result["tokens"] = original_tokens
            processed_newer.append(result)
            total_tokens_after += original_tokens

    token_reduction = total_tokens_before - total_tokens_after
    if total_tokens_before > 0:
        percent_reduction = (token_reduction / total_tokens_before) * 100
        logger.info(
            f"Stepped compression total results: {total_tokens_before} \u2192 {total_tokens_after} tokens "
            f"(saved {token_reduction} tokens, {percent_reduction:.1f}% reduction)"
        )

    return processed_older + processed_newer

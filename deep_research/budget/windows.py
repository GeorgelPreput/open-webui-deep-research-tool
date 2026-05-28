import logging
from typing import Any

from deep_research.budget.tokens import count_message_tokens, count_tokens
from deep_research.config.constants import REPORT_BUDGET_SAFETY_MARGIN
from deep_research.core.types import RunContext

logger = logging.getLogger(__name__)


def get_model_context_window(ctx: RunContext, model: str) -> int:
    """Return the configured context window for *model*.

    Resolution order: per-model override valve → cached ModelInfo from
    OWUI list_models → fallback 8192. The fallback matches the plan's
    B.2 auto-detect rule and is intentionally conservative.
    """
    if model == ctx.valves.models.synthesis_model:
        configured = ctx.valves.models.synthesis_context_window
    else:
        configured = ctx.valves.models.research_context_window
    if configured is not None:
        return configured

    models_cache = getattr(ctx.caches, "models", None) or {}
    entry = models_cache.get(model)
    if entry is not None:
        cw = getattr(entry, "context_window", None)
        if cw is None and isinstance(entry, dict):
            cw = entry.get("context_window")
        if cw is None and isinstance(entry, int):
            cw = entry
        if cw:
            return int(cw)
    return 8192


async def get_task_context_budget(
    ctx: RunContext,
    task_name: str,
    model: str,
    fixed_messages: list[dict[str, Any]],
    output_reserve: int | None = None,
) -> dict[str, int]:
    _task_output_reserves: dict[str, int] = {
        "titles": 250,
        "abstract": 700,
        "introduction": 600,
        "conclusion": 800,
        "review": 1200,
    }
    if output_reserve is None:
        output_reserve = _task_output_reserves.get(task_name, 600)

    context_window = get_model_context_window(ctx, model)
    prompt_tokens = await count_message_tokens(ctx, fixed_messages)
    safety_margin = REPORT_BUDGET_SAFETY_MARGIN
    input_budget = context_window - prompt_tokens - output_reserve - safety_margin
    input_budget = max(200, input_budget)

    return {
        "context_window": context_window,
        "prompt_tokens": prompt_tokens,
        "output_reserve": output_reserve,
        "safety_margin": safety_margin,
        "input_budget": input_budget,
    }


async def extract_token_window(
    ctx: RunContext, content: str, start_token: int, window_size: int
) -> str:
    total_tokens = 1
    try:
        total_tokens = await count_tokens(ctx, content)
        chars_per_token = len(content) / max(1, total_tokens)

        start_char = int(start_token * chars_per_token)
        window_chars = int(window_size * chars_per_token)

        start_char = max(0, min(start_char, len(content) - 1))
        end_char = min(len(content), start_char + window_chars)

        window_content = content[start_char:end_char]

        if start_char > 0:
            first_period = window_content.find(". ")
            if first_period > 0 and first_period < len(window_content) // 10:
                window_content = window_content[first_period + 2:]

        last_period = window_content.rfind(". ")
        if last_period > 0 and last_period > len(window_content) * 0.9:
            window_content = window_content[: last_period + 1]

        return window_content

    except Exception as e:
        logger.error(f"Error extracting token window: {e}")
        if len(content) > 0:
            safe_start = min(
                len(content) - 1,
                max(0, int(len(content) * (start_token / total_tokens))),
            )
            safe_end = min(len(content), safe_start + window_size)
            return content[safe_start:safe_end]
        return content

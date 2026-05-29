import logging
from typing import Any

from deep_research.core.text import response_text
from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import (
    checkpoint,
    extract_token_counts,
    get_dr_state,
    record_token_usage,
)
from deep_research.persistence.kb import (
    attach_collection_to_chat,
    kb_search,
)

logger = logging.getLogger("Deep Research")

POST_REPORT_SYSTEM_PROMPT = (
    "You are answering questions about a completed Deep Research corpus.\n\n"
    "Primary source of truth:\n"
    "- The attached knowledge collection for this chat.\n"
    "- The final markdown report stored in that same collection.\n"
    "- The original source documents stored in that same collection.\n\n"
    "Behavior:\n"
    "- Answer using the attached knowledge collection as the primary "
    "evidence base.\n"
    "- Prefer grounded answers over speculation.\n"
    "- If the answer is not supported by the attached knowledge, say so "
    "clearly.\n"
    "- Do not restart or continue the original deep-research crawl in "
    "this chat.\n"
    "- If the user asks for a retry, a broader/narrower report, different "
    "source-selection rules, a different synthesis style, or a new "
    "research attempt, tell them a new research run must start in a new "
    "chat.\n"
    "- In that case, offer to produce a short handoff summary that "
    "includes:\n"
    "  - the original research request,\n"
    "  - any follow-up refinements already made,\n"
    "  - any additional constraints the user has added,\n"
    "  so the user can copy-paste that summary into a new chat."
)


async def run_synthesis_time_kb_qa(
    ctx: RunContext,
    qa_prompt: str,
    *,
    k: int = 6,
    model: str | None = None,
) -> str:
    """Targeted KB-grounded QA call used during report synthesis.

    Retrieves chunks from the research KB for the given prompt, then
    asks the model for a grounded answer. Token usage is recorded under
    the dedicated 'synthesis_time_kb_qa' bucket — distinct from
    ordinary report-writer token counts.
    """
    dr = get_dr_state(ctx)
    if not dr or not dr.get("kb_id"):
        return ""
    kb_id = dr["kb_id"]
    chunks = await kb_search(ctx, kb_id, qa_prompt, k=k)
    if not chunks:
        return ""
    ctx_lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source") or {}
        origin = (
            src.get("source_url")
            or src.get("name")
            or src.get("file_id")
            or "kb-chunk"
        )
        text = (c.get("text") or "").strip()
        if not text:
            continue
        ctx_lines.append(f"[KB:{i}] ({origin})\n{text}")
    if not ctx_lines:
        return ""
    context_block = "\n\n---\n\n".join(ctx_lines)
    system = {
        "role": "system",
        "content": (
            "You are answering a precise verification question about a "
            "research corpus. Use ONLY the provided KB excerpts as evidence. "
            "If the excerpts do not support an answer, say so explicitly."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Question: {qa_prompt}\n\n"
            f"KB excerpts:\n\n{context_block}\n\n"
            "Answer the question using only these excerpts. Cite by [KB:i]."
        ),
    }
    chosen_model = model or ctx.valves.models.research_model
    try:
        response = await ctx.client.chat_completions(
            chosen_model,
            [system, user],
            stream=False,
            temperature=min(ctx.valves.models.temperature, 0.3),
        )
        content = response_text(response)
        prompt_tokens, completion_tokens = extract_token_counts(response)
        record_token_usage(ctx, "synthesis_time_kb_qa", prompt_tokens, completion_tokens)
        await checkpoint(ctx)
        return content
    except Exception as e:
        logger.warning(f"synthesis_time_kb_qa failed: {e}")
        return ""

async def answer_post_report_user_qa(ctx: RunContext, body: dict[str, Any]) -> str:
    """Post-report QA path: skip deep-research orchestration entirely
    and answer the user's latest message using the persisted KB."""
    dr = get_dr_state(ctx)
    if not dr or not dr.get("kb_id"):
        return (
            "The research knowledge base for this chat is not available. "
            "Please start a new research run in a new chat."
        )
    kb_id = dr["kb_id"]
    kb_name = dr.get("kb_name") or kb_id

    messages = body.get("messages") or []
    user_message = (messages[-1].get("content") or "").strip() if messages else ""
    if not user_message:
        return ""

    # Make sure the collection is attached on the chat for future turns,
    # and re-attach defensively in case it was stripped.
    await attach_collection_to_chat(ctx, ctx.chat_id, kb_id, kb_name)

    from deep_research.progress.events import StatusEvent

    await ctx.events.emit(StatusEvent(
        description="Post-report mode: answering from research KB...",
        level="info",
        done=False,
    ))
    chunks = await kb_search(ctx, kb_id, user_message, k=8)
    if not chunks:
        await ctx.events.emit(StatusEvent(
            description="No KB matches found; answering from system prompt only.",
            level="warning",
            done=False,
        ))
    ctx_lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("source") or {}
        origin = (
            src.get("source_url")
            or src.get("name")
            or src.get("file_id")
            or "kb-chunk"
        )
        text = (c.get("text") or "").strip()
        if text:
            ctx_lines.append(f"[KB:{i}] ({origin})\n{text}")
    context_block = "\n\n---\n\n".join(ctx_lines) if ctx_lines else "(none)"

    system = {"role": "system", "content": POST_REPORT_SYSTEM_PROMPT}
    user = {
        "role": "user",
        "content": (
            f"User question:\n{user_message}\n\n"
            f"Knowledge collection excerpts:\n\n{context_block}\n\n"
            "Answer using these excerpts and the rules in the system "
            "prompt. Cite excerpts by [KB:i] when relevant."
        ),
    }

    chosen_model = ctx.valves.models.synthesis_model or ctx.valves.models.research_model
    try:
        response = await ctx.client.chat_completions(
            chosen_model,
            [system, user],
            stream=False,
            temperature=ctx.valves.models.synthesis_temperature,
        )
        content = response_text(response)
        prompt_tokens, completion_tokens = extract_token_counts(response)
        record_token_usage(ctx, "synthesis_time_kb_qa", prompt_tokens, completion_tokens)
        await checkpoint(ctx)
        return content or ""
    except Exception as e:
        logger.error(f"Post-report QA generation failed: {e}")
        return (
            f"I couldn't generate an answer from the research knowledge base ({e})."
        )


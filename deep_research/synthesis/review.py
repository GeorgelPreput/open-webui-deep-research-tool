import json
from typing import Any

from loguru import logger

from deep_research.budget.tokens import count_tokens
from deep_research.budget.windows import get_task_context_budget
from deep_research.config.constants import REVIEW_WINDOW_OVERLAP_RATIO
from deep_research.core.text import response_text
from deep_research.progress.events import StatusEvent


async def review_synthesis(
    ctx,
    compiled_sections: dict[str, str],
    original_query: str,
    research_outline: list[dict[str, Any]],
    synthesis_model: str,
) -> dict[str, Any]:
    """Review the compiled synthesis and suggest edits.

    Uses the full verbatim report when it fits the review budget.
    Falls back to overlapping verbatim windows when it does not, to
    preserve exact find_text / replace_text applicability.
    """
    review_prompt = {
        "role": "system",
        "content": """You are a post-grad research editor reviewing a comprehensive research report assembled per-section in different model contexts.
    Your task is to identify any issues with this combination of multiple sections and the flow between them.

    Focus on:
    1. Identifying areas needing better transitions between sections
    2. Finding obvious anomalies in section generation or stylistic discrepancies large enough to be distracting
    3. Making the report read as though it were written by one author who compiled these topics together for good purpose

    Do NOT:
    1. Impart your own biases, interests, or preferences onto the report
    2. Re-interpret the research information or soften its conclusions
    3. Make useless or unnecessary revisions beyond the scope of ensuring flow from start to finish
4. Remove or edit ANY in-text citations or instances of applied strikethrough. These are for specific human review and MUST NOT be changed or decoupled

For each suggested edit, provide exact text to find, and exact replacement text.
Don't include any justification or reasoning for your replacements - they will be inserted directly, so please make sure they fit in context.

Format your response as a JSON object with the following structure:
{
  "global_edits": [
    {
      "find_text": "exact text to be replaced",
      "replace_text": "exact replacement text"
    }
  ]
}

The find_text must be the EXACT text string as it appears in the document, and the replace_text must be the EXACT text to replace it with.""",
    }

    # Build full verbatim review context (same as before)
    review_context = f"# Complete Research Report on: {original_query}\n\n"
    review_context += "## Research Outline:\n"
    for topic in research_outline:
        review_context += f"- {topic['topic']}\n"
        for subtopic in topic.get("subtopics", []):
            review_context += f"  - {subtopic}\n"
    review_context += "\n"

    review_context += "## Complete Report Content by Section:\n\n"
    state = ctx.state.get_state(ctx.conversation_id)
    memory_stats = state.get("memory_stats", {})
    section_tokens_map = memory_stats.get("section_tokens", {})

    for section_title, content in compiled_sections.items():
        tokens = section_tokens_map.get(section_title, 0)
        if tokens == 0:
            tokens = await count_tokens(ctx, content)
            section_tokens_map[section_title] = tokens
            memory_stats["section_tokens"] = section_tokens_map
            state["memory_stats"] = memory_stats

        review_context += f"### {section_title} [{tokens} tokens]\n\n"
        review_context += f"{content}\n\n"

    review_context += "\nReview this research report and respond with necessary edits with specified JSON structure. Please don't include any other text in your response but the edits."

    review_temperature = ctx.valves.models.synthesis_temperature * 0.5

    async def _run_single_review(window_text: str) -> dict[str, Any]:
        """Run the review prompt against one verbatim context string."""
        msgs = [review_prompt, {"role": "user", "content": window_text}]
        resp = await ctx.llm.chat_completions(
            synthesis_model, msgs, stream=False, temperature=review_temperature
        )
        if resp and "choices" in resp and len(resp["choices"]) > 0:
            raw = response_text(resp)
            try:
                js = raw[raw.find("{") : raw.rfind("}") + 1]
                return json.loads(js)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.error(f"Error parsing review JSON: {exc}")
        return {"global_edits": []}

    try:
        await ctx.events.emit(StatusEvent(
                "Reviewing and improving the synthesis...", "info", False
            ))

        budget = await get_task_context_budget(ctx,
            "review", synthesis_model, [review_prompt]
        )
        input_budget = budget["input_budget"]
        review_tokens = await count_tokens(ctx, review_context)

        use_windowed = review_tokens > input_budget

        logger.info(
            f"[budget] review | model={synthesis_model} | "
            f"ctx_window={budget['context_window']} | "
            f"prompt_tokens={budget['prompt_tokens']} | "
            f"output_reserve={budget['output_reserve']} | "
            f"safety_margin={budget['safety_margin']} | "
            f"input_budget={input_budget} | "
            f"review_context_tokens={review_tokens} | "
            f"windowed={use_windowed}"
        )

        if not use_windowed:
            # Full single-pass verbatim review
            return await _run_single_review(review_context)

        # ----------------------------------------------------------------
        # Windowed verbatim review fallback
        # ----------------------------------------------------------------
        overlap_ratio = REVIEW_WINDOW_OVERLAP_RATIO
        # Split review_context into characters; windows sized by budget
        # Estimate chars per token (rough average of 4)
        chars_per_token = max(1, len(review_context) // max(1, review_tokens))
        window_chars = input_budget * chars_per_token
        step_chars = int(window_chars * (1.0 - overlap_ratio))
        step_chars = max(1, step_chars)

        windows: list[str] = []
        pos = 0
        while pos < len(review_context):
            end = min(pos + int(window_chars), len(review_context))
            windows.append(review_context[pos:end])
            if end >= len(review_context):
                break
            pos += step_chars

        logger.info(
            f"[review] windowed into {len(windows)} windows "
            f"(~{int(window_chars)} chars each, {int(overlap_ratio * 100)}% overlap)"
        )

        all_edits: list[dict[str, str]] = []
        for _idx, window in enumerate(windows):
            window_data = await _run_single_review(window)
            window_edits = window_data.get("global_edits", [])
            all_edits.extend(window_edits)

        # Merge: deduplicate, discard empty find_text
        seen: set[tuple[str, str]] = set()
        merged: list[dict[str, str]] = []
        for edit in all_edits:
            find_text = edit.get("find_text", "").strip()
            if not find_text:
                continue
            key = (find_text, edit.get("replace_text", ""))
            if key not in seen:
                seen.add(key)
                merged.append(edit)

        return {"global_edits": merged}

    except Exception as e:
        logger.error(f"Error generating synthesis review: {e}")
        return {"global_edits": [], "section_edits": {}}

async def apply_review_edits(
    ctx,
    compiled_sections: dict[str, str],
    review_data: dict[str, Any],
    synthesis_model: str,
):
    """Apply the suggested edits from the review to improve the synthesis"""
    # Create deep copy of sections to modify
    edited_sections = compiled_sections.copy()

    # Track if we made any changes
    changes_made = False

    # Apply global edits
    global_edits = review_data.get("global_edits", [])
    if global_edits:
        changes_made = True
        await ctx.events.emit(StatusEvent(
                f"Applying {len(global_edits)} global edits to synthesis...",
                "info",
                False,
            ))

        for edit_idx, edit in enumerate(global_edits):
            find_text = edit.get("find_text", "")
            replace_text = edit.get("replace_text", "")

            if not find_text:
                logger.warning(f"Empty find_text in edit {edit_idx + 1}, skipping")
                continue

            # Apply to each section
            for section_title, content in edited_sections.items():
                if find_text in content:
                    edited_sections[section_title] = content.replace(
                        find_text, replace_text
                    )
                    logger.info(
                        f"Applied edit {edit_idx + 1} in section '{section_title}'"
                    )

    return edited_sections, changes_made


import logging
from typing import Any

from deep_research.budget.tokens import count_tokens
from deep_research.core.types import RunContext
from deep_research.progress.events import StatusEvent
from deep_research.synthesis.front_back import generate_abstract, generate_titles
from deep_research.synthesis.review import apply_review_edits, review_synthesis
from deep_research.synthesis.verify import add_verification_note

logger = logging.getLogger("deep_research.orchestrator.phases.front_back")


async def run_front_back(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    compiled_sections = ps.get("compiled_sections", {})
    synthesis_outline = ps.get("synthesis_outline", [])
    bibliography_data = ps.get("bibliography_data", {})
    bibliography_table = ps.get("bibliography_table", "")
    synthesis_model = ps.get("synthesis_model", ctx.valves.models.research_model)
    user_message = ps.get("user_message", "")

    await ctx.events.emit(StatusEvent(description="Generating report titles...", level="info", done=False))
    titles = await generate_titles(ctx, user_message, "".join(compiled_sections.values()), sections=compiled_sections)

    await ctx.events.emit(StatusEvent(description="Reviewing synthesis...", level="info", done=False))
    review_data = await review_synthesis(ctx, compiled_sections, user_message, synthesis_outline, synthesis_model)

    await ctx.events.emit(StatusEvent(description="Applying review edits...", level="info", done=False))
    edited_sections, changes = await apply_review_edits(ctx, compiled_sections, review_data, synthesis_model)

    await ctx.events.emit(StatusEvent(description="Generating abstract...", level="info", done=False))
    abstract = await generate_abstract(
        ctx, user_message, "".join(edited_sections.values()),
        bibliography_data.get("bibliography", []), sections=edited_sections,
    )

    comprehensive_answer = ""
    main_title = titles.get("main_title", f"Research Report: {user_message}")
    subtitle = titles.get("subtitle", "A Comprehensive Analysis and Synthesis")
    comprehensive_answer += f"# {main_title}\n\n## {subtitle}\n\n"
    comprehensive_answer += f"## Abstract\n\n{abstract}\n\n"

    comprehensive_answer += await _generate_introduction(ctx, user_message, synthesis_outline, edited_sections, synthesis_model)

    for section_title, content in edited_sections.items():
        if content.startswith(section_title) or content.startswith(f"# {section_title}") or content.startswith(f"## {section_title}"):
            content = content.split("\n", 1)[1].lstrip() if "\n" in content else content
        comprehensive_answer += f"## {section_title}\n\n{content}\n\n"

    comprehensive_answer += await _generate_conclusion(ctx, user_message, edited_sections, synthesis_model)

    comprehensive_answer = await add_verification_note(ctx, comprehensive_answer)

    research_date = getattr(ctx, "research_date", None)
    if research_date is None:
        from datetime import datetime
        research_date = datetime.now().strftime("%Y-%m-%d")
    comprehensive_answer += f"{bibliography_table}\n\n"
    comprehensive_answer += f"*Research conducted on: {research_date}*\n\n"

    memory_stats = conv_state.get("memory_stats", {})
    synthesis_tokens = await count_tokens(ctx, comprehensive_answer)
    memory_stats["synthesis_tokens"] = synthesis_tokens
    conv_state["memory_stats"] = memory_stats

    ps["comprehensive_answer"] = comprehensive_answer
    ps["main_title"] = main_title
    ps["subtitle"] = subtitle
    ps["titles"] = titles
    return ps


async def _generate_introduction(ctx, user_message, synthesis_outline, edited_sections, synthesis_model):
    intro_prompt = {
        "role": "system",
        "content": (
            f"You are a post-grad research assistant writing an introduction for a report on: \"{user_message}\".\n"
            "Create a concise introduction (2-3 paragraphs). Do not add bias or sentiment.\n"
            "Respond with only the introduction text."
        ),
    }
    from deep_research.budget.contexts import build_intro_context
    from deep_research.budget.windows import get_task_context_budget
    intro_budget = await get_task_context_budget(ctx, "introduction", synthesis_model, [intro_prompt])
    intro_context = await build_intro_context(ctx, user_message, synthesis_outline, edited_sections, intro_budget["input_budget"])
    intro_msg = {"role": "user", "content": intro_context}
    try:
        response = await ctx.client.chat_completions(
            synthesis_model,
            [intro_prompt, intro_msg],
            stream=False,
            temperature=ctx.valves.models.synthesis_temperature * 0.83,
        )
        if response and response.get("choices"):
            introduction = response["choices"][0]["message"]["content"]
            return f"## Introduction\n\n{introduction}\n\n"
    except Exception as e:
        logger.error(f"Introduction generation failed: {e}")
    return f"## Introduction\n\nThis report addresses: '{user_message}'.\n\n"


async def _generate_conclusion(ctx, user_message, edited_sections, synthesis_model):
    concl_prompt = {
        "role": "system",
        "content": (
            f"You are a post-grad research assistant writing a conclusion for a report on: \"{user_message}\".\n"
            "Create a concise conclusion (2-4 paragraphs). Synthesize key findings.\n"
            "Do not add bias, preach, or appeal to future research.\n"
            "Respond with only the conclusion text."
        ),
    }
    from deep_research.budget.contexts import build_conclusion_context
    from deep_research.budget.windows import get_task_context_budget
    concl_budget = await get_task_context_budget(ctx, "conclusion", synthesis_model, [concl_prompt])
    concl_context = await build_conclusion_context(ctx, user_message, edited_sections, concl_budget["input_budget"])
    concl_msg = {"role": "user", "content": concl_context}
    try:
        response = await ctx.client.chat_completions(
            synthesis_model,
            [concl_prompt, concl_msg],
            stream=False,
            temperature=ctx.valves.models.synthesis_temperature,
        )
        if response and response.get("choices"):
            conclusion = response["choices"][0]["message"]["content"]
            return f"## Conclusion\n\n{conclusion}\n\n"
    except Exception as e:
        logger.error(f"Conclusion generation failed: {e}")
    return ""

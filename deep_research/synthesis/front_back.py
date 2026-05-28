import asyncio
import json

from loguru import logger

from deep_research.budget.contexts import build_abstract_context, build_titles_context
from deep_research.budget.tokens import count_tokens
from deep_research.budget.windows import get_task_context_budget
from deep_research.core.types import BibliographyEntry
from deep_research.progress.events import MessageEvent
from deep_research.synthesis.utils import get_synthesis_model


async def generate_titles(
    ctx,
    user_message: str,
    comprehensive_answer: str,
    sections: dict[str, str] | None = None,
):
    """Generate a main title and subtitle for the research report."""
    titles_prompt = {
        "role": "system",
        "content": """You are a post-grad research writer creating compelling titles for research reports.

    Create a main title and subtitle for a comprehensive research report. The titles should:
    1. Be relevant and accurately reflect the content and focus of the research
    2. Be engaging and professional. Intriguing, even
    3. Follow academic/research paper conventions
    4. Avoid clickbait or sensationalism unless it's really begging for it

    For main title:
    - 5-12 words in length
    - Clear and focused
    - Appropriately formal for academic/research context

    For subtitle:
    - 8-15 words in length
    - Provides additional context and specificity
    - Complements the main title without redundancy

    Format your response as a JSON object with the following structure:
    {
      "main_title": "Your proposed main title",
      "subtitle": "Your proposed subtitle"
    }""",
    }

    try:
        research_model = ctx.valves.models.research_model

        # Compute task budget before building context
        budget = await get_task_context_budget(ctx,
            "titles", research_model, [titles_prompt]
        )
        input_budget = budget["input_budget"]

        if sections:
            titles_context = await build_titles_context(ctx,
                user_message, sections, input_budget
            )
        else:
            # Legacy fallback: use raw comprehensive_answer trimmed to budget
            raw_context = (
                f"Original Research Query: {user_message}\n\n"
                f"Research Report Content Summary:\n{comprehensive_answer}"
            )
            raw_tokens = await count_tokens(ctx, raw_context)
            if raw_tokens > input_budget:
                char_ratio = input_budget / raw_tokens
                raw_context = raw_context[: int(len(raw_context) * char_ratio)]
            titles_context = raw_context

        titles_context += "\n\nGenerate an appropriate main title and subtitle for this research report."

        logger.info(
            f"[budget] titles | model={research_model} | "
            f"ctx_window={budget['context_window']} | "
            f"prompt_tokens={budget['prompt_tokens']} | "
            f"output_reserve={budget['output_reserve']} | "
            f"safety_margin={budget['safety_margin']} | "
            f"input_budget={input_budget} | "
            f"context_tokens={await count_tokens(ctx, titles_context)}"
        )

        response = await ctx.client.chat_completions(
            research_model,
            [titles_prompt, {"role": "user", "content": titles_context}],
            temperature=0.7,
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            titles_content = response["choices"][0]["message"]["content"]

            try:
                json_str = titles_content[
                    titles_content.find("{") : titles_content.rfind("}") + 1
                ]
                titles_data = json.loads(json_str)

                main_title = titles_data.get(
                    "main_title", f"Research Report: {user_message}"
                )
                subtitle = titles_data.get(
                    "subtitle", "A Comprehensive Analysis and Synthesis"
                )

                return {"main_title": main_title, "subtitle": subtitle}
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing titles JSON: {e}")
                return {
                    "main_title": f"Research Report: {user_message[:50]}",
                    "subtitle": "A Comprehensive Analysis and Synthesis",
                }
        else:
            return {
                "main_title": f"Research Report: {user_message[:50]}",
                "subtitle": "A Comprehensive Analysis and Synthesis",
            }

    except Exception as e:
        logger.error(f"Error generating titles: {e}")
        return {
            "main_title": f"Research Report: {user_message[:50]}",
            "subtitle": "A Comprehensive Analysis and Synthesis",
        }

async def generate_abstract(
    ctx,
    user_message: str,
    comprehensive_answer: str,
    bibliography: list[BibliographyEntry],
    sections: dict[str, str] | None = None,
):
    """Generate an abstract for the research report."""
    abstract_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant writing an abstract for a comprehensive research report.

    Create a concise academic abstract (150-250 words) that summarizes the research report. The abstract should:
    1. Outline the research objective and original intent without simply restating the original query
    2. Summarize the key findings and their significance
    3. Be written in an academic yet interesting tone
    4. Be self-contained and understandable on its own
5. Draw you in by highlighting the interesting aspects of the research without being misleading or disingenuous

The abstract must NOT:
1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication.
2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.

    The abstract should follow scientific paper abstract structure but be accessible to an educated general audience.""",
    }

    try:
        synthesis_model = get_synthesis_model(ctx)

        budget = await get_task_context_budget(ctx,
            "abstract", synthesis_model, [abstract_prompt]
        )
        input_budget = budget["input_budget"]

        if sections:
            abstract_context = await build_abstract_context(ctx,
                user_message, sections, bibliography, input_budget
            )
        else:
            raw_context = (
                f"Research Query: {user_message}\n\n"
                f"Research Report Full Content:\n{comprehensive_answer}"
            )
            raw_tokens = await count_tokens(ctx, raw_context)
            if raw_tokens > input_budget:
                char_ratio = input_budget / raw_tokens
                raw_context = raw_context[: int(len(raw_context) * char_ratio)]
            abstract_context = raw_context

        abstract_context += "\n\nGenerate a concise, substantive abstract focusing on substantive content and key insights rather than how the research was conducted. Please don't include any other text in your response but the abstract."

        logger.info(
            f"[budget] abstract | model={synthesis_model} | "
            f"ctx_window={budget['context_window']} | "
            f"prompt_tokens={budget['prompt_tokens']} | "
            f"output_reserve={budget['output_reserve']} | "
            f"safety_margin={budget['safety_margin']} | "
            f"input_budget={input_budget} | "
            f"context_tokens={await count_tokens(ctx, abstract_context)}"
        )

        response = await asyncio.wait_for(
            ctx.client.chat_completions(
                synthesis_model,
                [abstract_prompt, {"role": "user", "content": abstract_context}],
                temperature=ctx.valves.models.synthesis_temperature,
            ),
            timeout=300,
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            abstract = response["choices"][0]["message"]["content"]
            if not ctx.valves.events.quiet_chat_mode:
                await ctx.events.emit(MessageEvent("*Abstract generation complete.*\n"))
            return abstract
        else:
            if not ctx.valves.events.quiet_chat_mode:
                await ctx.events.emit(MessageEvent("*Abstract generation fallback used.*\n"))
            return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."

    except TimeoutError:
        logger.error("Abstract generation timed out after 5 minutes")
        if not ctx.valves.events.quiet_chat_mode:
            await ctx.events.emit(MessageEvent("*Abstract generation timed out, using fallback.*\n"))
        return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."
    except Exception as e:
        logger.error(f"Error generating abstract: {e}")
        if not ctx.valves.events.quiet_chat_mode:
            await ctx.events.emit(MessageEvent("*Abstract generation error, using fallback.*\n"))
        return f"This research report addresses the query: '{user_message}'. It synthesizes information from {len(bibliography)} sources to provide a comprehensive analysis of the topic, examining key aspects and presenting relevant findings."


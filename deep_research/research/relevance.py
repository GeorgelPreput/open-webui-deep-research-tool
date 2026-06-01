import logging
import re
from typing import Any

from deep_research.core.text import response_text
from deep_research.core.types import RunContext
from deep_research.progress.events import MessageEvent, StatusEvent

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def extract_topic_relevant_info(ctx: RunContext, results, topics) -> str:
    """Extract information from search results specifically relevant to given topics"""
    if not results:
        return ""

    # Create a prompt for extracting relevant information
    extraction_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant extracting information from search results.
    Identify and extract information that is specifically relevant to the given topics.
    Format the extracted information as concise bullet points, focusing on facts, data, and insights.
    Ignore general information not directly related to the topics.""",
    }

    # Create context with search results and topics
    topics_str = ", ".join(topics)
    extraction_context = f"Topics: {topics_str}\n\nSearch Results:\n\n"

    for i, result in enumerate(results):
        extraction_context += f"Result {i + 1}:\n"
        extraction_context += f"Title: {result.get('title', 'Untitled')}\n"
        extraction_context += f"Content: {result.get('content', '')}...\n\n"

    extraction_context += "\nExtract relevant information for the listed topics from these search results."

    # Create messages for extraction
    extraction_messages = [
        extraction_prompt,
        {"role": "user", "content": extraction_context},
    ]

    # Extract relevant information
    try:
        response = await ctx.llm.chat_completions(
            ctx.valves.models.research_model,
            extraction_messages,
            temperature=ctx.valves.models.temperature
            * 0.4,  # Lower temperature for factual extraction
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            extracted_info = response_text(response)
            return extracted_info
        else:
            return "No relevant information found."

    except Exception as e:
        logger.error(f"Error extracting topic-relevant information: {e}")
        return "Error extracting information from search results."

async def refine_topics_with_research(
    ctx: RunContext, topics, relevant_info, pdv, original_query
):
    """Refine topics based on both user preferences and research results"""
    # Create a prompt for refining topics
    refine_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant refining research topics.
    Based on the extracted information and user preferences, revise each topic to:
    1. Be specific and targeted based on the research findings, while maintaining alignment with user preferences and the original query
    2. Prioritize topics that seem most relevant to answering the query and that will reasonably result in worthwhile expanded research
    3. Be phrased as clear, researchable topics in the same style as those to be replaced

    Your refined topics should incorporate new discoveries that heighten and expand upon the intent of the original query.
Avoid overstating the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
You do not need to include justification along with your refined topics.""",
    }

    # Create context with topics, research info, and preference direction
    pdv_context = ""
    if pdv is not None:
        pdv_context = "\nUser preferences are directing research toward topics similar to what was kept and away from what was removed."

    refine_context = f"""Original topics: {", ".join(topics)}

    Original query: {original_query}

    Extracted research information:
    {relevant_info}
    {pdv_context}

    Refine these topics based on the research findings and user preferences.
    Provide a list of the same number of refined topics."""

    # Create messages for refinement
    refine_messages = [refine_prompt, {"role": "user", "content": refine_context}]

    # Generate refined topics
    try:
        response = await ctx.llm.chat_completions(
            ctx.valves.models.research_model,
            refine_messages,
            temperature=ctx.valves.models.temperature
            * 0.7,  # Balanced temperature for creativity with focus
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            refined_content = response_text(response)

            # Extract topics using regex (looking for numbered or bulleted list items)
            refined_topics = re.findall(
                r"(?:^|\n)(?:\d+\.\s*|\*\s*|-\s*)([^\n]+)", refined_content
            )

            # If we couldn't extract enough topics, use the original ones
            if len(refined_topics) < len(topics):
                logger.warning(
                    f"Not enough refined topics extracted ({len(refined_topics)}), using originals"
                )
                return topics

            # Limit to the same number as original topics
            refined_topics = refined_topics[: len(topics)]
            return refined_topics
        else:
            return topics

    except Exception as e:
        logger.error(f"Error refining topics: {e}")
        return topics

async def is_follow_up_query(ctx: RunContext, messages: list[dict[str, Any]]) -> bool:
    """Determine if the current query is a follow-up to a previous research session"""
    # If we have a previous comprehensive summary and research has been completed,
    # treat any new query as a follow-up
    state = ctx.state.get_state(ctx.conversation_id)
    prev_comprehensive_summary = state.get("prev_comprehensive_summary", "")
    research_completed = state.get("research_completed", False)

    # Don't treat as a follow-up while we're still expecting outline
    # feedback on the current run.
    if state.get("waiting_for_outline_feedback", False):
        return False

    return bool(prev_comprehensive_summary and research_completed)


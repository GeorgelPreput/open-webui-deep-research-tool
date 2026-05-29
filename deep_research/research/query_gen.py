import json
import logging

from deep_research.core.types import RunContext
from deep_research.progress.events import MessageEvent, StatusEvent

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def improved_query_generation(
    ctx: RunContext, user_message, priority_topics, search_context
):
    """Generate refined search queries for research topics with improved context"""
    query_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant generating effective search queries.
Based on the user's original question, current research needs, and context provided, generate 4 precise search queries.
Each query should be specific, use relevant keywords, and be designed to find targeted information.

Your queries should:
1. Directly address the priority research topics
2. Avoid redundancy with previous queries
3. Target information gaps in the current research
4. Be concise (6-12 words) but specific
5. Include specialized terminology when appropriate

Focus on core conceptual terms with targeted expansions and don't return heavy, clunky queries.
Use quotes sparingly and as a last resort. Never use multiple sets of quotes in the same query.

Format your response as a valid JSON object with the following structure:
{
  "queries": [
    {"query": "search query 1", "topic": "related research topic"},
    {"query": "search query 2", "topic": "related research topic"},
    {"query": "search query 3", "topic": "related research topic"},
    {"query": "search query 4", "topic": "related research topic"}
  ]
}""",
    }

    message = {
        "role": "user",
        "content": f"""Original query: "{user_message}"\n\nResearch context: "{search_context}"\n\nGenerate 4 effective search queries to gather information for the priority research topics.""",
    }

    # Generate the queries first, without any embedding operations
    try:
        response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [query_prompt, message],
            temperature=ctx.valves.models.temperature,
        )

        query_content = response["choices"][0]["message"]["content"]

        # Extract JSON from response
        try:
            query_json_str = query_content[
                query_content.find("{") : query_content.rfind("}") + 1
            ]
            query_data = json.loads(query_json_str)
            queries = query_data.get("queries", [])

            # Check if queries is a list of strings or a list of objects
            if queries and isinstance(queries[0], str):
                # Convert to objects with query and topic
                query_strings = queries
                query_topics = (
                    priority_topics[: len(queries)]
                    if priority_topics
                    else ["Research"] * len(queries)
                )
                queries = [
                    {"query": q, "topic": t}
                    for q, t in zip(query_strings, query_topics, strict=False)
                ]

            return queries

        except Exception as e:
            logger.error(f"Error parsing query JSON: {e}")
            # Fallback: generate basic queries for priority topics
            queries = []
            for _i, topic in enumerate(priority_topics[:3]):
                queries.append({"query": f"{user_message} {topic}", "topic": topic})

            return queries

    except Exception as e:
        logger.error(f"Error generating improved queries: {e}")
        # Fallback: generate basic queries
        queries = []
        for _i, topic in enumerate(priority_topics[:3]):
            queries.append({"query": f"{user_message} {topic}", "topic": topic})

        return queries

async def generate_group_query(ctx: RunContext, topic_group, user_message):
    """Generate a search query that covers a group of related topics"""
    if not topic_group:
        return user_message

    topics_text = ", ".join(topic_group)

    # Create a prompt for generating the query
    prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant generating an effective search query.
    Create a search query that will find relevant information for a group of related topics aimed at addressing the original user input.
    The query should be specific enough to find targeted information while broadly representing all topics in the group.
    Make the query concise (maximum 10 words) and focused.""",
    }

    # Create the message content
    message = {
        "role": "user",
        "content": f"""Generate a search query for this group of topics:
    {topics_text}

    This is related to the original user query: "{user_message}"

    Generate a single concise search query that will find information relevant to these topics.
    Just respond with the search query text only.""",
    }

    # Generate the query
    try:
        response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [prompt, message],
            temperature=ctx.valves.models.temperature * 0.7,
        )

        query = response["choices"][0]["message"]["content"].strip()

        # Clean up the query: remove quotes and ensure it's not too long
        query = query.replace('"', "").replace('"', "").replace('"', "")

        # If the query is too long, truncate it
        if len(query.split()) > 12:
            query = " ".join(query.split()[:12])

        return query

    except Exception as e:
        logger.error(f"Error generating group query: {e}")
        # Fallback: combine the first topic with the user message
        return f"{user_message} {topic_group[0]}"


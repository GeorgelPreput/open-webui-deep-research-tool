import json
import re
from typing import Any

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from deep_research.core.text import response_text, stable_text_key
from deep_research.progress.events import StatusEvent
from deep_research.semantics.dimensions import translate_dimensions_to_words
from deep_research.semantics.embeddings import get_embedding
from deep_research.synthesis.utils import get_synthesis_model


async def generate_synthesis_outline(
    ctx,
    original_outline: list[dict[str, Any]],
    completed_topics: set[str],
    user_query: str,
    research_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate a refined research outline for synthesis that better integrates additional research areas"""

    state = ctx.state.get_state(ctx.conversation_id)

    # Get the number of elapsed cycles
    elapsed_cycles = len(state.get("cycle_summaries", []))

    # Create a prompt for generating the synthesis outline
    synthesis_outline_prompt = {
        "role": "system",
        "content": f"""You are a post-graduate academic scholar reorganizing a research outline to be used in writing a comprehensive research report.

    Create a refined outline that condenses key topics/subtopics and insights from the current outline, and focuses on addressing the original query in areas best supported by the research.
Aim to have approximately {round((elapsed_cycles * 0.25) + 2)} main topics and {round((elapsed_cycles * 0.8) + 5)} subtopics in your revised outline.

The original user query was: "{user_query}".

Your refined outline must:
    1. Appropriately incorporate relevant new topics discovered along the way that are directly relevant to the research "core" and original user query.
    2. Tailors the outline to reflect the progress and outcome of research activities without getting distracted by irrelevant results or specific examples, brands, locations, etc.
3. Unite how research has evolved, and the reference material obtained during research, with the initial purpose and scope, prioritizing the initial purpose and scope.
4. Where appropriate, reign in the representation of tangential research branches to refocus on topics more directly related to the original query.

Your refined outline must NOT:
1. Attempt to trump up, downplay, remove, soften, qualify, or otherwise modify the representation of research topics due to your own biases, preferences, or interests.
2. Include main topics intended to serve as an introduction or conclusion for the full report.
3. Focus on topics explored during research that don't actually serve to address the user's query or are fully tangent to it, or overly emphasize specific cases.
4. Include any other text - please only respond with the outline.

The goal is to create a refined outline reflecting a logical narrative and informational flow for the final comprehensive report based on the user's query and gathered research.

    Format your response as a valid JSON object with the following structure:
    {{"outline": [
      {{"topic": "Main topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]}},
      {{"topic": "Main topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}}
    ]}}""",
    }

    # Calculate similarity of research results to the research outline
    result_scores = []
    outline_text = "\n".join(
        [topic_item["topic"] for topic_item in original_outline]
    )

    # Check if we have a cached outline embedding
    state = ctx.state.get_state(ctx.conversation_id)
    outline_embedding_key = f"outline_embedding_{stable_text_key(outline_text)}"
    outline_embedding = state.get(outline_embedding_key)

    if not outline_embedding:
        outline_embedding = await get_embedding(ctx, outline_text)
        if outline_embedding:
            # Cache the outline embedding
            state[outline_embedding_key] = outline_embedding

    # Initialize outline_context
    outline_context = ""
    if outline_embedding:
        for i, result in enumerate(research_results):
            content = result.get("content", "")
            if not content:
                continue

            # Check cache first for result embedding
            result_key = f"result_embedding_{stable_text_key(result.get('url', ''))}"
            content_embedding = state.get(result_key)

            if not content_embedding:
                content_embedding = await get_embedding(ctx, content[:2000])
                if content_embedding:
                    # Cache the result embedding
                    state[result_key] = content_embedding

            if content_embedding:
                similarity = cosine_similarity(
                    [content_embedding], [outline_embedding]
                )[0][0]
                result_scores.append((i, similarity))

        # Sort results by similarity to outline in reverse order (most similar last)
        result_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_results = [research_results[i] for i, _ in result_scores]

        # Add sorted results to context
        outline_context += "\n### Research Results:\n\n"
        for result in sorted_results:
            outline_context += f"Title: {result.get('title', 'Untitled')}\n"
            outline_context += f"Content: {result.get('content', '')}\n\n"

    # Build context from the original outline and research results
    outline_context += "### Original Research Outline:\n\n"

    for topic_item in original_outline:
        outline_context += f"- {topic_item['topic']}\n"
        for subtopic in topic_item.get("subtopics", []):
            outline_context += f"  - {subtopic}\n"

    # Add semantic dimensions if available
    state = ctx.state.get_state(ctx.conversation_id)
    research_dimensions = state.get("research_dimensions")
    if research_dimensions:
        try:
            dimension_coverage = research_dimensions.get("coverage", [])

            # Create dimension labels for better context
            dimension_labels = await translate_dimensions_to_words(ctx,
                research_dimensions, dimension_coverage
            )

            if dimension_coverage:
                outline_context += "\n### Research Dimensions Coverage:\n"
                for dim in dimension_labels[:10]:  # Limit to top 10 dimensions
                    if isinstance(dim, dict):
                        outline_context += f"- {dim.get('words', 'Dimension ' + str(dim.get('dimension', 0)))}:  {dim.get('coverage', 0)}% covered\n"
                    else:
                        outline_context += f"- {dim}\n"

        except Exception as e:
            logger.error(
                f"Error adding research dimensions to outline context: {e}"
            )

    # Create messages for the model
    messages = [
        synthesis_outline_prompt,
        {
            "role": "user",
            "content": f"{outline_context}\n\nGenerate a refined research outline following the instructions and format in the system prompt.",
        },
    ]

    # Generate the synthesis outline
    try:
        await ctx.events.emit(StatusEvent(
                "Generating refined outline for synthesis...", "info", False
            ))

        # Use synthesis model for this task
        synthesis_model = get_synthesis_model(ctx)
        response = await ctx.client.chat_completions(
            synthesis_model, messages, temperature=ctx.valves.models.synthesis_temperature
        )
        outline_content = response_text(response)

        # Extract JSON from response
        try:
            # First try standard JSON extraction
            json_start = outline_content.find("{")
            json_end = outline_content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                outline_json_str = outline_content[json_start:json_end]
                try:
                    outline_data = json.loads(outline_json_str)
                    synthesis_outline = outline_data.get("outline", [])
                    if synthesis_outline:
                        return synthesis_outline
                except (json.JSONDecodeError, ValueError):
                    # If standard approach fails, try regex approach
                    pass

            # Use regex to find any JSON structure containing "outline" array
            json_pattern = r'(\{[^{}]*"outline"\s*:\s*\[[^\[\]]*\][^{}]*\})'
            matches = re.findall(json_pattern, outline_content, re.DOTALL)

            for match in matches:
                try:
                    outline_data = json.loads(match)
                    synthesis_outline = outline_data.get("outline", [])
                    if synthesis_outline:
                        return synthesis_outline
                except Exception:
                    continue

            # If no valid JSON found, try a more aggressive repair approach
            # Look for anything that resembles the outline structure
            topic_pattern = (
                r'"topic"\s*:\s*"([^"]*)"\s*,\s*"subtopics"\s*:\s*\[(.*?)\]'
            )
            topics_matches = re.findall(topic_pattern, outline_content, re.DOTALL)

            if topics_matches:
                synthetic_outline = []
                for topic_match in topics_matches:
                    topic = topic_match[0]
                    subtopics_str = topic_match[1]
                    # Extract subtopics strings - look for quoted strings
                    subtopics = re.findall(r'"([^"]*)"', subtopics_str)
                    synthetic_outline.append(
                        {"topic": topic, "subtopics": subtopics}
                    )

                if synthetic_outline:
                    return synthetic_outline

            # All extraction methods failed, return original outline
            return original_outline

        except Exception as e:
            logger.error(f"Error parsing synthesis outline JSON: {e}")
            return original_outline

    except Exception as e:
        logger.error(f"Error generating synthesis outline: {e}")
        return original_outline


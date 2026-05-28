import json
import logging
import re
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from deep_research.core.types import RunContext
from deep_research.progress.events import MessageEvent, StatusEvent
from deep_research.research.cycle import process_query
from deep_research.research.grouping import (
    generate_group_title,
    generate_replacement_topics,
    group_replacement_topics,
)
from deep_research.research.query_gen import generate_group_query
from deep_research.research.relevance import (
    extract_topic_relevant_info,
    refine_topics_with_research,
)
from deep_research.semantics.dimensions import initialize_research_dimensions
from deep_research.semantics.embeddings import get_embedding
from deep_research.semantics.preference import calculate_preference_direction_vector

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def process_user_outline_feedback(
    ctx: RunContext, outline_items: list[dict[str, Any]], original_query: str
) -> dict[str, Any]:
    """Process user feedback on research outline items by asking for feedback in chat"""
    # Number each outline item (maintain hierarchy but flatten for numbering)
    numbered_outline = []
    flat_items = []

    # Process the hierarchical outline structure
    item_num = 1
    for topic_item in outline_items:
        topic = topic_item.get("topic", "")
        subtopics = topic_item.get("subtopics", [])

        # Add main topic with number
        flat_items.append(topic)
        numbered_outline.append(f"{item_num}. {topic}")
        item_num += 1

        # Add subtopics with numbers
        for subtopic in subtopics:
            flat_items.append(subtopic)
            numbered_outline.append(f"{item_num}. {subtopic}")
            item_num += 1

    # Prepare the outline display
    outline_display = "\n".join(numbered_outline)

    # Emit a message with instructions using improved slash commands
    feedback_message = (
        "### Research Outline\n\n"
        f"{outline_display}\n\n"
        "**Please provide feedback on this research outline.**\n\n"
        "You can:\n"
        "- Use commands like `/keep 1,3,5-7` or `/remove 2,4,8-10` to select specific items by number\n"
        "- Or simply describe what topics you want to focus on or avoid in natural language\n\n"
        "Examples:\n"
        "- `/k 1,3,5-7` (keep only items 1,3,5,6,7)\n"
        "- `/r 2,4,8-10` (remove items 2,4,8,9,10)\n"
        '- "Focus on historical aspects and avoid technical details"\n'
        '- "I\'m more interested in practical applications than theoretical concepts"\n\n'
        "If you want to continue with all items, just reply 'continue' or leave your message empty.\n\n"
        "**I'll pause here to await your response before continuing the research.**"
    )

    await _emit_message(ctx, feedback_message)

    # Set flag to indicate we're waiting for feedback
    ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", True)
    ctx.state.update_state(ctx.conversation_id,
        "outline_feedback_data",
        {
            "outline_items": outline_items,
            "flat_items": flat_items,
            "numbered_outline": numbered_outline,
            "original_query": original_query,
        },
    )

    # Return a default response (this will be overridden in the next call).
    # _feedback_message is included so the caller can return it directly as the
    # pipe() return value — OWUI >=0.9 overwrites emitted content with the return
    # value on final save, so returning "" would produce an empty assistant message.
    return {
        "kept_items": flat_items,
        "removed_items": [],
        "kept_indices": list(range(len(flat_items))),
        "removed_indices": [],
        "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        "_feedback_message": feedback_message,
    }

async def process_natural_language_feedback(
    ctx: RunContext, user_message: str, flat_items: list[str]
) -> dict[str, Any]:
    """Process natural language feedback to determine which topics to keep/remove"""

    # Create a prompt for the model to interpret user feedback
    interpret_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant analyzing user feedback on a research outline.
    Based on the user's natural language input, determine which research topics should be kept or removed.

    The user's message expresses preferences about the research direction. Analyze this to identify:
    1. Which specific topics from the outline align with their interests
    2. Which specific topics should be removed based on their preferences

    Your task is to categorize each topic as EITHER "keep" OR "remove", NEVER both, based on the user's natural language feedback.
Don't allow your own biases or preferences to have any affect on your answer - please remain purely objective and user research-oriented.
    Provide your response as a JSON object with two lists: "keep" for indices to keep, and "remove" for indices to remove.
    Indices should be 0-based (first item is index 0).""",
    }

    # Prepare context with list of topics and user message
    topics_list = "\n".join([f"{i}. {topic}" for i, topic in enumerate(flat_items)])

    context = f"""Research outline topics:
    {topics_list}

    User feedback:
    "{user_message}"

    Based on this feedback, categorize each topic (by index) as either "keep" or "remove".
    If the user clearly expresses a preference to focus on certain topics or avoid others, use that to guide your decisions.
    If the user's feedback is ambiguous about some topics, categorize them based on their similarity to clearly mentioned preferences.
    """

    # Generate interpretation of user feedback
    try:
        response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [interpret_prompt, {"role": "user", "content": context}],
            temperature=ctx.valves.models.temperature
            * 0.3,  # Low temperature for consistent interpretation
        )

        result_content = response["choices"][0]["message"]["content"]

        # Extract JSON from response
        try:
            json_str = result_content[
                result_content.find("{") : result_content.rfind("}") + 1
            ]
            result_data = json.loads(json_str)

            # Get keep and remove lists
            keep_indices = result_data.get("keep", [])
            remove_indices = result_data.get("remove", [])

            # Ensure both keep_indices and remove_indices are lists
            if not isinstance(keep_indices, list):
                keep_indices = []
            if not isinstance(remove_indices, list):
                remove_indices = []

            # Ensure each index is in either keep or remove
            all_indices = set(range(len(flat_items)))
            missing_indices = all_indices - set(keep_indices) - set(remove_indices)

            # By default, keep missing indices
            keep_indices.extend(missing_indices)

            # Convert to kept and removed items
            kept_items = [
                flat_items[i] for i in keep_indices if i < len(flat_items)
            ]
            removed_items = [
                flat_items[i] for i in remove_indices if i < len(flat_items)
            ]

            logger.info(
                f"Natural language feedback interpretation: keep {len(kept_items)}, remove {len(removed_items)}"
            )

            return {
                "kept_items": kept_items,
                "removed_items": removed_items,
                "kept_indices": keep_indices,
                "removed_indices": remove_indices,
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing feedback interpretation: {e}")
            # Default to keeping all items
            return {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
            }

    except Exception as e:
        logger.error(f"Error interpreting natural language feedback: {e}")
        # Default to keeping all items
        return {
            "kept_items": flat_items,
            "removed_items": [],
            "kept_indices": list(range(len(flat_items))),
            "removed_indices": [],
        }

async def process_outline_feedback_continuation(ctx: RunContext, user_message: str):
    """Process the user feedback received in a continuation call"""
    # Get the data from the previous call
    state = ctx.state.get_state(ctx.conversation_id)
    feedback_data = state.get("outline_feedback_data", {})
    flat_items = feedback_data.get("flat_items", [])

    # Process the user input
    user_input = user_message.strip()

    # If user just wants to continue with all items
    if user_input.lower() == "continue" or not user_input:
        await _emit_message(ctx,
            "\n*Continuing with all research outline items.*\n\n"
        )
        return {
            "kept_items": flat_items,
            "removed_items": [],
            "kept_indices": list(range(len(flat_items))),
            "removed_indices": [],
            "preference_vector": {"pdv": None, "strength": 0.0, "impact": 0.0},
        }

    # Check if it's a slash command (keep or remove)
    slash_keep_patterns = [r"^/k\s", r"^/keep\s"]
    slash_remove_patterns = [r"^/r\s", r"^/remove\s"]

    is_keep_cmd = any(
        re.match(pattern, user_input) for pattern in slash_keep_patterns
    )
    is_remove_cmd = any(
        re.match(pattern, user_input) for pattern in slash_remove_patterns
    )

    # Process slash commands
    if is_keep_cmd or is_remove_cmd:
        # Extract the item indices/ranges part
        if is_keep_cmd:
            items_part = re.sub(r"^(/k|/keep)\s+", "", user_input).replace(",", " ")
        else:
            items_part = re.sub(r"^(/r|/remove)\s+", "", user_input).replace(
                ",", " "
            )

        # Process the indices and ranges
        selected_indices: set[int] = set()
        for part in items_part.split():
            part = part.strip()
            if not part:
                continue

            # Check if it's a range (e.g., 5-9)
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    # Validate range bounds before converting to 0-indexed
                    if (
                        start < 1
                        or start > len(flat_items)
                        or end < 1
                        or end > len(flat_items)
                    ):
                        await _emit_message(ctx,
                            f"Invalid range '{part}': valid range is 1-{len(flat_items)}. Skipping."
                        )
                        continue

                    # Convert to 0-indexed
                    start = start - 1
                    end = end - 1
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    await _emit_message(ctx,
                        f"Invalid range format: '{part}'. Skipping."
                    )
            else:
                # Single number
                try:
                    idx = int(part)
                    # Validate index before converting to 0-indexed
                    if idx < 1 or idx > len(flat_items):
                        await _emit_message(ctx,
                            f"Index {idx} out of range: valid range is 1-{len(flat_items)}. Skipping."
                        )
                        continue

                    # Convert to 0-indexed
                    idx = idx - 1
                    selected_indices.add(idx)
                except ValueError:
                    await _emit_message(ctx, f"Invalid number: '{part}'. Skipping.")

        # Convert to a sorted list for deterministic downstream slicing.
        sorted_selected: list[int] = sorted(selected_indices)

        # Determine kept and removed indices based on mode
        if is_keep_cmd:
            # Keep mode - selected indices are kept, others removed
            kept_indices = sorted_selected
            removed_indices = [
                i for i in range(len(flat_items)) if i not in kept_indices
            ]
        else:
            # Remove mode - selected indices are removed, others kept
            removed_indices = sorted_selected
            kept_indices = [
                i for i in range(len(flat_items)) if i not in removed_indices
            ]

        # Get the actual items
        kept_items = [flat_items[i] for i in kept_indices if i < len(flat_items)]
        removed_items = [
            flat_items[i] for i in removed_indices if i < len(flat_items)
        ]
    else:
        # Process natural language feedback
        nl_feedback = await process_natural_language_feedback(ctx,
            user_input, flat_items
        )

        # Make sure we have a valid response, not None
        if nl_feedback is None:
            # Default to keeping all items
            nl_feedback = {
                "kept_items": flat_items,
                "removed_items": [],
                "kept_indices": list(range(len(flat_items))),
                "removed_indices": [],
            }

        kept_items = nl_feedback.get("kept_items", flat_items)
        removed_items = nl_feedback.get("removed_items", [])
        kept_indices = nl_feedback.get("kept_indices", list(range(len(flat_items))))
        removed_indices = nl_feedback.get("removed_indices", [])

    # Calculate preference direction vector based on kept and removed items
    preference_vector = await calculate_preference_direction_vector(ctx,
        [str(x) for x in kept_items], [str(x) for x in removed_items], flat_items
    )

    # Update user_preferences in state with the new preference vector
    ctx.state.update_state(ctx.conversation_id, "user_preferences", preference_vector)
    logger.info(
        f"Updated user_preferences with PDV impact: {preference_vector.get('impact', 0.0):.3f}"
    )

    # Show the user what's happening
    await _emit_message(ctx, "\n### Feedback Processed\n")

    if kept_items:
        kept_list = "\n".join([f"✓ {item}" for item in kept_items])
        await _emit_message(ctx,
            f"**Keeping {len(kept_items)} items:**\n{kept_list}\n"
        )

    if removed_items:
        removed_list = "\n".join([f"✗ {item}" for item in removed_items])
        await _emit_message(ctx,
            f"**Removing {len(removed_items)} items:**\n{removed_list}\n"
        )

    await _emit_message(ctx, "Generating replacement items for removed topics...\n")

    return {
        "kept_items": kept_items,
        "removed_items": removed_items,
        "kept_indices": kept_indices,
        "removed_indices": removed_indices,
        "preference_vector": preference_vector,
    }

async def continue_research_after_feedback(
    ctx: RunContext,
    feedback_result,
    user_message,
    outline_items,
    all_topics,
    outline_embedding,
):
    """Continue the research process after receiving user feedback on the outline"""
    kept_items = feedback_result["kept_items"]
    removed_items = feedback_result["removed_items"]
    preference_vector = feedback_result["preference_vector"]

    # If there are no removed items, skip the replacement logic and return original outline
    if not removed_items:
        await _emit_message(ctx,
            "\n*No changes made to research outline. Continuing with original outline.*\n\n"
        )
        ctx.state.update_state(ctx.conversation_id,
            "research_state",
            {
                "research_outline": outline_items,
                "all_topics": all_topics,
                "outline_embedding": outline_embedding,
                "user_message": user_message,
            },
        )

        # Clear waiting flag
        ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", False)
        return outline_items, all_topics, outline_embedding

    # Generate replacement topics for removed items if needed
    if removed_items:
        await _emit_status(ctx, "info", "Generating replacement topics...", False)
        replacement_topics = await generate_replacement_topics(ctx,
            user_message,
            kept_items,
            removed_items,
            preference_vector,
            all_topics,
        )

        if replacement_topics:
            # Group replacement topics semantically
            topic_groups = await group_replacement_topics(ctx, replacement_topics)

            # Get state for tracking URLs
            state = ctx.state.get_state(ctx.conversation_id)
            url_selected_count = state.get("url_selected_count", {})

            # Get initial results to track URLs from previous cycles
            results_history = state.get("results_history", [])

            # Create a set of already seen URLs from all previous research
            previously_seen_urls = set()
            for result in results_history:
                url = result.get("url", "")
                if url:
                    previously_seen_urls.add(url)

            # Also track URLs we see during this replacement cycle
            replacement_cycle_seen_urls = set()

            # For each group, generate and execute targeted queries
            group_results = []
            for group in topic_groups:
                # Generate a query that covers this group of topics
                group_query = await generate_group_query(ctx, group, user_message)

                # Get query embedding
                query_embedding = await get_embedding(ctx, group_query)

                # Execute search for this group
                if not ctx.valves.events.quiet_chat_mode:
                    await _emit_message(ctx,
                        f"**Researching topics:** {', '.join(group)}\n**Query:** {group_query}\n\n"
                    )
                else:
                    await _emit_status(ctx,
                        "info",
                        f"Researching: {', '.join(group)}",
                        False,
                    )
                results = await process_query(ctx,
                    group_query, query_embedding, outline_embedding
                )

                # Filter out URLs we've seen in previous cycles or this replacement cycle
                filtered_results = []
                for result in results:
                    url = result.get("url", "")

                    # Skip if we've seen this URL in previous cycles or this replacement cycle
                    if url and (
                        url in previously_seen_urls
                        or url in replacement_cycle_seen_urls
                    ):
                        continue

                    # Keep new URLs we haven't seen before
                    filtered_results.append(result)
                    if url:
                        replacement_cycle_seen_urls.add(
                            url
                        )  # Mark as seen in this cycle

                # If we have no results after filtering but had some initially, use fallback
                if not filtered_results and results:
                    # Use a fallback approach - find the least seen URL
                    least_seen = None
                    min_seen_count = float("inf")

                    for result in results:
                        url = result.get("url", "")
                        seen_count = url_selected_count.get(url, 0)

                        if seen_count < min_seen_count:
                            min_seen_count = seen_count
                            least_seen = result

                    if least_seen:
                        filtered_results.append(least_seen)
                        if least_seen.get("url"):
                            replacement_cycle_seen_urls.add(least_seen.get("url"))
                        logger.info(
                            "Using least-seen URL as fallback to ensure research continues"
                        )

                group_results.append(
                    {
                        "topics": group,
                        "query": group_query,
                        "results": filtered_results,
                    }
                )

            # Now refine each topic based on both PDV and search results
            refined_topics = []
            for group in group_results:
                topics = group["topics"]
                results = group["results"]

                # Extract key information from results relevant to these topics
                relevant_info = await extract_topic_relevant_info(ctx,
                    results, topics
                )

                # Generate refined topics that incorporate both user preferences and new research
                refined = await refine_topics_with_research(ctx,
                    topics,
                    relevant_info,
                    ctx.state.get_state(ctx.conversation_id).get("user_preferences", {}).get("pdv"),
                    user_message,
                )

                refined_topics.extend(refined)

            # Use these refined topics in place of the original replacement topics
            replacement_topics = refined_topics

            # Create new research outline structure
            new_research_outline = []
            new_all_topics = []

            # Track the original hierarchy
            original_hierarchy = {}  # Store parent-child relationships
            original_main_topics = set()  # Track which items were main topics
            original_subtopics = set()  # Track which items were subtopics

            # Extract from the original outline structure
            for topic_item in outline_items:
                topic = topic_item["topic"]
                original_main_topics.add(topic)
                subtopics = topic_item.get("subtopics", [])

                # Track the hierarchy
                for subtopic in subtopics:
                    original_hierarchy[subtopic] = topic
                    original_subtopics.add(subtopic)

            # Process kept items to maintain hierarchy
            for topic_item in outline_items:
                topic = topic_item["topic"]
                subtopics = topic_item.get("subtopics", [])

                if topic in kept_items:
                    # Keep the original topic with its kept subtopics
                    kept_subtopics = [s for s in subtopics if s in kept_items]
                    if kept_subtopics:  # Only add if there are kept subtopics
                        new_topic_item = {
                            "topic": topic,
                            "subtopics": kept_subtopics,
                        }
                        new_research_outline.append(new_topic_item)
                        new_all_topics.append(topic)
                        new_all_topics.extend(kept_subtopics)
                    else:
                        # If main topic is kept but no subtopics, still add it
                        new_topic_item = {"topic": topic, "subtopics": []}
                        new_research_outline.append(new_topic_item)
                        new_all_topics.append(topic)
                else:
                    # For removed main topics, check if any subtopics were kept
                    kept_subtopics = [s for s in subtopics if s in kept_items]
                    if kept_subtopics:
                        # Just restore the original main topic name teehee
                        revised_topic = f"{topic}"
                        new_topic_item = {
                            "topic": revised_topic,
                            "subtopics": kept_subtopics,
                        }
                        new_research_outline.append(new_topic_item)
                        new_all_topics.append(revised_topic)
                        new_all_topics.extend(kept_subtopics)

            # Process orphaned kept items (not already added)
            orphaned_kept_items = [
                item for item in kept_items if item not in new_all_topics
            ]

            # Get embeddings for assignment
            if orphaned_kept_items and new_research_outline:
                try:
                    # Try to add orphaned items to existing topics based on semantic similarity
                    main_topic_embeddings = {}
                    for outline_item in new_research_outline:
                        topic = outline_item["topic"]
                        embedding = await get_embedding(ctx, topic)
                        if embedding:
                            main_topic_embeddings[topic] = embedding

                    for item in orphaned_kept_items:
                        item_embedding = await get_embedding(ctx, item)
                        if item_embedding:
                            # Find best match
                            best_match = None
                            best_score = 0.5  # Threshold

                            for (
                                topic,
                                topic_embedding,
                            ) in main_topic_embeddings.items():
                                similarity = cosine_similarity(
                                    [item_embedding], [topic_embedding]
                                )[0][0]
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = topic

                            if best_match:
                                # Add to existing topic
                                for outline_item in new_research_outline:
                                    if outline_item["topic"] == best_match:
                                        outline_item["subtopics"].append(item)
                                        new_all_topics.append(item)
                                        break
                            else:
                                # If no good match, create a new topic from the item
                                if item in original_main_topics:
                                    # It was a main topic, keep it that way
                                    new_research_outline.append(
                                        {"topic": item, "subtopics": []}
                                    )
                                    new_all_topics.append(item)
                                else:
                                    # It was a subtopic, but now it's orphaned, make it a main topic
                                    new_research_outline.append(
                                        {"topic": item, "subtopics": []}
                                    )
                                    new_all_topics.append(item)
                        else:
                            # No embedding, add as a main topic
                            new_research_outline.append(
                                {"topic": item, "subtopics": []}
                            )
                            new_all_topics.append(item)
                except Exception as e:
                    logger.error(f"Error assigning orphaned items: {e}")
                    # Add all orphaned items as main topics on error
                    for item in orphaned_kept_items:
                        new_research_outline.append(
                            {"topic": item, "subtopics": []}
                        )
                        new_all_topics.append(item)
            elif orphaned_kept_items:
                # No existing topics to add to, make each orphaned item a main topic
                for item in orphaned_kept_items:
                    new_research_outline.append({"topic": item, "subtopics": []})
                    new_all_topics.append(item)

            # Add replacement topics now
            # First, try to add them to semantically similar existing main topics
            if replacement_topics and new_research_outline:
                try:
                    # Get embeddings for existing main topics
                    main_topic_embeddings = {}
                    for outline_item in new_research_outline:
                        topic = outline_item["topic"]
                        embedding = await get_embedding(ctx, topic)
                        if embedding:
                            main_topic_embeddings[topic] = embedding

                    # Track which replacements have been assigned
                    assigned_replacements = set()

                    # Try to assign each replacement to a semantically similar main topic
                    for replacement in replacement_topics:
                        replacement_embedding = await get_embedding(ctx,
                            replacement
                        )
                        if replacement_embedding:
                            # Find best match
                            best_match = None
                            best_score = 0.65  # Higher threshold for replacements

                            for (
                                topic,
                                topic_embedding,
                            ) in main_topic_embeddings.items():
                                similarity = cosine_similarity(
                                    [replacement_embedding], [topic_embedding]
                                )[0][0]
                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = topic

                            if best_match:
                                # Add to existing topic
                                for outline_item in new_research_outline:
                                    if outline_item["topic"] == best_match:
                                        outline_item["subtopics"].append(
                                            replacement
                                        )
                                        new_all_topics.append(replacement)
                                        assigned_replacements.add(replacement)
                                        break

                    # Create new topics for unassigned replacements
                    unassigned_replacements = [
                        r
                        for r in replacement_topics
                        if r not in assigned_replacements
                    ]

                    # Group the unassigned replacements
                    replacement_groups = await group_replacement_topics(ctx,
                        unassigned_replacements
                    )

                    for group in replacement_groups:
                        # Generate title for the group
                        try:
                            group_title = await generate_group_title(ctx,
                                group, user_message
                            )
                        except Exception as e:
                            logger.error(f"Error generating group title: {e}")
                            group_title = f"Additional Research Area {len(new_research_outline) - len(outline_items) + 1}"

                        # Add as a new main topic
                        new_research_outline.append(
                            {"topic": group_title, "subtopics": group}
                        )
                        new_all_topics.append(group_title)
                        new_all_topics.extend(group)

                except Exception as e:
                    logger.error(f"Error during replacement topic assignment: {e}")
                    # Fallback: add all replacements as a new group
                    group_title = "Additional Research Topics"
                    new_research_outline.append(
                        {"topic": group_title, "subtopics": replacement_topics}
                    )
                    new_all_topics.append(group_title)
                    new_all_topics.extend(replacement_topics)
            elif replacement_topics:
                # No existing outline to add to, create groups from replacements
                replacement_groups = await group_replacement_topics(ctx,
                    replacement_topics
                )

                for i, group in enumerate(replacement_groups):
                    try:
                        group_title = await generate_group_title(ctx,
                            group, user_message
                        )
                    except Exception as e:
                        logger.error(f"Error generating group title: {e}")
                        group_title = f"Research Group {i + 1}"

                    new_research_outline.append(
                        {"topic": group_title, "subtopics": group}
                    )
                    new_all_topics.append(group_title)
                    new_all_topics.extend(group)

            # Update the research outline and topic list
            if new_research_outline:  # Only update if we have valid content
                research_outline = new_research_outline
                all_topics = new_all_topics

                # Update outline embedding based on all_topics
                outline_text = " ".join(all_topics)
                outline_embedding = await get_embedding(ctx, outline_text)

                # Re-initialize dimension tracking with new topics
                await initialize_research_dimensions(ctx, all_topics, user_message)

                # Make sure to store initial coverage for later display
                research_dimensions = state.get("research_dimensions")
                if research_dimensions:
                    # Make a copy to avoid reference issues
                    ctx.state.update_state(ctx.conversation_id,
                        "latest_dimension_coverage",
                        research_dimensions["coverage"].copy(),
                    )
                    logger.info(
                        f"Updated dimension coverage after feedback with {len(research_dimensions['coverage'])} values"
                    )

                    # Also update trajectory accumulator for consistency
                    ctx.trajectory_accumulator = (
                        None  # Reset for fresh accumulation
                    )

                # Show the updated outline to the user
                updated_outline = "### Updated Research Outline\n\n"
                for topic_item in research_outline:
                    updated_outline += f"**{topic_item['topic']}**\n"
                    for subtopic in topic_item.get("subtopics", []):
                        updated_outline += f"- {subtopic}\n"
                    updated_outline += "\n"

                await _emit_message(ctx, updated_outline)

                # Updated message about continuing with main research
                await _emit_message(ctx,
                    "\n*Updated research outline with user preferences. Continuing to main research cycles...*\n\n"
                )

                # Store the updated research state
                ctx.state.update_state(ctx.conversation_id,
                    "research_state",
                    {
                        "research_outline": research_outline,
                        "all_topics": all_topics,
                        "outline_embedding": outline_embedding,
                        "user_message": user_message,
                    },
                )

                # Clear waiting flag
                ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", False)
                return research_outline, all_topics, outline_embedding
            else:
                # If we couldn't create a valid outline, continue with original
                await _emit_message(ctx,
                    "\n*No valid outline could be created. Continuing with original outline.*\n\n"
                )
                ctx.state.update_state(ctx.conversation_id,
                    "research_state",
                    {
                        "research_outline": outline_items,
                        "all_topics": all_topics,
                        "outline_embedding": outline_embedding,
                        "user_message": user_message,
                    },
                )

                # Clear waiting flag
                ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", False)
                return outline_items, all_topics, outline_embedding
        else:
            # No items were removed, continue with original outline
            await _emit_message(ctx,
                "\n*No changes made to research outline. Continuing with original outline.*\n\n"
            )
            ctx.state.update_state(ctx.conversation_id,
                "research_state",
                {
                    "research_outline": outline_items,
                    "all_topics": all_topics,
                    "outline_embedding": outline_embedding,
                    "user_message": user_message,
                },
            )

            # Clear waiting flag
            ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", False)
            return outline_items, all_topics, outline_embedding


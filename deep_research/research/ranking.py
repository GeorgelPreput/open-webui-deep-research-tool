import logging
import re
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from deep_research.config.constants import RELEVANCY_SNIPPET_LENGTH
from deep_research.core.text import response_text
from deep_research.core.types import RunContext
from deep_research.progress.events import MessageEvent, StatusEvent
from deep_research.semantics.eigendecomposition import (
    apply_semantic_transformation,
)
from deep_research.semantics.embeddings import get_embedding
from deep_research.web.fetch import fetch_content

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def select_most_relevant_results(
    ctx: RunContext,
    results: list[dict[str, Any]],
    query: str,
    query_embedding: list[float] | None,
    outline_embedding: list[float] | None,
    summary_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Select the most relevant results from extra results pool using semantic transformations with similarity caching"""
    if not results:
        return results

    # If we only have the base needed amount or fewer, return them all
    base_results_per_query = ctx.valves.web.search_results_per_query
    if len(results) <= base_results_per_query:
        return results

    # Get state for URL tracking
    state = ctx.state.get_state(ctx.conversation_id)
    url_selected_count = state.get("url_selected_count", {})

    # Count URLs that have been repeated at REPEATS_BEFORE_EXPANSION times or more
    repeat_count = 0
    for count in url_selected_count.values():
        if count >= ctx.valves.web.repeats_before_expansion:
            repeat_count += 1

    # Calculate additional results to fetch based on repeat count
    additional_results = min(repeat_count, ctx.valves.web.extra_results_per_query)
    results_to_select = base_results_per_query + additional_results

    # Calculate relevance scores for each result
    relevance_scores = []

    # Get transformation if available
    state = ctx.state.get_state(ctx.conversation_id)
    transformation = state.get("semantic_transformations")

    # Get similarity cache
    similarity_cache = state.get("similarity_cache", {})

    # Process domain priority valve value (if provided)
    priority_domains = []
    if ctx.valves.web.domain_priority:
            # Split by commas and/or spaces
            domain_input = ctx.valves.web.domain_priority
            # Replace commas with spaces, then split by spaces
            domain_items = domain_input.replace(",", " ").split()
            # Remove empty items and add to priority domains
            priority_domains = [
                item.strip().lower() for item in domain_items if item.strip()
            ]
            if priority_domains:
                logger.info(f"Using priority domains: {priority_domains}")

    # Process content priority valve value (if provided)
    priority_keywords = []
    if ctx.valves.web.content_priority:
        # Split by commas and/or spaces, handling quoted phrases
        content_input = ctx.valves.web.content_priority

        # Function to parse keywords, respecting quotes
        def parse_keywords(text):
            keywords = []
            # Pattern for quoted phrases or words
            pattern = r"\'([^\']+)\'|\"([^\"]+)\"|(\S+)"

            matches = re.findall(pattern, text)
            for match in matches:
                # Each match is a tuple with three groups (one will contain the text)
                keyword = match[0] or match[1] or match[2]
                if keyword:
                    keywords.append(keyword.lower())
            return keywords

        priority_keywords = parse_keywords(content_input)
        if priority_keywords:
            logger.info(f"Using priority keywords: {priority_keywords}")

    # Get multiplier values from valves or use defaults
    domain_multiplier = getattr(ctx.valves.web, "domain_multiplier", 1.5)
    keyword_multiplier_per_match = getattr(
        ctx.valves.web, "keyword_multiplier_per_match", 1.1
    )
    max_keyword_multiplier = getattr(ctx.valves.web, "max_keyword_multiplier", 2.0)

    for i, result in enumerate(results):
        try:
            # Get a snippet for evaluation
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            # If snippet is too short and URL is available, fetch a bit of content
            if len(snippet) < RELEVANCY_SNIPPET_LENGTH and url:
                try:
                    await _emit_status(ctx,
                        "info",
                        f"Fetching snippet for relevance check: {url[:50]}...",
                        False,
                    )
                    # Only fetch the first part of the content for evaluation
                    content_preview = await fetch_content(ctx, url)
                    if content_preview:
                        snippet = content_preview[
                            : RELEVANCY_SNIPPET_LENGTH
                        ]
                except Exception as e:
                    logger.error(f"Error fetching content for relevance check: {e}")

            # Calculate relevance if we have enough content
            if snippet and len(snippet) > 100:
                # FIRST, CHECK FOR VOCABULARY LIST
                words = re.findall(r"\b\w+\b", snippet[:2000].lower())
                if len(words) > 150:  # Only check if enough words
                    unique_words = set(words)
                    unique_ratio = len(unique_words) / len(words)
                    if (
                        unique_ratio > 0.98
                    ):  # Extremely high uniqueness = vocabulary list
                        logger.warning(
                            f"Skipping likely vocabulary list: {unique_ratio:.3f} uniqueness ratio"
                        )
                        # Assign a very low similarity score
                        similarity = 0.01
                        relevance_scores.append((i, similarity))
                        result["similarity"] = similarity
                        continue  # Skip the expensive embedding calculation

                # Get embedding for the snippet
                snippet_embedding = await get_embedding(ctx, snippet)

                if snippet_embedding:
                    # Apply transformation to query only (Alternative A)
                    if transformation:
                        # Transform the query, not the content
                        transformed_query = (
                            await apply_semantic_transformation(ctx,
                                query_embedding, transformation
                            )
                        )

                        # Calculate similarity between untransformed content and transformed query
                        similarity = cosine_similarity(
                            [snippet_embedding], [transformed_query]
                        )[0][0]
                    else:
                        # Calculate basic similarity if no transformation
                        similarity = cosine_similarity(
                            [snippet_embedding], [query_embedding]
                        )[0][0]

                    # Track original similarity for logging
                    original_similarity = similarity

                    # Apply domain multiplier if priority domains are set
                    if priority_domains and url:
                        url_lower = url.lower()
                        if any(domain in url_lower for domain in priority_domains):
                            similarity *= domain_multiplier
                            logger.debug(
                                f"Applied domain multiplier {domain_multiplier}x to URL: {url}"
                            )

                    # Apply keyword multiplier if priority keywords are set
                    if priority_keywords and snippet:
                        snippet_lower = snippet.lower()
                        # Count matching keywords
                        keyword_matches = [
                            keyword
                            for keyword in priority_keywords
                            if keyword in snippet_lower
                        ]
                        keyword_count = len(keyword_matches)

                        if keyword_count > 0:
                            # Calculate cumulative multiplier (multiply by keyword_multiplier_per_match for each match)
                            # But cap at max_keyword_multiplier
                            cumulative_multiplier = min(
                                max_keyword_multiplier,
                                keyword_multiplier_per_match**keyword_count,
                            )
                            similarity *= cumulative_multiplier
                            logger.debug(
                                f"Applied keyword multiplier {cumulative_multiplier:.2f}x "
                                f"({keyword_count} keywords matched: {', '.join(keyword_matches[:3])}) to result {i}"
                            )

                    # Cap at 0.99 to avoid perfect scores
                    similarity = min(0.99, similarity)

                    # Log the full transformation if multipliers were applied
                    if similarity != original_similarity:
                        logger.info(
                            f"Result {i} multiplied: {original_similarity:.3f} → {similarity:.3f}"
                        )

                    # Store similarity in the result object for later use in topic dampening
                    result["similarity"] = similarity

                    # Apply penalty for repeated URLs
                    repeat_penalty = 1.0
                    url_repeats = url_selected_count.get(url, 0)
                    if url_repeats > 0:
                        # Apply a progressive penalty based on number of repeats
                        # More repeats = lower score (0.9, 0.8, 0.7, etc.)
                        repeat_penalty = max(0.5, 1.0 - (0.1 * url_repeats))
                        logger.debug(
                            f"Applied repeat penalty of {repeat_penalty} to URL: {url}"
                        )

                    # Apply penalty to similarity score
                    similarity *= repeat_penalty

                    # Store score for sorting
                    relevance_scores.append((i, similarity))

                    # Also store in the result for future use
                    result["similarity"] = similarity
                else:
                    # No embedding, assign low score
                    relevance_scores.append((i, 0.1))
                    result["similarity"] = 0.1
            else:
                # Insufficient content, assign low score
                relevance_scores.append((i, 0.0))
                result["similarity"] = 0.0

        except Exception as e:
            logger.error(f"Error calculating relevance for result {i}: {e}")
            relevance_scores.append((i, 0.0))
            result["similarity"] = 0.0

    # Update similarity cache
    ctx.state.update_state(ctx.conversation_id, "similarity_cache", similarity_cache)

    # Sort by relevance score (highest first)
    relevance_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top results based on the dynamic count
    selected_indices = [x[0] for x in relevance_scores[:results_to_select]]
    selected_results = [results[i] for i in selected_indices]

    # Log selection information
    logger.info(
        f"Selected {len(selected_results)} most relevant results from {len(results)} total (added {additional_results} due to repeats)"
    )
    # Collect all content and quality factors first
    all_content = []
    for result in selected_results:
        content = result.get("content", "")[:2000]
        if content:
            # Use similarity as quality factor, normalize between 0.5-1.0
            quality = 0.5
            if "similarity" in result:
                quality = 0.5 + (result["similarity"] * 0.5)
            all_content.append((content, quality))

    # Update ALL coverage in a single call
    if all_content:
        # Just grab dimensions once
        state = ctx.state.get_state(ctx.conversation_id)
        dims = state.get("research_dimensions")
        if dims and "coverage" in dims:
            coverage = np.array(dims["coverage"])

            # Process each content item sequentially
            for content, quality in all_content:
                embed = await get_embedding(ctx, content[:2000])
                if not embed:
                    continue
                projection = np.dot(embed, np.array(dims["eigenvectors"]).T)
                contribution = np.abs(projection) * quality

                # Update coverage directly
                for i in range(min(len(contribution), len(coverage))):
                    coverage[i] += contribution[i] * (1 - coverage[i] / 2)

            # Normalize once at the end
            coverage = np.minimum(coverage, 3.0) / 3.0

            # Save back
            dims["coverage"] = coverage.tolist()
            ctx.state.update_state(ctx.conversation_id, "research_dimensions", dims)
            ctx.state.update_state(ctx.conversation_id, "latest_dimension_coverage", coverage.tolist())

            # Log dimension updates for debugging
            state = ctx.state.get_state(ctx.conversation_id)
            research_dimensions = state.get("research_dimensions")
            if research_dimensions:
                coverage = research_dimensions.get("coverage", [])
                logger.debug(
                    f"Dimension coverage after result: {[round(c * 100) for c in coverage[:3]]}%..."
                )

    return selected_results

async def check_result_relevance(
    ctx: RunContext,
    result: dict[str, Any],
    query: str,
    outline_items: list[str] | None = None,
) -> bool:
    """Check if a search result is relevant to the query and research outline using a lightweight model"""
    if not ctx.valves.web.quality_filter_enabled:
        return True  # Skip filtering if disabled

    # Get similarity score from result - access it correctly
    similarity = result.get("similarity", 0.0)

    # Skip filtering for very high similarity scores
    if similarity >= ctx.valves.web.quality_similarity_threshold:
        logger.info(
            f"Result passed quality filter automatically with similarity {similarity:.3f}"
        )
        return True

    # Get content from the result
    content = result.get("content", "")
    title = result.get("title", "")
    url = result.get("url", "")

    if not content or len(content) < 200:
        logger.warning(
            "Content too short for quality filtering, accepting by default"
        )
        return True

    # Create prompt for relevance checking
    relevance_prompt = {
        "role": "system",
        "content": """You are evaluating the relevance of a search result to a research query.
Your task is to determine if the content is actually relevant to what the user is researching.

Answer with ONLY "Yes" if the content is relevant to the research query or "No" if it is:
- Not related to the core topic
- An advertisement disguised as content
- About a different product/concept with similar keywords
- So general or vague that it provides no substantive information
- Littered with HTML or CSS to the point of being unreadable

Reply with JUST "Yes" or "No" - no explanation or other text.""",
    }

    # Create context with query, outline, and full content
    context = f"Research Query: {query}\n\n"

    if outline_items and len(outline_items) > 0:
        context += "Research Outline Topics:\n"
        for item in outline_items[:5]:  # Limit to first 5 items
            context += f"- {item}\n"
        context += "\n"

    context += f"Result Title: {title}\n"
    context += f"Result URL: {url}\n\n"
    context += f"Content:\n{content}\n\n"
    context += f"""Is the above content relevant to this query: "{query}"? Answer with ONLY 'Yes' or 'No'."""

    try:
        # Use quality filter model
        quality_model = ctx.valves.models.quality_filter_model

        response = await ctx.client.chat_completions(
            quality_model,
            [relevance_prompt, {"role": "user", "content": context}],
            temperature=ctx.valves.models.temperature
            * 0.2,  # Use your valve system with adjustment
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            answer = response_text(response).strip().lower()

            # Parse the response to get yes/no
            is_relevant = "yes" in answer.lower() and "no" not in answer.lower()

            logger.info(
                f"Quality check for result: {'RELEVANT' if is_relevant else 'NOT RELEVANT'} (sim={similarity:.3f})"
            )

            return is_relevant
        else:
            logger.warning(
                "Failed to get response from quality model, accepting by default"
            )
            return True

    except Exception as e:
        logger.error(f"Error in quality filtering: {e}")
        return True  # Accept by default on error

async def rank_topics_by_research_priority(
    ctx: RunContext,
    active_topics: list[str],
    gap_vector: list[float] | None = None,
    completed_topics: set[str] | None = None,
    research_results: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Rank research topics by priority using semantic dimensions and gap analysis with dampening for frequently used topics"""
    if not active_topics:
        return []

    # If we only have a few topics, keep the original order
    if len(active_topics) <= 3:
        return active_topics

    # Get cache of topic alignments
    state = ctx.state.get_state(ctx.conversation_id)
    topic_alignment_cache = state.get("topic_alignment_cache", {})

    # Get topic usage counts for dampening
    topic_usage_counts = state.get("topic_usage_counts", {})
    dampening_factor = 0.9  # Each use reduces priority by 10%

    # Initialize scores for each topic
    topic_scores = {}

    # Get embeddings for all topics
    logger.info(f"Getting embeddings for {len(active_topics)} topics")
    topic_embeddings = {}

    # Get embeddings for each topic
    for topic in active_topics:
        embedding = await get_embedding(ctx, topic)
        if embedding:
            topic_embeddings[topic] = embedding

    # Get research trajectory for alignment calculation
    research_trajectory = state.get("research_trajectory")

    # Get user preferences
    user_preferences = state.get("user_preferences", {})
    pdv = user_preferences.get("pdv")
    pdv_impact = user_preferences.get("impact", 0.0)

    # Get current cycle for adaptive weights
    current_cycle = len(state.get("cycle_summaries", [])) + 1
    max_cycles = ctx.valves.cycles.max_cycles

    # Calculate weights for different factors based on research progress
    trajectory_weight = ctx.valves.cycles.trajectory_momentum

    # PDV weight calculation
    pdv_weight = 0.0
    if pdv is not None and pdv_impact > 0.1:
        pdv_alignment_history = state.get("pdv_alignment_history", [])
        if pdv_alignment_history:
            recent_alignment = sum(pdv_alignment_history[-3:]) / max(
                1, len(pdv_alignment_history[-3:])
            )
            alignment_factor = min(1.0, recent_alignment * 2)
            pdv_weight = pdv_impact * alignment_factor

            # Apply adaptive fade-out
            fade_start_cycle = min(5, int(0.33 * max_cycles))
            if current_cycle > fade_start_cycle:
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_ratio = remaining_cycles / total_fade_cycles
                    pdv_weight *= max(0.0, fade_ratio)
                else:
                    pdv_weight = 0.0
        else:
            pdv_weight = pdv_impact

    # Gap weight calculation
    gap_weight = 0.0
    if gap_vector is not None:
        fade_start_cycle = min(5, int(0.5 * max_cycles))
        if current_cycle <= fade_start_cycle:
            gap_weight = ctx.valves.cycles.gap_exploration_weight
        else:
            remaining_cycles = max_cycles - current_cycle
            total_fade_cycles = max_cycles - fade_start_cycle
            if total_fade_cycles > 0:
                fade_ratio = remaining_cycles / total_fade_cycles
                gap_weight = ctx.valves.cycles.gap_exploration_weight * max(
                    0.0, fade_ratio
                )

    # Content relevance weight increases over time
    relevance_weight = 0.2 + (0.3 * min(1.0, current_cycle / (max_cycles * 0.7)))

    # Normalize weights to sum to 1.0
    total_weight = trajectory_weight + pdv_weight + gap_weight + relevance_weight
    if total_weight > 0:
        trajectory_weight /= total_weight
        pdv_weight /= total_weight
        gap_weight /= total_weight
        relevance_weight /= total_weight

    logger.info(
        f"Priority weights: trajectory={trajectory_weight:.2f}, pdv={pdv_weight:.2f}, gap={gap_weight:.2f}, relevance={relevance_weight:.2f}"
    )

    # Prepare completed topics embeddings for relevance scoring
    completed_embeddings = {}
    if completed_topics and len(completed_topics) > 0 and relevance_weight > 0.0:
        # Limit number of completed topics to consider for efficiency
        completed_sample_size = min(10, len(completed_topics))
        completed_topics_list = list(completed_topics)[:completed_sample_size]

        # Get all completed topics embeddings sequentially
        completed_embed_results = []
        for topic in completed_topics_list:
            embedding = await get_embedding(ctx, topic)
            if embedding:
                completed_embed_results.append(embedding)

        # Store valid embeddings with topic keys
        for i, embedding in enumerate(completed_embed_results):
            if embedding and i < len(completed_topics_list):
                completed_embeddings[completed_topics_list[i]] = embedding

    # Prepare recent result embeddings for relevance scoring
    result_embeddings = {}
    if research_results and len(research_results) > 0 and relevance_weight > 0.0:
        # Get limited recent results (last 8 for efficiency)
        recent_results = research_results[-8:]

        # Prepare content for embedding
        result_contents = []
        for result in recent_results:
            content = result.get("content", "")[:2000]
            result_contents.append(content)

        # Get embeddings sequentially
        result_embed_results = []
        for content in result_contents:
            embedding = await get_embedding(ctx, content)
            if embedding:
                result_embed_results.append(embedding)

        # Store valid embeddings with result index as key
        for i, embedding in enumerate(result_embed_results):
            if embedding and i < len(recent_results):
                result_id = recent_results[i].get("url", "") or f"result_{i}"
                result_embeddings[result_id] = embedding

    # Calculate scores for each topic
    for topic, topic_embedding in topic_embeddings.items():
        component_scores = {}

        # Factor 1: Alignment with trajectory (research direction)
        if research_trajectory is not None and trajectory_weight > 0.0:
            # Check cache first
            cache_key = f"traj_{topic}"
            if cache_key in topic_alignment_cache:
                traj_alignment = topic_alignment_cache[cache_key]
            else:
                traj_alignment = np.dot(topic_embedding, research_trajectory)
                # Normalize to 0-1 range
                traj_alignment = (traj_alignment + 1) / 2
                # Cache the result
                topic_alignment_cache[cache_key] = traj_alignment

            component_scores["trajectory"] = traj_alignment * trajectory_weight

        # Factor 2: Alignment with user preference direction vector
        if pdv is not None and pdv_weight > 0.0:
            # Check cache first
            cache_key = f"pdv_{topic}"
            if cache_key in topic_alignment_cache:
                pdv_alignment = topic_alignment_cache[cache_key]
            else:
                pdv_alignment = np.dot(topic_embedding, pdv)
                # Normalize to 0-1 range
                pdv_alignment = (pdv_alignment + 1) / 2
                # Cache the result
                topic_alignment_cache[cache_key] = pdv_alignment

            component_scores["pdv"] = pdv_alignment * pdv_weight

        # Factor 3: Alignment with gap vector (unexplored areas)
        if gap_vector is not None and gap_weight > 0.0:
            # Check cache first
            cache_key = f"gap_{topic}"
            if cache_key in topic_alignment_cache:
                gap_alignment = topic_alignment_cache[cache_key]
            else:
                gap_alignment = np.dot(topic_embedding, gap_vector)
                # Normalize to 0-1 range
                gap_alignment = (gap_alignment + 1) / 2
                # Cache the result
                topic_alignment_cache[cache_key] = gap_alignment

            component_scores["gap"] = gap_alignment * gap_weight

        # Factor 4: Topic novelty compared to completed research
        if completed_embeddings and relevance_weight > 0.0:
            # Calculate average similarity to completed topics
            similarity_sum = 0
            count = 0

            for (
                completed_topic,
                completed_embedding,
            ) in completed_embeddings.items():
                # Check cache first
                cache_key = f"comp_{topic}_{completed_topic}"
                if cache_key in topic_alignment_cache:
                    sim = topic_alignment_cache[cache_key]
                else:
                    sim = cosine_similarity(
                        [topic_embedding], [completed_embedding]
                    )[0][0]
                    # Cache the result
                    topic_alignment_cache[cache_key] = sim

                similarity_sum += sim
                count += 1

            if count > 0:
                avg_similarity = similarity_sum / count
                # Invert - lower similarity means higher novelty
                novelty = 1.0 - avg_similarity
                component_scores["novelty"] = novelty * (relevance_weight * 0.5)

        # Factor 5: Information need based on search results
        if result_embeddings and relevance_weight > 0.0:
            # Calculate average relevance to results
            relevance_sum = 0
            count = 0

            for result_id, result_embedding in result_embeddings.items():
                # Create cache key using result ID
                cache_key = f"res_{topic}_{hash(result_id) % 10000}"

                if cache_key in topic_alignment_cache:
                    rel = topic_alignment_cache[cache_key]
                else:
                    rel = cosine_similarity([topic_embedding], [result_embedding])[
                        0
                    ][0]
                    # Cache the result
                    topic_alignment_cache[cache_key] = rel

                relevance_sum += rel
                count += 1

            if count > 0:
                avg_relevance = relevance_sum / count
                # Invert - lower relevance means higher information need
                info_need = 1.0 - avg_relevance
                component_scores["info_need"] = info_need * (relevance_weight * 0.5)

        # Calculate final score as sum of all component scores
        final_score = sum(component_scores.values())
        if not component_scores:
            final_score = 0.5  # Default if no components were calculated

        # Apply dampening based on usage count and result quality
        usage_count = topic_usage_counts.get(topic, 0)
        if usage_count > 0:
            # Get all results related to this topic
            topic_results = []

            # Look for results where the topic appears in the query or result content
            for result in research_results or []:
                # Check if this result is relevant to this topic
                result_content = result.get("content", "")[
                    :500
                ]  # Use first 500 chars for efficiency
                if topic in result.get("query", "") or topic in result_content:
                    topic_results.append(result)

            # If we have results for this topic, calculate quality-based dampening
            if topic_results:
                # Calculate average similarity for this topic's results
                avg_similarity = 0.0
                count = 0
                for result in topic_results:
                    similarity = result.get("similarity", 0.0)
                    if similarity > 0:  # Only count results with valid similarity
                        avg_similarity += similarity
                        count += 1

                if count > 0:
                    avg_similarity /= count

                # Scale dampening factor based on result quality
                # similarity > 0.8: no penalty (dampening_multiplier = 1.0)
                # similarity < 0.3: 50% penalty (dampening_multiplier = 0.5)
                # Linear scaling between
                if avg_similarity >= 0.8:
                    dampening_multiplier = 1.0
                elif avg_similarity <= 0.3:
                    dampening_multiplier = 0.5
                else:
                    # Linear scaling between 0.5 and 1.0
                    dampening_multiplier = 0.5 + (
                        0.5 * (avg_similarity - 0.3) / 0.5
                    )

                logger.debug(
                    f"Topic '{topic}' quality-based dampening: {dampening_multiplier:.3f} (avg similarity: {avg_similarity:.3f}, from {count} results)"
                )
            else:
                # If no results yet, use the default dampening
                dampening_multiplier = dampening_factor**usage_count
                logger.debug(
                    f"Topic '{topic}' default dampening: {dampening_multiplier:.3f} (used {usage_count} times)"
                )

            # Apply the dampening multiplier
            final_score *= dampening_multiplier

        # Store the score
        topic_scores[topic] = final_score

    # Update alignment cache with size limiting
    if len(topic_alignment_cache) > 300:  # Limit cache size
        # Create new cache with only the 200 most recent entries
        items = list(topic_alignment_cache.items())
        topic_alignment_cache = dict(reversed(items[-200:]))

    ctx.state.update_state(ctx.conversation_id, "topic_alignment_cache", topic_alignment_cache)

    # Sort topics by score (highest first)
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_topics = [topic for topic, score in sorted_topics]

    logger.info(f"Ranked {len(ranked_topics)} topics by research priority")
    return ranked_topics


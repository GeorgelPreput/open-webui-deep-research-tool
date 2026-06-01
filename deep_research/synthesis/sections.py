import math
import re
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from deep_research.budget.tokens import count_tokens
from deep_research.config.constants import VERIFY_CITATIONS
from deep_research.core.text import response_text, stable_text_key
from deep_research.progress.events import StatusEvent
from deep_research.semantics.embeddings import get_embedding
from deep_research.synthesis.verify import verify_citation_batch
from deep_research.web.fetch import fetch_content


async def generate_subtopic_content_with_citations(
    ctx,
    section_title: str,
    subtopic: str,
    original_query: str,
    research_results: list[dict[str, Any]],
    synthesis_model: str,
    is_follow_up: bool = False,
    previous_summary: str = "",
) -> dict[str, Any]:
    """Generate content for a single subtopic with numbered citations"""
    # Only emit status if we haven't seen this subtopic yet
    if subtopic not in ctx.seen_subtopics:
        await ctx.events.emit(StatusEvent(
                f"Generating content for subtopic: {subtopic}...", "info", False
            ))
        ctx.seen_subtopics.add(subtopic)

    # Get state
    state = ctx.state.get_state(ctx.conversation_id)

    # Get relevance cache or initialize it
    relevance_cache = state.get("subtopic_relevance_cache", {})

    # Create embedding cache keys for efficiency
    query_embedding_key = f"query_embedding_{stable_text_key(original_query)}"
    subtopic_embedding_key = f"subtopic_embedding_{stable_text_key(subtopic)}"
    combined_embedding_key = (
        f"combined_embedding_{stable_text_key(original_query)}_{stable_text_key(subtopic)}"
    )

    # Create a prompt specific to this subtopic
    subtopic_prompt = {
        "role": "system",
        "content": f"""You are a post-grad research assistant writing a concise subsection (1-3 paragraphs) about "{subtopic}"
    for a comprehensive combined research report addressing this query: "{original_query}" based on internet research results.

    Your subsection MUST:
    1. Focus specifically on the subtopic "{subtopic}" within the broader section "{section_title}".
    2. Make FULL use of the provided research sources, and ONLY the provided sources.
    3. Include IN-TEXT CITATIONS for all information from sources, using ONLY the numerical IDs provided in the source list, e.g. [1], [4], etc.
    4. Follow a structure that best fits the subtopic subject matter. Aim for an academic report style while tolerating flexibility as appropriate.
    5. Only be written on the subtopic matter - consider how your subsection will be combined with others in a greater research report.
    6. Be written to a length, between 1 medium paragraph and 3 long paragraphs, based on the subtopic's perceived importance to the research query.

    Your subsection must NOT:
    1. Interpret the content in a lofty way that exaggerates its importance or profundity, or contrives a narrative with empty sophistication.
    2. Attempt to portray the subject matter in any particular sort of light, good or bad, especially by using apologetic or dismissive language.
    3. Focus on perceived complexities or challenges related to the topic or research process, or include appeals to future research.
    4. Ever take a preachy or moralizing tone, or take a "stance" for or against/"side" with or against anything not driven by the provided data.
    5. Overstate the significance of specific services, providers, locations, brands, or other entities beyond examples of some type or category.
    6. Sound to the reader as though it is overtly attempting to be diplomatic, considerate, enthusiastic, or overly-generalized.

    You must accurately cite your sources to avoid plagiarizing. Citations MUST be numerical and correspond to the correct source ID in the provided list.
    Do not combine multiple IDs in one citation tag. Please respond with just the subsection body, no intro or title.""",
    }

    # Create a combined embedding for query + subtopic with state-level caching
    combined_embedding = state.get(combined_embedding_key)

    if combined_embedding is None:
        # Check if we already have the individual embeddings cached in state
        query_embedding = state.get(query_embedding_key)
        subtopic_embedding = state.get(subtopic_embedding_key)

        # Get query embedding if not already cached
        if query_embedding is None:
            try:
                query_embedding = await get_embedding(ctx, original_query)
                if query_embedding:
                    state[query_embedding_key] = query_embedding
            except Exception as e:
                logger.error(f"Error getting query embedding: {e}")

        if subtopic_embedding is None:
            try:
                subtopic_embedding = await get_embedding(ctx, subtopic)
                if subtopic_embedding:
                    state[subtopic_embedding_key] = subtopic_embedding
            except Exception as e:
                logger.error(f"Error getting subtopic embedding: {e}")


        if query_embedding and subtopic_embedding:
            try:
                # Combine with equal weight
                combined_array = (
                    np.array(query_embedding) * 0.5
                    + np.array(subtopic_embedding) * 0.5
                )
                # Normalize
                norm = np.linalg.norm(combined_array)
                if norm > 1e-10:
                    combined_array = combined_array / norm
                combined_embedding = combined_array.tolist()
                # Cache the combined embedding
                state[combined_embedding_key] = combined_embedding
            except Exception as e:
                logger.error(f"Error creating combined embedding: {e}")

    # If combined embedding failed or doesn't exist, fall back to subtopic embedding
    if not combined_embedding:
        # Check if we have cached subtopic embedding
        subtopic_embedding = state.get(subtopic_embedding_key)
        if not subtopic_embedding:
            # Try to get it now
            subtopic_embedding = await get_embedding(ctx, subtopic)
            # Cache if successful
            if subtopic_embedding:
                state[subtopic_embedding_key] = subtopic_embedding

        combined_embedding = subtopic_embedding

    # Build context from research results that might be relevant to this subtopic
    subtopic_context = f"# Subtopic to Write: {subtopic}\n"
    subtopic_context += f"# Within Section: {section_title}\n\n"

    # Add the research outline for context
    subtopic_context += "## Research Outline Context:\n"
    state = ctx.state.get_state(ctx.conversation_id)
    synthesis_outline = state.get("research_state", {}).get("research_outline", [])
    if synthesis_outline:
        for topic_item in synthesis_outline:
            topic = topic_item.get("topic", "")
            if topic == section_title:
                subtopic_context += f"**Current Section: {topic}**\n"
            else:
                subtopic_context += f"Section: {topic}\n"

            for st in topic_item.get("subtopics", []):
                if st == subtopic:
                    subtopic_context += f"  - **Current Subtopic: {st}**\n"
                else:
                    subtopic_context += f"  - {st}\n"
        subtopic_context += "\n"

    # Create a unique cache key for this subtopic
    subtopic_key = f"{section_title}_{subtopic}"

    # Calculate relevance scores for each result
    subtopic_results = []
    result_scores = []

    # Check if we have cached relevance scores for this subtopic
    if subtopic_key in relevance_cache:
        logger.info(f"Using cached relevance scores for subtopic: {subtopic}")
        result_scores = relevance_cache[subtopic_key]
        # Sort by relevance score (highest first)
        result_scores.sort(key=lambda x: x[1], reverse=True)
        # Map back to research results
        subtopic_results = [
            research_results[i]
            for i, _ in result_scores
            if i < len(research_results)
        ]
    elif combined_embedding:
        # Calculate relevance scores using combined query+subtopic embedding
        for i, result in enumerate(research_results):
            content = result.get("content", "")
            if not content:
                continue

            # Create a cache key for this result's embedding
            result_key = f"result_{stable_text_key(result.get('url', ''))}"
            content_embedding = state.get(result_key)

            if not content_embedding:
                content_embedding = await get_embedding(ctx, content[:2000])
                # Cache the content embedding if valid
                if content_embedding:
                    state[result_key] = content_embedding

            if content_embedding:
                similarity = cosine_similarity(
                    [content_embedding], [combined_embedding]
                )[0][0]
                result_scores.append((i, similarity))

        # Cache the relevance scores for this subtopic
        relevance_cache[subtopic_key] = result_scores
        state["subtopic_relevance_cache"] = relevance_cache

        # Sort by relevance score (highest first)
        result_scores.sort(key=lambda x: x[1], reverse=True)
        # Map to research results
        subtopic_results = [research_results[i] for i, _ in result_scores]
    else:
        # If no embedding, just use all results
        subtopic_results = research_results

    # Calculate how many results to include based on number of cycles and vibes
    top_results_count = max(
        3, min(len(subtopic_results), math.ceil(0.5 * ctx.valves.cycles.max_cycles + 3))
    )
    top_results = subtopic_results[:top_results_count]

    # Create source list with assigned IDs
    sources_for_subtopic = {}
    source_id = 1

    # Extract URLs and titles from top results, sort alphabetically by title
    sorted_results = sorted(top_results, key=lambda x: x.get("title", "").lower())

    for result in sorted_results:
        url = result.get("url", "")
        title = result.get("title", "Untitled Source")

        if url and url not in sources_for_subtopic:
            sources_for_subtopic[url] = {
                "id": source_id,
                "title": title,
                "url": url,
                "subtopic": subtopic,
                "section": section_title,
            }
            source_id += 1

    # Add source list to context (at the beginning)
    subtopic_context += (
        "## Available Source List (Use ONLY these numerical citations):\n\n"
    )
    for url, source_data in sorted(
        sources_for_subtopic.items(), key=lambda x: x[1]["title"]
    ):
        subtopic_context += (
            f"[{source_data['id']}] {source_data['title']} - {url}\n"
        )

    subtopic_context += "\n## Research Results:\n\n"

    # Reorder results to have most relevant last (most recent in context)
    top_results.reverse()

    # Add the top results to context (most relevant last)
    for result in top_results:
        url = result.get("url", "")
        title = result.get("title", "Untitled Source")
        content = result.get("content", "")

        # Skip results without content
        if not content:
            continue

        # Get the source ID for this URL
        source_id = sources_for_subtopic.get(url, {}).get("id", "?")

        subtopic_context += f"Source ID: [{source_id}] {title}\n"
        subtopic_context += f"Content: {content}\n\n"

    # Include previous summary if this is a follow-up
    if is_follow_up and previous_summary:
        subtopic_context += "## Previous Research Summary:\n\n"
        subtopic_context += f"{previous_summary}...\n\n"

    # Prepare final instruction
    subtopic_context += f"""Using the provided research sources and referencing them with numerical citations [#], write a concise subsection about "{subtopic}" per the system prompt."""
    subtopic_context += """Every citation MUST be numerical (e.g., [1], [2]) corresponding to the source list provided."""
    subtopic_context += """Please use proper Markdown and write 1-3 focused paragraphs exclusively on this specific subtopic."""

    # Create messages array for completion
    messages = [subtopic_prompt, {"role": "user", "content": subtopic_context}]

    # Generate subtopic content
    try:
        # Calculate scaled temperature from the synthesis temperature valve
        scaled_temperature = (
            ctx.valves.models.temperature
        )  # Use research model temperature for subtopics

        # Use research model for generating subtopics
        response = await ctx.llm.chat_completions(
            synthesis_model,
            messages,
            stream=False,
            temperature=scaled_temperature,
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            subtopic_content = response_text(response)

            # Count tokens in the subtopic content
            tokens = await count_tokens(ctx, subtopic_content)

            # Store content for later use
            subtopic_synthesized_content = state.get(
                "subtopic_synthesized_content", {}
            )
            subtopic_synthesized_content[subtopic] = subtopic_content
            state["subtopic_synthesized_content"] = subtopic_synthesized_content

            # Store source mapping for this subtopic
            subtopic_sources = state.get("subtopic_sources", {})
            subtopic_sources[subtopic] = sources_for_subtopic
            state["subtopic_sources"] = subtopic_sources

            # Identify citations in this subtopic content
            subtopic_citations = []
            for url, source_data in sources_for_subtopic.items():
                local_id = source_data.get("id")
                if local_id is not None:
                    # Find all instances of this citation in the text
                    pattern = (
                        r"([^.!?]*(?:\["
                        + str(local_id)
                        + r"\]|&#"
                        + str(local_id)
                        + r")[^.!?]*[.!?])"
                    )
                    context_matches = re.findall(pattern, subtopic_content)

                    for match in context_matches:
                        citation = {
                            "marker": str(local_id),
                            "raw_text": f"[{local_id}]",
                            "text": match,
                            "url": url,
                            "section": section_title,
                            "subtopic": subtopic,
                            "suggested_title": source_data.get("title", ""),
                        }
                        subtopic_citations.append(citation)

            # Log the sources used
            logger.info(
                f"Subtopic '{subtopic}' uses {len(sources_for_subtopic)} sources"
            )

            return {
                "content": subtopic_content,  # Return original content with local IDs
                "tokens": tokens,
                "sources": sources_for_subtopic,
                "citations": subtopic_citations,
                "verified_citations": [],  # Verification happens later
                "flagged_citations": [],  # Flagging happens later
            }
        else:
            return {
                "content": f"*Error generating content for subtopic: {subtopic}*",
                "tokens": 0,
                "sources": {},
                "citations": [],
                "verified_citations": [],
                "flagged_citations": [],
            }

    except Exception as e:
        logger.error(f"Error generating subtopic content for '{subtopic}': {e}")
        return {
            "content": f"*Error generating content for subtopic: {subtopic}*",
            "tokens": 0,
            "sources": {},
            "citations": [],
            "verified_citations": [],
            "flagged_citations": [],
        }

async def generate_section_content_with_citations(
    ctx,
    section_title: str,
    subtopics: list[str],
    original_query: str,
    research_results: list[dict[str, Any]],
    synthesis_model: str,
    is_follow_up: bool = False,
    previous_summary: str = "",
) -> dict[str, Any]:
    """Generate content for a section by combining subtopics with citations"""
    # Only emit status if we haven't seen this section yet
    if section_title not in ctx.seen_sections:
        await ctx.events.emit(StatusEvent(
                f"Generating content for section: {section_title}...", "info", False
            ))
        ctx.seen_sections.add(section_title)

    # Get state
    state = ctx.state.get_state(ctx.conversation_id)

    # Generate content for each subtopic independently
    subtopic_contents = {}
    section_sources = {}
    all_section_citations = []
    total_tokens = 0

    for subtopic in subtopics:
        subtopic_result = await generate_subtopic_content_with_citations(ctx,
            section_title,
            subtopic,
            original_query,
            research_results,
            synthesis_model,
            is_follow_up,
            previous_summary if is_follow_up else "",
        )

        subtopic_contents[subtopic] = subtopic_result["content"]
        total_tokens += subtopic_result.get("tokens", 0)

        # Collect all citations from this subtopic
        if "citations" in subtopic_result:
            all_section_citations.extend(subtopic_result["citations"])

        # Merge sources with section sources, maintaining unique source IDs and tracking originals
        for url, source_data in subtopic_result["sources"].items():
            if url not in section_sources:
                section_sources[url] = (
                    source_data.copy()
                )  # Use copy to avoid reference issues

            # Store the original local ID with subtopic context for precise replacement
            # Add this information to the source data record
            if "original_ids" not in section_sources[url]:
                section_sources[url]["original_ids"] = {}

            # Track which local ID was used in which subtopic
            local_id = source_data.get("id")
            if local_id is not None:
                section_sources[url]["original_ids"][subtopic] = local_id

    # Build or update the global citation map with sources from this section
    master_source_table = state.get("master_source_table", {})
    global_citation_map = state.get("global_citation_map", {})

    # Add all section sources to global map if not already present
    for url, source_data in section_sources.items():
        if url not in global_citation_map:
            global_citation_map[url] = len(global_citation_map) + 1

        # Also add to master source table if not already there
        if url not in master_source_table:
            source_id = f"S{len(master_source_table) + 1}"
            master_source_table[url] = {
                "id": source_id,
                "title": source_data.get("title", "Untitled Source"),
                "content_preview": "",
                "source_type": "web" if not url.endswith(".pdf") else "pdf",
                "accessed_date": ctx.research_date or datetime.now().strftime("%Y-%m-%d"),
                "cited_in_sections": set([section_title]),
            }
        elif section_title not in master_source_table[url].get(
            "cited_in_sections", set()
        ):
            # Update sections where this source is cited
            master_source_table[url]["cited_in_sections"].add(section_title)

    # Update state
    state["global_citation_map"] = global_citation_map
    state["master_source_table"] = master_source_table

    # Verify citations if enabled
    verified_citations = []
    flagged_citations = []

    if VERIFY_CITATIONS and all_section_citations:
        # Group citations by URL for efficient verification
        citations_by_url: dict[str, list[dict[str, Any]]] = {}
        for citation in all_section_citations:
            url = citation.get("url")
            if url:
                if url not in citations_by_url:
                    citations_by_url[url] = []
                citations_by_url[url].append(citation)

        # Verify each URL's citations
        for url, citations in citations_by_url.items():
            try:
                # Get source content
                url_results_cache = state.get("url_results_cache", {})

                # Check cache first
                source_content = None
                if url in url_results_cache:
                    source_content = url_results_cache[url]

                # If not in cache, fetch source content
                if not source_content or len(source_content) < 200:
                    source_content = await fetch_content(ctx, url)

                if source_content and len(source_content) >= 200:
                    # Add global ID to each citation for verification tracking
                    if url in global_citation_map:
                        global_id = global_citation_map[url]
                        for citation in citations:
                            citation["global_id"] = global_id

                    # Verify citations against source content
                    verification_results = await verify_citation_batch(ctx,
                        url, citations, source_content
                    )

                    # Sort verified and flagged citations
                    for result in verification_results:
                        if result.get("verified", False):
                            verified_citations.append(result)
                        elif result.get("flagged", False):
                            flagged_citations.append(result)
                else:
                    # Mark as unverified but not flagged
                    for citation in citations:
                        citation["verified"] = False
                        citation["flagged"] = False
            except Exception as e:
                logger.error(f"Error verifying citations for URL {url}: {e}")
                # Mark as unverified but not flagged
                for citation in citations:
                    citation["verified"] = False
                    citation["flagged"] = False

    # Now process each subtopic content to:
    # 1. Apply strikethrough to flagged citations
    # 2. Replace local citation IDs with global IDs
    processed_subtopic_contents = {}

    for subtopic, content in subtopic_contents.items():
        processed_content = content

        # Apply strikethrough to flagged citations
        flagged_sentences_for_subtopic = set()
        for citation in flagged_citations:
            if citation.get("subtopic") == subtopic and citation.get("text"):
                flagged_sentences_for_subtopic.add(citation.get("text"))

        if flagged_sentences_for_subtopic:
            # Split content into sentences
            sentences = re.split(r"(?<=[.!?])\s+", processed_content)
            modified_sentences = []

            for sentence in sentences:
                modified_sentence = sentence

                # Check if this is a flagged sentence
                for flagged_text in flagged_sentences_for_subtopic:
                    if flagged_text in sentence:
                        # Apply strikethrough
                        modified_sentence = f"~~{modified_sentence}~~"
                        break

                modified_sentences.append(modified_sentence)

            # Join sentences back together
            processed_content = " ".join(modified_sentences)

            # Track applied strikethroughs
            citation_fixes = state.get("citation_fixes", [])
            for flagged_text in flagged_sentences_for_subtopic:
                citation_fixes.append(
                    {
                        "section": section_title,
                        "subtopic": subtopic,
                        "reason": "Citation could not be verified",
                        "original_text": flagged_text,
                    }
                )
            state["citation_fixes"] = citation_fixes

        # Now replace all local citation IDs with global IDs using context-aware replacement
        # First handle single citations - standard pattern [n]
        for url, source_data in section_sources.items():
            # Check if this URL has a local ID for this specific subtopic
            original_ids = source_data.get("original_ids", {})
            local_id = original_ids.get(subtopic)

            if local_id is not None and url in global_citation_map:
                global_id = global_citation_map[url]

                # Replace local citation ID with global ID
                pattern = r"\[" + re.escape(str(local_id)) + r"\]"
                processed_content = re.sub(
                    pattern, f"[{global_id}]", processed_content
                )

        # Now handle combined citations like [1, 2] or [1,2]
        # First, extract all citation groups from content
        combined_citation_pattern = r"\[(\d+(?:\s*,\s*\d+)+)\]"
        combined_matches = re.finditer(combined_citation_pattern, processed_content)

        # Process each combined citation group
        for match in combined_matches:
            original_citation = match.group(
                0
            )  # The full citation group e.g. "[1, 2, 3]"
            citation_ids = match.group(1)  # Just the IDs part e.g. "1, 2, 3"

            # Extract individual IDs (handles both [1,2] and [1, 2] formats)
            local_ids = [
                int(id_str.strip()) for id_str in re.split(r"\s*,\s*", citation_ids)
            ]

            # Convert each local ID to its global ID
            global_ids = []
            for local_id in local_ids:
                # Find the URL(s) that had this local ID in this subtopic
                for url, source_data in section_sources.items():
                    original_ids = source_data.get("original_ids", {})
                    if (
                        subtopic in original_ids
                        and original_ids[subtopic] == local_id
                    ) and url in global_citation_map:
                        global_ids.append(str(global_citation_map[url]))

            # If we found global IDs, create the replacement citation
            if global_ids:
                global_citation = f"[{', '.join(global_ids)}]"
                # Replace just this specific citation instance
                processed_content = processed_content.replace(
                    original_citation, global_citation, 1
                )

        processed_subtopic_contents[subtopic] = processed_content

    # Combine subtopic contents into a section draft
    combined_content = ""
    for subtopic, content in processed_subtopic_contents.items():
        # Add subtopic heading
        combined_content += f"\n\n### {subtopic}\n\n"
        combined_content += f"{content}\n\n"

    # Only do smoothing if we have multiple subtopics
    if len(subtopics) > 1:
        # Review and smooth transitions between subtopics
        section_content = await smooth_section_transitions(ctx,
            section_title,
            subtopics,
            combined_content,
            original_query,
            synthesis_model,
        )
    else:
        section_content = combined_content

    # Track token counts
    memory_stats = state.get("memory_stats", {})
    section_tokens = memory_stats.get("section_tokens", {})
    section_tokens[section_title] = total_tokens
    memory_stats["section_tokens"] = section_tokens
    state["memory_stats"] = memory_stats

    # Store content for later use
    section_synthesized_content = state.get("section_synthesized_content", {})
    section_synthesized_content[section_title] = section_content
    state["section_synthesized_content"] = section_synthesized_content

    # Store section sources for later citation correlation
    section_sources_map = state.get("section_sources_map", {})
    section_sources_map[section_title] = section_sources
    state["section_sources_map"] = section_sources_map

    # Store all citations for this section
    section_citations = state.get("section_citations", {})
    section_citations[section_title] = all_section_citations
    state["section_citations"] = section_citations

    # Show section completion status
    await ctx.events.emit(StatusEvent(
        f"Section generated: {section_title}",
        "info",
        False,
    ))

    return {
        "content": section_content,
        "tokens": total_tokens,
        "sources": section_sources,
        "citations": all_section_citations,
        "verified_citations": verified_citations,
        "flagged_citations": flagged_citations,
    }

async def smooth_section_transitions(
    ctx,
    section_title: str,
    subtopics: list[str],
    combined_content: str,
    original_query: str,
    synthesis_model: str,
) -> str:
    """Review and smooth transitions between subtopics in a section"""

    # Create a prompt for smoothing transitions
    smoothing_prompt = {
        "role": "system",
        "content": """You are a post-grad research editor editing a section that combines multiple subtopics.

Review the section content and improve it by:
1. Restructuring subtopic content and makeup to better fit the greater context of the section and full report
2. Ensuring consistent style and tone throughout the section and ensuring consistent use of proper Markdown
3. Maintaining the exact factual content in sentences with numerical citations [#]
4. Removing duplicate subtopic headings
5. Moving sentences or concepts between subsections as appropriate and revising subsection headers to fit the content
6. Removing any meta-commentary, e.g. "Okay, here's the section" or "I wrote the section while considering..."
7. Making the section read as though it were written by one person with a cohesive strategy for assembling the section

DO NOT:
1. Remove, change, or edit ANY in-text citations or applied strikethrough
2. Alter, censor, re-analyze, or edit the factual content in ANY way
3. Add new information or qualifiers not present in the original
4. Decouple the factual content of a sentence from its specific citation
5. Include any introduction, conclusion, main title header, or meta-commentary - please return the section as requested with no other text
6. Combine sentences containing in-text citations and/or strikethrough

It is vitally important that your edits preserve the direct connection between any sentence and its in-text citation and/or applied strikethrough.
You may relocate or lightly edit sentences with in-text citations or strikethrough if appropriate, as long as they maintain these features.""",
    }

    # Create context with the combined subtopics
    smoothing_context = f"# Section to Improve: '{section_title}'\n\n"
    smoothing_context += (
        f"This section is part of a research paper on: '{original_query}'\n\n"
    )

    # Add the research outline for better context
    state = ctx.state.get_state(ctx.conversation_id)
    research_outline = state.get("research_state", {}).get("research_outline", [])
    if research_outline:
        smoothing_context += "## Full Research Outline:\n"
        for topic_item in research_outline:
            topic = topic_item.get("topic", "")
            if topic == section_title:
                smoothing_context += f"**Current Section: {topic}**\n"
            else:
                smoothing_context += f"Section: {topic}\n"

            for st in topic_item.get("subtopics", []):
                smoothing_context += f"  - {st}\n"
        smoothing_context += "\n"

    smoothing_context += "## Subtopics in this section:\n"
    for subtopic in subtopics:
        smoothing_context += f"- '{subtopic}'\n"

    smoothing_context += f"\n## Combined Section Content:\n\n{combined_content}\n\n"
    smoothing_context += "Please improve this section by ensuring smooth transitions between subtopics while preserving all factual content and numerical citations."

    # Create messages for completion
    messages = [smoothing_prompt, {"role": "user", "content": smoothing_context}]

    try:
        # Use synthesis model for smoothing
        response = await ctx.llm.chat_completions(
            synthesis_model,
            messages,
            stream=False,
            temperature=ctx.valves.models.synthesis_temperature
            * 0.7,  # Lower temperature for editing
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            improved_content = response_text(response)
            return improved_content
        else:
            # Return original if synthesis fails
            return combined_content

    except Exception as e:
        logger.error(
            f"Error smoothing transitions for section '{section_title}': {e}"
        )
        # Return original content on error
        return combined_content


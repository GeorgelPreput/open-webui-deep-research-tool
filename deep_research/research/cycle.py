import logging
from typing import Any

from deep_research.budget.tokens import count_tokens
from deep_research.config.constants import MAX_FAILED_RESULTS
from deep_research.core.text import clean_text_formatting
from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import update_token_counts
from deep_research.progress.events import MessageEvent, StatusEvent
from deep_research.research.ranking import check_result_relevance, select_most_relevant_results
from deep_research.web.search import process_search_result, sanitize_query, search_web

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def process_query(
    ctx: RunContext,
    query: str,
    query_embedding: list[float] | None,
    outline_embedding: list[float] | None,
    cycle_feedback: dict[str, Any] | None = None,
    summary_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Process a single search query and get results with quality filtering"""
    await _emit_status(ctx, "info", f"Searching for: {query}", False)

    # Sanitize the query to make it safer for search engines
    sanitized_query = await sanitize_query(ctx, query)

    # Get search results for the query
    search_results = await search_web(ctx, sanitized_query)
    if not search_results:
        if not ctx.valves.events.quiet_chat_mode:
            await _emit_message(ctx, f"*No results found for query: {query}*\n\n")
        else:
            await _emit_status(ctx,
                "warning", f"No results found for query: {query}", False
            )
        return []

    # Always select the most relevant results - this adds similarity scores
    search_results = await select_most_relevant_results(ctx,
        search_results,
        query,
        query_embedding,
        outline_embedding,
        summary_embedding,
    )

    # Process each search result until we have enough successful results
    successful_results: list[dict[str, Any]] = []
    failed_count = 0

    # Get state for access to research outline
    state = ctx.state.get_state(ctx.conversation_id)
    all_topics = state.get("all_topics", [])

    # Track rejected results for logging
    rejected_results = []

    for result in search_results:
        # Stop if we've reached our target of successful results
        if len(successful_results) >= ctx.valves.web.successful_results_per_query:
            break

        # Stop if we've had too many consecutive failures
        if failed_count >= MAX_FAILED_RESULTS:
            if not ctx.valves.events.quiet_chat_mode:
                await _emit_message(ctx,
                    f"*Skipping remaining results for query: {query} after {failed_count} failures*\n\n"
                )
            else:
                await _emit_status(ctx,
                    "warning",
                    f"Skipping remaining results after {failed_count} failures",
                    False,
                )
            break

        try:
            # Process the result
            processed_result = await process_search_result(ctx,
                result,
                query,
                query_embedding,
                outline_embedding,
                summary_embedding,
            )

            # Make sure similarity is preserved from original result
            if "similarity" in result and "similarity" not in processed_result:
                processed_result["similarity"] = result["similarity"]

            # Check if processing was successful (has substantial content and valid URL)
            if (
                processed_result
                and processed_result.get("content")
                and len(processed_result.get("content", "")) > 200
                and processed_result.get("valid", False)
                and processed_result.get("url", "")
            ):
                # Add token count if not already present
                if "tokens" not in processed_result:
                    processed_result["tokens"] = await count_tokens(ctx,
                        processed_result["content"]
                    )

                # Skip results with less than 200 tokens
                if processed_result["tokens"] < 200:
                    failed_count += 1
                    logger.info(
                        f"Skipping result with only {processed_result['tokens']} tokens (less than minimum 200)"
                    )
                    continue

                # Only apply quality filter for results with low similarity
                if (
                    ctx.valves.web.quality_filter_enabled
                    and "similarity" in processed_result
                    and processed_result["similarity"]
                    < ctx.valves.web.quality_similarity_threshold
                ):
                    # Check if result is relevant using quality filter
                    is_relevant = await check_result_relevance(ctx,
                        processed_result,
                        query,
                        all_topics,
                    )

                    if not is_relevant:
                        # Track rejected result
                        rejected_results.append(
                            {
                                "url": processed_result.get("url", ""),
                                "title": processed_result.get("title", ""),
                                "similarity": processed_result.get("similarity", 0),
                                "processed_result": processed_result,
                            }
                        )
                        logger.warning(
                            f"Rejected irrelevant result: {processed_result.get('url', '')}"
                        )
                        continue
                else:
                    # Skip filter for high similarity or when filtering is disabled
                    logger.info(
                        f"Skipping quality filter for result: {processed_result.get('similarity', 0):.3f}"
                    )

                # Add to successful results
                successful_results.append(processed_result)

                # Get the document title for display
                document_title = processed_result["title"]
                if document_title == f"'{query}'" and processed_result["url"]:
                    # Try to get a better title from the URL
                    from urllib.parse import urlparse

                    parsed_url = urlparse(processed_result["url"])
                    path_parts = parsed_url.path.split("/")
                    if path_parts[-1]:
                        file_name = path_parts[-1]
                        # Clean up filename to use as title
                        if file_name.endswith(".pdf"):
                            document_title = (
                                file_name[:-4].replace("-", " ").replace("_", " ")
                            )
                        elif "." in file_name:
                            document_title = (
                                file_name.split(".")[0]
                                .replace("-", " ")
                                .replace("_", " ")
                            )
                        else:
                            document_title = file_name.replace("-", " ").replace(
                                "_", " "
                            )
                    else:
                        # Use domain as title if no useful path
                        document_title = parsed_url.netloc

                # Get token count for displaying
                token_count = processed_result.get("tokens", 0)
                if token_count == 0:
                    token_count = await count_tokens(ctx,
                        processed_result["content"]
                    )

                # Display the result to the user with improved formatting
                if not ctx.valves.events.quiet_chat_mode:
                    if processed_result["url"]:
                        # Show full URL in the result header
                        url = processed_result["url"]

                        # Check if this is a PDF (either by extension or by content type detection)
                        if (
                            url.endswith(".pdf")
                            or "application/pdf" in url
                        ):
                            prefix = "PDF: "
                        else:
                            prefix = "Site: "

                        result_text = (
                            f"#### {prefix}{url}\n**Tokens:** {token_count}\n\n"
                        )
                    else:
                        result_text = (
                            f"#### {document_title} [{token_count} tokens]\n\n"
                        )

                    result_text += f"*Search query: {query}*\n\n"

                    # Format content with short line merging
                    _full_content = processed_result["content"]
                    _max_tokens = ctx.valves.web.max_result_tokens
                    _content_tokens = await count_tokens(ctx, _full_content)
                    if _content_tokens > _max_tokens:
                        _char_ratio = _max_tokens / max(_content_tokens, 1)
                        _char_limit = int(len(_full_content) * _char_ratio)
                        content_to_display = _full_content[:_char_limit]
                    else:
                        content_to_display = _full_content
                    formatted_content = await clean_text_formatting(
                        content_to_display
                    )
                    result_text += f"{formatted_content}...\n\n"

                    # Add repeat indicator if this is a repeated URL
                    repeat_count = processed_result.get("repeat_count", 0)
                    if repeat_count > 1:
                        result_text += f"*Note: This URL has been processed {repeat_count} times*\n\n"

                    await _emit_message(ctx, result_text)

                # Reset failed count on success
                failed_count = 0
            else:
                # Count as a failure
                failed_count += 1
                logger.warning(
                    f"Failed to get substantial content from result {len(successful_results) + failed_count} for query: {query}"
                )

        except Exception as e:
            # Count as a failure
            failed_count += 1
            logger.error(f"Error processing result for query '{query}': {e}")
            if not ctx.valves.events.quiet_chat_mode:
                await _emit_message(ctx,
                    f"*Error processing a result for query: {query}*\n\n"
                )
            else:
                await _emit_status(ctx,
                    "warning",
                    f"Error processing a result for query: {query}",
                    False,
                )

    # If we didn't get any successful results but had rejected ones, use the top rejected result
    if not successful_results and rejected_results:
        # Sort rejected results by similarity (descending)
        sorted_rejected = sorted(
            rejected_results, key=lambda x: x.get("similarity", 0), reverse=True
        )
        top_rejected = sorted_rejected[0]

        logger.info(
            f"Using top rejected result as fallback: {top_rejected.get('url', '')}"
        )

        # Get the processed result directly from the rejection record
        if "processed_result" in top_rejected:
            processed_result = top_rejected["processed_result"]
            successful_results.append(processed_result)

            if not ctx.valves.events.quiet_chat_mode:
                # Display the result with a note that it might not be fully relevant
                document_title = processed_result.get(
                    "title", f"Result for '{query}'"
                )
                token_count = processed_result.get(
                    "tokens", 0
                ) or await count_tokens(ctx, processed_result["content"])
                url = processed_result.get("url", "")

                result_text = f"#### {document_title} [{token_count} tokens]\n\n"
                if url:
                    result_text = f"#### {'PDF: ' if url.endswith('.pdf') else 'Site: '}{url}\n**Tokens:** {token_count}\n\n"

                result_text += f"*Search query: {query}*\n\n"
                result_text += "*Note: This result was initially filtered but is used as a fallback.*\n\n"

                # Format content
                content_to_display = processed_result["content"][
                    : ctx.valves.web.max_result_tokens
                ]
                formatted_content = await clean_text_formatting(
                    content_to_display
                )
                result_text += f"{formatted_content}...\n\n"

                await _emit_message(ctx, result_text)

    # If we still didn't get any successful results, log this
    if not successful_results:
        logger.warning(f"No valid results obtained for query: {query}")
        if not ctx.valves.events.quiet_chat_mode:
            await _emit_message(ctx,
                f"*No valid results found for query: {query}*\n\n"
            )
        else:
            await _emit_status(ctx,
                "warning", f"No valid results found for query: {query}", False
            )

    # Update token counts with new results
    await update_token_counts(ctx, successful_results)

    return successful_results


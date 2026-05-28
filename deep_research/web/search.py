import json
import logging
from datetime import datetime
from typing import Any

from deep_research.core.types import RunContext
from deep_research.persistence.kb import persist_selected_source
from deep_research.progress.events import StatusEvent
from deep_research.semantics.similarity import scale_token_limit_by_relevance

logger = logging.getLogger("deep_research.web.search")


async def sanitize_query(ctx: RunContext, query: str) -> str:
    """Sanitize search query by removing quotes and handling special characters"""
    # Remove quotes that might cause problems with search engines
    sanitized = query.replace('"', " ").replace('"', " ").replace('"', " ")

    # Replace multiple spaces with a single space
    sanitized = " ".join(sanitized.split())

    # Ensure the query isn't too long
    if len(sanitized) > 250:
        sanitized = sanitized[:250]

    logger.info(f"Sanitized query: '{query}' -> '{sanitized}'")
    return sanitized



async def try_owui_search(ctx: RunContext, query: str, total_results: int) -> tuple[list[dict[str, Any]], str | None]:
    """Call Open WebUI's integrated web search via REST.

    Returns (results, failure_reason), where None means the call completed
    and a string means the OWUI search call failed.
    """
    try:
        data = await ctx.client.web_search([query])
        if not data:
            return [], None

        results: list[dict[str, Any]] = []

        items = data.get("items") or data.get("results") or []
        for it in items[:total_results]:
            if not isinstance(it, dict):
                continue
            url = it.get("url") or it.get("link") or ""
            snippet = it.get("snippet") or it.get("content") or it.get("description") or ""
            title = it.get("title") or f"'{query}'"
            results.append({"title": title, "url": url, "snippet": snippet})

        if not results:
            filenames = data.get("filenames") or []
            docs = data.get("docs") or []
            if docs:
                for i, doc in enumerate(docs[:total_results]):
                    url = filenames[i] if i < len(filenames) else ""
                    if isinstance(doc, dict):
                        snippet = doc.get("content") or ""
                        meta = doc.get("metadata") or {}
                        title = meta.get("title") or meta.get("name") or f"'{query}'"
                        if not url:
                            url = meta.get("source") or meta.get("url") or ""
                    else:
                        snippet = str(doc) if doc is not None else ""
                        title = f"'{query}'"
                    results.append({"title": title, "url": url, "snippet": snippet})

        return results, None

    except TimeoutError:
        logger.error(f"OpenWebUI search timed out for query: {query}")
        return [], "timeout"
    except Exception as e:
        logger.error(f"Error in try_owui_search: {str(e)}")
        return [], f"error: {e}"



async def search_web(ctx: RunContext, query: str) -> list[dict[str, Any]]:
    """Perform an Open WebUI integrated web search.

    Returns normalized {title, url, snippet} results, or one synthetic
    "No results" placeholder so downstream processing degrades gracefully.
    """
    logger.debug(f"Starting web search for query: {query}")

    state = ctx.state.get_state(ctx.conversation_id)
    url_selected_count = state.get("url_selected_count", {})

    repeat_count = 0
    for _url, count in url_selected_count.items():
        if count >= ctx.valves.web.repeats_before_expansion:
            repeat_count += 1

    base_results = ctx.valves.web.search_results_per_query
    additional_results = min(repeat_count, ctx.valves.web.extra_results_per_query)
    total_results = (
        base_results + ctx.valves.web.extra_results_per_query + additional_results
    )

    logger.debug(
        f"Requesting {total_results} search results (added {additional_results} due to repeats)"
    )

    results, failure_reason = await try_owui_search(query, total_results)

    if results:
        logger.debug(
            f"Search successful, found {len(results)} results for: {query}"
        )
        return results

    if failure_reason is not None:
        logger.warning(
            f"OpenWebUI search failed ({failure_reason}) for query: {query}; returning synthetic placeholder"
        )
    else:
        logger.info(
            f"OpenWebUI search returned no results for query: {query}; returning synthetic placeholder"
        )
    return [
        {
            "title": f"No results for '{query}'",
            "url": "",
            "snippet": f"No search results were found for the query: {query}",
        }
    ]



async def process_search_result(ctx: RunContext, result: dict[str, Any], query: str, query_embedding: list[float] | None, outline_embedding: list[float] | None, summary_embedding: list[float] | None = None) -> dict[str, Any]:
    """Process a search result to extract and compress content with token limiting"""
    title = result.get("title", "")
    url = result.get("url", "")
    snippet = result.get("snippet", "")

    # Require a URL for all results
    if not url:
        return {
            "title": title or f"Result for '{query}'",
            "url": "",
            "content": "This result has no associated URL and cannot be processed.",
            "query": query,
            "valid": False,
        }

    await ctx.events.emit(StatusEvent(description=f"Processing result: {title[:50]}...", level="info", done=False))

    repeat_count = 0
    tokens = 0
    from deep_research.budget.tokens import count_tokens
    from deep_research.compression.repeated import handle_repeated_content
    from deep_research.web.fetch import fetch_content
    try:
        # Get state
        state = ctx.state.get_state(ctx.conversation_id)
        url_selected_count = state.get("url_selected_count", {})
        url_token_counts = state.get("url_token_counts", {})
        master_source_table = state.get("master_source_table", {})

        # Check if this is a repeated URL
        repeat_count = url_selected_count.get(url, 0)

        # If the snippet is empty or short but we have a URL, try to fetch content
        if (not snippet or len(snippet) < 200) and url:
            await ctx.events.emit(StatusEvent(description=f"Fetching content from URL: {url}...", level="info", done=False))
            content = await fetch_content(url)

            if content and len(content) > 200:
                snippet = content
                logger.debug(
                    f"Successfully fetched content from URL: {url} ({len(content)} chars)"
                )
            else:
                logger.warning(f"Failed to fetch useful content from URL: {url}")

        # If we still don't have useful content, mark as invalid
        if not snippet or len(snippet) < 200:
            return {
                "title": title or f"Result for '{query}'",
                "url": url,
                "content": snippet
                or "No substantial content available for this result.",
                "query": query,
                "valid": False,
            }

        # For repeated URLs, apply special sliding window treatment
        if repeat_count > 0:
            snippet = await handle_repeated_content(
                snippet, url, query_embedding, repeat_count
            )

        # Calculate tokens in the content
        content_tokens = await count_tokens(ctx, snippet)

        # Get user preferences for PDV
        state = ctx.state.get_state(ctx.conversation_id)
        user_preferences = state.get("user_preferences", {})
        pdv = user_preferences.get("pdv")

        # Apply token limit if needed with adaptive scaling based on relevance
        max_tokens = await scale_token_limit_by_relevance(
            result, query_embedding, pdv
        )

        if content_tokens > max_tokens:
            # Process the content with token limiting using simple truncation with some padding
            try:
                await ctx.events.emit(StatusEvent(description="Truncating content to token limit...", level="info", done=False))

                # Calculate character position based on token limit
                char_ratio = max_tokens / content_tokens
                char_limit = int(len(snippet) * char_ratio)

                # Pad the limit to ensure we have complete sentences
                padded_limit = min(len(snippet), int(char_limit * 1.1))

                # Truncate content
                truncated_content = snippet[:padded_limit]

                # Find a good sentence break point
                last_period = truncated_content.rfind(".")
                if (
                    last_period > char_limit * 0.9
                ):  # Only use period if it's near the target limit
                    truncated_content = truncated_content[: last_period + 1]

                # If we got useful truncated content, use it
                if truncated_content and len(truncated_content) > 100:
                    # Mark URL as actually selected (shown to user)
                    url_selected_count[url] = url_selected_count.get(url, 0) + 1
                    ctx.state.update_state(ctx.conversation_id, "url_selected_count", url_selected_count)

                    # Store total tokens for this URL if not already done
                    if url not in url_token_counts:
                        url_token_counts[url] = content_tokens
                        ctx.state.update_state(ctx.conversation_id, "url_token_counts", url_token_counts)

                    # Make sure this URL is in the master source table
                    if url not in master_source_table:
                        # (unchanged source table code)
                        source_type = "web"
                        if url.endswith(".pdf") or False:
                            source_type = "pdf"

                        # Try to get or create a good title
                        if not title or title == f"Result for '{query}'":
                            from urllib.parse import urlparse

                            parsed_url = urlparse(url)
                            if source_type == "pdf":
                                file_name = parsed_url.path.split("/")[-1]
                                title = (
                                    file_name.replace(".pdf", "")
                                    .replace("-", " ")
                                    .replace("_", " ")
                                )
                            else:
                                title = parsed_url.netloc

                        source_id = f"S{len(master_source_table) + 1}"
                        master_source_table[url] = {
                            "id": source_id,
                            "title": title,
                            "content_preview": truncated_content[:500],
                            "source_type": source_type,
                            "accessed_date": datetime.now().strftime("%Y-%m-%d"),
                            "cited_in_sections": set(),
                        }
                        ctx.state.update_state(ctx.conversation_id, "master_source_table", master_source_table
                        )

                        # Count tokens in truncated content
                        tokens = await count_tokens(ctx, truncated_content)

                        # Add timestamp to the result
                        result["timestamp"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        # Write-through to research KB with the FULL snippet
                        # (not the truncated_content). Idempotent on hash.
                        if isinstance(snippet, str) and snippet.strip():
                            try:
                                await persist_selected_source(
                                    url=url,
                                    full_text=snippet,
                                    title=title,
                                    source_type=source_type,
                                    archived=master_source_table.get(url, {}).get(
                                        "archived", False
                                    ),
                                    search_query=query,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"KB persistence failed for {url}: {e}"
                                )

                    return {
                        "title": title,
                        "url": url,
                        "content": truncated_content,
                        "query": query,
                        "repeat_count": repeat_count,
                        "tokens": tokens,
                        "valid": True,
                    }
            except Exception as e:
                logger.error(f"Error in token-based truncation: {e}")
                # If truncation fails, we'll fall back to using original content with hard limit

        # If we haven't returned yet, use the original content with token limiting
        # Mark URL as actually selected (shown to user)
        url_selected_count[url] = url_selected_count.get(url, 0) + 1
        ctx.state.update_state(ctx.conversation_id, "url_selected_count", url_selected_count)

        # Store total tokens for this URL if not already done
        if url not in url_token_counts:
            url_token_counts[url] = content_tokens
            ctx.state.update_state(ctx.conversation_id, "url_token_counts", url_token_counts)

        # Make sure this URL is in the master source table
        if url not in master_source_table:
            source_type = "web"
            if url.endswith(".pdf") or False:
                source_type = "pdf"

            # Try to get or create a good title
            if not title or title == f"Result for '{query}'":
                from urllib.parse import urlparse

                parsed_url = urlparse(url)
                if source_type == "pdf":
                    file_name = parsed_url.path.split("/")[-1]
                    title = (
                        file_name.replace(".pdf", "")
                        .replace("-", " ")
                        .replace("_", " ")
                    )
                else:
                    title = parsed_url.netloc

            source_id = f"S{len(master_source_table) + 1}"
            master_source_table[url] = {
                "id": source_id,
                "title": title,
                "content_preview": snippet[:500],
                "source_type": source_type,
                "accessed_date": datetime.now().strftime("%Y-%m-%d"),
                "cited_in_sections": set(),
            }
            ctx.state.update_state(ctx.conversation_id, "master_source_table", master_source_table)

        # Write-through to research KB with the FULL snippet. Idempotent
        # on hash. Cheap when fetch_content already persisted upstream.
        if isinstance(snippet, str) and snippet.strip():
            try:
                await persist_selected_source(
                    url=url,
                    full_text=snippet,
                    title=title,
                    source_type=master_source_table.get(url, {}).get(
                        "source_type", "web"
                    ),
                    archived=master_source_table.get(url, {}).get(
                        "archived", False
                    ),
                    search_query=query,
                )
            except Exception as e:
                logger.warning(f"KB persistence failed for {url}: {e}")

        # If over token limit, truncate
        if content_tokens > max_tokens:
            # Estimate character position based on token limit
            char_ratio = max_tokens / content_tokens
            char_limit = int(len(snippet) * char_ratio)
            limited_content = snippet[:char_limit]
            # Actually count tokens rather than assuming max_tokens
            tokens = await count_tokens(ctx, limited_content)
        else:
            limited_content = snippet
            tokens = content_tokens

            # Add timestamp to the result
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "title": title,
            "url": url,
            "content": limited_content,
            "query": query,
            "repeat_count": repeat_count,
            "tokens": tokens,
            "valid": True,
        }

    except Exception as e:
        logger.error(f"Unhandled error in process_search_result: {e}")
        # Return a failure result
        error_msg = f"Error processing search result: {str(e)}\n\nOriginal snippet: {snippet[:1000] if snippet else 'No content available'}"
        tokens = await count_tokens(ctx, error_msg)

        return {
            "title": title or f"Error processing result for '{query}'",
            "url": url,
            "content": error_msg,
            "query": query,
            "repeat_count": repeat_count,
            "tokens": tokens,
            "valid": False,
        }



async def improved_query_generation(ctx: RunContext, user_message, priority_topics, search_context):
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
{"queries": [
  "query": "search query 1", "topic": "related research topic",
  "query": "search query 2", "topic": "related research topic",
  "query": "search query 3", "topic": "related research topic",
  "query": "search query 4", "topic": "related research topic"
]}""",
    }

    message = {
        "role": "user",
        "content": f"""Original query: "{user_message}"\n\nResearch context: "{search_context}"\n\nGenerate 4 effective search queries to gather information for the priority research topics.""",
    }

    # Generate the queries first, without any embedding operations
    try:
        response = await ctx.client.chat_completions(model=
            ctx.valves.models.research_model, messages=[query_prompt, message],
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

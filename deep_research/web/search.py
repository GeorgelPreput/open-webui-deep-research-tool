import logging
from datetime import datetime
from typing import Any

import httpx

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

    except (TimeoutError, httpx.TimeoutException):
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

    results, failure_reason = await try_owui_search(ctx, query, total_results)

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



def _register_source(
    ctx: RunContext,
    *,
    url: str,
    title: str,
    query: str,
    content_preview: str,
    master_source_table: dict[str, Any],
    url_selected_count: dict[str, int],
    url_token_counts: dict[str, int],
    content_tokens: int,
) -> tuple[str, str, bool]:
    """Bump per-URL counters and register the URL in master_source_table on first sight.

    Returns (resolved_title, source_type, newly_registered).
    """
    url_selected_count[url] = url_selected_count.get(url, 0) + 1
    ctx.state.update_state(ctx.conversation_id, "url_selected_count", url_selected_count)

    if url not in url_token_counts:
        url_token_counts[url] = content_tokens
        ctx.state.update_state(ctx.conversation_id, "url_token_counts", url_token_counts)

    source_type = "pdf" if url.endswith(".pdf") else "web"
    if not title or title == f"Result for '{query}'":
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        if source_type == "pdf":
            file_name = parsed_url.path.split("/")[-1]
            title = (
                file_name.replace(".pdf", "").replace("-", " ").replace("_", " ")
            )
        else:
            title = parsed_url.netloc

    newly_registered = url not in master_source_table
    if newly_registered:
        source_id = f"S{len(master_source_table) + 1}"
        master_source_table[url] = {
            "id": source_id,
            "title": title,
            "content_preview": content_preview[:500],
            "source_type": source_type,
            "accessed_date": datetime.now().strftime("%Y-%m-%d"),
            "cited_in_sections": set(),
        }
        ctx.state.update_state(
            ctx.conversation_id, "master_source_table", master_source_table
        )

    return title, source_type, newly_registered


async def process_search_result(ctx: RunContext, result: dict[str, Any], query: str, query_embedding: list[float] | None, outline_embedding: list[float] | None, summary_embedding: list[float] | None = None) -> dict[str, Any]:
    """Process a search result to extract and compress content with token limiting"""
    title = result.get("title", "")
    url = result.get("url", "")
    snippet = result.get("snippet", "")

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
        state = ctx.state.get_state(ctx.conversation_id)
        url_selected_count = state.get("url_selected_count", {})
        url_token_counts = state.get("url_token_counts", {})
        master_source_table = state.get("master_source_table", {})

        repeat_count = url_selected_count.get(url, 0)

        if (not snippet or len(snippet) < 200) and url:
            await ctx.events.emit(StatusEvent(description=f"Fetching content from URL: {url}...", level="info", done=False))
            content = await fetch_content(ctx, url)
            if content and len(content) > 200:
                snippet = content
                logger.debug(
                    f"Successfully fetched content from URL: {url} ({len(content)} chars)"
                )
            else:
                logger.warning(f"Failed to fetch useful content from URL: {url}")

        if not snippet or len(snippet) < 200:
            return {
                "title": title or f"Result for '{query}'",
                "url": url,
                "content": snippet
                or "No substantial content available for this result.",
                "query": query,
                "valid": False,
            }

        if repeat_count > 0:
            snippet = await handle_repeated_content(
                ctx, snippet, url, query_embedding, repeat_count
            )

        content_tokens = await count_tokens(ctx, snippet)

        state = ctx.state.get_state(ctx.conversation_id)
        user_preferences = state.get("user_preferences", {})
        pdv = user_preferences.get("pdv")

        max_tokens = await scale_token_limit_by_relevance(
            ctx, result, query_embedding, pdv
        )

        source_registered = False
        if content_tokens > max_tokens:
            try:
                await ctx.events.emit(StatusEvent(description="Truncating content to token limit...", level="info", done=False))

                char_ratio = max_tokens / content_tokens
                char_limit = int(len(snippet) * char_ratio)
                padded_limit = min(len(snippet), int(char_limit * 1.1))
                truncated_content = snippet[:padded_limit]

                last_period = truncated_content.rfind(".")
                if last_period > char_limit * 0.9:
                    truncated_content = truncated_content[: last_period + 1]

                if truncated_content and len(truncated_content) > 100:
                    title, source_type, newly_registered = _register_source(
                        ctx,
                        url=url,
                        title=title,
                        query=query,
                        content_preview=truncated_content,
                        master_source_table=master_source_table,
                        url_selected_count=url_selected_count,
                        url_token_counts=url_token_counts,
                        content_tokens=content_tokens,
                    )
                    source_registered = True

                    if newly_registered:
                        # Truncation branch performs these only on first registration,
                        # matching the original semantics. KB write-through uses the
                        # FULL snippet (not the truncated copy); idempotent on hash.
                        tokens = await count_tokens(ctx, truncated_content)
                        result["timestamp"] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        if isinstance(snippet, str) and snippet.strip():
                            try:
                                await persist_selected_source(
                                    ctx=ctx,
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
                # Fall through to non-truncation path with hard limit

        # Skip if the truncation branch already registered this URL (it may have
        # fallen through here via its except handler), to avoid double-counting.
        if not source_registered:
            title, _source_type, _newly = _register_source(
                ctx,
                url=url,
                title=title,
                query=query,
                content_preview=snippet,
                master_source_table=master_source_table,
                url_selected_count=url_selected_count,
                url_token_counts=url_token_counts,
                content_tokens=content_tokens,
            )

        if isinstance(snippet, str) and snippet.strip():
            try:
                await persist_selected_source(
                    ctx=ctx,
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

        if content_tokens > max_tokens:
            char_ratio = max_tokens / content_tokens
            char_limit = int(len(snippet) * char_ratio)
            limited_content = snippet[:char_limit]
            tokens = await count_tokens(ctx, limited_content)
        else:
            limited_content = snippet
            tokens = content_tokens
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




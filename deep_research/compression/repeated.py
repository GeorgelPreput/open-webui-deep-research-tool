import logging

from sklearn.metrics.pairwise import cosine_similarity

from deep_research.budget.tokens import count_tokens
from deep_research.budget.windows import extract_token_window
from deep_research.config.constants import REPEAT_WINDOW_FACTOR
from deep_research.core.text import chunk_text
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.compression.repeated")


async def handle_repeated_content(
    ctx: RunContext,
    content: str,
    url: str,
    query_embedding: list[float] | None,
    repeat_count: int,
) -> str:
    state = ctx.state.get_state(ctx.conversation_id)
    url_selected_count = state.get("url_selected_count", {})
    url_token_counts = state.get("url_token_counts", {})

    selected_count = url_selected_count.get(url, 0)

    if selected_count < 1:
        total_tokens = await count_tokens(ctx, content)
        url_token_counts[url] = total_tokens
        ctx.state.update_state(ctx.conversation_id, "url_token_counts", url_token_counts)
        return content

    total_tokens = url_token_counts.get(url, 0)
    if total_tokens == 0:
        total_tokens = await count_tokens(ctx, content)
        url_token_counts[url] = total_tokens
        ctx.state.update_state(ctx.conversation_id, "url_token_counts", url_token_counts)

    max_tokens = ctx.valves.web.max_result_tokens
    window_factor = REPEAT_WINDOW_FACTOR

    if total_tokens > max_tokens:
        window_start = int((repeat_count - 1) * window_factor * max_tokens)

        if window_start >= total_tokens:
            cycles_completed = window_start // total_tokens

            shrink_factor = 0.7**cycles_completed

            window_size = int(max_tokens * shrink_factor)
            window_size = max(200, window_size)

            window_start = window_start % total_tokens

            logger.info(
                f"Repeat URL {url} (count: {selected_count}): applying shrinkage after full cycle. "
                f"Factor: {shrink_factor:.2f}, window size: {window_size} tokens"
            )
        else:
            window_size = max_tokens
            logger.info(
                f"Repeat URL {url} (count: {selected_count}): sliding window, "
                f"starting at token {window_start}, window size {window_size}"
            )

        window_content = await extract_token_window(
            ctx, content, window_start, window_size
        )

        return window_content
    else:
        logger.info(
            f"Repeat URL {url} (count: {selected_count}): applying compression/centering for content already within token limit"
        )

        # Re-centering is purely relevance-driven; with no query vector there is
        # nothing to center on, so return the content unchanged.
        if query_embedding is None:
            return content

        content_embedding = await get_embedding(ctx, content[:2000])
        if not content_embedding:
            return content

        try:
            chunks = chunk_text(content, ctx.valves.compression.chunk_level)
            if len(chunks) <= 3:
                return content

            chunk_embeddings = []
            relevance_scores = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = await get_embedding(ctx, chunk[:2000])
                if chunk_embedding:
                    chunk_embeddings.append(chunk_embedding)
                    relevance = cosine_similarity(
                        [chunk_embedding], [query_embedding]
                    )[0][0]
                    relevance_scores.append((i, relevance))

            relevance_scores.sort(key=lambda x: x[1], reverse=True)

            if relevance_scores:
                most_relevant_idx = relevance_scores[0][0]

                start_idx = max(0, most_relevant_idx - len(chunks) // 4)
                end_idx = min(len(chunks), most_relevant_idx + len(chunks) // 4 + 1)

                recentered_content = "\n".join(chunks[start_idx:end_idx])
                return recentered_content

        except Exception as e:
            logger.error(f"Error re-centering window: {e}")

        return content

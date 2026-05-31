import logging
from typing import Any

import numpy as np

from deep_research.core.text import stable_text_key
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.semantics.similarity")


def _safe_normalize(vec) -> np.ndarray | None:
    """Return the unit vector for ``vec``, or None if it is empty / zero / non-finite.

    Guards against the degenerate cases (zero vector, NaN/inf) that would
    otherwise make ``np.linalg.norm`` divide by zero and poison every
    similarity score with NaN.
    """
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return None
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return arr / norm


async def update_topic_usage_counts(ctx: RunContext, used_topics):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    topic_usage_counts = conv_state.get("topic_usage_counts", {})

    for topic in used_topics:
        topic_usage_counts[topic] = topic_usage_counts.get(topic, 0) + 1

    ctx.state.update_state(
        ctx.conversation_id, "topic_usage_counts", topic_usage_counts
    )


async def calculate_query_similarity(
    ctx: RunContext,
    content_embedding: list[float],
    query_embedding: list[float],
    outline_embedding: list[float] | None = None,
    summary_embedding: list[float] | None = None,
) -> float:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    similarity_cache = conv_state.get("similarity_cache", {})

    content_key = stable_text_key(str(np.array(content_embedding).round(2)))
    query_key = stable_text_key(str(np.array(query_embedding).round(2)))

    combined_key = f"combined_{content_key}_{query_key}"
    if outline_embedding:
        outline_key = stable_text_key(str(np.array(outline_embedding).round(2)))
        combined_key += f"_{outline_key}"
    if summary_embedding:
        summary_key = stable_text_key(str(np.array(summary_embedding).round(2)))
        combined_key += f"_{summary_key}"

    if combined_key in similarity_cache:
        return similarity_cache[combined_key]

    c_emb = _safe_normalize(content_embedding)
    q_emb = _safe_normalize(query_embedding)

    base_key = f"{content_key}_{query_key}"
    if base_key in similarity_cache:
        query_sim = similarity_cache[base_key]
    elif c_emb is None or q_emb is None:
        query_sim = 0.0
        similarity_cache[base_key] = query_sim
    else:
        query_sim = np.dot(c_emb, q_emb)
        similarity_cache[base_key] = query_sim

    query_weight = 0.4
    outline_weight = 0.3
    summary_weight = 0.3

    outline_sim = 0.0
    if outline_embedding is not None:
        outline_key = stable_text_key(str(np.array(outline_embedding).round(2)))
        outline_cache_key = f"{content_key}_{outline_key}"

        if outline_cache_key in similarity_cache:
            outline_sim = similarity_cache[outline_cache_key]
        else:
            o_emb = _safe_normalize(outline_embedding)
            outline_sim = 0.0 if c_emb is None or o_emb is None else np.dot(c_emb, o_emb)
            similarity_cache[outline_cache_key] = outline_sim
    else:
        query_weight += outline_weight
        outline_weight = 0.0

    summary_sim = 0.0
    if summary_embedding is not None:
        summary_key = stable_text_key(str(np.array(summary_embedding).round(2)))
        summary_cache_key = f"{content_key}_{summary_key}"

        if summary_cache_key in similarity_cache:
            summary_sim = similarity_cache[summary_cache_key]
        else:
            s_emb = _safe_normalize(summary_embedding)
            summary_sim = 0.0 if c_emb is None or s_emb is None else np.dot(c_emb, s_emb)
            similarity_cache[summary_cache_key] = summary_sim
    else:
        query_weight += summary_weight
        summary_weight = 0.0

    combined_sim = (
        (query_sim * query_weight)
        + (outline_sim * outline_weight)
        + (summary_sim * summary_weight)
    )

    similarity_cache[combined_key] = combined_sim

    if len(similarity_cache) > 1000:
        keys_to_remove = list(similarity_cache.keys())[:200]
        for k in keys_to_remove:
            del similarity_cache[k]

    ctx.state.update_state(
        ctx.conversation_id, "similarity_cache", similarity_cache
    )

    return combined_sim


async def scale_token_limit_by_relevance(
    ctx: RunContext,
    result: dict[str, Any],
    query_embedding: list[float] | None,
    pdv: list[float] | None = None,
) -> int:
    base_token_limit = ctx.valves.web.max_result_tokens

    if "similarity" not in result:
        return base_token_limit

    similarity = result.get("similarity", 0.5)

    pdv_alignment = 0.5
    if pdv is not None:
        try:
            content = result.get("content", "")
            content_embedding = await get_embedding(ctx, content[:2000])

            if content_embedding and len(content_embedding) == len(pdv):
                alignment = np.dot(content_embedding, pdv)
                pdv_alignment = (alignment + 1) / 2
        except Exception as e:
            logger.error(f"Error calculating PDV alignment: {e}")

    combined_relevance = (similarity * 0.7) + (pdv_alignment * 0.3)

    scaling_factor = 0.5 + (combined_relevance * 1.0)
    scaled_limit = int(base_token_limit * scaling_factor)

    min_limit = int(base_token_limit * 0.5)
    max_limit = int(base_token_limit * 1.5)

    scaled_limit = max(min_limit, min(max_limit, scaled_limit))

    logger.info(
        f"Scaled token limit for result: {scaled_limit} tokens "
        f"(similarity: {similarity:.2f}, scaling factor: {scaling_factor:.2f})"
    )

    return scaled_limit

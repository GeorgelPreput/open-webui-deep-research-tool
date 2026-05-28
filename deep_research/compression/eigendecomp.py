import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from deep_research.budget.tokens import count_tokens
from deep_research.compression.local_similarity import (
    compress_content_with_local_similarity,
)
from deep_research.config.constants import LOCAL_INFLUENCE_RADIUS, MAX_REPORT_FIT_PASSES
from deep_research.core.text import chunk_text
from deep_research.core.types import RunContext
from deep_research.semantics.eigendecomposition import (
    apply_semantic_transformation,
    compute_semantic_eigendecomposition,
    create_semantic_transformation,
)
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.compression.eigendecomp")


async def compress_content_with_eigendecomposition(
    ctx: RunContext,
    content: str,
    query_embedding: list[float] | None,
    summary_embedding: list[float] | None = None,
    ratio: float | None = None,
    max_tokens: int | None = None,
    _retry_depth: int = 0,
) -> str:
    if len(content) < 200:
        return content

    if max_tokens:
        content_tokens = await count_tokens(ctx, content)
        if content_tokens <= max_tokens:
            return content

        if not ratio:
            ratio = max_tokens / content_tokens

    chunks = chunk_text(content, ctx.valves.compression.chunk_level)

    if len(chunks) <= 2:
        return content

    chunk_embeddings = []
    for chunk in chunks:
        embedding = await get_embedding(ctx, chunk)
        if embedding:
            chunk_embeddings.append(embedding)

    if len(chunk_embeddings) <= 2:
        return content

    if ratio is None:
        compress_ratios = {
            1: 0.9,
            2: 0.8,
            3: 0.7,
            4: 0.6,
            5: 0.5,
            6: 0.4,
            7: 0.3,
            8: 0.2,
            9: 0.15,
            10: 0.1,
        }
        level = ctx.valves.compression.compression_level
        ratio = compress_ratios.get(level, 0.5)

    n_chunks = len(chunks)
    n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

    if n_keep >= n_chunks:
        n_keep = max(1, n_chunks - 1)

    try:
        eigendecomposition = await compute_semantic_eigendecomposition(
            ctx, chunks, chunk_embeddings
        )

        if eigendecomposition:
            importance_scores = []

            directions = {}
            if query_embedding:
                directions["query"] = query_embedding
            if summary_embedding:
                directions["summary"] = summary_embedding

            state = ctx.state.get_state(ctx.conversation_id)
            user_preferences = state.get(
                "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
            )
            if user_preferences["pdv"] is not None:
                directions["pdv"] = user_preferences["pdv"]

            transformation = await create_semantic_transformation(
                ctx,
                eigendecomposition,
                pdv=(
                    user_preferences["pdv"]
                    if user_preferences["impact"] > 0.1
                    else None
                ),
            )

            projected_chunks = eigendecomposition["projected_embeddings"]

            local_coherence = []
            local_radius = LOCAL_INFLUENCE_RADIUS

            for i in range(len(projected_chunks)):
                local_sim = 0.0
                count = 0

                for j in range(
                    max(0, i - local_radius),
                    min(len(projected_chunks), i + local_radius + 1),
                ):
                    if i == j:
                        continue

                    sim = 0.0
                    for k in range(eigendecomposition["n_components"]):
                        weight = eigendecomposition["explained_variance"][k]
                        dim_sim = 1.0 - abs(
                            projected_chunks[i][k] - projected_chunks[j][k]
                        )
                        sim += weight * dim_sim

                    local_sim += sim
                    count += 1

                if count > 0:
                    local_sim /= count
                local_coherence.append(local_sim)

            if query_embedding:
                try:
                    transformed_query = query_embedding
                    if transformation:
                        _tq = await apply_semantic_transformation(
                            ctx, query_embedding, transformation
                        )
                        if _tq:
                            transformed_query = _tq
                            query_embedding = transformed_query

                    query_relevance = []
                    for chunk_embedding in chunk_embeddings:
                        if chunk_embedding:
                            similarity = cosine_similarity(
                                [chunk_embedding], [transformed_query]
                            )[0][0]
                            query_relevance.append(similarity)
                        else:
                            query_relevance.append(
                                0.5
                            )
                except Exception as e:
                    logger.warning(f"Error calculating query relevance: {e}")
                    query_relevance = [0.5] * len(projected_chunks)
            else:
                query_relevance = [0.5] * len(projected_chunks)

            for i in range(len(chunks)):
                if i >= len(local_coherence) or i >= len(query_relevance):
                    continue

                coherence_weight = 0.4
                relevance_weight = 0.6

                if (
                    user_preferences["pdv"] is not None
                    and user_preferences["impact"] > 0.1
                ):
                    pdv_weight = min(0.3, user_preferences["impact"])
                    coherence_weight *= 1.0 - pdv_weight
                    relevance_weight *= 1.0 - pdv_weight

                    if i < len(chunk_embeddings):
                        try:
                            chunk_embed = chunk_embeddings[i]
                            pdv_alignment = np.dot(
                                chunk_embed, user_preferences["pdv"]
                            )
                            pdv_alignment = (pdv_alignment + 1) / 2
                        except Exception as e:
                            logger.warning(f"Error calculating PDV alignment: {e}")
                            pdv_alignment = 0.5
                    else:
                        pdv_alignment = 0.5

                    final_score = (
                        (local_coherence[i] * coherence_weight)
                        + (query_relevance[i] * relevance_weight)
                        + (pdv_alignment * pdv_weight)
                    )
                else:
                    final_score = (local_coherence[i] * coherence_weight) + (
                        query_relevance[i] * relevance_weight
                    )

                importance_scores.append((i, final_score))

            importance_scores.sort(key=lambda x: x[1], reverse=True)

            selected_indices = [x[0] for x in importance_scores[:n_keep]]

            selected_indices.sort()

            selected_chunks = [
                chunks[i] for i in selected_indices if i < len(chunks)
            ]

            chunk_level = ctx.valves.compression.chunk_level
            if chunk_level == 1:
                compressed_content = " ".join(selected_chunks)
            elif chunk_level == 2:
                processed_sentences = []
                for sentence in selected_chunks:
                    if not sentence.endswith((".", "!", "?", ":", ";")):
                        sentence += "."
                    processed_sentences.append(sentence)
                compressed_content = " ".join(processed_sentences)
            else:
                compressed_content = "\n".join(selected_chunks)

            if max_tokens:
                final_tokens = await count_tokens(ctx, compressed_content)
                if final_tokens > max_tokens:
                    if _retry_depth < MAX_REPORT_FIT_PASSES - 1:
                        new_ratio = max_tokens / final_tokens
                        compressed_content = (
                            await compress_content_with_eigendecomposition(
                                ctx,
                                compressed_content,
                                query_embedding,
                                summary_embedding,
                                ratio=new_ratio,
                                max_tokens=max_tokens,
                                _retry_depth=_retry_depth + 1,
                            )
                        )
                    else:
                        char_ratio = max_tokens / final_tokens
                        compressed_content = compressed_content[
                            : int(len(compressed_content) * char_ratio)
                        ]

            return compressed_content

        logger.warning(
            "Eigendecomposition compression failed, using original method"
        )
        return await compress_content_with_local_similarity(
            ctx, content, query_embedding, summary_embedding, ratio, max_tokens
        )

    except Exception as e:
        logger.error(f"Error during compression with eigendecomposition: {e}")
        try:
            return await compress_content_with_local_similarity(
                ctx, content, query_embedding, summary_embedding, ratio, max_tokens
            )
        except Exception as fallback_error:
            logger.error(f"Fallback compression also failed: {fallback_error}")

            if max_tokens and content:
                content_tokens = await count_tokens(ctx, content)
                if content_tokens > max_tokens:
                    char_ratio = max_tokens / content_tokens
                    char_limit = int(len(content) * char_ratio)
                    return content[:char_limit]

            return content

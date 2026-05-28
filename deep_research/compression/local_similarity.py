import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from deep_research.budget.tokens import count_tokens
from deep_research.config.constants import LOCAL_INFLUENCE_RADIUS, MAX_REPORT_FIT_PASSES
from deep_research.core.text import chunk_text
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.compression.local_similarity")


async def compress_content_with_local_similarity(
    ctx: RunContext,
    content: str,
    query_embedding: list[float] | None,
    summary_embedding: list[float] | None = None,
    ratio: float | None = None,
    max_tokens: int | None = None,
    _retry_depth: int = 0,
) -> str:
    if len(content) < 100:
        return content

    if max_tokens:
        content_tokens = await count_tokens(ctx, content)
        if content_tokens <= max_tokens:
            return content

        if not ratio:
            ratio = max_tokens / content_tokens

    chunks = chunk_text(content, ctx.valves.compression.chunk_level)

    if len(chunks) <= 1:
        return content

    chunk_embeddings = []
    for chunk in chunks:
        embedding = await get_embedding(ctx, chunk)
        if embedding:
            chunk_embeddings.append(embedding)

    if len(chunk_embeddings) <= 1:
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

    n_chunks = len(chunk_embeddings)
    n_keep = max(1, min(n_chunks - 1, int(n_chunks * ratio)))

    if n_keep >= n_chunks:
        n_keep = max(1, n_chunks - 1)

    try:
        embeddings_array = np.array(chunk_embeddings)

        document_centroid = np.mean(embeddings_array, axis=0)

        local_similarities = []
        local_radius = LOCAL_INFLUENCE_RADIUS

        for i in range(len(embeddings_array)):
            local_sim = 0.0
            count = 0

            for j in range(max(0, i - local_radius), i):
                local_sim += cosine_similarity(
                    [embeddings_array[i]], [embeddings_array[j]]
                )[0][0]
                count += 1

            for j in range(i + 1, min(len(embeddings_array), i + local_radius + 1)):
                local_sim += cosine_similarity(
                    [embeddings_array[i]], [embeddings_array[j]]
                )[0][0]
                count += 1

            if count > 0:
                local_sim /= count

            local_similarities.append(local_sim)

        importance_scores = []
        state = ctx.state.get_state(ctx.conversation_id)
        user_preferences = state.get(
            "user_preferences", {"pdv": None, "strength": 0.0, "impact": 0.0}
        )

        for i, embedding in enumerate(embeddings_array):
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                embedding = np.nan_to_num(
                    embedding, nan=0.0, posinf=1.0, neginf=-1.0
                )

            doc_similarity = cosine_similarity([embedding], [document_centroid])[0][
                0
            ]

            query_similarity = cosine_similarity([embedding], [query_embedding])[0][
                0
            ]

            summary_similarity = 0.0
            if summary_embedding is not None:
                summary_similarity = cosine_similarity(
                    [embedding], [summary_embedding]
                )[0][0]
                query_similarity = (
                    query_similarity * ctx.valves.cycles.followup_weight
                ) + (summary_similarity * (1.0 - ctx.valves.cycles.followup_weight))

            local_influence = local_similarities[i]

            pdv_alignment = 0.5
            if (
                ctx.valves.persistence.user_preference_throughout
                and user_preferences["pdv"] is not None
            ):
                chunk_embedding_np = np.array(embedding)
                pdv_np = np.array(user_preferences["pdv"])
                alignment = np.dot(chunk_embedding_np, pdv_np)
                pdv_alignment = (alignment + 1) / 2

                pdv_influence = min(0.3, user_preferences["strength"] / 10)
            else:
                pdv_influence = 0.0

            doc_weight = (
                1.0 - ctx.valves.advanced.query_weight
            ) * 0.4
            local_weight = (
                1.0 - ctx.valves.advanced.query_weight
            ) * 0.8
            query_weight = ctx.valves.advanced.query_weight * (1.0 - pdv_influence)

            final_score = (
                (doc_similarity * doc_weight)
                + (query_similarity * query_weight)
                + (local_influence * local_weight)
                + (pdv_alignment * pdv_influence)
            )

            importance_scores.append((i, final_score))

        importance_scores.sort(key=lambda x: x[1], reverse=True)

        selected_indices = [x[0] for x in importance_scores[:n_keep]]

        selected_indices.sort()

        selected_chunks = [chunks[i] for i in selected_indices if i < len(chunks)]

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
                        await compress_content_with_local_similarity(
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

    except Exception as e:
        logger.error(f"Error during compression with local similarity: {e}")

        if max_tokens and content:
            content_tokens = await count_tokens(ctx, content)
            if content_tokens > max_tokens:
                char_ratio = max_tokens / content_tokens
                char_limit = int(len(content) * char_ratio)
                return content[:char_limit]

        return content

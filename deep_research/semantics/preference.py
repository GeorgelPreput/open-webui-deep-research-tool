import logging
from typing import Any

import numpy as np

from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding
from deep_research.semantics.vocabulary import load_vocabulary_embeddings

logger = logging.getLogger("deep_research.semantics.preference")


async def calculate_preference_impact(
    ctx: RunContext, kept_items, removed_items, all_topics
):
    if not kept_items or not removed_items:
        return 0.0

    total_items = len(all_topics)
    if total_items == 0:
        return 0.0

    impact = len(removed_items) / total_items
    logger.info(
        f"User preference impact: {impact:.3f} ({len(removed_items)}/{total_items} items removed)"
    )
    return impact


async def calculate_preference_direction_vector(
    ctx: RunContext, kept_items: list[str], removed_items: list[str], all_topics: list[str]
) -> dict[str, Any]:
    if not kept_items or not removed_items:
        return {"pdv": None, "strength": 0.0, "impact": 0.0}

    kept_embeddings = []
    for item in kept_items:
        embedding = await get_embedding(ctx, item)
        if embedding:
            kept_embeddings.append(embedding)

    removed_embeddings = []
    for item in removed_items:
        embedding = await get_embedding(ctx, item)
        if embedding:
            removed_embeddings.append(embedding)

    if not kept_embeddings or not removed_embeddings:
        return {"pdv": None, "strength": 0.0, "impact": 0.0}

    try:
        kept_mean = np.mean(kept_embeddings, axis=0)
        removed_mean = np.mean(removed_embeddings, axis=0)

        if (
            np.isnan(kept_mean).any()
            or np.isnan(removed_mean).any()
            or np.isinf(kept_mean).any()
            or np.isinf(removed_mean).any()
        ):
            logger.warning("Invalid values in kept or removed mean vectors")
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        pdv = kept_mean - removed_mean

        pdv_norm = np.linalg.norm(pdv)
        if pdv_norm < 1e-10:
            logger.warning("PDV has near-zero norm")
            return {"pdv": None, "strength": 0.0, "impact": 0.0}

        pdv = pdv / pdv_norm

        strength = np.linalg.norm(kept_mean - removed_mean)

        impact = await calculate_preference_impact(
            ctx, kept_items, removed_items, all_topics
        )

        return {"pdv": pdv.tolist(), "strength": float(strength), "impact": impact}
    except Exception as e:
        logger.error(f"Error calculating PDV: {e}")
        return {"pdv": None, "strength": 0.0, "impact": 0.0}


async def translate_pdv_to_words(ctx: RunContext, pdv):
    if not pdv:
        return None

    vocab_embeddings = await load_vocabulary_embeddings(ctx)
    if not vocab_embeddings:
        return None

    try:
        pdv_array = np.array(pdv)

        word_alignments = []
        for word, embedding in vocab_embeddings.items():
            alignment = np.dot(pdv_array, embedding)
            word_alignments.append((word, alignment))

        top_words = sorted(word_alignments, key=lambda x: x[1], reverse=True)[:10]

        return ", ".join([word for word, _ in top_words])
    except Exception as e:
        logger.error(f"Error translating PDV to words: {e}")
        return None


async def calculate_preference_alignment(ctx: RunContext, content_embedding, pdv):
    if not pdv or not content_embedding:
        return 0.5

    try:
        alignment = np.dot(content_embedding, pdv)
        normalized = (alignment + 1) / 2
        return normalized
    except Exception as e:
        logger.error(f"Error calculating preference alignment: {e}")
        return 0.5

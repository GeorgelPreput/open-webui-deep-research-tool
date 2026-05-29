import json
import logging

from deep_research.core.text import stable_text_key
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.semantics.embeddings")


async def get_embedding(ctx: RunContext, text: str) -> list[float] | None:
    if not text or not text.strip():
        return None

    text = text[:2000]
    text = text.replace(":", " - ")

    cached_embedding = await ctx.caches.embedding.get(text)
    if cached_embedding is not None:
        return cached_embedding

    try:
        model = ctx.valves.models.embedding_model
        result = await ctx.client.embeddings(model, [text])
        embedding = result[0] if result else None
        if embedding:
            await ctx.caches.embedding.set(text, embedding)
            return embedding
        return None
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


async def get_transformed_embedding(
    ctx: RunContext, text: str, transformation=None
) -> list[float] | None:
    if not text or not text.strip():
        return None

    if transformation is None:
        return await get_embedding(ctx, text)

    if isinstance(transformation, dict):
        transform_id = transformation.get(
            "id",
            stable_text_key(
                json.dumps(transformation, sort_keys=True, default=str)
            ),
        )
    else:
        transform_id = transformation
    cached_transformed = await ctx.caches.transformation.get(text, transform_id)
    if cached_transformed is not None:
        return cached_transformed

    base_embedding = await get_embedding(ctx, text)
    if not base_embedding:
        return None

    from deep_research.semantics.eigendecomposition import apply_semantic_transformation

    transformed = await apply_semantic_transformation(
        ctx, base_embedding, transformation
    )

    if transformed:
        await ctx.caches.transformation.set(text, transform_id, transformed)

    return transformed

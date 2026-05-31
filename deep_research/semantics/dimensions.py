import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from deep_research.core.text import stable_text_key
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding
from deep_research.semantics.vocabulary import load_vocabulary_embeddings

logger = logging.getLogger("deep_research.semantics.dimensions")


async def initialize_research_dimensions(
    ctx: RunContext, outline_items: list[str], user_query: str
):
    try:
        item_embeddings = []
        for item in outline_items:
            embedding = await get_embedding(ctx, item[:2000])
            if embedding:
                item_embeddings.append(embedding)

        if len(item_embeddings) < 3:
            logger.warning(
                f"Not enough valid embeddings for research dimensions: {len(item_embeddings)}/3 required"
            )
            ctx.state.update_state(ctx.conversation_id, "research_dimensions", None)
            return

        pca = PCA(n_components=min(10, len(item_embeddings)))
        embedding_array = np.array(item_embeddings)
        pca.fit(embedding_array)

        assert (
            pca.components_ is not None
            and pca.explained_variance_ is not None
            and pca.explained_variance_ratio_ is not None
        )
        research_dimensions = {
            "eigenvectors": pca.components_.tolist(),
            "eigenvalues": pca.explained_variance_.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "total_variance": pca.explained_variance_ratio_.sum(),
            "dimensions": pca.n_components_,
            "coverage": np.zeros(pca.n_components_).tolist(),
        }

        ctx.state.update_state(
            ctx.conversation_id, "research_dimensions", research_dimensions
        )

        ctx.state.update_state(
            ctx.conversation_id,
            "latest_dimension_coverage",
            research_dimensions["coverage"],
        )

        logger.info(
            f"Initialized research dimensions with {pca.n_components_} dimensions"
        )
    except Exception as e:
        logger.error(f"Error initializing research dimensions: {e}")
        ctx.state.update_state(ctx.conversation_id, "research_dimensions", None)


async def update_dimension_coverage(
    ctx: RunContext, content: str, quality_factor: float = 1.0
):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    research_dimensions = conv_state.get("research_dimensions")
    if not research_dimensions:
        return

    try:
        content_embedding = await get_embedding(ctx, content[:2000])
        if not content_embedding:
            return

        current_coverage = research_dimensions.get("coverage", [])
        eigenvectors = research_dimensions.get("eigenvectors", [])

        if not current_coverage or not eigenvectors:
            return

        coverage_array = np.array(current_coverage)
        eigenvectors_array = np.array(eigenvectors)

        projection = np.dot(np.array(content_embedding), eigenvectors_array.T)
        contribution = np.abs(projection) * quality_factor

        for i in range(min(len(contribution), len(coverage_array))):
            current_value = coverage_array[i]
            new_contribution = contribution[i] * (1 - current_value / 2)
            coverage_array[i] = min(1.0, current_value + new_contribution)

        research_dimensions["coverage"] = coverage_array.tolist()

        ctx.state.update_state(
            ctx.conversation_id, "research_dimensions", research_dimensions
        )
        ctx.state.update_state(
            ctx.conversation_id,
            "latest_dimension_coverage",
            coverage_array.tolist(),
        )

        logger.debug(
            f"Updated dimension coverage: {[round(c * 100) for c in coverage_array.tolist()]}%"
        )

    except Exception as e:
        logger.error(f"Error updating dimension coverage: {e}")


async def identify_research_gaps(ctx: RunContext) -> list[int]:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    research_dimensions = conv_state.get("research_dimensions")
    if not research_dimensions:
        return []

    try:
        coverage = np.array(research_dimensions["coverage"])

        sorted_dims = np.argsort(coverage)

        gaps = [int(i) for i in sorted_dims[:3] if coverage[i] < 0.5]

        return gaps
    except Exception as e:
        logger.error(f"Error identifying research gaps: {e}")
        return []


async def translate_dimensions_to_words(
    ctx: RunContext, dimensions: dict[str, Any], coverage: list[float]
):
    if not dimensions or not coverage:
        return []

    conv_state = ctx.state.get_state(ctx.conversation_id)
    dimensions_cache = conv_state.get("dimensions_translation_cache", {})

    dim_hash = stable_text_key(str(dimensions.get("eigenvectors", [])[:3]))
    coverage_hash = stable_text_key(str(coverage))
    cache_key = f"dim_{dim_hash}_{coverage_hash}"

    if cache_key in dimensions_cache:
        logger.info("Using cached dimension translation")
        return dimensions_cache[cache_key]

    dimension_labels = []

    vocab_embeddings = await load_vocabulary_embeddings(ctx)

    if not vocab_embeddings:
        default_labels = [f"Dimension {i + 1}" for i in range(len(coverage))]
        dimensions_cache[cache_key] = default_labels
        ctx.state.update_state(
            ctx.conversation_id, "dimensions_translation_cache", dimensions_cache
        )
        return default_labels

    eigenvectors = np.array(dimensions.get("eigenvectors", []))

    if len(eigenvectors) == 0 or len(eigenvectors) != len(coverage):
        default_labels = [f"Dimension {i + 1}" for i in range(len(coverage))]
        dimensions_cache[cache_key] = default_labels
        ctx.state.update_state(
            ctx.conversation_id, "dimensions_translation_cache", dimensions_cache
        )
        return default_labels

    try:
        for i, eigen_vector in enumerate(eigenvectors):
            word_alignments = []
            for word, embedding in vocab_embeddings.items():
                alignment = np.dot(eigen_vector, embedding)
                word_alignments.append((word, alignment))

            top_words = sorted(word_alignments, key=lambda x: x[1], reverse=True)[:3]
            top_words_str = ", ".join([word for word, _ in top_words])

            cov_percentage = coverage[i]
            dimension_labels.append(
                {
                    "dimension": i + 1,
                    "words": top_words_str,
                    "coverage": cov_percentage,
                }
            )

        dimensions_cache[cache_key] = dimension_labels
        ctx.state.update_state(
            ctx.conversation_id, "dimensions_translation_cache", dimensions_cache
        )

        return dimension_labels
    except Exception as e:
        logger.error(f"Error translating dimensions to words: {e}")
        default_labels = [f"Dimension {i + 1}" for i in range(len(coverage))]
        dimensions_cache[cache_key] = default_labels
        ctx.state.update_state(
            ctx.conversation_id, "dimensions_translation_cache", dimensions_cache
        )
        return default_labels


async def update_research_dimensions_display(ctx: RunContext):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    research_dimensions = conv_state.get("research_dimensions")

    if research_dimensions:
        coverage = research_dimensions.get("coverage", [])
        if coverage:
            ctx.state.update_state(
                ctx.conversation_id, "latest_dimension_coverage", coverage
            )
            logger.info(
                f"Updated latest dimension coverage with {len(coverage)} values"
            )
        else:
            logger.warning("Research dimensions exist but coverage is empty")
    else:
        logger.warning("No research dimensions available for display")

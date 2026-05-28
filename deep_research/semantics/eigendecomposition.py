import logging

import numpy as np

from deep_research.config.constants import SEMANTIC_TRANSFORMATION_STRENGTH
from deep_research.core.types import RunContext

logger = logging.getLogger("deep_research.semantics.eigendecomposition")


async def compute_semantic_eigendecomposition(
    ctx: RunContext, chunks, embeddings, cache_key=None
):
    if not chunks or not embeddings or len(chunks) < 3:
        return None

    if cache_key is None:
        embeddings_concat = np.concatenate(
            embeddings[: min(5, len(embeddings))], axis=0
        )
        fingerprint = np.mean(embeddings_concat, axis=0)
        cache_key = hash(str(fingerprint.round(2)))

    conv_state = ctx.state.get_state(ctx.conversation_id)
    eigendecomposition_cache = conv_state.get("eigendecomposition_cache", {})
    if cache_key in eigendecomposition_cache:
        return eigendecomposition_cache[cache_key]

    try:
        embeddings_array = np.array(embeddings)

        if np.isnan(embeddings_array).any() or np.isinf(embeddings_array).any():
            logger.warning(
                "Invalid values in embeddings, cannot perform eigendecomposition"
            )
            return None

        centered_embeddings = embeddings_array - np.mean(embeddings_array, axis=0)

        cov_matrix = np.cov(centered_embeddings, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_variance = np.sum(eigenvalues)
        if total_variance <= 0:
            logger.warning(
                "Total variance is zero or negative, cannot continue with eigendecomposition"
            )
            return None

        explained_variance_ratio = eigenvalues / total_variance

        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components = int(np.argmax(cumulative_variance >= 0.8)) + 1
        n_components = max(3, min(n_components, 10))

        principal_components = eigenvectors[:, :n_components]
        projected_embeddings = np.dot(centered_embeddings, principal_components)

        result = {
            "eigenvalues": eigenvalues[:n_components].tolist(),
            "eigenvectors": principal_components.tolist(),
            "explained_variance": explained_variance_ratio[:n_components].tolist(),
            "projected_embeddings": projected_embeddings.tolist(),
            "n_components": n_components,
        }

        eigendecomposition_cache[cache_key] = result
        if len(eigendecomposition_cache) > 50:
            oldest_key = next(iter(eigendecomposition_cache))
            del eigendecomposition_cache[oldest_key]
        ctx.state.update_state(
            ctx.conversation_id, "eigendecomposition_cache", eigendecomposition_cache
        )

        return result
    except Exception as e:
        logger.error(f"Error in semantic eigendecomposition: {e}")
        return None


async def create_semantic_transformation(
    ctx: RunContext, semantic_eigendecomposition, pdv=None, trajectory=None, gap_vector=None
):
    if not semantic_eigendecomposition:
        return None

    conv_state = ctx.state.get_state(ctx.conversation_id)
    transformation_id = (
        f"transform_{str(hash(str(pdv)))[:8]}_"
        f"{str(hash(str(trajectory)))[:8]}_"
        f"{str(hash(str(gap_vector)))[:8]}"
    )

    try:
        eigenvectors = np.array(semantic_eigendecomposition["eigenvectors"])
        eigenvalues = np.array(semantic_eigendecomposition["eigenvalues"])

        embedding_dim = eigenvectors.shape[0]
        transformation = np.eye(embedding_dim)

        variance_importance = eigenvalues / np.sum(eigenvalues)

        for i, importance in enumerate(variance_importance):
            eigenvector = eigenvectors[:, i]
            amplification = 1.0 + importance * 2.0
            transformation += (amplification - 1.0) * np.outer(
                eigenvector, eigenvector
            )

        pdv_weight = (
            SEMANTIC_TRANSFORMATION_STRENGTH
            * conv_state.get("user_preferences", {}).get("impact", 0.0)
            if pdv is not None
            else 0.0
        )

        trajectory_weight = (
            ctx.valves.cycles.trajectory_momentum if trajectory is not None else 0.0
        )

        gap_weight = 0.0
        if gap_vector is not None:
            current_cycle = len(conv_state.get("cycle_summaries", [])) + 1
            max_cycles = ctx.valves.cycles.max_cycles
            fade_start_cycle = min(5, int(0.5 * max_cycles))

            if current_cycle <= fade_start_cycle:
                gap_weight = ctx.valves.cycles.gap_exploration_weight
            else:
                remaining_cycles = max_cycles - current_cycle
                total_fade_cycles = max_cycles - fade_start_cycle
                if total_fade_cycles > 0:
                    fade_ratio = remaining_cycles / total_fade_cycles
                    gap_weight = ctx.valves.cycles.gap_exploration_weight * max(
                        0.0, fade_ratio
                    )
                else:
                    gap_weight = 0.0

        total_weight = pdv_weight + trajectory_weight + gap_weight
        if total_weight > 0.8:
            scale_factor = 0.8 / total_weight
            pdv_weight *= scale_factor
            trajectory_weight *= scale_factor
            gap_weight *= scale_factor

        if pdv is not None and pdv_weight > 0.1:
            pdv_array = np.array(pdv)
            norm = np.linalg.norm(pdv_array)
            if norm > 1e-10:
                pdv_array = pdv_array / norm
                transformation += pdv_weight * np.outer(pdv_array, pdv_array)

        if trajectory is not None and trajectory_weight > 0.1:
            trajectory_array = np.array(trajectory)
            norm = np.linalg.norm(trajectory_array)
            if norm > 1e-10:
                trajectory_array = trajectory_array / norm
                transformation += trajectory_weight * np.outer(
                    trajectory_array, trajectory_array
                )

        if gap_vector is not None and gap_weight > 0.1:
            gap_array = np.array(gap_vector)
            norm = np.linalg.norm(gap_array)
            if norm > 1e-10:
                gap_array = gap_array / norm
                transformation += gap_weight * np.outer(gap_array, gap_array)

        return {
            "id": transformation_id,
            "matrix": transformation.tolist(),
            "dimension": embedding_dim,
            "pdv_weight": pdv_weight,
            "trajectory_weight": trajectory_weight,
            "gap_weight": gap_weight,
        }

    except Exception as e:
        logger.error(f"Error creating semantic transformation: {e}")
        return None


async def apply_semantic_transformation(ctx: RunContext, embedding, transformation):
    if not transformation or not embedding:
        return embedding

    try:
        embedding_array = np.array(embedding)

        if isinstance(transformation, str):
            logger.warning(f"Transformation ID not found: {transformation}")
            return embedding

        transform_matrix = np.array(transformation["matrix"])

        if (
            np.isnan(embedding_array).any()
            or np.isnan(transform_matrix).any()
            or np.isinf(embedding_array).any()
            or np.isinf(transform_matrix).any()
        ):
            logger.warning("Invalid values in embedding or transformation matrix")
            return embedding

        transformed = np.dot(embedding_array, transform_matrix)

        if np.isnan(transformed).any() or np.isinf(transformed).any():
            logger.warning("Transformation produced invalid values")
            return embedding

        norm = np.linalg.norm(transformed)
        if norm > 1e-10:
            transformed = transformed / norm
            return transformed.tolist()
        else:
            logger.warning("Transformation produced zero vector")
            return embedding
    except Exception as e:
        logger.error(f"Error applying semantic transformation: {e}")
        return embedding

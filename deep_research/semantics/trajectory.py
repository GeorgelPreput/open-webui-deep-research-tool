import collections
import logging

import numpy as np

from deep_research.core.state import TrajectoryAccumulator
from deep_research.core.text import stable_text_key
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.semantics.trajectory")

# Bounded LRU, mirroring ResearchStateManager.MAX_CONVERSATIONS, so this
# process-global does not grow without limit over a long-lived instance.
_TRAJ_MAX_CONVERSATIONS = 256
_trajectory_accumulators: "collections.OrderedDict[str, TrajectoryAccumulator]" = (
    collections.OrderedDict()
)


def _store_trajectory_accumulator(
    conversation_id: str, acc: TrajectoryAccumulator
) -> None:
    _trajectory_accumulators[conversation_id] = acc
    _trajectory_accumulators.move_to_end(conversation_id)
    while len(_trajectory_accumulators) > _TRAJ_MAX_CONVERSATIONS:
        _trajectory_accumulators.popitem(last=False)


async def calculate_research_trajectory(
    ctx: RunContext, previous_queries, successful_results
):
    if not previous_queries or not successful_results:
        return None

    conv_state = ctx.state.get_state(ctx.conversation_id)
    trajectory_cache = conv_state.get("trajectory_cache", {})

    recent_query_key = stable_text_key(
        str(
            previous_queries[-3:]
            if len(previous_queries) >= 3
            else previous_queries
        )
    )
    recent_result_key = stable_text_key(
        str([r.get("url", "") for r in successful_results[-5:] if "url" in r])
    )
    cache_key = f"{recent_query_key}_{recent_result_key}"

    if cache_key in trajectory_cache:
        logger.info(f"Using cached trajectory for key: {cache_key}")
        return trajectory_cache[cache_key]

    traj_acc = _trajectory_accumulators.get(ctx.conversation_id)
    if traj_acc is not None:
        _trajectory_accumulators.move_to_end(ctx.conversation_id)
    if traj_acc is None:
        sample_embedding = None
        for result in successful_results[:6]:
            content = result.get("content", "")[:2000]
            if content:
                sample_embedding = await get_embedding(ctx, content)
                if sample_embedding:
                    embedding_dim = len(sample_embedding)
                    traj_acc = TrajectoryAccumulator(embedding_dim)
                    _store_trajectory_accumulator(ctx.conversation_id, traj_acc)
                    break

        # If no sample embedding was available, do NOT create an accumulator
        # with a guessed dimension: a wrongly-sized accumulator cached here
        # would crash with a numpy broadcast error once real embeddings arrive.
        # Defer creation until the cycle embeddings below reveal the real dim.

    try:
        max_history_cycles = 5
        queries_per_cycle = ctx.valves.web.search_results_per_query
        results_per_query = ctx.valves.web.successful_results_per_query

        max_queries = max_history_cycles * queries_per_cycle
        max_results = max_queries * results_per_query

        recent_queries = (
            previous_queries[-max_queries:]
            if len(previous_queries) > max_queries
            else previous_queries
        )
        recent_results = (
            successful_results[-max_results:]
            if len(successful_results) > max_results
            else successful_results
        )

        logger.info(
            f"Calculating research trajectory with {len(recent_queries)} recent queries and {len(recent_results)} recent results"
        )

        query_embeddings = []
        for query in recent_queries:
            embedding = await get_embedding(ctx, query)
            if embedding:
                query_embeddings.append(embedding)

        result_embeddings = []
        for result in recent_results:
            content = result.get("content", "")
            if not content:
                continue
            embedding = await get_embedding(ctx, content[:2000])
            if embedding:
                result_embeddings.append(embedding)

        if not query_embeddings or not result_embeddings:
            return None

        # Lazily create the accumulator now that we know the real embedding
        # dimension from an actual vector (the fallback path leaves it None).
        if traj_acc is None:
            traj_acc = TrajectoryAccumulator(len(result_embeddings[0]))
            _store_trajectory_accumulator(ctx.conversation_id, traj_acc)

        traj_acc.add_cycle_data(query_embeddings, result_embeddings)

        trajectory = traj_acc.get_trajectory()

        if trajectory:
            trajectory_cache[cache_key] = trajectory
            if len(trajectory_cache) > 10:
                oldest_key = next(iter(trajectory_cache))
                del trajectory_cache[oldest_key]
            ctx.state.update_state(
                ctx.conversation_id, "trajectory_cache", trajectory_cache
            )

            pdv = conv_state.get("user_preferences", {}).get("pdv")
            if pdv:
                pdv_array = np.array(pdv)
                trajectory_array = np.array(trajectory)
                alignment = np.dot(trajectory_array, pdv_array)
                alignment = (alignment + 1) / 2

                pdv_alignment_history = conv_state.get("pdv_alignment_history", [])
                pdv_alignment_history.append(alignment)
                if len(pdv_alignment_history) > 5:
                    pdv_alignment_history = pdv_alignment_history[-5:]
                ctx.state.update_state(
                    ctx.conversation_id,
                    "pdv_alignment_history",
                    pdv_alignment_history,
                )

                logger.info(f"PDV-Trajectory alignment: {alignment:.3f}")

        return trajectory

    except Exception as e:
        logger.error(f"Error calculating research trajectory: {e}")
        return None


async def calculate_gap_vector(ctx: RunContext):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    research_dimensions = conv_state.get("research_dimensions")
    if not research_dimensions:
        return None

    try:
        coverage = np.array(research_dimensions["coverage"])
        components = np.array(research_dimensions["eigenvectors"])

        current_cycle = len(conv_state.get("cycle_summaries", [])) + 1
        max_cycles = ctx.valves.cycles.max_cycles
        fade_start_cycle = min(5, int(0.5 * max_cycles))

        fade_factor = 1.0
        if current_cycle > fade_start_cycle:
            remaining_cycles = max_cycles - current_cycle
            total_fade_cycles = max_cycles - fade_start_cycle
            if total_fade_cycles > 0:
                fade_factor = max(0.0, remaining_cycles / total_fade_cycles)
            else:
                fade_factor = 0.0

        if fade_factor <= 0.01:
            logger.info("Gap vector faded out completely, returning None")
            return None

        gap_coverage_history = conv_state.get("gap_coverage_history", [])
        gap_coverage_history.append(np.mean(coverage).item())
        if len(gap_coverage_history) > 5:
            gap_coverage_history = gap_coverage_history[-5:]
        ctx.state.update_state(
            ctx.conversation_id, "gap_coverage_history", gap_coverage_history
        )

        gap_vector = np.zeros(components.shape[1])

        for i, cov in enumerate(coverage):
            gap = 1.0 - cov
            if gap > 0.3:
                if isinstance(components, np.ndarray) and i < len(components):
                    gap_vector += gap * components[i]
                else:
                    logger.warning(f"Invalid components at index {i}")

        gap_vector *= fade_factor

        if np.isnan(gap_vector).any() or np.isinf(gap_vector).any():
            logger.warning("Invalid values in gap vector")
            return None

        norm = np.linalg.norm(gap_vector)
        if norm > 1e-10:
            gap_vector = gap_vector / norm
            return gap_vector.tolist()
        else:
            logger.warning("Gap vector has zero norm")
            return None
    except Exception as e:
        logger.error(f"Error calculating gap vector: {e}")
        return None

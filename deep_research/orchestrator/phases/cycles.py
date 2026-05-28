import json
import logging
from typing import Any

from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import checkpoint, update_token_counts
from deep_research.progress.embed import refresh_progress_embed
from deep_research.progress.events import StatusEvent
from deep_research.research.cycle import process_query
from deep_research.research.query_gen import improved_query_generation
from deep_research.research.ranking import rank_topics_by_research_priority
from deep_research.semantics.dimensions import (
    update_dimension_coverage,
)
from deep_research.semantics.embeddings import get_embedding
from deep_research.semantics.trajectory import (
    calculate_gap_vector,
    calculate_research_trajectory,
)

logger = logging.getLogger("deep_research.orchestrator.phases.cycles")


async def run_cycles(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    user_message = ps.get("user_message", "")
    all_topics = ps.get("all_topics", conv_state.get("all_topics", []))
    outline_embedding = ps.get("outline_embedding")
    summary_embedding = ps.get("summary_embedding")
    initial_results = ps.get("initial_results", [])

    min_cycles = ctx.valves.cycles.min_cycles
    max_cycles = ctx.valves.cycles.max_cycles

    completed_topics = set(conv_state.get("completed_topics", []))
    irrelevant_topics = set(conv_state.get("irrelevant_topics", []))
    search_history = list(conv_state.get("search_history", []))
    results_history = list(conv_state.get("results_history", [])) + initial_results
    cycle_summaries = list(conv_state.get("cycle_summaries", []))
    active_outline = list(set(all_topics) - completed_topics - irrelevant_topics)

    await update_token_counts(ctx)

    cycle = ps.get("cycle", 1)
    while cycle < max_cycles and active_outline:
        cycle += 1
        await ctx.events.emit(StatusEvent(
            description=f"Research cycle {cycle}/{max_cycles}: Generating search queries...",
            level="info", done=False,
        ))

        if cycle > 2 and results_history:
            trajectory = await calculate_research_trajectory(ctx, search_history, results_history)
            conv_state["research_trajectory"] = trajectory

        gap_vector = await calculate_gap_vector(ctx)
        prioritized_topics = await rank_topics_by_research_priority(
            ctx, active_outline, gap_vector, set(completed_topics), results_history
        )
        priority_topics = prioritized_topics[:10]

        search_context = _build_search_context(ctx, conv_state, user_message, priority_topics,
                                                active_outline, search_history, results_history,
                                                cycle_summaries, cycle)

        query_objects = await improved_query_generation(ctx, user_message, priority_topics, search_context)
        current_cycle_queries = list(query_objects)
        conv_state["current_cycle_queries"] = current_cycle_queries
        conv_state["progress_cycle"] = cycle

        query_strings = [q.get("query", "") for q in current_cycle_queries]
        search_history.extend(query_strings)
        conv_state["search_history"] = search_history

        cycle_results = []
        for query_obj in current_cycle_queries:
            query = query_obj.get("query", "")
            query_embedding = await get_embedding(ctx, query)
            if not query_embedding:
                query_embedding = [0.0] * 384

            semantic_transformations = conv_state.get("semantic_transformations")
            if semantic_transformations:
                from deep_research.semantics.eigendecomposition import apply_semantic_transformation
                transformed = await apply_semantic_transformation(ctx, query_embedding, semantic_transformations)
                if transformed:
                    query_embedding = transformed

            results = await process_query(ctx, query, query_embedding, outline_embedding,
                                          cycle_feedback=None, summary_embedding=summary_embedding)
            cycle_results.extend(results)
            results_history.extend(results)

        conv_state["results_history"] = results_history

        if cycle_results:
            await _analyze_cycle_results(ctx, conv_state, cycle, max_cycles, user_message,
                                         cycle_results, completed_topics, irrelevant_topics,
                                         active_outline, all_topics, cycle_summaries,
                                         results_history, outline_embedding)

        if not active_outline:
            await ctx.events.emit(StatusEvent(description="All research topics addressed", level="info", done=False))
            break

        coverage_ratio = len(completed_topics) / max(len(all_topics), 1)
        if cycle >= min_cycles and coverage_ratio > 0.7:
            await ctx.events.emit(StatusEvent(description="Most topics addressed. Finalizing...", level="info", done=False))
            break

        if cycle >= max_cycles:
            await ctx.events.emit(StatusEvent(description=f"Max cycles ({max_cycles}) reached", level="info", done=False))
            break

    ps["results_history"] = results_history
    ps["cycle_summaries"] = cycle_summaries
    ps["completed_topics"] = list(completed_topics)
    ps["irrelevant_topics"] = list(irrelevant_topics)
    ps["all_topics"] = all_topics
    ps["cycle"] = cycle
    return ps


def _build_search_context(ctx, conv_state, user_message, priority_topics,
                          active_outline, search_history, results_history, cycle_summaries, cycle):
    context = f"### Original Query:\n{user_message}\n\n"
    prefs = conv_state.get("user_preferences", {})
    if prefs.get("pdv") is not None:
        context += "### User preferences active\n\n"
    context += "### Priority topics:\n" + "\n".join(f"- {t}" for t in priority_topics) + "\n"
    if len(active_outline) > len(priority_topics):
        remaining = [t for t in active_outline if t not in priority_topics]
        context += "\n### Additional topics:\n" + "\n".join(f"- {t}" for t in remaining) + "\n"
    if search_history:
        context += "\n### Recent queries:\n" + ", ".join(f"'{q}'" for q in search_history[-9:]) + "\n"
    if results_history:
        recent = results_history[-6:]
        context += "\n### Recent results:\n"
        for i, r in enumerate(recent):
            context += f"Result {i+1} (Query: '{r.get('query','')}'): {r.get('content','')[:200]}...\n"
    if cycle_summaries:
        context += "\n### Previous summaries:\n"
        for i, s in enumerate(cycle_summaries[-3:]):
            context += f"Cycle {cycle - 3 + i}: {s}\n"
    gaps = conv_state.get("research_dimensions")
    if gaps:
        context += "\n### Research gaps identified\n"
    follow_up = conv_state.get("prev_comprehensive_summary", "")
    if conv_state.get("follow_up_mode") and follow_up:
        context += f"\n### Previous summary:\n{follow_up[:2000]}...\n"
    return context


async def _analyze_cycle_results(ctx, conv_state, cycle, max_cycles, user_message,
                                 cycle_results, completed_topics, irrelevant_topics,
                                 active_outline, all_topics, cycle_summaries,
                                 results_history, outline_embedding):
    analysis_prompt = {
        "role": "system",
        "content": (
            f"You are a post-grad researcher analyzing search results.\n"
            f"This is cycle {cycle} out of {max_cycles}.\n"
            f"Original query: \"{user_message}\".\n\n"
            "Classify topics as COMPLETED, PARTIAL, IRRELEVANT, or NEW.\n"
            'Format: {"completed_topics":[],"partial_topics":[],"irrelevant_topics":[],'
            '"new_topics":[],"analysis":"..."}'
        ),
    }
    analysis_context = _build_analysis_context(cycle_results, active_outline,
                                                completed_topics, irrelevant_topics, cycle_summaries)
    analysis_msg = {"role": "user", "content": f"Original query: {user_message}\n\n{analysis_context}\n\nAnalyze."}
    response = await ctx.client.chat_completions(
        ctx.valves.models.research_model,
        [analysis_prompt, analysis_msg],
        temperature=ctx.valves.models.temperature,
    )
    content = response["choices"][0]["message"]["content"]
    try:
        json_str = content[content.find("{"):content.rfind("}") + 1]
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        data = {}

    newly_completed = set(data.get("completed_topics", []))
    completed_topics.update(newly_completed)
    newly_irrelevant = set(data.get("irrelevant_topics", []))
    irrelevant_topics.update(newly_irrelevant)
    new_topics = data.get("new_topics", [])

    for t in new_topics:
        if t not in all_topics and t not in completed_topics and t not in irrelevant_topics:
            active_outline.append(t)
            all_topics.append(t)

    active_outline[:] = [t for t in active_outline if t not in completed_topics and t not in irrelevant_topics]

    conv_state["completed_topics"] = list(completed_topics)
    conv_state["irrelevant_topics"] = list(irrelevant_topics)
    conv_state["active_outline"] = active_outline
    conv_state["all_topics"] = all_topics
    summary = data.get("analysis", f"Analysis for cycle {cycle}")
    cycle_summaries.append(summary)
    conv_state["cycle_summaries"] = cycle_summaries
    conv_state["partial_topics"] = list(data.get("partial_topics", []))
    conv_state["latest_new_topics"] = list(new_topics)
    conv_state["latest_completed_topics"] = list(newly_completed)
    conv_state["latest_irrelevant_topics"] = list(newly_irrelevant)
    conv_state["progress_cycle"] = cycle
    from datetime import datetime
    conv_state["progress_last_updated_at"] = datetime.now().isoformat(timespec="seconds")

    dims = conv_state.get("research_dimensions")
    if dims:
        conv_state["latest_dimension_coverage"] = dict(dims.get("coverage", {}))

    for result in cycle_results:
        content = result.get("content", "")
        if content:
            quality = 0.5
            if "similarity" in result:
                quality = 0.5 + result["similarity"] * 0.5
            await update_dimension_coverage(ctx, content, quality)

    await refresh_progress_embed(ctx, cycle=cycle, force=True)
    await checkpoint(ctx)


def _build_analysis_context(cycle_results, active_outline, completed_topics, irrelevant_topics, cycle_summaries):
    ctx_text = "### Current Outline:\n" + "\n".join(f"- {t}" for t in active_outline) + "\n\n"
    ctx_text += "### Latest Results:\n"
    for i, r in enumerate(cycle_results):
        ctx_text += f"Result {i+1}: {r.get('title','')} - {r.get('content','')[:2000]}...\n"
    if cycle_summaries:
        ctx_text += "\n### Previous summaries:\n"
        for i, s in enumerate(cycle_summaries):
            ctx_text += f"Cycle {i+1}: {s}\n"
    if completed_topics:
        ctx_text += "\n### Completed:\n" + "\n".join(f"- {t}" for t in completed_topics) + "\n"
    if irrelevant_topics:
        ctx_text += "\n### Irrelevant:\n" + "\n".join(f"- {t}" for t in irrelevant_topics) + "\n"
    return ctx_text

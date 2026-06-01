import logging
from typing import Any

from deep_research.core.text import response_text
from deep_research.core.types import RunContext
from deep_research.persistence.chat_state import initialize_research_state
from deep_research.progress.events import MessageEvent, StatusEvent
from deep_research.research.cycle import process_query
from deep_research.research.relevance import is_follow_up_query
from deep_research.semantics.dimensions import initialize_research_dimensions
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger("deep_research.orchestrator.phases.initial_queries")


async def run_initial_queries(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    user_message = ps.get("user_message", conv_state.get("last_user_message", ""))
    is_follow_up = await is_follow_up_query(ctx, [{"role": "user", "content": user_message}])
    ctx.state.update_state(ctx.conversation_id, "follow_up_mode", is_follow_up)

    summary_embedding = None
    if is_follow_up:
        prev_summary = conv_state.get("prev_comprehensive_summary", "")
        if prev_summary:
            summary_embedding = await get_embedding(ctx, prev_summary)

    await ctx.events.emit(StatusEvent(description="Generating initial search queries...", level="info", done=False))

    if is_follow_up:
        research_outline, all_topics, outline_embedding, initial_results = await _handle_follow_up(ctx, user_message, summary_embedding)
    else:
        research_outline, all_topics, outline_embedding, initial_results = await _handle_fresh(ctx, user_message, summary_embedding)

    await ctx.events.emit(StatusEvent(description="Initial research outline generated", level="info", done=False))

    ps["user_message"] = user_message
    ps["research_outline"] = research_outline
    ps["all_topics"] = all_topics
    ps["outline_embedding"] = outline_embedding
    ps["summary_embedding"] = summary_embedding
    ps["is_follow_up"] = is_follow_up
    ps["initial_results"] = initial_results
    ps["cycle"] = 1

    if ctx.valves.persistence.interactive_research and not is_follow_up:
        outline_lines = ["### Research Outline\n"]
        for topic in research_outline:
            outline_lines.append(f"**{topic.get('topic', '')}**")
            for sub in topic.get("subtopics", []):
                outline_lines.append(f"- {sub}")
            outline_lines.append("")
        outline_report = "\n".join(outline_lines) + (
            "\n\n*Reply with adjustments to the outline, or send any message "
            "to confirm and proceed with research cycles.*"
        )
        ctx.state.update_state(ctx.conversation_id, "waiting_for_outline_feedback", True)
        ctx.state.update_state(
            ctx.conversation_id,
            "outline_feedback_data",
            {
                "original_query": user_message,
                "outline_items": research_outline,
            },
        )
        ps["awaiting_outline_feedback"] = True
        ps["outline_report"] = outline_report

    return ps


async def _handle_fresh(ctx, user_message, summary_embedding):
    research_outline, all_topics, outline_embedding, initial_results = await _generate_outline_and_initial(ctx, user_message)

    await initialize_research_state(ctx, user_message, research_outline, all_topics, outline_embedding, initial_results)
    await _emit_outline(ctx, research_outline)
    await ctx.events.emit(StatusEvent(description="Research outline generated. Beginning research cycles...", level="info", done=False))

    return research_outline, all_topics, outline_embedding, initial_results


async def _handle_follow_up(ctx, user_message, summary_embedding):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    research_outline, all_topics, outline_embedding, initial_results = await _generate_outline_and_initial(ctx, user_message, summary_embedding=summary_embedding)

    await initialize_research_state(ctx, user_message, research_outline, all_topics, outline_embedding, initial_results)
    await _emit_outline(ctx, research_outline)

    conv_state["follow_up_mode"] = True
    return research_outline, all_topics, outline_embedding, initial_results


async def _generate_outline_and_initial(ctx, user_message, summary_embedding=None):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    is_follow_up = conv_state.get("follow_up_mode", False)

    if is_follow_up:
        prev_summary = conv_state.get("prev_comprehensive_summary", "")
        prompt_text = (
            "You are a post-grad research assistant generating effective search queries for continued research.\n"
            f"Follow-up question: {user_message}\n"
            f"Previous summary: {prev_summary[:2000]}\n"
            "Generate 6 targeted search queries that build on previous findings. "
            "Format as JSON: {\"queries\": [\"q1\", \"q2\", ...]}"
        )
    else:
        prompt_text = (
            "You are a post-grad research assistant generating effective search queries.\n"
            f"User query: {user_message}\n"
            "Generate 8 initial search queries (half broad, half specific). "
            "Format as JSON: {\"queries\": [\"q1\", \"q2\", ...]}"
        )

    import json
    system_msg = {"role": "system", "content": "You generate search queries as JSON."}
    user_msg = {"role": "user", "content": prompt_text}
    response = await ctx.llm.chat_completions(
        ctx.valves.models.research_model,
        [system_msg, user_msg],
        temperature=ctx.valves.models.temperature,
    )
    content = response_text(response)
    try:
        json_str = content[content.find("{"):content.rfind("}") + 1]
        query_data = json.loads(json_str)
        initial_queries = query_data.get("queries", [])
    except (json.JSONDecodeError, ValueError):
        import re
        initial_queries = re.findall(r'"([^"]+)"', content)[:4]
        if not initial_queries:
            initial_queries = [f"Information about {user_message}"]

    initial_results = []
    seen_urls = set()
    diag = getattr(ctx, "embeddings_diagnostics", None)
    embedding_degraded = diag is not None and diag.degraded

    # Best-effort path: under embedding-throttle pressure, run search without
    # query embedding rather than dropping the query entirely. Otherwise we'd
    # collapse the outline whenever the provider rate-limits us.
    embedding_failures = 0
    for query in initial_queries:
        query_embedding = None
        if not embedding_degraded:
            query_embedding = await get_embedding(ctx, query)
            if not query_embedding:
                embedding_failures += 1
                # If a majority of embeddings have failed, switch to
                # embedding-less mode for the rest of the batch — likely the
                # provider is rate-limiting and further attempts only burn TPM.
                if embedding_failures * 2 > len(initial_queries):
                    embedding_degraded = True
                    logger.warning(
                        "Initial queries: %d/%d embedding failures, switching "
                        "to embedding-less best-effort search",
                        embedding_failures,
                        len(initial_queries),
                    )
        if not query_embedding and not embedding_degraded:
            logger.warning(f"Skipping initial query with no embedding: {query!r}")
            if diag is not None:
                diag.record_skipped()
            continue
        results = await process_query(
            ctx,
            query,
            query_embedding,
            None,
            cycle_feedback=None,
            summary_embedding=summary_embedding,
        )
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                initial_results.append(r)

    outline_data = await _generate_outline(ctx, user_message, initial_results, is_follow_up)
    research_outline = outline_data.get("outline", [])
    all_topics = []
    for item in research_outline:
        all_topics.append(item["topic"])
        all_topics.extend(item.get("subtopics", []))
    outline_text = " ".join(all_topics)
    outline_embedding = await get_embedding(ctx, outline_text)
    await initialize_research_dimensions(ctx, all_topics, user_message)

    return research_outline, all_topics, outline_embedding, initial_results


async def _generate_outline(ctx, user_message, initial_results, is_follow_up):
    conv_state = ctx.state.get_state(ctx.conversation_id)
    if is_follow_up:
        prev_summary = conv_state.get("prev_comprehensive_summary", "")
        context = f"Previous summary: {prev_summary[:2000]}\n\nNew results:\n"
        for i, r in enumerate(initial_results):
            context += f"Result {i+1}: {r.get('title','')} - {r.get('content','')[:500]}\n"
    else:
        context = "\n".join(
            f"Result {i+1}: {r.get('title','')} - {r.get('content','')[:500]}"
            for i, r in enumerate(initial_results)
        )

    prompt = {
        "role": "system",
        "content": "Generate a research outline as JSON with format: "
        '{"outline": [{"topic": "...", "subtopics": ["..."]}]}',
    }
    msg = {
        "role": "user",
        "content": f"Query: {user_message}\n\nResults:\n{context}\n\nCreate outline.",
    }
    response = await ctx.llm.chat_completions(
        ctx.valves.models.research_model,
        [prompt, msg],
        temperature=ctx.valves.models.temperature,
    )
    import json
    content = response_text(response)
    try:
        json_str = content[content.find("{"):content.rfind("}") + 1]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return {"outline": [{"topic": "Research Findings", "subtopics": ["Key Aspects", "Detailed Analysis"]}]}


async def _emit_outline(ctx, research_outline):
    await ctx.events.emit(MessageEvent(content="### Research Outline\n\n"))
    for topic in research_outline:
        t = topic["topic"]
        subs = topic.get("subtopics", [])
        line = f"**{t}**\n" + "".join(f"- {s}\n" for s in subs) + "\n"
        await ctx.events.emit(MessageEvent(content=line))
    await ctx.events.emit(MessageEvent(content="\n*Continuing with research...*\n\n"))

import logging
from typing import Any

from deep_research.core.types import RunContext
from deep_research.progress.events import StatusEvent
from deep_research.synthesis.citations import (
    format_bibliography_list,
    generate_bibliography,
    identify_and_correlate_citations,
)
from deep_research.synthesis.sections import generate_section_content_with_citations
from deep_research.synthesis.utils import get_synthesis_model

logger = logging.getLogger("deep_research.orchestrator.phases.synthesize")


async def run_synthesize(ctx: RunContext, ps: dict[str, Any]) -> dict[str, Any]:
    conv_state = ctx.state.get_state(ctx.conversation_id)
    synthesis_outline = ps.get("synthesis_outline", [])
    results_history = ps.get("results_history", [])
    user_message = ps.get("user_message", "")
    is_follow_up = ps.get("is_follow_up", False)

    if not synthesis_outline:
        logger.warning("No synthesis outline, skipping synthesis")
        ps["compiled_sections"] = {}
        return ps

    synthesis_model = get_synthesis_model(ctx)
    await ctx.events.emit(StatusEvent(description=f"Synthesizing with {synthesis_model}...", level="info", done=False))

    irrelevant_topics = set(conv_state.get("irrelevant_topics", []))
    relevant_topics = [t for t in synthesis_outline if t["topic"] not in irrelevant_topics]
    if not relevant_topics:
        relevant_topics = [{"topic": "Research Summary", "subtopics": ["General Information"]}]

    conv_state["section_synthesized_content"] = {}
    conv_state["subtopic_synthesized_content"] = {}
    conv_state["section_sources_map"] = {}
    conv_state["section_citations"] = {}
    if "global_citation_map" not in conv_state:
        conv_state["global_citation_map"] = {}

    compiled_sections: dict[str, str] = {}
    all_verified = []
    all_flagged = []
    global_citation_map = dict(conv_state.get("global_citation_map", {}))
    master_source_table = dict(conv_state.get("master_source_table", {}))
    prev_summary = conv_state.get("prev_comprehensive_summary", "")

    for topic_item in relevant_topics:
        section_title = str(topic_item["topic"])
        subtopics = [st for st in topic_item.get("subtopics", []) if st not in irrelevant_topics]

        section_data = await generate_section_content_with_citations(
            ctx, section_title, subtopics, user_message, results_history,
            synthesis_model, is_follow_up,
            prev_summary if is_follow_up else "",
        )
        compiled_sections[section_title] = section_data["content"]
        all_verified.extend(section_data.get("verified_citations", []))
        all_flagged.extend(section_data.get("flagged_citations", []))

    conv_state["verification_results"] = {"verified": all_verified, "flagged": all_flagged}

    additional_citations = []
    for section_title, content in list(compiled_sections.items()):
        section_cits = await identify_and_correlate_citations(ctx, section_title, content, master_source_table)
        if section_cits:
            additional_citations.extend(section_cits)
            all_section_cits = dict(conv_state.get("section_citations", {}))
            if section_title not in all_section_cits:
                all_section_cits[section_title] = []
            all_section_cits[section_title].extend(section_cits)
            conv_state["section_citations"] = all_section_cits
            for cit in section_cits:
                url = cit.get("url", "")
                if url and url not in global_citation_map:
                    global_citation_map[url] = len(global_citation_map) + 1

    conv_state["global_citation_map"] = global_citation_map

    bibliography_data = await generate_bibliography(ctx, master_source_table, global_citation_map)

    for section_title, content in list(compiled_sections.items()):
        modified = content
        section_cits = [c for c in additional_citations if c.get("section") == section_title]
        for cit in section_cits:
            url = cit.get("url", "")
            raw = cit.get("raw_text", "")
            if url and url in global_citation_map and raw:
                modified = modified.replace(raw, f"[{global_citation_map[url]}]", 1)
        compiled_sections[section_title] = modified

    bibliography_table = await format_bibliography_list(ctx, bibliography_data.get("bibliography", []))

    ps["compiled_sections"] = compiled_sections
    ps["bibliography_data"] = bibliography_data
    ps["bibliography_table"] = bibliography_table
    ps["synthesis_model"] = synthesis_model
    return ps

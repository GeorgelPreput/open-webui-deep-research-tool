import logging
from typing import Any

from deep_research.budget.packing import pack_sections_to_budget
from deep_research.budget.tokens import count_tokens
from deep_research.core.types import BibliographyEntry, RunContext

logger = logging.getLogger(__name__)


async def build_titles_context(
    ctx: RunContext,
    user_message: str,
    sections: dict[str, str],
    input_budget: int,
) -> str:
    """Compact context for title generation: query + section titles + 1 key chunk each."""
    section_titles_line = "\n".join(f"- {t}" for t in sections)
    anchor = f"{user_message}\n{section_titles_line}"
    pinned = (
        f"Research Query: {user_message}\n\nSection Titles:\n{section_titles_line}"
    )
    body = await pack_sections_to_budget(
        ctx,
        sections,
        input_budget=max(50, input_budget - await count_tokens(ctx, pinned) - 4),
        anchor_text=anchor,
        task_name="titles",
        include_query_line=False,
        include_section_headings=False,
        pinned_prefix="",
        min_chunks_per_section=1,
    )
    if body.strip():
        return f"{pinned}\n\nKey content per section:\n{body}"
    return pinned


async def build_abstract_context(
    ctx: RunContext,
    user_message: str,
    sections: dict[str, str],
    bibliography: list[BibliographyEntry],
    input_budget: int,
) -> str:
    """Context for abstract: query + section coverage + source count."""
    section_titles_line = "\n".join(f"- {t}" for t in sections)
    anchor = f"{user_message}\n{section_titles_line}"
    source_line = f"\n\nTotal sources: {len(bibliography)}" if bibliography else ""
    pinned = f"Research Query: {user_message}\n\nSections:\n{section_titles_line}{source_line}"
    body = await pack_sections_to_budget(
        ctx,
        sections,
        input_budget=max(50, input_budget - await count_tokens(ctx, pinned) - 4),
        anchor_text=anchor,
        task_name="abstract",
        include_query_line=False,
        include_section_headings=True,
        pinned_prefix="",
        min_chunks_per_section=1,
    )
    if body.strip():
        return f"{pinned}\n\nSection content:\n{body}"
    return pinned


async def build_intro_context(
    ctx: RunContext,
    user_message: str,
    outline: list[dict[str, Any]],
    sections: dict[str, str],
    input_budget: int,
) -> str:
    """Context for introduction: query + outline bullets + section chunks."""
    section_titles_line = "\n".join(f"- {t}" for t in sections)
    outline_bullets = ""
    for topic in outline:
        outline_bullets += f"- {topic.get('topic', '')}\n"
        for sub in topic.get("subtopics", []):
            outline_bullets += f"  - {sub}\n"

    anchor = f"{user_message}\n{section_titles_line}"
    pinned = (
        f"Research Query: {user_message}\n\n"
        f"Research Outline:\n{outline_bullets}\n"
        f"Sections:\n{section_titles_line}"
    )
    body = await pack_sections_to_budget(
        ctx,
        sections,
        input_budget=max(50, input_budget - await count_tokens(ctx, pinned) - 4),
        anchor_text=anchor,
        task_name="introduction",
        include_query_line=False,
        include_section_headings=True,
        pinned_prefix="",
        min_chunks_per_section=1,
    )
    if body.strip():
        return f"{pinned}\n\nSection content:\n{body}"
    return pinned


async def build_conclusion_context(
    ctx: RunContext,
    user_message: str,
    sections: dict[str, str],
    input_budget: int,
) -> str:
    """Context for conclusion: query + section-aware packed content."""
    section_titles_line = "\n".join(f"- {t}" for t in sections)
    anchor = f"{user_message}\n{section_titles_line}"
    pinned = f"Research Query: {user_message}\n\nSections:\n{section_titles_line}"
    body = await pack_sections_to_budget(
        ctx,
        sections,
        input_budget=max(50, input_budget - await count_tokens(ctx, pinned) - 4),
        anchor_text=anchor,
        task_name="conclusion",
        include_query_line=False,
        include_section_headings=True,
        pinned_prefix="",
        min_chunks_per_section=1,
    )
    if body.strip():
        return f"{pinned}\n\nKey findings:\n{body}"
    return pinned

import logging
from collections import defaultdict
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from deep_research.budget.tokens import count_tokens
from deep_research.config.constants import (
    MAX_REPORT_FIT_PASSES,
    MIN_SECTION_COVERAGE_CHUNKS,
)
from deep_research.core.text import chunk_text
from deep_research.core.types import RunContext
from deep_research.semantics.embeddings import get_embedding

logger = logging.getLogger(__name__)


async def score_section_chunks(
    ctx: RunContext,
    sections: dict[str, str],
    anchor_text: str,
    summary_text: str | None = None,
    task_name: str | None = None,
) -> list[dict[str, Any]]:
    """Score all chunks across sections for relevance to *anchor_text*.

    Returns a flat list of chunk records, each with keys:
        section_title, chunk_index, text, tokens, score, pinned
    """
    anchor_embedding = await get_embedding(ctx, anchor_text[:2000])
    summary_embedding: list[float] | None = None
    if summary_text:
        summary_embedding = await get_embedding(ctx, summary_text[:2000])

    records: list[dict[str, Any]] = []

    for section_title, content in sections.items():
        if not content or not content.strip():
            continue
        chunks = chunk_text(content, chunk_level=ctx.valves.compression.chunk_level)
        n = len(chunks)

        chunk_embeddings: list[list[float] | None] = []
        for chunk in chunks:
            emb = await get_embedding(ctx, chunk)
            chunk_embeddings.append(emb)

        for idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings, strict=False)):
            token_count = await count_tokens(ctx, chunk)

            if emb and anchor_embedding:
                try:
                    sim = cosine_similarity([emb], [anchor_embedding])[0][0]
                except Exception:
                    sim = 0.3
            else:
                sim = 0.3

            score = float(sim)

            if emb and summary_embedding:
                try:
                    sum_sim = cosine_similarity([emb], [summary_embedding])[0][0]
                    score = score * 0.6 + float(sum_sim) * 0.4
                except Exception:
                    pass

            if idx == 0:
                score += 0.05
            if idx == n - 1 and n > 1:
                score += 0.03

            records.append(
                {
                    "section_title": section_title,
                    "chunk_index": idx,
                    "text": chunk,
                    "tokens": token_count,
                    "score": score,
                    "pinned": False,
                }
            )

    return records


async def pack_sections_to_budget(
    ctx: RunContext,
    sections: dict[str, str],
    input_budget: int,
    anchor_text: str,
    task_name: str,
    include_query_line: bool = True,
    include_section_headings: bool = True,
    pinned_prefix: str = "",
    min_chunks_per_section: int | None = None,
    allow_lossy: bool = True,
) -> str:
    """Pack section content into *input_budget* tokens preserving cross-section coverage."""
    if min_chunks_per_section is None:
        min_chunks_per_section = MIN_SECTION_COVERAGE_CHUNKS

    chunk_records = await score_section_chunks(
        ctx, sections, anchor_text, task_name=task_name
    )

    pinned_tokens = await count_tokens(ctx, pinned_prefix) if pinned_prefix else 0

    heading_tokens: dict[str, int] = {}
    if include_section_headings:
        for title in sections:
            heading_tokens[title] = await count_tokens(ctx, f"## {title}\n")

    total_heading_tokens = (
        sum(heading_tokens.values()) if include_section_headings else 0
    )
    remaining_budget = input_budget - pinned_tokens - total_heading_tokens
    remaining_budget = max(50, remaining_budget)

    selected: list[dict[str, Any]] = []
    used_tokens = 0
    covered_sections: set[str] = set()

    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in chunk_records:
        by_section[rec["section_title"]].append(rec)
    for title in by_section:
        by_section[title].sort(key=lambda r: r["score"], reverse=True)

    for title, recs in by_section.items():
        for rec in recs[:min_chunks_per_section]:
            if used_tokens + rec["tokens"] <= remaining_budget:
                selected.append(rec)
                used_tokens += rec["tokens"]
                covered_sections.add(title)
                break

    selected_keys = {(r["section_title"], r["chunk_index"]) for r in selected}

    remaining_records = sorted(
        [
            r
            for r in chunk_records
            if (r["section_title"], r["chunk_index"]) not in selected_keys
        ],
        key=lambda r: r["score"],
        reverse=True,
    )
    for rec in remaining_records:
        if used_tokens + rec["tokens"] > remaining_budget:
            continue
        selected.append(rec)
        used_tokens += rec["tokens"]

    section_order = list(sections.keys())
    selected.sort(
        key=lambda r: (
            (
                section_order.index(r["section_title"])
                if r["section_title"] in section_order
                else 9999
            ),
            r["chunk_index"],
        )
    )

    parts: list[str] = []
    if pinned_prefix:
        parts.append(pinned_prefix)

    current_section: str | None = None
    for rec in selected:
        if include_section_headings and rec["section_title"] != current_section:
            parts.append(f"## {rec['section_title']}")
            current_section = rec["section_title"]
        parts.append(rec["text"])

    assembled = "\n\n".join(parts)

    for _pass in range(MAX_REPORT_FIT_PASSES):
        actual_tokens = await count_tokens(ctx, assembled)
        if actual_tokens <= input_budget:
            break
        if not selected:
            break
        min_idx = min(
            range(len(selected)),
            key=lambda i: selected[i]["score"],
        )
        selected.pop(min_idx)

        parts = []
        if pinned_prefix:
            parts.append(pinned_prefix)
        current_section = None
        for rec in selected:
            if include_section_headings and rec["section_title"] != current_section:
                parts.append(f"## {rec['section_title']}")
                current_section = rec["section_title"]
            parts.append(rec["text"])
        assembled = "\n\n".join(parts)
    else:
        actual_tokens = await count_tokens(ctx, assembled)
        if actual_tokens > input_budget:
            char_ratio = input_budget / actual_tokens
            assembled = assembled[: int(len(assembled) * char_ratio)]

    return assembled

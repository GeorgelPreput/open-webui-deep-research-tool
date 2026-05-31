"""Regression tests for research-layer bug fixes.

Covers:
  BUG 36 - extract_topic_relevant_info returns "" (a str) on empty input,
           not [] (a list), so callers' f-strings don't render "[]".
  BUG 21 - group_replacement_topics re-examines split halves until no group
           exceeds 5 topics.
"""
import pytest


# --- BUG 36: consistent return type -----------------------------------------

@pytest.mark.asyncio
async def test_extract_topic_relevant_info_empty_returns_empty_string(run_context):
    from deep_research.research.relevance import extract_topic_relevant_info

    result = await extract_topic_relevant_info(run_context, [], ["some topic"])
    assert result == ""
    assert isinstance(result, str)


# --- BUG 21: large-group split invariant ------------------------------------

@pytest.mark.asyncio
async def test_group_split_never_leaves_group_over_five(run_context, monkeypatch):
    # Two tight embedding clusters (12 + 6) so KMeans yields >1 group with a
    # large member that must be repeatedly split down to <=5.
    async def fake_embedding(ctx, text):
        return [1.0, 0.0, 0.0] if text.startswith("A") else [0.0, 1.0, 0.0]

    monkeypatch.setattr(
        "deep_research.research.grouping.get_embedding", fake_embedding
    )
    from deep_research.research.grouping import group_replacement_topics

    topics = [f"A{i}" for i in range(12)] + [f"B{i}" for i in range(6)]
    groups = await group_replacement_topics(run_context, topics)

    assert all(len(g) <= 5 for g in groups), [len(g) for g in groups]
    # No topics lost or duplicated.
    flat = [t for g in groups for t in g]
    assert sorted(flat) == sorted(topics)

"""Regression tests for research-layer bug fixes.

Covers:
  BUG 36 - extract_topic_relevant_info returns "" (a str) on empty input,
           not [] (a list), so callers' f-strings don't render "[]".
  BUG 21 - group_replacement_topics re-examines split halves until no group
           exceeds 5 topics.
  BUG 50 - _analyze_cycle_results crashed with TypeError when writing
           latest_dimension_coverage: coverage is a list[float] and the
           previous code wrapped it in dict() which expects (k, v) pairs.
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


# --- BUG 50: dimension-coverage write must not call dict() on a list -------

@pytest.mark.asyncio
async def test_analyze_cycle_results_writes_dimension_coverage_as_list(
    run_context, monkeypatch
):
    """``_analyze_cycle_results`` previously crashed with TypeError when
    ``research_dimensions["coverage"]`` (a ``list[float]``) reached the
    ``dict(...)`` wrapper at the end of the function. This pin asserts
    the write succeeds and that the stored shape is a list, consistent
    with every other writer in the engine.
    """
    from deep_research.orchestrator.phases import cycles as cycles_mod

    async def fake_chat_completions(model, messages, **kwargs):
        # The function only reads response_text() of the result and then
        # tolerates JSON-decode failures. An empty-text response is the
        # simplest stub that drives the function to its tail.
        return {"choices": [{"message": {"content": ""}}]}

    run_context.llm = type("StubLLM", (), {
        "chat_completions": staticmethod(fake_chat_completions),
    })()

    conv_state = run_context.state.get_state(run_context.conversation_id)
    coverage = [0.25, 0.5, 0.75]
    conv_state["research_dimensions"] = {
        "coverage": coverage,
        "eigenvectors": [],
        "eigenvalues": [],
        "explained_variance": [],
        "total_variance": 0.0,
        "dimensions": 3,
    }

    await cycles_mod._analyze_cycle_results(
        ctx=run_context,
        conv_state=conv_state,
        cycle=2,
        max_cycles=5,
        user_message="anything",
        cycle_results=[{"content": "x" * 300, "similarity": 0.4}],
        completed_topics=set(),
        irrelevant_topics=set(),
        active_outline=["t1"],
        all_topics=["t1"],
        cycle_summaries=[],
        results_history=[],
        outline_embedding=None,
    )

    stored = conv_state["latest_dimension_coverage"]
    assert isinstance(stored, list)
    assert stored == coverage
    # Confirm it's an independent list (mutation should not leak back).
    stored.append(99.0)
    assert conv_state["research_dimensions"]["coverage"] == coverage

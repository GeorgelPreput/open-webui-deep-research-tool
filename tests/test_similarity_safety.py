import math

import numpy as np
import pytest

from deep_research.semantics.similarity import _safe_normalize, calculate_query_similarity


def test_safe_normalize_unit_vector():
    out = _safe_normalize([3.0, 4.0])
    assert out is not None
    assert np.linalg.norm(out) == pytest.approx(1.0)


def test_safe_normalize_zero_vector_returns_none():
    assert _safe_normalize([0.0] * 768) is None


def test_safe_normalize_empty_returns_none():
    assert _safe_normalize([]) is None


def test_safe_normalize_non_finite_returns_none():
    assert _safe_normalize([float("nan"), 1.0]) is None
    assert _safe_normalize([float("inf"), 1.0]) is None


@pytest.mark.asyncio
async def test_calculate_query_similarity_zero_vector_is_finite(run_context):
    # A zero content embedding must NOT poison the score with NaN.
    sim = await calculate_query_similarity(
        run_context,
        content_embedding=[0.0] * 8,
        query_embedding=[1.0] * 8,
    )
    assert math.isfinite(sim)
    assert sim == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_calculate_query_similarity_normal_vectors(run_context):
    sim = await calculate_query_similarity(
        run_context,
        content_embedding=[1.0, 0.0, 0.0],
        query_embedding=[1.0, 0.0, 0.0],
    )
    assert math.isfinite(sim)
    # identical direction → high similarity
    assert sim > 0.9

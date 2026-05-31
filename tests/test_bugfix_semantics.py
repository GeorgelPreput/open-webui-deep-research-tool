"""Regression tests for semantics-layer bug fixes.

Covers:
  BUG 8  - TrajectoryAccumulator rejects dimension mismatch instead of a raw
           numpy broadcast error.
  BUG 17 - _trajectory_accumulators is an LRU-bounded map (no unbounded growth).
  BUG 28 - create_semantic_transformation guards a ~0 eigenvalue sum (no NaN/inf).
  BUG 15 - dimension coverage is clamped to [0, 1].
"""
import numpy as np
import pytest

from deep_research.core.state import TrajectoryAccumulator
from deep_research.semantics import trajectory as traj_mod


# --- BUG 8: TrajectoryAccumulator dimension safety --------------------------

def test_accumulator_dimension_mismatch_raises_valueerror():
    acc = TrajectoryAccumulator(4)
    with pytest.raises(ValueError):
        acc.add_cycle_data([[1.0] * 8], [[1.0] * 8])  # 8-d vectors into a 4-d acc


def test_accumulator_accumulates_and_returns_unit_trajectory():
    acc = TrajectoryAccumulator(3)
    acc.add_cycle_data([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    traj = acc.get_trajectory()
    assert traj is not None
    assert len(traj) == 3
    assert abs(float(np.linalg.norm(traj)) - 1.0) < 1e-6


def test_accumulator_empty_inputs_are_noop():
    acc = TrajectoryAccumulator(3)
    acc.add_cycle_data([], [])
    assert acc.count == 0
    assert acc.get_trajectory() is None


# --- BUG 17: bounded LRU map of accumulators --------------------------------

def test_trajectory_accumulators_lru_evicts_oldest(monkeypatch):
    monkeypatch.setattr(traj_mod, "_TRAJ_MAX_CONVERSATIONS", 3)
    traj_mod._trajectory_accumulators.clear()
    for i in range(5):
        traj_mod._store_trajectory_accumulator(f"c{i}", TrajectoryAccumulator(3))
    assert len(traj_mod._trajectory_accumulators) == 3
    assert "c0" not in traj_mod._trajectory_accumulators  # oldest evicted
    assert "c4" in traj_mod._trajectory_accumulators
    traj_mod._trajectory_accumulators.clear()


def test_trajectory_accumulators_move_to_end_on_restore(monkeypatch):
    monkeypatch.setattr(traj_mod, "_TRAJ_MAX_CONVERSATIONS", 3)
    traj_mod._trajectory_accumulators.clear()
    for k in ("a", "b", "c"):
        traj_mod._store_trajectory_accumulator(k, TrajectoryAccumulator(3))
    # Re-storing 'a' refreshes its recency; the next insert should evict 'b'.
    traj_mod._store_trajectory_accumulator("a", TrajectoryAccumulator(3))
    traj_mod._store_trajectory_accumulator("d", TrajectoryAccumulator(3))
    assert "b" not in traj_mod._trajectory_accumulators
    assert "a" in traj_mod._trajectory_accumulators
    assert "d" in traj_mod._trajectory_accumulators
    traj_mod._trajectory_accumulators.clear()


# --- BUG 28: variance-importance division guard -----------------------------

@pytest.mark.asyncio
async def test_transform_returns_none_on_zero_eigenvalue_sum(run_context):
    from deep_research.semantics.eigendecomposition import create_semantic_transformation

    eig = {"eigenvectors": [[1.0, 0.0], [0.0, 1.0]], "eigenvalues": [0.0, 0.0]}
    result = await create_semantic_transformation(run_context, eig, pdv=None)
    assert result is None


@pytest.mark.asyncio
async def test_transform_returns_none_on_negative_only_eigenvalues(run_context):
    from deep_research.semantics.eigendecomposition import create_semantic_transformation

    # Tiny negative eigenvalues (numerical noise from eigh) clip to 0 -> sum 0.
    eig = {"eigenvectors": [[1.0, 0.0], [0.0, 1.0]], "eigenvalues": [-1e-15, -2e-15]}
    result = await create_semantic_transformation(run_context, eig, pdv=None)
    assert result is None


@pytest.mark.asyncio
async def test_transform_returns_matrix_on_positive_eigenvalues(run_context):
    from deep_research.semantics.eigendecomposition import create_semantic_transformation

    eig = {"eigenvectors": [[1.0, 0.0], [0.0, 1.0]], "eigenvalues": [1.0, 0.5]}
    result = await create_semantic_transformation(run_context, eig, pdv=None)
    assert result is not None
    assert result["dimension"] == 2
    mat = np.array(result["matrix"])
    assert mat.shape == (2, 2)
    assert np.isfinite(mat).all()


# --- BUG 15: dimension coverage clamp ---------------------------------------

@pytest.mark.asyncio
async def test_dimension_coverage_clamped_to_one(run_context, monkeypatch):
    async def fake_embedding(ctx, text):
        # Large vector -> large projection -> would push coverage well past 1.0.
        return [10.0, 10.0, 10.0]

    monkeypatch.setattr(
        "deep_research.semantics.dimensions.get_embedding", fake_embedding
    )
    from deep_research.semantics.dimensions import update_dimension_coverage

    state = run_context.state.get_state(run_context.conversation_id)
    state["research_dimensions"] = {
        "coverage": [0.0, 0.0, 0.0],
        "eigenvectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }

    for _ in range(5):
        await update_dimension_coverage(run_context, "content goes here")

    coverage = run_context.state.get_state(run_context.conversation_id)[
        "research_dimensions"
    ]["coverage"]
    assert coverage  # non-empty
    assert all(0.0 <= c <= 1.0 for c in coverage)
    assert max(coverage) == pytest.approx(1.0)

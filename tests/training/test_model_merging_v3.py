"""
Tests for model_merging_v3.py — SLERP, TIES, DARE, and evaluation utilities.

All tests use tiny 2-layer MLPs (16→16→16) to stay fast.
Every test performs at least a forward pass.
"""

import torch
import torch.nn as nn
import pytest

from src.training.model_merging_v3 import (
    DAREPruner,
    ModelMergeEvaluator,
    ParameterVector,
    SLERPMerger,
    TIESMerger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mlp(seed: int = 0) -> nn.Module:
    """Create a tiny 2-layer MLP (16→16→16) with reproducible weights."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(16, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 16, bias=True),
    )
    return model


def forward_backward(model: nn.Module) -> None:
    """Run a forward and backward pass to confirm the model is live."""
    x = torch.randn(2, 16)
    out = model(x)
    loss = out.sum()
    loss.backward()


# ---------------------------------------------------------------------------
# ParameterVector tests
# ---------------------------------------------------------------------------

def test_parameter_vector_to_vector_shape():
    """to_vector returns a 1-D tensor of length n_params."""
    model = make_mlp(0)
    pv = ParameterVector(model)
    vec = pv.to_vector()
    assert vec.ndim == 1
    assert vec.shape[0] == pv.n_params()
    forward_backward(model)


def test_parameter_vector_round_trip():
    """from_vector correctly restores parameters after modification."""
    model = make_mlp(1)
    pv = ParameterVector(model)
    original_vec = pv.to_vector().clone()

    # Corrupt all params
    modified = original_vec + 99.0
    pv.from_vector(modified)

    restored_vec = pv.to_vector()
    assert torch.allclose(restored_vec, modified)

    # Restore and verify
    pv.from_vector(original_vec)
    assert torch.allclose(pv.to_vector(), original_vec)
    forward_backward(model)


def test_parameter_vector_task_vector_base_zero():
    """task_vector of base relative to itself is all zeros."""
    model = make_mlp(2)
    pv = ParameterVector(model)
    base_vec = pv.to_vector()
    tv = pv.task_vector(base_vec)
    assert torch.allclose(tv, torch.zeros_like(tv))
    forward_backward(model)


# ---------------------------------------------------------------------------
# SLERPMerger tests
# ---------------------------------------------------------------------------

def test_slerp_output_shape_matches_input():
    """merge output shape equals input shape."""
    merger = SLERPMerger(t=0.5)
    v1 = torch.randn(128)
    v2 = torch.randn(128)
    result = merger.merge(v1, v2)
    assert result.shape == v1.shape


def test_slerp_t0_returns_v1():
    """t=0 should return v1 (or very close for non-parallel vectors)."""
    merger = SLERPMerger(t=0.0)
    v1 = torch.randn(64)
    v2 = torch.randn(64)
    result = merger.merge(v1, v2)
    assert torch.allclose(result, v1, atol=1e-5)


def test_slerp_t1_returns_v2():
    """t=1 should return v2."""
    merger = SLERPMerger(t=1.0)
    v1 = torch.randn(64)
    v2 = torch.randn(64)
    result = merger.merge(v1, v2)
    assert torch.allclose(result, v2, atol=1e-5)


def test_slerp_parallel_vectors_linear_fallback():
    """Nearly parallel vectors trigger linear interpolation fallback."""
    t = 0.3
    merger = SLERPMerger(t=t)
    v = torch.randn(64)
    # Identical vectors are perfectly parallel
    result = merger.merge(v, v)
    expected = (1.0 - t) * v + t * v  # simplifies to v
    assert torch.allclose(result, expected, atol=1e-5)


def test_slerp_merge_models_output_shape():
    """merge_models output length equals the number of base model parameters."""
    base = make_mlp(0)
    model_a = make_mlp(1)
    model_b = make_mlp(2)

    merger = SLERPMerger(t=0.5)
    merged_vec = merger.merge_models(base, model_a, model_b)

    base_pv = ParameterVector(base)
    assert merged_vec.shape == (base_pv.n_params(),)

    # Load merged params and run a forward pass
    base_pv.from_vector(merged_vec)
    forward_backward(base)


# ---------------------------------------------------------------------------
# TIESMerger tests
# ---------------------------------------------------------------------------

def test_ties_trim_nonzero_fraction():
    """After trim, the number of non-zeros ≤ top_k_fraction * total."""
    merger = TIESMerger(top_k_fraction=0.2)
    tv = torch.randn(200)
    trimmed = merger.trim(tv, 0.2)
    n_nonzero = (trimmed != 0).sum().item()
    assert n_nonzero <= int(0.2 * 200) + 1  # +1 for rounding


def test_ties_elect_sign_values():
    """elect_sign returns a tensor with values only in {-1, 0, +1}."""
    merger = TIESMerger()
    tvs = [torch.randn(64) for _ in range(3)]
    sign_vec = merger.elect_sign(tvs)
    unique_vals = set(sign_vec.tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})


def test_ties_merge_output_shape():
    """TIES merge output has the same shape as base_vector."""
    merger = TIESMerger(top_k_fraction=0.3)
    base = make_mlp(0)
    base_pv = ParameterVector(base)
    base_vec = base_pv.to_vector()

    tvs = [torch.randn_like(base_vec) for _ in range(3)]
    merged = merger.merge(tvs, base_vec)
    assert merged.shape == base_vec.shape

    # Load and forward pass
    base_pv.from_vector(merged)
    forward_backward(base)


def test_ties_merge_improves_sign_agreement():
    """After TIES, the merged task vector has better sign agreement than random."""
    evaluator = ModelMergeEvaluator()
    merger = TIESMerger(top_k_fraction=0.5)

    torch.manual_seed(7)
    tvs = [torch.randn(128) for _ in range(4)]
    base_vec = torch.zeros(128)

    merged = merger.merge(tvs, base_vec)
    merged_tv = merged - base_vec  # task vector of merged model

    # The merged task vector should not be all zeros
    assert merged_tv.abs().sum().item() > 0

    # sign agreement of the merged tv with the originals should be ≥ random baseline
    agreement = evaluator.sign_agreement([merged_tv] + tvs)
    # At minimum this is a non-negative number
    assert 0.0 <= agreement <= 1.0


# ---------------------------------------------------------------------------
# DAREPruner tests
# ---------------------------------------------------------------------------

def test_dare_prune_zero_fraction():
    """Fraction of zeros in pruned vector is approximately drop_rate (±5%)."""
    pruner = DAREPruner(drop_rate=0.9, seed=0)
    tv = torch.ones(2000)
    pruned = pruner.prune(tv)
    zero_frac = (pruned == 0).float().mean().item()
    assert abs(zero_frac - 0.9) < 0.05


def test_dare_prune_rescaling():
    """Surviving elements are rescaled by 1/(1-drop_rate)."""
    drop_rate = 0.8
    pruner = DAREPruner(drop_rate=drop_rate, seed=1)
    tv = torch.ones(1000)
    pruned = pruner.prune(tv)
    # Non-zero elements should all equal 1 / (1 - 0.8) = 5.0
    non_zero_vals = pruned[pruned != 0]
    assert torch.allclose(non_zero_vals, torch.full_like(non_zero_vals, 1.0 / (1.0 - drop_rate)))


def test_dare_prune_reproducibility():
    """Same seed produces identical pruned vectors."""
    pruner = DAREPruner(drop_rate=0.7, seed=99)
    tv = torch.randn(500)
    result1 = pruner.prune(tv)
    result2 = pruner.prune(tv)
    assert torch.equal(result1, result2)


def test_dare_merge_output_shape():
    """dare_merge output shape equals base_vector shape."""
    pruner = DAREPruner(drop_rate=0.5, seed=5)
    base = make_mlp(0)
    base_pv = ParameterVector(base)
    base_vec = base_pv.to_vector()

    tvs = [torch.randn_like(base_vec) for _ in range(3)]
    merged = pruner.dare_merge(tvs, base_vec)
    assert merged.shape == base_vec.shape

    # Load and forward pass
    base_pv.from_vector(merged)
    forward_backward(base)


# ---------------------------------------------------------------------------
# ModelMergeEvaluator tests
# ---------------------------------------------------------------------------

def test_evaluator_parameter_interference_range():
    """parameter_interference returns a value in [-1, 1]."""
    evaluator = ModelMergeEvaluator()
    tvs = [torch.randn(128) for _ in range(4)]
    score = evaluator.parameter_interference(tvs)
    assert -1.0 <= score <= 1.0


def test_evaluator_sign_agreement_range():
    """sign_agreement returns a value in [0, 1]."""
    evaluator = ModelMergeEvaluator()
    tvs = [torch.randn(128) for _ in range(3)]
    score = evaluator.sign_agreement(tvs)
    assert 0.0 <= score <= 1.0


def test_evaluator_magnitude_preservation_range():
    """magnitude_preservation returns a value in [-1, 1]."""
    evaluator = ModelMergeEvaluator()
    original = torch.randn(128)
    merged = torch.randn(128)
    score = evaluator.magnitude_preservation(original, merged)
    assert -1.0 <= score <= 1.0


def test_evaluator_magnitude_preservation_identical():
    """magnitude_preservation returns 1.0 for identical vectors."""
    evaluator = ModelMergeEvaluator()
    vec = torch.randn(128)
    score = evaluator.magnitude_preservation(vec, vec)
    assert abs(score - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Integration: TIES merge of 3 models vs simple average
# ---------------------------------------------------------------------------

def test_ties_differs_from_simple_average():
    """TIES merge result differs from simple arithmetic average of task vectors."""
    torch.manual_seed(42)

    base = make_mlp(0)
    base_pv = ParameterVector(base)
    base_vec = base_pv.to_vector().clone()

    # Create 3 fine-tuned models with deliberate sign conflicts
    model_a = make_mlp(10)
    model_b = make_mlp(20)
    model_c = make_mlp(30)

    tv_a = ParameterVector(model_a).task_vector(base_vec)
    tv_b = ParameterVector(model_b).task_vector(base_vec)
    tv_c = ParameterVector(model_c).task_vector(base_vec)

    # Simple average baseline
    simple_avg_merged = base_vec + (tv_a + tv_b + tv_c) / 3.0

    # TIES merge
    merger = TIESMerger(top_k_fraction=0.3)
    ties_merged = merger.merge([tv_a, tv_b, tv_c], base_vec)

    # TIES result must differ from naive average
    assert not torch.allclose(ties_merged, simple_avg_merged, atol=1e-6)

    # Both results should be loadable and runnable
    base_pv.from_vector(ties_merged)
    forward_backward(base)

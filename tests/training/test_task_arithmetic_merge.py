"""Tests for task_arithmetic_merge.py (Ilharco et al. 2023).

All tests use small nn.Linear or nn.Sequential models (no large dependencies).
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.training.task_arithmetic_merge import (
    MergeConfig,
    TaskArithmeticMerger,
    TaskVector,
    add_task_vectors,
    apply_task_vector,
    extract_task_vector,
    negate_task_vector,
    resolve_conflicts,
    scale_task_vector,
    task_vector_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear(in_f: int = 8, out_f: int = 4, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def _seq(in_f: int = 8, hidden: int = 16, out_f: int = 4, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(), nn.Linear(hidden, out_f))


def _perturb(model: nn.Module, scale: float = 0.1, seed: int = 42) -> nn.Module:
    """Return a deep-copied model with weights perturbed by Gaussian noise."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for p in m.parameters():
        p.data += torch.randn_like(p) * scale
    return m


def _state(model: nn.Module):
    return model.state_dict()


# ---------------------------------------------------------------------------
# 1. extract_task_vector returns dict with correct keys
# ---------------------------------------------------------------------------

def test_extract_task_vector_keys():
    pre = _linear()
    ft = _perturb(pre)
    tv = extract_task_vector(_state(pre), _state(ft))
    assert set(tv.keys()) == set(pre.state_dict().keys())


# ---------------------------------------------------------------------------
# 2. extract_task_vector: τ of identical models is all zeros
# ---------------------------------------------------------------------------

def test_extract_task_vector_identical_is_zero():
    pre = _linear()
    tv = extract_task_vector(_state(pre), _state(pre))
    for k, delta in tv.items():
        assert torch.all(delta == 0), f"Expected zero delta for {k}"


# ---------------------------------------------------------------------------
# 3. apply_task_vector with scaling=0.0 returns pretrained weights
# ---------------------------------------------------------------------------

def test_apply_task_vector_scaling_zero():
    pre = _linear(seed=1)
    ft = _perturb(pre, seed=10)
    tv = extract_task_vector(_state(pre), _state(ft))
    merged = apply_task_vector(_state(pre), tv, scaling=0.0)
    for k in pre.state_dict():
        assert torch.allclose(merged[k], pre.state_dict()[k]), (
            f"Weights differ at {k} with scaling=0"
        )


# ---------------------------------------------------------------------------
# 4. apply_task_vector with scaling=1.0 returns finetuned weights
# ---------------------------------------------------------------------------

def test_apply_task_vector_scaling_one():
    pre = _linear(seed=2)
    ft = _perturb(pre, seed=20)
    tv = extract_task_vector(_state(pre), _state(ft))
    merged = apply_task_vector(_state(pre), tv, scaling=1.0)
    for k in ft.state_dict():
        assert torch.allclose(merged[k].float(), ft.state_dict()[k].float(), atol=1e-6), (
            f"Merged weights ≠ finetuned at {k}"
        )


# ---------------------------------------------------------------------------
# 5. add_task_vectors with 2 identical vectors = 2x one vector (uniform weights)
# ---------------------------------------------------------------------------

def test_add_task_vectors_two_identical():
    pre = _linear(seed=3)
    ft = _perturb(pre, seed=30)
    tv = extract_task_vector(_state(pre), _state(ft))
    combined = add_task_vectors([tv, tv])
    # With 2 identical vectors, uniform weight = 0.5 each → sum = 1.0 * tv
    # BUT uniform means each weighted 1/n=0.5, so result = 0.5*tv + 0.5*tv = tv
    # Test that 2*tv == add_task_vectors([tv, tv], weights=[1,1])
    combined_unit = add_task_vectors([tv, tv], weights=[1.0, 1.0])
    for k in tv:
        assert torch.allclose(combined_unit[k], 2.0 * tv[k], atol=1e-6), (
            f"Expected 2x vector at {k}"
        )


# ---------------------------------------------------------------------------
# 6. add_task_vectors uniform weights = mean
# ---------------------------------------------------------------------------

def test_add_task_vectors_uniform_is_mean():
    pre = _linear(seed=4)
    ft1 = _perturb(pre, scale=0.1, seed=41)
    ft2 = _perturb(pre, scale=0.1, seed=42)
    tv1 = extract_task_vector(_state(pre), _state(ft1))
    tv2 = extract_task_vector(_state(pre), _state(ft2))

    # Uniform (default) weights: result[k] = 0.5*tv1[k] + 0.5*tv2[k]
    result = add_task_vectors([tv1, tv2])
    for k in tv1:
        expected = 0.5 * tv1[k] + 0.5 * tv2[k]
        assert torch.allclose(result[k], expected, atol=1e-6), (
            f"Uniform mean mismatch at {k}"
        )


# ---------------------------------------------------------------------------
# 7. negate_task_vector flips signs
# ---------------------------------------------------------------------------

def test_negate_task_vector():
    pre = _linear(seed=5)
    ft = _perturb(pre, seed=50)
    tv = extract_task_vector(_state(pre), _state(ft))
    neg = negate_task_vector(tv)
    for k in tv:
        assert torch.allclose(neg[k], -tv[k], atol=1e-6), (
            f"Negation mismatch at {k}"
        )


# ---------------------------------------------------------------------------
# 8. scale_task_vector doubles all values with scale=2.0
# ---------------------------------------------------------------------------

def test_scale_task_vector_double():
    pre = _linear(seed=6)
    ft = _perturb(pre, seed=60)
    tv = extract_task_vector(_state(pre), _state(ft))
    scaled = scale_task_vector(tv, 2.0)
    for k in tv:
        assert torch.allclose(scaled[k], 2.0 * tv[k], atol=1e-6), (
            f"Scaling mismatch at {k}"
        )


# ---------------------------------------------------------------------------
# 9. resolve_conflicts "mean" = add_task_vectors with uniform weights
# ---------------------------------------------------------------------------

def test_resolve_conflicts_mean_equals_uniform_add():
    pre = _linear(seed=7)
    ft1 = _perturb(pre, scale=0.1, seed=71)
    ft2 = _perturb(pre, scale=0.1, seed=72)
    tv1 = extract_task_vector(_state(pre), _state(ft1))
    tv2 = extract_task_vector(_state(pre), _state(ft2))

    via_resolve = resolve_conflicts([tv1, tv2], method="mean")
    via_add = add_task_vectors([tv1, tv2])
    for k in tv1:
        assert torch.allclose(via_resolve[k], via_add[k], atol=1e-6), (
            f"'mean' resolve ≠ uniform add at {k}"
        )


# ---------------------------------------------------------------------------
# 10. resolve_conflicts "ties": conflicting sign params zeroed out
# ---------------------------------------------------------------------------

def test_resolve_conflicts_ties_zeros_conflicts():
    # Construct task vectors with guaranteed sign conflict.
    # tv1 has large positive delta, tv2 has large negative delta → no majority
    pre = _linear(seed=8)
    pre_state = _state(pre)

    # All-positive delta
    pos_state = {k: v.clone() + 5.0 for k, v in pre_state.items()}
    # All-negative delta
    neg_state = {k: v.clone() - 5.0 for k, v in pre_state.items()}

    tv_pos = extract_task_vector(pre_state, pos_state)
    tv_neg = extract_task_vector(pre_state, neg_state)

    result = resolve_conflicts([tv_pos, tv_neg], method="ties")
    for k, v in result.items():
        # With equal positive and negative, no majority → should be zeroed
        assert torch.all(v == 0), (
            f"Expected zeros due to sign conflict at {k}, got {v}"
        )


# ---------------------------------------------------------------------------
# 11. resolve_conflicts "dare": output has some zeros (dropout effect)
# ---------------------------------------------------------------------------

def test_resolve_conflicts_dare_has_zeros():
    torch.manual_seed(0)
    pre = _seq(seed=0)
    ft = _perturb(pre, scale=0.5, seed=1)
    tv = extract_task_vector(_state(pre), _state(ft))
    # Even a single task vector goes through DARE dropout
    result = resolve_conflicts([tv], method="dare")
    total_elements = sum(v.numel() for v in result.values())
    zero_elements = sum((v == 0).sum().item() for v in result.values())
    # With p=0.5 and enough elements, expect some zeros
    assert zero_elements > 0, "DARE should produce some zeros via dropout"
    assert zero_elements < total_elements, "DARE should not zero everything"


# ---------------------------------------------------------------------------
# 12. task_vector_similarity(tv, tv) ≈ 1.0
# ---------------------------------------------------------------------------

def test_task_vector_similarity_self():
    pre = _linear(seed=9)
    ft = _perturb(pre, seed=90)
    tv = extract_task_vector(_state(pre), _state(ft))
    sim = task_vector_similarity(tv, tv)
    assert abs(sim - 1.0) < 1e-5, f"Self-similarity should be ≈1.0, got {sim}"


# ---------------------------------------------------------------------------
# 13. task_vector_similarity(tv, -tv) ≈ -1.0
# ---------------------------------------------------------------------------

def test_task_vector_similarity_negated():
    pre = _linear(seed=10)
    ft = _perturb(pre, seed=100)
    tv = extract_task_vector(_state(pre), _state(ft))
    neg = negate_task_vector(tv)
    sim = task_vector_similarity(tv, neg)
    assert abs(sim - (-1.0)) < 1e-5, f"Anti-similarity should be ≈-1.0, got {sim}"


# ---------------------------------------------------------------------------
# 14. TaskArithmeticMerger.n_tasks increments after add_finetuned
# ---------------------------------------------------------------------------

def test_merger_n_tasks_increments():
    pre = _linear(seed=11)
    merger = TaskArithmeticMerger(pre)
    assert merger.n_tasks == 0
    merger.add_finetuned(_perturb(pre, seed=111))
    assert merger.n_tasks == 1
    merger.add_finetuned(_perturb(pre, seed=112))
    assert merger.n_tasks == 2


# ---------------------------------------------------------------------------
# 15. TaskArithmeticMerger.merge returns nn.Module
# ---------------------------------------------------------------------------

def test_merger_merge_returns_module():
    pre = _linear(seed=12)
    merger = TaskArithmeticMerger(pre)
    merger.add_finetuned(_perturb(pre, seed=121))
    result = merger.merge()
    assert isinstance(result, nn.Module), "merge() should return an nn.Module"
    # Should be a different object
    assert result is not pre


# ---------------------------------------------------------------------------
# 16. TaskArithmeticMerger.forget returns model with negated task vector applied
# ---------------------------------------------------------------------------

def test_merger_forget_negates_task_vector():
    pre = _linear(seed=13)
    ft = _perturb(pre, seed=130)
    merger = TaskArithmeticMerger(pre)

    forgotten = merger.forget(ft, scaling=1.0)

    # θ_result = θ_pre - 1.0 * (θ_ft - θ_pre) = 2*θ_pre - θ_ft
    pre_state = pre.state_dict()
    ft_state = ft.state_dict()
    forgotten_state = forgotten.state_dict()

    for k in pre_state:
        expected = 2.0 * pre_state[k].float() - ft_state[k].float()
        assert torch.allclose(forgotten_state[k].float(), expected, atol=1e-5), (
            f"Forget mismatch at {k}"
        )


# ---------------------------------------------------------------------------
# 17. merge does not modify original pretrained model
# ---------------------------------------------------------------------------

def test_merger_merge_does_not_mutate_pretrained():
    pre = _linear(seed=14)
    pre_state_before = {k: v.clone() for k, v in pre.state_dict().items()}
    merger = TaskArithmeticMerger(pre)
    merger.add_finetuned(_perturb(pre, seed=141))
    _ = merger.merge()
    for k in pre_state_before:
        assert torch.equal(pre.state_dict()[k], pre_state_before[k]), (
            f"Pretrained model was mutated at {k}"
        )


# ---------------------------------------------------------------------------
# 18. Merger with MergeConfig scaling=0.0 returns pretrained weights
# ---------------------------------------------------------------------------

def test_merger_scaling_zero_returns_pretrained():
    pre = _linear(seed=15)
    ft = _perturb(pre, seed=150)
    cfg = MergeConfig(scaling=0.0)
    merger = TaskArithmeticMerger(pre, config=cfg)
    merger.add_finetuned(ft)
    merged = merger.merge()

    for k in pre.state_dict():
        assert torch.allclose(
            merged.state_dict()[k].float(),
            pre.state_dict()[k].float(),
            atol=1e-6,
        ), f"Expected pretrained weights at {k} when scaling=0"

"""Tests for task vector arithmetic (Ilharco et al. 2022, arXiv:2212.04089)."""
import copy

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.task_vector import (
    TaskVector,
    apply_task_vectors,
    extract_task_vector,
    negation_edit,
)


def _small_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def _finetuned_model(pretrained: AureliusTransformer, seed: int = 42) -> AureliusTransformer:
    """Create a fine-tuned variant by perturbing weights."""
    finetuned = copy.deepcopy(pretrained)
    torch.manual_seed(seed)
    for p in finetuned.parameters():
        p.data += torch.randn_like(p) * 0.05
    return finetuned


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_task_vector_from_models():
    """TaskVector created from (pretrained, finetuned) has expected keys."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)

    param_names = {name for name, _ in pretrained.named_parameters()}
    assert set(tv.vector.keys()) == param_names


def test_task_vector_from_dict():
    """TaskVector created from a pre-computed dict stores the dict as-is."""
    pretrained = _small_model()
    vec = {name: torch.zeros_like(p) for name, p in pretrained.named_parameters()}
    tv = TaskVector(vector=vec)
    assert set(tv.vector.keys()) == set(vec.keys())


def test_task_vector_zero_from_same_model():
    """TV of (model, model) yields all-zero diffs."""
    model = _small_model()
    tv = TaskVector(model, model)
    for name, diff in tv.vector.items():
        assert torch.all(diff == 0), f"Non-zero diff for {name}"


# ---------------------------------------------------------------------------
# Arithmetic tests
# ---------------------------------------------------------------------------

def test_task_vector_addition():
    """(tv1 + tv2).vector keys match tv1.vector keys."""
    pretrained = _small_model()
    ft1 = _finetuned_model(pretrained, seed=1)
    ft2 = _finetuned_model(pretrained, seed=2)
    tv1 = TaskVector(pretrained, ft1)
    tv2 = TaskVector(pretrained, ft2)
    tv_sum = tv1 + tv2
    assert set(tv_sum.vector.keys()) == set(tv1.vector.keys())


def test_task_vector_scalar_multiply():
    """(2.0 * tv).vector[k] == 2.0 * tv.vector[k] for all k."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)
    tv2 = 2.0 * tv
    for name in tv.vector:
        assert torch.allclose(tv2.vector[name], 2.0 * tv.vector[name])


def test_task_vector_negation():
    """(-tv).vector[k] == -tv.vector[k] for all k."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)
    neg_tv = -tv
    for name in tv.vector:
        assert torch.allclose(neg_tv.vector[name], -tv.vector[name])


# ---------------------------------------------------------------------------
# Norm test
# ---------------------------------------------------------------------------

def test_task_vector_norm_nonneg():
    """norm() >= 0."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)
    assert tv.norm() >= 0.0


# ---------------------------------------------------------------------------
# apply() tests
# ---------------------------------------------------------------------------

def test_apply_returns_new_model():
    """apply() returns a different Python object from pretrained."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)
    new_model = tv.apply(pretrained)
    assert new_model is not pretrained


def test_apply_zero_tv_same_as_pretrained():
    """Applying a zero TV with coef=1 gives weights identical to pretrained."""
    pretrained = _small_model()
    zero_vec = {name: torch.zeros_like(p) for name, p in pretrained.named_parameters()}
    tv = TaskVector(vector=zero_vec)
    new_model = tv.apply(pretrained, scaling_coef=1.0)

    for (name, p_orig), (_, p_new) in zip(
        pretrained.named_parameters(), new_model.named_parameters()
    ):
        assert torch.allclose(p_orig, p_new), f"Weights differ for {name}"


def test_apply_nonzero_tv_changes_weights():
    """Applying a nonzero TV changes at least one weight."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    tv = TaskVector(pretrained, finetuned)
    new_model = tv.apply(pretrained, scaling_coef=1.0)

    any_changed = False
    for (_, p_orig), (_, p_new) in zip(
        pretrained.named_parameters(), new_model.named_parameters()
    ):
        if not torch.equal(p_orig, p_new):
            any_changed = True
            break
    assert any_changed


# ---------------------------------------------------------------------------
# apply_task_vectors() test
# ---------------------------------------------------------------------------

def test_apply_task_vectors_combines():
    """Two TVs applied together equal TV sum applied once."""
    pretrained = _small_model()
    ft1 = _finetuned_model(pretrained, seed=1)
    ft2 = _finetuned_model(pretrained, seed=2)
    tv1 = TaskVector(pretrained, ft1)
    tv2 = TaskVector(pretrained, ft2)

    combined = apply_task_vectors(pretrained, [tv1, tv2], scaling_coef=1.0)
    tv_sum = tv1 + tv2
    expected = tv_sum.apply(pretrained, scaling_coef=1.0)

    for (name, p_comb), (_, p_exp) in zip(
        combined.named_parameters(), expected.named_parameters()
    ):
        assert torch.allclose(p_comb, p_exp, atol=1e-6), f"Mismatch at {name}"


# ---------------------------------------------------------------------------
# negation_edit() test
# ---------------------------------------------------------------------------

def test_negation_edit_removes_delta():
    """negation_edit with coef=1 gives 2*pretrained - finetuned weights."""
    pretrained = _small_model()
    finetuned = _finetuned_model(pretrained)
    result = negation_edit(pretrained, finetuned, scaling_coef=1.0)

    for (name, p_pre), (_, p_ft), (_, p_res) in zip(
        pretrained.named_parameters(),
        finetuned.named_parameters(),
        result.named_parameters(),
    ):
        expected = 2.0 * p_pre.data - p_ft.data
        assert torch.allclose(p_res.data.float(), expected.float(), atol=1e-5), (
            f"negation_edit mismatch at {name}"
        )

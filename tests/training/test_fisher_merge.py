"""Tests for Fisher information-weighted model merging."""

from __future__ import annotations

import copy
from collections.abc import Generator

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.fisher_merge import (
    FisherMergeResult,
    analyze_fisher_merge,
    compute_fisher,
    fisher_merge,
    fisher_merge_models,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_model(seed: int = 0) -> AureliusTransformer:
    """Return a tiny AureliusTransformer for fast tests."""
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


def _make_dataloader(
    n_batches: int = 5,
    batch_size: int = 2,
    seq_len: int = 8,
    vocab_size: int = 256,
    seed: int = 42,
) -> Generator:
    """Yield random (input_ids, labels) batches."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    for _ in range(n_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng)
        yield input_ids, labels


def _param_names(model: nn.Module) -> set[str]:
    """Return set of trainable parameter names (excludes tied aliases in state_dict)."""
    return {name for name, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# compute_fisher tests
# ---------------------------------------------------------------------------


def test_compute_fisher_returns_dict():
    """compute_fisher returns a dict with same keys as model.named_parameters()."""
    model = _small_model()
    dl = _make_dataloader(n_batches=3)
    fisher = compute_fisher(model, dl, n_batches=3)

    expected_keys = _param_names(model)
    assert set(fisher.keys()) == expected_keys


def test_compute_fisher_shapes_match():
    """Each Fisher tensor has the same shape as the corresponding parameter."""
    model = _small_model()
    dl = _make_dataloader(n_batches=3)
    fisher = compute_fisher(model, dl, n_batches=3)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher, f"Missing Fisher for {name}"
            assert fisher[name].shape == param.shape, (
                f"{name}: fisher shape {fisher[name].shape} != param shape {param.shape}"
            )


def test_compute_fisher_positive():
    """All Fisher values must be >= 0 (they are squared gradients)."""
    model = _small_model()
    dl = _make_dataloader(n_batches=3)
    fisher = compute_fisher(model, dl, n_batches=3, normalize=False)

    for name, fval in fisher.items():
        assert (fval >= 0).all(), f"Negative Fisher values for {name}"


def test_compute_fisher_normalized():
    """With normalize=True, max Fisher value per parameter must be <= 1.0."""
    model = _small_model()
    dl = _make_dataloader(n_batches=3)
    fisher = compute_fisher(model, dl, n_batches=3, normalize=True)

    for name, fval in fisher.items():
        assert fval.max().item() <= 1.0 + 1e-6, f"{name} max Fisher = {fval.max().item()} > 1.0"


# ---------------------------------------------------------------------------
# fisher_merge tests
# ---------------------------------------------------------------------------


def test_fisher_merge_output_shape():
    """Merged state dict has same parameter shapes as the input models."""
    m1 = _small_model(0)
    m2 = _small_model(1)
    dl1 = _make_dataloader(seed=0)
    dl2 = _make_dataloader(seed=1)

    f1 = compute_fisher(m1, dl1, n_batches=3)
    f2 = compute_fisher(m2, dl2, n_batches=3)

    merged_state = fisher_merge([m1, m2], [f1, f2])

    ref_state = m1.state_dict()
    for name, tensor in merged_state.items():
        assert tensor.shape == ref_state[name].shape, (
            f"{name}: merged shape {tensor.shape} != expected {ref_state[name].shape}"
        )


def test_fisher_merge_interpolates():
    """Merged params should lie between model A and B values (for 2-model merge).

    With uniform Fisher weights, the result should be the arithmetic mean,
    which lies element-wise between the two models' parameters.
    """
    torch.manual_seed(0)
    m1 = _small_model(0)
    m2 = copy.deepcopy(m1)

    # Perturb m2 significantly
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 0.5

    # Uniform Fisher: result should be arithmetic mean (between m1 and m2)
    f_uniform = {
        name: torch.ones_like(param) for name, param in m1.named_parameters() if param.requires_grad
    }

    merged_state = fisher_merge([m1, m2], [f_uniform, f_uniform], normalize=False)

    s1 = m1.state_dict()
    s2 = m2.state_dict()
    param_keys = _param_names(m1)

    # Check one floating-point parameter from the named params
    for name in merged_state:
        if name not in param_keys:
            continue
        if merged_state[name].dtype.is_floating_point and not merged_state[name].is_complex():
            merged = merged_state[name].float()
            lo = torch.minimum(s1[name].float(), s2[name].float())
            hi = torch.maximum(s1[name].float(), s2[name].float())
            # Merged values should be between lo and hi (within float tolerance)
            assert ((merged >= lo - 1e-4) & (merged <= hi + 1e-4)).all(), (
                f"Merged param {name} has values outside [model_A, model_B] range"
            )
            break  # one parameter check is enough


def test_fisher_merge_uniform_fisher():
    """With equal Fisher weights, result equals the arithmetic mean."""
    torch.manual_seed(42)
    m1 = _small_model(0)
    m2 = copy.deepcopy(m1)
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 0.3

    # Uniform Fisher (all 1s) — only for true named parameters
    uniform_fisher = {
        name: torch.ones_like(param) for name, param in m1.named_parameters() if param.requires_grad
    }

    merged_state = fisher_merge([m1, m2], [uniform_fisher, uniform_fisher], normalize=False)

    s1 = m1.state_dict()
    s2 = m2.state_dict()
    param_keys = _param_names(m1)

    for name in merged_state:
        # Skip state_dict entries that are tied aliases (not real parameters)
        if name not in param_keys:
            continue
        if s1[name].dtype.is_floating_point and not s1[name].is_complex():
            expected = (s1[name].float() + s2[name].float()) / 2.0
            actual = merged_state[name].float()
            assert torch.allclose(actual, expected, atol=1e-5), (
                f"{name}: merged != arithmetic mean under uniform Fisher"
            )


def test_fisher_merge_skewed_fisher():
    """A model with higher Fisher weights should dominate the merge."""
    torch.manual_seed(7)
    m1 = _small_model(0)
    m2 = copy.deepcopy(m1)
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 1.0

    param_keys = _param_names(m1)

    # m1 gets very high Fisher, m2 gets near-zero -- only for true named params
    high_fisher = {
        name: torch.full_like(param, 1000.0)
        for name, param in m1.named_parameters()
        if param.requires_grad
    }
    low_fisher = {
        name: torch.full_like(param, 1e-6)
        for name, param in m2.named_parameters()
        if param.requires_grad
    }

    merged_state = fisher_merge([m1, m2], [high_fisher, low_fisher], normalize=False)

    s1 = m1.state_dict()

    # Merged should be very close to m1 (the dominant-Fisher model)
    for name in merged_state:
        # Skip tied aliases that aren't in Fisher dicts
        if name not in param_keys:
            continue
        if s1[name].dtype.is_floating_point and not s1[name].is_complex():
            assert torch.allclose(merged_state[name].float(), s1[name].float(), atol=1e-2), (
                f"{name}: skewed-Fisher merge did not favour the high-Fisher model"
            )


def test_fisher_merge_two_identical():
    """Merging two identical models yields the same params."""
    m = _small_model(0)
    m_copy = copy.deepcopy(m)

    uniform_fisher = {
        name: torch.ones_like(param) for name, param in m.named_parameters() if param.requires_grad
    }

    merged_state = fisher_merge([m, m_copy], [uniform_fisher, uniform_fisher])

    ref = m.state_dict()
    param_keys = _param_names(m)
    for name in merged_state:
        if name not in param_keys:
            continue
        if ref[name].dtype.is_floating_point:
            assert torch.allclose(merged_state[name].float(), ref[name].float(), atol=1e-5), (
                f"{name}: merging identical models changed parameter values"
            )


# ---------------------------------------------------------------------------
# fisher_merge_models tests
# ---------------------------------------------------------------------------


def test_fisher_merge_models_returns_model():
    """fisher_merge_models returns an nn.Module with a valid forward pass."""
    base = _small_model(0)
    m1 = copy.deepcopy(base)
    m2 = copy.deepcopy(base)
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 0.1

    dl1 = list(_make_dataloader(n_batches=3, seed=10))
    dl2 = list(_make_dataloader(n_batches=3, seed=11))

    merged = fisher_merge_models(base, [m1, m2], [dl1, dl2], n_batches=3)

    assert isinstance(merged, nn.Module)

    input_ids = torch.randint(0, 256, (1, 8))
    _, logits, _ = merged(input_ids)
    assert logits.shape == (1, 8, 256)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# analyze_fisher_merge tests
# ---------------------------------------------------------------------------


def test_analyze_fisher_merge_fields():
    """FisherMergeResult has all required fields with sensible values."""
    m1 = _small_model(0)
    m2 = copy.deepcopy(m1)

    f1 = compute_fisher(m1, _make_dataloader(n_batches=2), n_batches=2)
    f2 = compute_fisher(m2, _make_dataloader(n_batches=2), n_batches=2)

    result = analyze_fisher_merge([m1, m2], [f1, f2])

    assert isinstance(result, FisherMergeResult)
    assert result.n_models == 2
    assert result.n_params > 0
    assert result.mean_fisher_weight >= 0.0
    assert result.merge_time_ms >= 0.0
    assert isinstance(result.param_names, list)
    assert len(result.param_names) == result.n_params

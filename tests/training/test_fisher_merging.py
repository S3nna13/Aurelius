"""Tests for fisher_merging.py -- Fisher information-weighted model merging."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.fisher_merging import (
    FisherMergeConfig,
    FisherMerger,
    apply_state_dict,
    compute_diagonal_fisher,
    fisher_merge_two,
    regmean_merge,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
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
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _make_data(n: int = 4, seq_len: int = 8, vocab_size: int = 256, seed: int = 0) -> list[Tensor]:
    """Return list of random input_ids tensors."""
    torch.manual_seed(seed)
    return [torch.randint(0, vocab_size, (1, seq_len)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test 1: FisherMergeConfig defaults
# ---------------------------------------------------------------------------


def test_fisher_merge_config_defaults():
    cfg = FisherMergeConfig()
    assert cfg.n_samples == 64
    assert cfg.fisher_floor == pytest.approx(1e-6)
    assert cfg.normalize is True
    assert cfg.merge_strategy == "fisher_weighted"


# ---------------------------------------------------------------------------
# Test 2: compute_diagonal_fisher returns dict with param names
# ---------------------------------------------------------------------------


def test_compute_diagonal_fisher_returns_dict():
    model = _small_model(0)
    data = _make_data(4)
    fisher = compute_diagonal_fisher(model, data, n_samples=4)
    assert isinstance(fisher, dict)
    expected_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3: compute_diagonal_fisher all values >= 0
# ---------------------------------------------------------------------------


def test_compute_diagonal_fisher_nonnegative():
    model = _small_model(0)
    data = _make_data(4)
    fisher = compute_diagonal_fisher(model, data, n_samples=4)
    for name, fval in fisher.items():
        assert (fval >= 0).all(), f"Negative Fisher values for parameter '{name}'"


# ---------------------------------------------------------------------------
# Test 4: compute_diagonal_fisher shape matches parameter shape
# ---------------------------------------------------------------------------


def test_compute_diagonal_fisher_shapes():
    model = _small_model(0)
    data = _make_data(4)
    fisher = compute_diagonal_fisher(model, data, n_samples=4)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher
            assert fisher[name].shape == param.shape, (
                f"{name}: fisher shape {fisher[name].shape} != param shape {param.shape}"
            )


# ---------------------------------------------------------------------------
# Test 5: fisher_merge_two returns state dict
# ---------------------------------------------------------------------------


def test_fisher_merge_two_returns_state_dict():
    model_a = _small_model(0)
    model_b = _small_model(1)
    data = _make_data(4)
    fa = compute_diagonal_fisher(model_a, data, n_samples=4)
    fb = compute_diagonal_fisher(model_b, data, n_samples=4)
    cfg = FisherMergeConfig()
    merged = fisher_merge_two(model_a, model_b, fa, fb, cfg)
    assert isinstance(merged, dict)
    # Should have at least as many keys as the state dict
    assert len(merged) >= len({n for n, _ in model_a.named_parameters() if True})


# ---------------------------------------------------------------------------
# Test 6: fisher_merge_two merged values between model_a and model_b
# ---------------------------------------------------------------------------


def test_fisher_merge_two_values_between_models():
    torch.manual_seed(7)
    model_a = _small_model(0)
    model_b = copy.deepcopy(model_a)
    for p in model_b.parameters():
        p.data += torch.randn_like(p) * 0.5

    # Use equal Fisher so merged = arithmetic mean which is between a and b
    uniform_fa = {n: torch.ones_like(p) for n, p in model_a.named_parameters() if p.requires_grad}
    uniform_fb = {n: torch.ones_like(p) for n, p in model_b.named_parameters() if p.requires_grad}
    cfg = FisherMergeConfig(normalize=False)
    merged = fisher_merge_two(model_a, model_b, uniform_fa, uniform_fb, cfg)

    sa = model_a.state_dict()
    sb = model_b.state_dict()
    param_keys = {n for n, p in model_a.named_parameters() if p.requires_grad}

    checked = 0
    for name in merged:
        if name not in param_keys:
            continue
        if not merged[name].dtype.is_floating_point or merged[name].is_complex():
            continue
        m = merged[name].float()
        lo = torch.minimum(sa[name].float(), sb[name].float())
        hi = torch.maximum(sa[name].float(), sb[name].float())
        assert ((m >= lo - 1e-4) & (m <= hi + 1e-4)).all(), (
            f"Merged param '{name}' outside [model_a, model_b] range"
        )
        checked += 1
        if checked >= 3:
            break
    assert checked > 0, "No floating-point parameters were checked"


# ---------------------------------------------------------------------------
# Test 7: fisher_merge_two equal Fisher -> midpoint merge
# ---------------------------------------------------------------------------


def test_fisher_merge_two_equal_fisher_midpoint():
    torch.manual_seed(42)
    model_a = _small_model(0)
    model_b = copy.deepcopy(model_a)
    for p in model_b.parameters():
        p.data += torch.randn_like(p) * 0.3

    uniform_f = {n: torch.ones_like(p) for n, p in model_a.named_parameters() if p.requires_grad}
    cfg = FisherMergeConfig(normalize=False)
    merged = fisher_merge_two(model_a, model_b, uniform_f, uniform_f, cfg)

    sa = model_a.state_dict()
    sb = model_b.state_dict()
    param_keys = {n for n, p in model_a.named_parameters() if p.requires_grad}

    for name in merged:
        if name not in param_keys:
            continue
        if not sa[name].dtype.is_floating_point or sa[name].is_complex():
            continue
        expected = (sa[name].float() + sb[name].float()) / 2.0
        actual = merged[name].float()
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"'{name}': equal-Fisher merge is not the midpoint"
        )


# ---------------------------------------------------------------------------
# Test 8: apply_state_dict modifies model parameters
# ---------------------------------------------------------------------------


def test_apply_state_dict_modifies_model():
    model = _small_model(0)
    new_model = _small_model(1)

    original_vals = {n: p.data.clone() for n, p in model.named_parameters()}
    new_state = new_model.state_dict()

    apply_state_dict(model, new_state)

    # At least one parameter should have changed
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param.data, original_vals[name]):
            changed = True
            break
    assert changed, "apply_state_dict did not change any parameter"


# ---------------------------------------------------------------------------
# Test 9: regmean_merge returns state dict
# ---------------------------------------------------------------------------


def test_regmean_merge_returns_state_dict():
    model_a = _small_model(0)
    model_b = _small_model(1)
    data = _make_data(3)
    merged = regmean_merge(model_a, model_b, data)
    assert isinstance(merged, dict)
    assert len(merged) > 0


# ---------------------------------------------------------------------------
# Test 10: regmean_merge shapes match model parameters
# ---------------------------------------------------------------------------


def test_regmean_merge_shapes_match():
    model_a = _small_model(0)
    model_b = _small_model(1)
    data = _make_data(3)
    merged = regmean_merge(model_a, model_b, data)
    ref_state = model_a.state_dict()
    for name, tensor in merged.items():
        assert name in ref_state, f"Extra key '{name}' in merged state dict"
        assert tensor.shape == ref_state[name].shape, (
            f"'{name}': merged shape {tensor.shape} != expected {ref_state[name].shape}"
        )


# ---------------------------------------------------------------------------
# Test 11: FisherMerger.compute_fisher returns dict
# ---------------------------------------------------------------------------


def test_fisher_merger_compute_fisher_returns_dict():
    cfg = FisherMergeConfig(n_samples=3)
    merger = FisherMerger(cfg)
    model = _small_model(0)
    data = _make_data(3)
    fisher = merger.compute_fisher(model, data)
    assert isinstance(fisher, dict)
    assert len(fisher) > 0


# ---------------------------------------------------------------------------
# Test 12: FisherMerger.merge returns state dict for 2 models
# ---------------------------------------------------------------------------


def test_fisher_merger_merge_two_models():
    cfg = FisherMergeConfig(n_samples=3)
    merger = FisherMerger(cfg)
    model_a = _small_model(0)
    model_b = _small_model(1)
    data_a = _make_data(3, seed=0)
    data_b = _make_data(3, seed=1)
    merged = merger.merge([model_a, model_b], [data_a, data_b])
    assert isinstance(merged, dict)
    # Merged state should cover at least the trainable parameters
    param_keys = {n for n, _ in model_a.named_parameters()}
    assert param_keys.issubset(set(merged.keys())), (
        "Merged state dict is missing some parameter keys"
    )


# ---------------------------------------------------------------------------
# Test 13: FisherMerger.evaluate_merge_quality returns required keys
# ---------------------------------------------------------------------------


def test_evaluate_merge_quality_keys():
    cfg = FisherMergeConfig(n_samples=3)
    merger = FisherMerger(cfg)
    model_a = _small_model(0)
    model_b = _small_model(1)
    data_a = _make_data(3, seed=0)
    data_b = _make_data(3, seed=1)
    merged_state = merger.merge([model_a, model_b], [data_a, data_b])
    test_data = _make_data(2, seed=99)
    quality = merger.evaluate_merge_quality(merged_state, [model_a, model_b], test_data)
    assert "merged_loss" in quality
    assert "avg_original_loss" in quality
    assert "degradation" in quality


# ---------------------------------------------------------------------------
# Test 14: FisherMerger.evaluate_merge_quality degradation is float
# ---------------------------------------------------------------------------


def test_evaluate_merge_quality_degradation_is_float():
    cfg = FisherMergeConfig(n_samples=3)
    merger = FisherMerger(cfg)
    model_a = _small_model(0)
    model_b = _small_model(1)
    data_a = _make_data(3, seed=0)
    data_b = _make_data(3, seed=1)
    merged_state = merger.merge([model_a, model_b], [data_a, data_b])
    test_data = _make_data(2, seed=99)
    quality = merger.evaluate_merge_quality(merged_state, [model_a, model_b], test_data)
    assert isinstance(quality["degradation"], float)
    # degradation = merged_loss - avg_original_loss
    expected_deg = quality["merged_loss"] - quality["avg_original_loss"]
    assert abs(quality["degradation"] - expected_deg) < 1e-6


# ---------------------------------------------------------------------------
# Test 15: compute_diagonal_fisher with 1 sample still works
# ---------------------------------------------------------------------------


def test_compute_diagonal_fisher_one_sample():
    model = _small_model(0)
    data = _make_data(1)
    fisher = compute_diagonal_fisher(model, data, n_samples=1)
    assert isinstance(fisher, dict)
    assert len(fisher) > 0
    for name, fval in fisher.items():
        assert (fval >= 0).all(), f"Negative Fisher value for '{name}'"
        assert fval.shape == dict(model.named_parameters())[name].shape

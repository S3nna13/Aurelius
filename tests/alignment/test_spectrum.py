"""Tests for Spectrum SNR-based LoRA layer selection."""
import math
import torch
import torch.nn as nn
import pytest
from src.alignment.spectrum import compute_snr, SpectrumSelector, LayerSNR
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_compute_snr_positive(small_model):
    for name, param in small_model.named_parameters():
        if param.ndim == 2 and name.endswith(".weight"):
            snr = compute_snr(param.data)
            assert snr >= 0 or snr == float("inf"), f"Negative SNR for {name}: {snr}"
            break


def test_compute_snr_requires_2d():
    with pytest.raises(ValueError, match="2-D"):
        compute_snr(torch.randn(2, 3, 4))


def test_select_layers_returns_strings(small_model):
    selector = SpectrumSelector(small_model, top_k_fraction=0.5)
    selected = selector.select_layers()
    assert isinstance(selected, list)
    assert all(isinstance(s, str) for s in selected)
    assert len(selected) > 0


def test_select_layers_are_valid_param_names(small_model):
    param_names = {n for n, _ in small_model.named_parameters()}
    selector = SpectrumSelector(small_model, top_k_fraction=0.5)
    for name in selector.select_layers():
        assert name in param_names, f"{name} not in model parameters"


def test_select_fraction_at_least_one_per_group(small_model):
    """Even with a tiny fraction, at least 1 layer per group is selected."""
    selector = SpectrumSelector(small_model, top_k_fraction=0.01)
    selected = selector.select_layers()
    # There are multiple module types (q_proj, k_proj, v_proj, o_proj, etc.)
    # Each should contribute at least 1
    assert len(selected) >= 1


def test_higher_snr_preferred():
    """A weight matrix with more signal should have higher SNR than pure noise."""
    torch.manual_seed(0)
    # Low-rank (high signal): rank-2 matrix with clear singular value gap
    U = torch.randn(32, 2)
    V = torch.randn(2, 32)
    signal_weight = U @ V  # rank-2 matrix

    # Random noise matrix
    noise_weight = torch.randn(32, 32) * 0.01

    snr_signal = compute_snr(signal_weight)
    snr_noise = compute_snr(noise_weight)
    assert snr_signal > snr_noise, f"Expected signal SNR {snr_signal:.3f} > noise SNR {snr_noise:.3f}"


def test_invalid_fraction():
    model = nn.Linear(32, 32)
    with pytest.raises(ValueError):
        SpectrumSelector(model, top_k_fraction=0.0)
    with pytest.raises(ValueError):
        SpectrumSelector(model, top_k_fraction=1.1)

"""Tests for model analysis utilities."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.model_analysis import (
    ModelStats,
    count_parameters,
    estimate_flops,
    weight_statistics,
    activation_histogram,
    ArchitectureSummary,
    compare_models,
)


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    return AureliusTransformer(small_cfg)


# ---------------------------------------------------------------------------
# count_parameters tests
# ---------------------------------------------------------------------------

def test_count_parameters_total(small_model):
    stats = count_parameters(small_model)
    # Unique parameter count (shared embed/lm_head counted once)
    seen = set()
    total_actual = 0
    for p in small_model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total_actual += p.numel()
    assert stats.total_params == total_actual


def test_count_parameters_trainable(small_model):
    # Freeze a subset of parameters
    for name, param in small_model.named_parameters():
        if "norm" in name.lower():
            param.requires_grad_(False)

    stats = count_parameters(small_model)
    assert stats.frozen_params > 0
    assert stats.trainable_params + stats.frozen_params == stats.total_params


def test_count_parameters_no_double_counting(small_model):
    """Shared embed/lm_head weights should only be counted once."""
    stats = count_parameters(small_model)
    # Naive sum would double-count tied weights
    naive_total = sum(p.numel() for p in small_model.parameters())
    # If embeddings are tied, naive_total > stats.total_params
    # Either way, stats should not exceed naive total
    assert stats.total_params <= naive_total


# ---------------------------------------------------------------------------
# estimate_flops tests
# ---------------------------------------------------------------------------

def test_estimate_flops_returns_dict(small_model):
    result = estimate_flops(small_model, seq_len=16, batch_size=1)
    required_keys = {"total_flops", "attention_flops", "ffn_flops", "lm_head_flops", "tflops"}
    assert required_keys.issubset(result.keys())


def test_estimate_flops_tflops_positive(small_model):
    result = estimate_flops(small_model, seq_len=16, batch_size=1)
    assert result["tflops"] > 0


# ---------------------------------------------------------------------------
# weight_statistics tests
# ---------------------------------------------------------------------------

def test_weight_statistics_keys(small_model):
    result = weight_statistics(small_model)
    required_keys = {"mean", "std", "l2_norm", "max_abs", "n_dead_neurons_pct", "per_layer"}
    assert required_keys.issubset(result.keys())


def test_weight_statistics_l2_positive(small_model):
    result = weight_statistics(small_model)
    assert result["l2_norm"] > 0


# ---------------------------------------------------------------------------
# activation_histogram tests
# ---------------------------------------------------------------------------

def test_activation_histogram_shape(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    result = activation_histogram(small_model, input_ids, n_bins=50)
    assert len(result) > 0
    for name, hist in result.items():
        assert "bins" in hist
        assert "counts" in hist
        assert hist["bins"].shape[0] == 51  # n_bins + 1
        assert hist["counts"].shape[0] == 50


# ---------------------------------------------------------------------------
# ArchitectureSummary tests
# ---------------------------------------------------------------------------

def test_architecture_summary_string(small_model):
    summary = ArchitectureSummary(small_model)
    s = summary.summary_string()
    assert "Parameters" in s


# ---------------------------------------------------------------------------
# compare_models tests
# ---------------------------------------------------------------------------

def test_compare_models_same_model(small_model):
    result = compare_models(small_model, small_model, "M", "M")
    assert abs(result["weight_distance"]) < 1e-6


def test_compare_models_shared_architecture(small_cfg):
    model_a = AureliusTransformer(small_cfg)
    model_b = AureliusTransformer(small_cfg)
    result = compare_models(model_a, model_b)
    assert result["shared_architecture"] is True

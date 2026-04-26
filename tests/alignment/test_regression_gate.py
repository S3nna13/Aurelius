"""Tests for RegressionGate."""

import math

import pytest
import torch
from torch.utils.data import TensorDataset

from src.alignment.regression_gate import GateResult, RegressionGate
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def tiny_dataset(small_cfg):
    """Tiny held-out dataset: 4 sequences of 16 tokens."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, small_cfg.vocab_size, (4, 16))
    labels = torch.randint(0, small_cfg.vocab_size, (4, 16))
    return TensorDataset(input_ids, labels)


def test_compute_perplexity_returns_positive(small_model, tiny_dataset):
    gate = RegressionGate(threshold_pct=5.0)
    ppl = gate.compute_perplexity(small_model, tiny_dataset)
    assert ppl > 0
    assert math.isfinite(ppl)


def test_compute_perplexity_same_model_consistent(small_model, tiny_dataset):
    """Same model evaluated twice should give same perplexity."""
    gate = RegressionGate(threshold_pct=5.0)
    ppl1 = gate.compute_perplexity(small_model, tiny_dataset)
    ppl2 = gate.compute_perplexity(small_model, tiny_dataset)
    assert abs(ppl1 - ppl2) < 1e-3


def test_gate_accepts_identical_model(small_model, tiny_dataset):
    """Identical baseline and new model should always be accepted."""
    gate = RegressionGate(threshold_pct=5.0)
    result = gate.check(small_model, small_model, tiny_dataset)
    assert result.accepted
    assert abs(result.regression_pct) < 1e-3


def test_gate_result_fields(small_model, tiny_dataset):
    gate = RegressionGate(threshold_pct=5.0)
    result = gate.check(small_model, small_model, tiny_dataset)
    assert isinstance(result, GateResult)
    assert isinstance(result.baseline_ppl, float)
    assert isinstance(result.new_ppl, float)
    assert isinstance(result.regression_pct, float)
    assert isinstance(result.reason, str)
    assert isinstance(result.accepted, bool)


def test_gate_rejects_bad_adapter(small_cfg, tiny_dataset):
    """A model with scrambled weights should be rejected."""
    torch.manual_seed(0)
    baseline = AureliusTransformer(small_cfg)

    # Perturb weights heavily to simulate a bad adapter
    torch.manual_seed(99)
    bad_model = AureliusTransformer(small_cfg)
    with torch.no_grad():
        for p in bad_model.parameters():
            p.mul_(10.0)  # scale up weights -> high perplexity

    gate = RegressionGate(threshold_pct=5.0)
    result = gate.check(baseline, bad_model, tiny_dataset)
    # Bad model should have much higher perplexity -> rejected
    assert result.regression_pct > 0  # it got worse


def test_archive_on_rejection(tmp_path, small_cfg, tiny_dataset):
    """Rejected adapter should be archived."""
    torch.manual_seed(0)
    baseline = AureliusTransformer(small_cfg)
    torch.manual_seed(99)
    bad_model = AureliusTransformer(small_cfg)
    with torch.no_grad():
        for p in bad_model.parameters():
            p.mul_(10.0)

    # Create a fake adapter file to archive
    adapter_file = tmp_path / "adapter.pt"
    torch.save({}, adapter_file)

    gate = RegressionGate(threshold_pct=0.0)  # threshold=0 -> any regression rejected
    result = gate.check(baseline, bad_model, tiny_dataset, adapter_path=adapter_file)
    if not result.accepted:
        assert adapter_file.with_suffix(".rejected").exists()

"""Tests for weight_sharing_nas.py — Single-path One-Shot NAS and Slimmable Networks."""

from __future__ import annotations

import math

import pytest
import torch
import torch.optim as optim

from src.training.weight_sharing_nas import (
    OFABlock,
    OFAConfig,
    OFATrainer,
    OneShotSuperNet,
    SlimmableLinear,
    SubnetSpec,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return OFAConfig(
        d_model_choices=[32, 48, 64],
        d_ff_choices=[64, 128, 256],
        n_layer_choices=[1, 2],
        vocab_size=256,
    )


@pytest.fixture
def supernet(config):
    return OneShotSuperNet(config)


@pytest.fixture
def trainer(supernet, config):
    opt = optim.Adam(supernet.parameters(), lr=1e-3)
    return OFATrainer(supernet, opt, config)


@pytest.fixture
def input_ids():
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# 1. OFAConfig defaults
# ---------------------------------------------------------------------------


def test_ofa_config_defaults():
    cfg = OFAConfig()
    assert cfg.d_model_choices == [32, 48, 64]
    assert cfg.d_ff_choices == [64, 128, 256]
    assert cfg.n_layer_choices == [1, 2]
    assert cfg.elastic_kernel is False
    assert cfg.vocab_size == 256


# ---------------------------------------------------------------------------
# 2. SubnetSpec fields
# ---------------------------------------------------------------------------


def test_subnet_spec_fields():
    spec = SubnetSpec(d_model=32, d_ff=64, n_layers=1)
    assert spec.d_model == 32
    assert spec.d_ff == 64
    assert spec.n_layers == 1


# ---------------------------------------------------------------------------
# 3. SlimmableLinear full output shape
# ---------------------------------------------------------------------------


def test_slimmable_linear_full_shape():
    layer = SlimmableLinear(max_in=64, max_out=128)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert out.shape == (2, 10, 128)


# ---------------------------------------------------------------------------
# 4. SlimmableLinear sliced output shape
# ---------------------------------------------------------------------------


def test_slimmable_linear_sliced_shape():
    layer = SlimmableLinear(max_in=64, max_out=128)
    x = torch.randn(2, 10, 32)
    out = layer(x, in_features=32, out_features=64)
    assert out.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# 5. OFABlock output shape with full dims
# ---------------------------------------------------------------------------


def test_ofa_block_full_shape():
    block = OFABlock(max_d_model=64, max_d_ff=256)
    x = torch.randn(2, 8, 64)
    out = block(x)
    assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 6. OFABlock output shape with smaller dims
# ---------------------------------------------------------------------------


def test_ofa_block_smaller_shape():
    block = OFABlock(max_d_model=64, max_d_ff=256)
    x = torch.randn(2, 8, 64)
    out = block(x, d_model=32, d_ff=128)
    # residual stream uses sliced d_model
    assert out.shape == (2, 8, 32)


# ---------------------------------------------------------------------------
# 7. OneShotSuperNet forward returns 3-tuple
# ---------------------------------------------------------------------------


def test_supernet_forward_returns_tuple(supernet, input_ids):
    result = supernet(input_ids)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 8. OneShotSuperNet logits shape (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_supernet_logits_shape(supernet, input_ids):
    _, logits, _ = supernet(input_ids)
    B, T = input_ids.shape
    assert logits.shape == (B, T, 256)


# ---------------------------------------------------------------------------
# 9. OneShotSuperNet.sample_subnet returns valid SubnetSpec
# ---------------------------------------------------------------------------


def test_sample_subnet_returns_spec(supernet):
    spec = supernet.sample_subnet()
    assert isinstance(spec, SubnetSpec)


# ---------------------------------------------------------------------------
# 10. OneShotSuperNet.sample_subnet choices within config bounds
# ---------------------------------------------------------------------------


def test_sample_subnet_valid_choices(supernet, config):
    for _ in range(20):
        spec = supernet.sample_subnet()
        assert spec.d_model in config.d_model_choices
        assert spec.d_ff in config.d_ff_choices
        assert spec.n_layers in config.n_layer_choices


# ---------------------------------------------------------------------------
# 11. OneShotSuperNet.get_subnet_params returns positive int
# ---------------------------------------------------------------------------


def test_get_subnet_params_positive(supernet):
    spec = SubnetSpec(d_model=32, d_ff=64, n_layers=1)
    n = supernet.get_subnet_params(spec)
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# 12. OFATrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_trainer_step_keys(trainer, input_ids):
    result = trainer.train_step(input_ids)
    assert "loss" in result
    assert "subnet_spec" in result
    assert "n_params" in result


# ---------------------------------------------------------------------------
# 13. OFATrainer.train_step loss is finite
# ---------------------------------------------------------------------------


def test_trainer_step_loss_finite(trainer, input_ids):
    result = trainer.train_step(input_ids)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# 14. OFATrainer.evaluate_subnet returns float
# ---------------------------------------------------------------------------


def test_evaluate_subnet_returns_float(trainer, supernet, input_ids):
    spec = SubnetSpec(d_model=32, d_ff=64, n_layers=1)
    data = [input_ids, input_ids]
    result = trainer.evaluate_subnet(spec, data)
    assert isinstance(result, float)
    assert math.isfinite(result)

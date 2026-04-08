"""Tests for dense-to-MoE upcycling (src/model/moe_upcycle.py)."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.ffn import SwiGLUFFN
from src.model.moe import MoEConfig, SparseMoEFFN
from src.model.moe_upcycle import (
    UpcycleConfig,
    count_parameters,
    upcycle_ffn,
    upcycle_model,
)
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
        max_seq_len=32,
    )


@pytest.fixture
def dense_ffn(small_cfg):
    return SwiGLUFFN(small_cfg)


@pytest.fixture
def dense_model(small_cfg):
    return AureliusTransformer(small_cfg)


# ---------------------------------------------------------------------------
# upcycle_ffn tests
# ---------------------------------------------------------------------------

def test_upcycle_ffn_creates_n_experts(small_cfg, dense_ffn):
    """upcycle_ffn with n_experts=4 produces a SparseMoEFFN with 4 experts."""
    moe_cfg = MoEConfig(n_experts=4, top_k=2)
    moe = upcycle_ffn(dense_ffn, moe_cfg, small_cfg)
    assert isinstance(moe, SparseMoEFFN)
    assert len(moe.experts) == 4


def test_upcycle_ffn_copies_weights(small_cfg, dense_ffn):
    """Expert 0 weights must exactly match the original dense FFN."""
    moe_cfg = MoEConfig(n_experts=4, top_k=2)
    moe = upcycle_ffn(dense_ffn, moe_cfg, small_cfg)
    expert0 = moe.experts[0]
    assert torch.equal(expert0.gate_proj.weight, dense_ffn.gate_proj.weight)
    assert torch.equal(expert0.up_proj.weight, dense_ffn.up_proj.weight)
    assert torch.equal(expert0.down_proj.weight, dense_ffn.down_proj.weight)


def test_upcycle_ffn_experts_identical(small_cfg, dense_ffn):
    """All experts must have identical weights at initialization."""
    moe_cfg = MoEConfig(n_experts=4, top_k=2)
    moe = upcycle_ffn(dense_ffn, moe_cfg, small_cfg)
    ref_gate = moe.experts[0].gate_proj.weight
    for i in range(1, 4):
        assert torch.equal(moe.experts[i].gate_proj.weight, ref_gate), (
            f"Expert {i} gate_proj differs from expert 0"
        )


def test_upcycle_ffn_router_is_small(small_cfg, dense_ffn):
    """Router weight std must be small (< 0.1) — sanity check for small init."""
    moe_cfg = MoEConfig(n_experts=4, top_k=2)
    moe = upcycle_ffn(dense_ffn, moe_cfg, small_cfg)
    router_std = moe.router.weight.std().item()
    assert router_std < 0.1, f"Router std {router_std:.4f} is unexpectedly large"


# ---------------------------------------------------------------------------
# upcycle_model tests
# ---------------------------------------------------------------------------

def test_upcycle_model_replaces_ffn_layers(small_cfg, dense_model):
    """After upcycling, every layer's ffn is a SparseMoEFFN."""
    upcycle_cfg = UpcycleConfig(n_experts=4, top_k=2)
    upcycle_model(dense_model, small_cfg, upcycle_cfg)
    for i, layer in enumerate(dense_model.layers):
        assert isinstance(layer.ffn, SparseMoEFFN), (
            f"Layer {i} ffn is {type(layer.ffn).__name__}, expected SparseMoEFFN"
        )


def test_upcycle_model_increases_param_count(small_cfg):
    """Upcycled model (n_experts=4 > 1) must have more parameters than dense."""
    dense = AureliusTransformer(small_cfg)
    dense_params = count_parameters(dense)["total"]

    upcycle_cfg = UpcycleConfig(n_experts=4, top_k=2)
    upcycle_model(dense, small_cfg, upcycle_cfg)
    moe_params = count_parameters(dense)["total"]

    assert moe_params > dense_params, (
        f"MoE params ({moe_params}) should exceed dense params ({dense_params})"
    )


def test_upcycle_model_partial_layers(small_cfg):
    """upcycle_all_layers=False with layer_indices=[0] — only layer 0 is upcycled."""
    model = AureliusTransformer(small_cfg)
    upcycle_cfg = UpcycleConfig(
        n_experts=4,
        top_k=2,
        upcycle_all_layers=False,
        layer_indices=[0],
    )
    upcycle_model(model, small_cfg, upcycle_cfg)

    assert isinstance(model.layers[0].ffn, SparseMoEFFN), "Layer 0 should be MoE"
    assert isinstance(model.layers[1].ffn, SwiGLUFFN), "Layer 1 should remain dense"


def test_upcycle_model_forward_still_works(small_cfg):
    """Upcycled model forward pass returns (loss, logits, pkv) without error."""
    model = AureliusTransformer(small_cfg)
    upcycle_cfg = UpcycleConfig(n_experts=4, top_k=2)
    upcycle_model(model, small_cfg, upcycle_cfg)

    model.eval()
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, small_cfg.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, small_cfg.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        loss, logits, pkv = model(input_ids, labels=labels)

    assert loss is not None
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert logits.shape == (batch_size, seq_len, small_cfg.vocab_size)
    assert len(pkv) == small_cfg.n_layers


# ---------------------------------------------------------------------------
# count_parameters tests
# ---------------------------------------------------------------------------

def test_count_parameters_returns_dict(small_cfg, dense_model):
    """count_parameters returns dict with total, trainable, frozen keys."""
    result = count_parameters(dense_model)
    assert "total" in result
    assert "trainable" in result
    assert "frozen" in result
    assert result["total"] == result["trainable"] + result["frozen"]
    assert result["total"] > 0

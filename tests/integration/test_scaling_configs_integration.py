"""Integration tests for scaling config YAMLs → TrainConfig → AureliusTransformer.

Loads each production scaling config, verifies dimensions, and confirms a
reduced model can be instantiated and run a forward pass.
"""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.trainer import TrainConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_aurelius_config(train_cfg: TrainConfig) -> AureliusConfig:
    """Return a tiny AureliusConfig derived from a loaded TrainConfig.

    Keeps the structural dimensions (d_model, n_heads, head_dim, d_ff, MoE
    flags, etc.) so the architecture is representative, but shrinks n_layers
    and vocab_size so the test stays fast on CPU.
    """
    return AureliusConfig(
        d_model=train_cfg.model_d_model,
        n_layers=min(2, train_cfg.model_n_layers),
        n_heads=train_cfg.model_n_heads,
        n_kv_heads=train_cfg.model_n_kv_heads,
        head_dim=train_cfg.model_head_dim,
        d_ff=train_cfg.model_d_ff,
        vocab_size=256,  # tiny for speed
        max_seq_len=64,  # tiny for speed
        rope_theta=train_cfg.model_rope_theta,
        tie_embeddings=train_cfg.model_tie_embeddings,
        moe_enabled=train_cfg.model_moe_enabled,
        moe_num_experts=train_cfg.model_moe_num_experts,
        moe_top_k=train_cfg.model_moe_top_k,
        moe_every_n_layers=train_cfg.model_moe_every_n_layers,
        moe_capacity_factor=train_cfg.model_moe_capacity_factor,
    )


def _forward_pass(model: AureliusTransformer) -> torch.Tensor:
    """Run a tiny forward pass and return logits."""
    batch_size = 1
    seq_len = 8
    vocab_size = model.config.vocab_size
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        loss, logits, _ = model(tokens)
    return loss, logits


# ---------------------------------------------------------------------------
# 2.7B dense
# ---------------------------------------------------------------------------


def test_train_2_7b_yaml_dimensions():
    """TrainConfig.from_yaml parses 2.7B dimensions correctly."""
    cfg = TrainConfig.from_yaml("configs/train_2.7b.yaml")
    assert cfg.model_name == "aurelius-2.7b"
    assert cfg.model_d_model == 2560
    assert cfg.model_n_layers == 32
    assert cfg.model_n_heads == 20
    assert cfg.model_n_kv_heads == 5
    assert cfg.model_head_dim == 128
    assert cfg.model_d_ff == 7168
    assert cfg.model_vocab_size == 128_000
    assert cfg.model_max_seq_len == 8192
    assert cfg.model_rope_theta == 500_000.0
    assert cfg.model_tie_embeddings is True
    assert cfg.model_moe_enabled is False


def test_train_2_7b_model_forward():
    """AureliusTransformer instantiated from 2.7B config runs a forward pass."""
    train_cfg = TrainConfig.from_yaml("configs/train_2.7b.yaml")
    tiny_cfg = _make_tiny_aurelius_config(train_cfg)
    model = AureliusTransformer(tiny_cfg)
    loss, logits = _forward_pass(model)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)
    assert loss is None  # no labels provided


# ---------------------------------------------------------------------------
# 3B dense
# ---------------------------------------------------------------------------


def test_train_3b_yaml_dimensions():
    """TrainConfig.from_yaml parses 3B dimensions correctly."""
    cfg = TrainConfig.from_yaml("configs/train_3b.yaml")
    assert cfg.model_name == "aurelius-3b"
    assert cfg.model_d_model == 3072
    assert cfg.model_n_layers == 28
    assert cfg.model_n_heads == 24
    assert cfg.model_n_kv_heads == 6
    assert cfg.model_head_dim == 128
    assert cfg.model_d_ff == 8192
    assert cfg.model_vocab_size == 128_000
    assert cfg.model_max_seq_len == 4096
    assert cfg.model_rope_theta == 500_000.0
    assert cfg.model_tie_embeddings is True
    assert cfg.model_moe_enabled is False


def test_train_3b_model_forward():
    """AureliusTransformer instantiated from 3B config runs a forward pass."""
    train_cfg = TrainConfig.from_yaml("configs/train_3b.yaml")
    tiny_cfg = _make_tiny_aurelius_config(train_cfg)
    model = AureliusTransformer(tiny_cfg)
    loss, logits = _forward_pass(model)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)
    assert loss is None


# ---------------------------------------------------------------------------
# MoE 5B
# ---------------------------------------------------------------------------


def test_train_moe_5b_yaml_dimensions():
    """TrainConfig.from_yaml parses MoE 5B dimensions correctly."""
    cfg = TrainConfig.from_yaml("configs/train_moe_5b.yaml")
    assert cfg.model_name == "aurelius-moe-5b"
    assert cfg.model_d_model == 2048
    assert cfg.model_n_layers == 24
    assert cfg.model_n_heads == 16
    assert cfg.model_n_kv_heads == 8
    assert cfg.model_head_dim == 128
    assert cfg.model_d_ff == 5632
    assert cfg.model_vocab_size == 128_000
    assert cfg.model_max_seq_len == 8192
    assert cfg.model_rope_theta == 500_000.0
    assert cfg.model_tie_embeddings is True
    assert cfg.model_moe_enabled is True
    assert cfg.model_moe_num_experts == 8
    assert cfg.model_moe_top_k == 2
    assert cfg.model_moe_every_n_layers == 2
    assert cfg.model_moe_capacity_factor == 1.25


def test_train_moe_5b_model_forward():
    """AureliusTransformer instantiated from MoE 5B config runs a forward pass."""
    train_cfg = TrainConfig.from_yaml("configs/train_moe_5b.yaml")
    tiny_cfg = _make_tiny_aurelius_config(train_cfg)
    model = AureliusTransformer(tiny_cfg)
    loss, logits = _forward_pass(model)
    assert logits.shape == (1, 8, tiny_cfg.vocab_size)
    assert loss is None

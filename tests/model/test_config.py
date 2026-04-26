"""Tests for AureliusConfig."""

import pytest

from src.model.config import AureliusConfig


def test_default_config_values():
    cfg = AureliusConfig()
    assert cfg.d_model == 2048
    assert cfg.n_layers == 24
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.d_ff == 5632
    assert cfg.vocab_size == 128_000
    assert cfg.max_seq_len == 8192
    assert cfg.rope_theta == 500_000.0
    assert cfg.tie_embeddings is True


def test_d_model_consistency():
    cfg = AureliusConfig()
    assert cfg.d_model == cfg.n_heads * cfg.head_dim


def test_gqa_ratio_valid():
    cfg = AureliusConfig()
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.n_heads // cfg.n_kv_heads == 2


def test_invalid_d_model_raises():
    with pytest.raises(AssertionError):
        AureliusConfig(d_model=2048, n_heads=16, head_dim=64)  # 16*64=1024 ≠ 2048


def test_invalid_kv_heads_raises():
    with pytest.raises(AssertionError):
        AureliusConfig(n_heads=16, n_kv_heads=5)  # not divisible


def test_small_config_for_tests():
    """Small config used in tests to keep things fast."""
    cfg = AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64, d_ff=512, vocab_size=1000
    )
    assert cfg.d_model == cfg.n_heads * cfg.head_dim

"""Tests for NoPE layers and Differential Attention."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer, TransformerBlock
from src.model.attention import DifferentialAttention, precompute_rope_frequencies, apply_rope


@pytest.fixture
def nope_cfg():
    return AureliusConfig(
        n_layers=8, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        d_ff=512, vocab_size=1000, max_seq_len=128, nope_every_n_layers=4,
    )


@pytest.fixture
def diff_cfg():
    return AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        d_ff=512, vocab_size=1000, max_seq_len=128, use_diff_attn=True,
    )


def test_nope_layers_have_no_rope(nope_cfg):
    """Layers at (idx+1) % nope_every_n_layers == 0 should have apply_rope=False."""
    for i in range(nope_cfg.n_layers):
        block = TransformerBlock(nope_cfg, layer_idx=i)
        expected_rope = ((i + 1) % nope_cfg.nope_every_n_layers != 0)
        assert block.attn.apply_rope == expected_rope, (
            f"Layer {i}: expected apply_rope={expected_rope}, got {block.attn.apply_rope}"
        )


def test_nope_model_forward_shape(nope_cfg):
    model = AureliusTransformer(nope_cfg)
    tokens = torch.randint(0, nope_cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, nope_cfg.vocab_size)


def test_nope_disabled_all_layers_have_rope():
    cfg = AureliusConfig(
        n_layers=4, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        d_ff=512, vocab_size=1000, nope_every_n_layers=0,
    )
    for i in range(4):
        block = TransformerBlock(cfg, layer_idx=i)
        assert block.attn.apply_rope is True


def test_diff_attn_output_shape(diff_cfg):
    model = AureliusTransformer(diff_cfg)
    tokens = torch.randint(0, diff_cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, diff_cfg.vocab_size)


def test_diff_attn_no_nan(diff_cfg):
    model = AureliusTransformer(diff_cfg)
    tokens = torch.randint(0, diff_cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(tokens)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_diff_attn_has_lambda_param(diff_cfg):
    """DifferentialAttention must have learnable lambda parameters."""
    block = TransformerBlock(diff_cfg, layer_idx=0)
    assert hasattr(block.attn, 'lambda_param')
    assert block.attn.lambda_param.shape == (diff_cfg.n_heads,)


def test_diff_attn_no_bias(diff_cfg):
    block = TransformerBlock(diff_cfg, layer_idx=0)
    for name, _ in block.attn.named_parameters():
        if 'lambda' not in name:
            assert 'bias' not in name


def test_nope_and_diff_attn_combined():
    cfg = AureliusConfig(
        n_layers=8, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
        d_ff=512, vocab_size=1000, max_seq_len=64,
        nope_every_n_layers=4, use_diff_attn=True,
    )
    model = AureliusTransformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (1, 16, cfg.vocab_size)
    assert not torch.isnan(logits).any()

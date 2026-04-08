import pytest
import torch
import torch.nn as nn
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer, TransformerBlock
from src.model.attention import precompute_rope_frequencies
from src.model.mod import MoDConfig, MoDRouter, MoDLayer

@pytest.fixture
def small_cfg():
    return AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                          head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64)

@pytest.fixture
def block_and_freqs(small_cfg):
    block = TransformerBlock(small_cfg)
    freqs = precompute_rope_frequencies(small_cfg.head_dim, small_cfg.max_seq_len)
    return block, freqs

def test_router_output_shape(small_cfg):
    router = MoDRouter(small_cfg.d_model)
    x = torch.randn(2, 10, small_cfg.d_model)
    scores = router(x)
    assert scores.shape == (2, 10)

def test_mod_layer_output_shape(block_and_freqs, small_cfg):
    block, freqs = block_and_freqs
    mod = MoDLayer(block, small_cfg.d_model, MoDConfig(capacity_fraction=0.5))
    x = torch.randn(1, 8, small_cfg.d_model)
    out, _, aux = mod(x, freqs[:8])
    assert out.shape == x.shape

def test_mod_layer_aux_loss_finite(block_and_freqs, small_cfg):
    block, freqs = block_and_freqs
    mod = MoDLayer(block, small_cfg.d_model, MoDConfig(capacity_fraction=0.5))
    x = torch.randn(1, 8, small_cfg.d_model)
    _, _, aux = mod(x, freqs[:8])
    assert aux.ndim == 0
    import math
    assert math.isfinite(aux.item())

def test_mod_layer_rejects_kv_cache(block_and_freqs, small_cfg):
    block, freqs = block_and_freqs
    mod = MoDLayer(block, small_cfg.d_model)
    x = torch.randn(1, 1, small_cfg.d_model)
    fake_kv = (torch.zeros(1, 1, 1, small_cfg.head_dim),
               torch.zeros(1, 1, 1, small_cfg.head_dim))
    with pytest.raises(ValueError):
        mod(x, freqs[:1], past_kv=fake_kv)

def test_mod_capacity_one_token(block_and_freqs, small_cfg):
    block, freqs = block_and_freqs
    mod = MoDLayer(block, small_cfg.d_model, MoDConfig(capacity_fraction=0.1))
    x = torch.randn(1, 8, small_cfg.d_model)
    out, _, _ = mod(x, freqs[:8])
    assert out.shape == x.shape  # even with 1 token processed, output is same shape

def test_mod_gradients_flow(block_and_freqs, small_cfg):
    block, freqs = block_and_freqs
    mod = MoDLayer(block, small_cfg.d_model)
    x = torch.randn(1, 8, small_cfg.d_model, requires_grad=True)
    out, _, aux = mod(x, freqs[:8])
    (out.sum() + aux).backward()
    assert x.grad is not None

"""Tests for activation checkpointing (gradient checkpointing) utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.activation_checkpointing import (
    ActivationCheckpointTrainer,
    CheckpointConfig,
    CheckpointedLayer,
    estimate_memory_savings,
    get_checkpoint_stats,
    wrap_layers_with_checkpointing,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def tiny_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def small_model():
    torch.manual_seed(42)
    return AureliusTransformer(tiny_config())


# ---------------------------------------------------------------------------
# 1. CheckpointConfig defaults
# ---------------------------------------------------------------------------

def test_checkpoint_config_defaults():
    cfg = CheckpointConfig()
    assert cfg.checkpoint_every_n_layers == 1
    assert cfg.use_reentrant is False
    assert cfg.offload_to_cpu is False


def test_checkpoint_config_custom():
    cfg = CheckpointConfig(checkpoint_every_n_layers=2, use_reentrant=True, offload_to_cpu=True)
    assert cfg.checkpoint_every_n_layers == 2
    assert cfg.use_reentrant is True
    assert cfg.offload_to_cpu is True


# ---------------------------------------------------------------------------
# 2. CheckpointedLayer forward produces same output as unwrapped layer
# ---------------------------------------------------------------------------

def test_checkpointed_layer_same_output():
    """CheckpointedLayer should produce identical outputs to the raw layer."""
    torch.manual_seed(0)
    model = small_model()
    layer = model.layers[0]  # raw TransformerBlock

    # Build a simple wrapper that mirrors the TransformerBlock signature
    ckpt_layer = CheckpointedLayer(layer, use_reentrant=False)

    B, S, D = 1, 8, 64
    x = torch.randn(B, S, D)
    freqs = model.freqs_cis[:S]

    with torch.no_grad():
        out_raw, kv_raw = layer(x, freqs)
        out_ckpt, kv_ckpt = ckpt_layer(x, freqs)

    assert torch.allclose(out_raw, out_ckpt, atol=1e-5), "Outputs differ between raw and checkpointed layer"


def test_checkpointed_layer_output_shape():
    """CheckpointedLayer should preserve output shape."""
    model = small_model()
    layer = model.layers[0]
    ckpt_layer = CheckpointedLayer(layer, use_reentrant=False)

    B, S, D = 2, 16, 64
    x = torch.randn(B, S, D)
    freqs = model.freqs_cis[:S]

    with torch.no_grad():
        out, _kv = ckpt_layer(x, freqs)

    assert out.shape == (B, S, D), f"Expected ({B}, {S}, {D}) got {out.shape}"


# ---------------------------------------------------------------------------
# 3. wrap_layers_with_checkpointing wraps correct number of layers
# ---------------------------------------------------------------------------

def test_wrap_all_layers():
    """With checkpoint_every_n_layers=1, every layer should be wrapped."""
    model = small_model()
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    n_wrapped = wrap_layers_with_checkpointing(model, cfg)
    assert n_wrapped == 2
    assert all(isinstance(l, CheckpointedLayer) for l in model.layers)


def test_wrap_returns_correct_count():
    """wrap_layers_with_checkpointing should return the wrapped count."""
    model = small_model()
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    n_wrapped = wrap_layers_with_checkpointing(model, cfg)
    assert isinstance(n_wrapped, int)
    assert n_wrapped > 0


# ---------------------------------------------------------------------------
# 4. checkpoint_every_n_layers=2 wraps ~half the layers
# ---------------------------------------------------------------------------

def test_wrap_every_other_layer():
    """With checkpoint_every_n_layers=2 and 4 layers, 2 should be wrapped."""
    cfg_model = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg_model)
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    n_wrapped = wrap_layers_with_checkpointing(model, cfg)
    assert n_wrapped == 2, f"Expected 2 wrapped layers, got {n_wrapped}"


def test_wrap_every_other_layer_correct_indices():
    """With checkpoint_every_n_layers=2, even-indexed layers should be wrapped."""
    cfg_model = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg_model)
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    wrap_layers_with_checkpointing(model, cfg)
    # Indices 0 and 2 should be wrapped; 1 and 3 should not
    assert isinstance(model.layers[0], CheckpointedLayer)
    assert not isinstance(model.layers[1], CheckpointedLayer)
    assert isinstance(model.layers[2], CheckpointedLayer)
    assert not isinstance(model.layers[3], CheckpointedLayer)


# ---------------------------------------------------------------------------
# 5. estimate_memory_savings returns positive floats
# ---------------------------------------------------------------------------

def test_estimate_memory_savings_returns_floats():
    full_mb, ckpt_mb = estimate_memory_savings(n_layers=24, d_model=2048, seq_len=512, batch_size=4)
    assert isinstance(full_mb, float)
    assert isinstance(ckpt_mb, float)


def test_estimate_memory_savings_positive():
    full_mb, ckpt_mb = estimate_memory_savings(n_layers=24, d_model=2048, seq_len=512, batch_size=4)
    assert full_mb > 0.0
    assert ckpt_mb > 0.0


# ---------------------------------------------------------------------------
# 6. Memory savings > 0 for any valid input
# ---------------------------------------------------------------------------

def test_memory_savings_nonzero_small():
    full_mb, ckpt_mb = estimate_memory_savings(n_layers=2, d_model=64, seq_len=8, batch_size=1)
    assert full_mb > 0.0
    assert ckpt_mb > 0.0


def test_memory_savings_full_greater_than_checkpointed():
    full_mb, ckpt_mb = estimate_memory_savings(n_layers=12, d_model=512, seq_len=256, batch_size=2)
    assert full_mb > ckpt_mb, "Full activation memory should exceed checkpointed memory"


def test_memory_savings_formula():
    """Verify the formula: full = n*seq*B*d*4/1e6, ckpt = full/n."""
    n, d, s, b = 8, 128, 32, 2
    expected_full = n * s * b * d * 4 / 1e6
    expected_ckpt = expected_full / n
    full_mb, ckpt_mb = estimate_memory_savings(n_layers=n, d_model=d, seq_len=s, batch_size=b)
    assert abs(full_mb - expected_full) < 1e-9
    assert abs(ckpt_mb - expected_ckpt) < 1e-9


# ---------------------------------------------------------------------------
# 7. ActivationCheckpointTrainer.train_step returns loss
# ---------------------------------------------------------------------------

def test_trainer_train_step_returns_loss():
    """train_step should return a dict containing a scalar 'loss' key."""
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = ActivationCheckpointTrainer(model, optimizer, CheckpointConfig())

    input_ids = torch.randint(0, 256, (1, 16))
    result = trainer.train_step(input_ids)

    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0.0


def test_trainer_train_step_returns_n_checkpointed():
    """train_step should return 'n_checkpointed_layers' in result dict."""
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = ActivationCheckpointTrainer(model, optimizer, CheckpointConfig())

    input_ids = torch.randint(0, 256, (1, 16))
    result = trainer.train_step(input_ids)

    assert "n_checkpointed_layers" in result
    assert result["n_checkpointed_layers"] == 2


def test_trainer_wraps_layers_on_init():
    """ActivationCheckpointTrainer should wrap all layers at construction time."""
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _trainer = ActivationCheckpointTrainer(model, optimizer, CheckpointConfig())
    assert all(isinstance(l, CheckpointedLayer) for l in model.layers)


# ---------------------------------------------------------------------------
# 8. get_checkpoint_stats returns correct ratio
# ---------------------------------------------------------------------------

def test_get_checkpoint_stats_all_wrapped():
    """With all layers wrapped, ratio should be 1.0."""
    model = small_model()
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    wrap_layers_with_checkpointing(model, cfg)
    stats = get_checkpoint_stats(model)

    assert stats["total_layers"] == 2
    assert stats["checkpointed_layers"] == 2
    assert stats["checkpoint_ratio"] == 1.0


def test_get_checkpoint_stats_none_wrapped():
    """With no wrapping done, ratio should be 0.0."""
    model = small_model()
    stats = get_checkpoint_stats(model)
    assert stats["checkpoint_ratio"] == 0.0
    assert stats["checkpointed_layers"] == 0


def test_get_checkpoint_stats_partial():
    """With half layers wrapped, ratio should be 0.5."""
    cfg_model = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg_model)
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    wrap_layers_with_checkpointing(model, cfg)
    stats = get_checkpoint_stats(model)

    assert stats["total_layers"] == 4
    assert stats["checkpointed_layers"] == 2
    assert abs(stats["checkpoint_ratio"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# 9. CheckpointedLayer preserves gradient flow (backward works)
# ---------------------------------------------------------------------------

def test_checkpointed_layer_gradient_flow():
    """Gradients should flow through CheckpointedLayer during backward."""
    model = small_model()
    layer = model.layers[0]
    ckpt_layer = CheckpointedLayer(layer, use_reentrant=False)

    B, S, D = 1, 8, 64
    x = torch.randn(B, S, D, requires_grad=True)
    freqs = model.freqs_cis[:S]

    out, _kv = ckpt_layer(x, freqs)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad should not be None — gradient did not flow"
    assert x.grad.shape == x.shape


def test_full_model_backward_with_checkpointing():
    """A full forward+backward through a checkpointed model should succeed."""
    model = small_model()
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    wrap_layers_with_checkpointing(model, cfg)
    model.train()

    input_ids = torch.randint(0, 256, (1, 16))
    loss, _logits, _pkv = model(input_ids, labels=input_ids)

    assert loss is not None
    loss.backward()

    # At least one parameter should have a gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed after backward through checkpointed model"

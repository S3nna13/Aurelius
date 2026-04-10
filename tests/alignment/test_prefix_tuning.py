"""Tests for PrefixTuning: layer-wise soft prefix tuning (Li & Liang 2021)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.alignment.prefix_tuning import (
    PrefixConfig,
    PrefixTuning,
    apply_prefix_to_attention,
    PrefixTuningTrainer,
    PrefixEncoderConfig,
    PrefixEncoder,
    prepend_prefix_to_kv,
    PrefixTuningModel,
    PrefixTuner,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# Small dimensions for fast CPU tests
CFG_SMALL = PrefixConfig(
    prefix_length=4,
    n_layers=3,
    n_kv_heads=2,
    head_dim=8,
    dropout=0.0,
    use_mlp_reparameterization=True,
)

CFG_SMALL_NO_MLP = PrefixConfig(
    prefix_length=4,
    n_layers=3,
    n_kv_heads=2,
    head_dim=8,
    dropout=0.0,
    use_mlp_reparameterization=False,
)


# Tiny Aurelius config for integration tests
TINY_AURELIUS = AureliusConfig(
    n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
    head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
)

# Matching PrefixEncoderConfig
TINY_PREFIX_CFG = PrefixEncoderConfig(
    prefix_length=10, d_model=64, n_layers=2,
    dropout=0.0, reparameterize=True, reparam_hidden=512,
)


# ---------------------------------------------------------------------------
# Helper: make a tiny backbone
# ---------------------------------------------------------------------------

def make_simple_backbone() -> nn.Module:
    """Tiny linear backbone for trainer tests."""
    return nn.Linear(16, 16)


def make_tiny_model() -> AureliusTransformer:
    """Create a tiny Aurelius model for integration tests."""
    return AureliusTransformer(TINY_AURELIUS)


# ===========================================================================
# Original tests (1-12) -- preserved from the original test file
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. test_prefix_config_defaults
# ---------------------------------------------------------------------------

def test_prefix_config_defaults():
    cfg = PrefixConfig()
    assert cfg.prefix_length == 10
    assert cfg.n_layers == 24
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.dropout == 0.1
    assert cfg.use_mlp_reparameterization is True


# ---------------------------------------------------------------------------
# 2. test_prefix_tuning_get_prefix_kv_shape
# ---------------------------------------------------------------------------

def test_prefix_tuning_get_prefix_kv_shape():
    model = PrefixTuning(CFG_SMALL)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    expected = (CFG_SMALL.prefix_length, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    assert prefix_k.shape == expected, f"prefix_k shape {prefix_k.shape} != {expected}"
    assert prefix_v.shape == expected, f"prefix_v shape {prefix_v.shape} != {expected}"


# ---------------------------------------------------------------------------
# 3. test_prefix_tuning_all_layers_count
# ---------------------------------------------------------------------------

def test_prefix_tuning_all_layers_count():
    model = PrefixTuning(CFG_SMALL)
    all_kvs = model.get_all_prefix_kvs()
    assert len(all_kvs) == CFG_SMALL.n_layers, (
        f"Expected {CFG_SMALL.n_layers} entries, got {len(all_kvs)}"
    )


# ---------------------------------------------------------------------------
# 4. test_prefix_tuning_with_mlp
# ---------------------------------------------------------------------------

def test_prefix_tuning_with_mlp():
    cfg = PrefixConfig(
        prefix_length=4, n_layers=3, n_kv_heads=2, head_dim=8,
        dropout=0.0, use_mlp_reparameterization=True,
    )
    model = PrefixTuning(cfg)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    assert prefix_k.shape == (4, 2, 8)
    assert prefix_v.shape == (4, 2, 8)


# ---------------------------------------------------------------------------
# 5. test_prefix_tuning_without_mlp
# ---------------------------------------------------------------------------

def test_prefix_tuning_without_mlp():
    cfg = PrefixConfig(
        prefix_length=4, n_layers=3, n_kv_heads=2, head_dim=8,
        dropout=0.0, use_mlp_reparameterization=False,
    )
    model = PrefixTuning(cfg)
    prefix_k, prefix_v = model.get_prefix_kv(0)
    assert prefix_k.shape == (4, 2, 8)
    assert prefix_v.shape == (4, 2, 8)


# ---------------------------------------------------------------------------
# 6. test_apply_prefix_to_attention_shape
# ---------------------------------------------------------------------------

def test_apply_prefix_to_attention_shape():
    B, S = 2, 6
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=0)

    expected_seq = CFG_SMALL.prefix_length + S
    assert new_k.shape == (B, expected_seq, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim), (
        f"Expected k shape (B, {expected_seq}, n_kv_heads, head_dim), got {new_k.shape}"
    )
    assert new_v.shape == (B, expected_seq, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim), (
        f"Expected v shape (B, {expected_seq}, n_kv_heads, head_dim), got {new_v.shape}"
    )


# ---------------------------------------------------------------------------
# 7. test_apply_prefix_expands_batch
# ---------------------------------------------------------------------------

def test_apply_prefix_expands_batch():
    B, S = 4, 10
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=1)

    # Prefix portion should be the same for all batch items
    prefix_k_slice = new_k[:, : CFG_SMALL.prefix_length]  # (B, prefix_len, kv_heads, head_dim)
    assert prefix_k_slice.shape[0] == B, "Batch dim should be B after expansion"
    # All batch items have the same prefix values
    assert torch.allclose(prefix_k_slice[0], prefix_k_slice[1]), (
        "Prefix should be identical across batch items"
    )


# ---------------------------------------------------------------------------
# 8. test_prefix_tuning_trainer_freeze
# ---------------------------------------------------------------------------

def test_prefix_tuning_trainer_freeze():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    frozen_count = trainer.freeze_backbone()
    assert frozen_count > 0, "freeze_backbone should return > 0 frozen params"
    for p in backbone.parameters():
        assert not p.requires_grad, "Backbone params should be frozen after freeze_backbone()"


# ---------------------------------------------------------------------------
# 9. test_prefix_tuning_trainer_unfreeze
# ---------------------------------------------------------------------------

def test_prefix_tuning_trainer_unfreeze():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    trainer.unfreeze_backbone()
    for p in backbone.parameters():
        assert p.requires_grad, "Backbone params should be trainable after unfreeze_backbone()"


# ---------------------------------------------------------------------------
# 10. test_prefix_tuning_trainer_trainable_params
# ---------------------------------------------------------------------------

def test_prefix_tuning_trainer_trainable_params():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    trainable = trainer.trainable_params()

    # All returned params must have requires_grad=True
    for p in trainable:
        assert p.requires_grad, "trainable_params() should only return grad-enabled params"

    # Must be exactly the prefix_tuning parameters
    prefix_param_ids = {id(p) for p in prefix_model.parameters()}
    trainable_ids = {id(p) for p in trainable}
    assert trainable_ids == prefix_param_ids, (
        "trainable_params() should return exactly the prefix_tuning parameters"
    )


# ---------------------------------------------------------------------------
# 11. test_prefix_tuning_param_count
# ---------------------------------------------------------------------------

def test_prefix_tuning_param_count():
    backbone = make_simple_backbone()
    prefix_model = PrefixTuning(CFG_SMALL)
    optimizer = torch.optim.Adam(prefix_model.parameters(), lr=1e-3)
    trainer = PrefixTuningTrainer(backbone, prefix_model, optimizer)

    trainer.freeze_backbone()
    counts = trainer.param_count()

    assert "total" in counts
    assert "trainable" in counts
    assert "frozen" in counts
    assert counts["total"] == counts["trainable"] + counts["frozen"], (
        "total must equal trainable + frozen"
    )
    assert counts["trainable"] > 0, "trainable count should be > 0"
    assert counts["frozen"] > 0, "frozen count should be > 0 (backbone is frozen)"


# ---------------------------------------------------------------------------
# 12. test_prefix_tuning_gradients_flow
# ---------------------------------------------------------------------------

def test_prefix_tuning_gradients_flow():
    B, S = 2, 5
    prefix_model = PrefixTuning(CFG_SMALL)
    k = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)
    v = torch.randn(B, S, CFG_SMALL.n_kv_heads, CFG_SMALL.head_dim)

    new_k, new_v = apply_prefix_to_attention(prefix_model, k, v, layer_idx=0)
    loss = new_k.sum() + new_v.sum()
    loss.backward()

    # At least one prefix parameter should have a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in prefix_model.parameters()
    )
    assert has_grad, "At least one prefix parameter should have a non-zero gradient after backward()"


# ===========================================================================
# New tests (13-31) -- PrefixEncoderConfig, PrefixEncoder, prepend_prefix_to_kv,
#                      PrefixTuningModel, PrefixTuner
# ===========================================================================

# ---------------------------------------------------------------------------
# 13. test_prefix_encoder_config_defaults
# ---------------------------------------------------------------------------

def test_prefix_encoder_config_defaults():
    cfg = PrefixEncoderConfig()
    assert cfg.prefix_length == 10
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
    assert cfg.dropout == 0.1
    assert cfg.reparameterize is True
    assert cfg.reparam_hidden == 512


# ---------------------------------------------------------------------------
# 14. test_prefix_encoder_output_shape_with_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_output_shape_with_reparam():
    cfg = PrefixEncoderConfig(
        prefix_length=10, d_model=64, n_layers=2,
        dropout=0.0, reparameterize=True, reparam_hidden=512,
    )
    encoder = PrefixEncoder(cfg)
    out = encoder()
    assert out.shape == (2, 2, 10, 64), (
        f"Expected (n_layers=2, 2, prefix_length=10, d_model=64), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 15. test_prefix_encoder_output_shape_without_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_output_shape_without_reparam():
    cfg = PrefixEncoderConfig(
        prefix_length=8, d_model=32, n_layers=3,
        dropout=0.0, reparameterize=False, reparam_hidden=256,
    )
    encoder = PrefixEncoder(cfg)
    out = encoder()
    assert out.shape == (3, 2, 8, 32), (
        f"Expected (n_layers=3, 2, prefix_length=8, d_model=32), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 16. test_prefix_encoder_gradients_flow_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_gradients_flow_reparam():
    cfg = PrefixEncoderConfig(
        prefix_length=5, d_model=16, n_layers=2,
        dropout=0.0, reparameterize=True, reparam_hidden=64,
    )
    encoder = PrefixEncoder(cfg)
    out = encoder()
    loss = out.sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in encoder.parameters()
    )
    assert has_grad, "Gradients should flow through reparameterized PrefixEncoder"


# ---------------------------------------------------------------------------
# 17. test_prefix_encoder_gradients_flow_no_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_gradients_flow_no_reparam():
    cfg = PrefixEncoderConfig(
        prefix_length=5, d_model=16, n_layers=2,
        dropout=0.0, reparameterize=False, reparam_hidden=64,
    )
    encoder = PrefixEncoder(cfg)
    out = encoder()
    loss = out.sum()
    loss.backward()

    assert encoder.prefix_params.grad is not None, (
        "Direct prefix_params should have gradients"
    )
    assert encoder.prefix_params.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 18. test_prefix_encoder_has_mlp_when_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_has_mlp_when_reparam():
    cfg = PrefixEncoderConfig(reparameterize=True)
    encoder = PrefixEncoder(cfg)
    assert hasattr(encoder, 'mlp'), "Reparameterized encoder should have MLP"
    assert hasattr(encoder, 'embedding'), "Reparameterized encoder should have embedding"


# ---------------------------------------------------------------------------
# 19. test_prefix_encoder_has_direct_params_when_no_reparam
# ---------------------------------------------------------------------------

def test_prefix_encoder_has_direct_params_when_no_reparam():
    cfg = PrefixEncoderConfig(reparameterize=False)
    encoder = PrefixEncoder(cfg)
    assert hasattr(encoder, 'prefix_params'), (
        "Non-reparameterized encoder should have prefix_params"
    )
    assert not hasattr(encoder, 'mlp'), (
        "Non-reparameterized encoder should not have MLP"
    )


# ---------------------------------------------------------------------------
# 20. test_prepend_prefix_to_kv_with_past
# ---------------------------------------------------------------------------

def test_prepend_prefix_to_kv_with_past():
    cfg = PrefixEncoderConfig(
        prefix_length=4, d_model=16, n_layers=2, dropout=0.0,
    )
    encoder = PrefixEncoder(cfg)
    prefix_kv = encoder()  # (2, 2, 4, 16)

    B, S, D = 3, 10, 16
    past_k = torch.randn(B, S, D)
    past_v = torch.randn(B, S, D)

    new_k, new_v = prepend_prefix_to_kv((past_k, past_v), prefix_kv, layer_idx=0)

    assert new_k.shape == (B, 4 + S, D), f"Expected (3, 14, 16), got {new_k.shape}"
    assert new_v.shape == (B, 4 + S, D), f"Expected (3, 14, 16), got {new_v.shape}"


# ---------------------------------------------------------------------------
# 21. test_prepend_prefix_to_kv_without_past
# ---------------------------------------------------------------------------

def test_prepend_prefix_to_kv_without_past():
    cfg = PrefixEncoderConfig(
        prefix_length=6, d_model=32, n_layers=3, dropout=0.0,
    )
    encoder = PrefixEncoder(cfg)
    prefix_kv = encoder()  # (3, 2, 6, 32)

    new_k, new_v = prepend_prefix_to_kv(None, prefix_kv, layer_idx=1)

    # Should have batch dim = 1
    assert new_k.shape == (1, 6, 32), f"Expected (1, 6, 32), got {new_k.shape}"
    assert new_v.shape == (1, 6, 32), f"Expected (1, 6, 32), got {new_v.shape}"


# ---------------------------------------------------------------------------
# 22. test_prepend_prefix_to_kv_preserves_original
# ---------------------------------------------------------------------------

def test_prepend_prefix_to_kv_preserves_original():
    cfg = PrefixEncoderConfig(
        prefix_length=3, d_model=8, n_layers=2, dropout=0.0,
    )
    encoder = PrefixEncoder(cfg)
    prefix_kv = encoder()

    B, S, D = 2, 5, 8
    past_k = torch.randn(B, S, D)
    past_v = torch.randn(B, S, D)

    new_k, new_v = prepend_prefix_to_kv((past_k, past_v), prefix_kv, layer_idx=0)

    # The original KV should be preserved in the latter part
    assert torch.allclose(new_k[:, 3:, :], past_k), "Original keys should be preserved"
    assert torch.allclose(new_v[:, 3:, :], past_v), "Original values should be preserved"


# ---------------------------------------------------------------------------
# 23. test_prefix_tuning_model_freezes_backbone
# ---------------------------------------------------------------------------

def test_prefix_tuning_model_freezes_backbone():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)

    assert pt_model._is_backbone_frozen(), "Backbone should be frozen after init"

    # Prefix encoder should be trainable
    for p in pt_model.prefix_encoder.parameters():
        assert p.requires_grad, "Prefix encoder params should be trainable"


# ---------------------------------------------------------------------------
# 24. test_prefix_tuning_model_trainable_count
# ---------------------------------------------------------------------------

def test_prefix_tuning_model_trainable_count():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)

    n_trainable = pt_model.trainable_param_count()
    n_total = pt_model.total_param_count()

    assert n_trainable > 0, "Should have trainable prefix parameters"
    assert n_total > n_trainable, "Total should exceed trainable (backbone is frozen but counted)"
    # Prefix params should be fewer than total (backbone has its own params)
    assert n_trainable < n_total, (
        "Trainable prefix params should be fewer than total params"
    )


# ---------------------------------------------------------------------------
# 25. test_prefix_tuning_model_get_prefix_logits_shape
# ---------------------------------------------------------------------------

def test_prefix_tuning_model_get_prefix_logits_shape():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)
    pt_model.eval()

    B, T = 2, 20
    input_ids = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        logits = pt_model.get_prefix_logits(input_ids)

    assert logits.shape == (B, T, 256), (
        f"Expected logits shape (2, 20, 256), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# 26. test_prefix_tuner_train_step_returns_dict
# ---------------------------------------------------------------------------

def test_prefix_tuner_train_step_returns_dict():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)
    tuner = PrefixTuner(pt_model, lr=1e-3)

    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))

    result = tuner.train_step(input_ids)

    assert isinstance(result, dict), "train_step should return a dict"
    assert "loss" in result, "Result should contain 'loss'"
    assert "n_prefix_params" in result, "Result should contain 'n_prefix_params'"
    assert isinstance(result["loss"], float), "Loss should be a float"
    assert result["n_prefix_params"] > 0, "Should have positive prefix param count"


# ---------------------------------------------------------------------------
# 27. test_prefix_tuner_loss_decreases
# ---------------------------------------------------------------------------

def test_prefix_tuner_loss_decreases():
    torch.manual_seed(42)
    backbone = make_tiny_model()
    cfg = PrefixEncoderConfig(
        prefix_length=5, d_model=64, n_layers=2,
        dropout=0.0, reparameterize=True, reparam_hidden=128,
    )
    pt_model = PrefixTuningModel(backbone, cfg)
    tuner = PrefixTuner(pt_model, lr=1e-2)

    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))

    losses = []
    for _ in range(5):
        result = tuner.train_step(input_ids)
        losses.append(result["loss"])

    # Loss should generally decrease over steps (at least final < initial)
    assert losses[-1] < losses[0], (
        f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 28. test_prefix_tuner_backbone_stays_frozen
# ---------------------------------------------------------------------------

def test_prefix_tuner_backbone_stays_frozen():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)
    tuner = PrefixTuner(pt_model, lr=1e-3)

    # Record backbone weights before training
    backbone_weights_before = {
        name: p.clone()
        for name, p in backbone.named_parameters()
    }

    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))
    tuner.train_step(input_ids)

    # Verify backbone weights did not change
    for name, p in backbone.named_parameters():
        assert torch.equal(p, backbone_weights_before[name]), (
            f"Backbone parameter '{name}' changed during training!"
        )


# ---------------------------------------------------------------------------
# 29. test_prefix_tuner_with_custom_optimizer
# ---------------------------------------------------------------------------

def test_prefix_tuner_with_custom_optimizer():
    backbone = make_tiny_model()
    pt_model = PrefixTuningModel(backbone, TINY_PREFIX_CFG)
    custom_opt = torch.optim.SGD(pt_model.prefix_encoder.parameters(), lr=0.01)
    tuner = PrefixTuner(pt_model, optimizer=custom_opt)

    assert tuner.optimizer is custom_opt, "Should use the custom optimizer"

    B, T = 2, 16
    input_ids = torch.randint(0, 256, (B, T))
    result = tuner.train_step(input_ids)
    assert "loss" in result


# ---------------------------------------------------------------------------
# 30. test_prefix_encoder_different_layer_counts
# ---------------------------------------------------------------------------

def test_prefix_encoder_different_layer_counts():
    for n_layers in [1, 4, 8]:
        cfg = PrefixEncoderConfig(
            prefix_length=5, d_model=16, n_layers=n_layers,
            dropout=0.0, reparameterize=True, reparam_hidden=64,
        )
        encoder = PrefixEncoder(cfg)
        out = encoder()
        assert out.shape == (n_layers, 2, 5, 16), (
            f"For n_layers={n_layers}: expected ({n_layers}, 2, 5, 16), got {out.shape}"
        )


# ---------------------------------------------------------------------------
# 31. test_prepend_prefix_to_kv_different_layers
# ---------------------------------------------------------------------------

def test_prepend_prefix_to_kv_different_layers():
    cfg = PrefixEncoderConfig(
        prefix_length=4, d_model=16, n_layers=3, dropout=0.0,
    )
    encoder = PrefixEncoder(cfg)
    prefix_kv = encoder()

    B, S, D = 2, 8, 16
    past_k = torch.randn(B, S, D)
    past_v = torch.randn(B, S, D)

    # Check that different layers produce different prefix values
    k0, v0 = prepend_prefix_to_kv((past_k, past_v), prefix_kv, layer_idx=0)
    k1, v1 = prepend_prefix_to_kv((past_k, past_v), prefix_kv, layer_idx=1)
    k2, v2 = prepend_prefix_to_kv((past_k, past_v), prefix_kv, layer_idx=2)

    # Prefix portions should differ across layers (with overwhelming probability)
    prefix_k0 = k0[:, :4, :]
    prefix_k1 = k1[:, :4, :]
    prefix_k2 = k2[:, :4, :]

    assert not torch.allclose(prefix_k0, prefix_k1, atol=1e-5), (
        "Different layers should have different prefix keys"
    )
    assert not torch.allclose(prefix_k1, prefix_k2, atol=1e-5), (
        "Different layers should have different prefix keys"
    )

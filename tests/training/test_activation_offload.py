"""Tests for src/training/activation_offload.py — 12 tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.activation_offload import (
    CPUOffloadTensor,
    GradientCheckpointWrapper,
    MemoryEfficientTrainer,
    MemoryTracker,
    OffloadConfig,
    apply_selective_checkpointing,
    get_dtype,
    selective_checkpoint_layers,
)

# ---------------------------------------------------------------------------
# Shared small config
# ---------------------------------------------------------------------------

SMALL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


# ---------------------------------------------------------------------------
# 1. OffloadConfig defaults
# ---------------------------------------------------------------------------


def test_offload_config_defaults():
    cfg = OffloadConfig()
    assert cfg.offload_to_cpu is True
    assert cfg.pin_memory is True
    assert cfg.checkpoint_layers == []
    assert cfg.checkpoint_ratio == 0.5
    assert cfg.profile_memory is False
    assert cfg.dtype == "float32"


# ---------------------------------------------------------------------------
# 2. get_dtype — float32
# ---------------------------------------------------------------------------


def test_get_dtype_float32():
    assert get_dtype("float32") is torch.float32


# ---------------------------------------------------------------------------
# 3. get_dtype — bfloat16
# ---------------------------------------------------------------------------


def test_get_dtype_bfloat16():
    assert get_dtype("bfloat16") is torch.bfloat16


# ---------------------------------------------------------------------------
# 4. CPUOffloadTensor — restored tensor has same shape
# ---------------------------------------------------------------------------


def test_cpu_offload_tensor_restore_shape():
    t = torch.randn(3, 4, 5)
    offloaded = CPUOffloadTensor(t, pin_memory=False)
    restored = offloaded.restore()
    assert restored.shape == t.shape


# ---------------------------------------------------------------------------
# 5. CPUOffloadTensor — restored values match original
# ---------------------------------------------------------------------------


def test_cpu_offload_tensor_restore_values():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    offloaded = CPUOffloadTensor(t, pin_memory=False)
    restored = offloaded.restore()
    assert torch.allclose(t, restored), "Restored tensor values must match original"


# ---------------------------------------------------------------------------
# 6. selective_checkpoint_layers — explicit checkpoint_layers respected
# ---------------------------------------------------------------------------


def test_selective_checkpoint_explicit():
    cfg = OffloadConfig(checkpoint_layers=[0, 3, 5])
    result = selective_checkpoint_layers(8, cfg)
    assert result == [0, 3, 5]


# ---------------------------------------------------------------------------
# 7. selective_checkpoint_layers — ratio=0.5 gives ~half the layers
# ---------------------------------------------------------------------------


def test_selective_checkpoint_ratio():
    cfg = OffloadConfig(checkpoint_layers=[], checkpoint_ratio=0.5)
    n_layers = 8
    result = selective_checkpoint_layers(n_layers, cfg)
    # ratio=0.5 → every other layer: [0, 2, 4, 6]
    assert result == [0, 2, 4, 6]
    assert len(result) == n_layers // 2


# ---------------------------------------------------------------------------
# 8. MemoryTracker.snapshot — returns a float
# ---------------------------------------------------------------------------


def test_memory_tracker_snapshot():
    tracker = MemoryTracker()
    value = tracker.snapshot()
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# 9. MemoryTracker.summary — has expected keys
# ---------------------------------------------------------------------------


def test_memory_tracker_summary_keys():
    tracker = MemoryTracker()
    tracker.snapshot()
    stats = tracker.summary()
    assert "peak_mb" in stats
    assert "mean_mb" in stats
    assert "n_snapshots" in stats


# ---------------------------------------------------------------------------
# 10. GradientCheckpointWrapper — output shape matches unwrapped
# ---------------------------------------------------------------------------


def test_gradient_checkpoint_wrapper_output_shape():
    linear = nn.Linear(16, 16)
    wrapper = GradientCheckpointWrapper(linear, enabled=True)
    # Use eval mode to avoid reentrant issues in tests
    wrapper.eval()

    x = torch.randn(2, 16)
    with torch.no_grad():
        out_wrapped = wrapper(x)
        out_plain = linear(x)

    assert out_wrapped.shape == out_plain.shape


# ---------------------------------------------------------------------------
# 11. apply_selective_checkpointing — selected layers are wrapped
# ---------------------------------------------------------------------------


def test_apply_selective_checkpointing_wraps_layers():
    layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(8)])
    cfg = OffloadConfig(checkpoint_layers=[], checkpoint_ratio=0.5)

    new_layers = apply_selective_checkpointing(layers, cfg)

    assert len(new_layers) == len(layers)
    checkpoint_indices = {0, 2, 4, 6}
    for i, layer in enumerate(new_layers):
        if i in checkpoint_indices:
            assert isinstance(layer, GradientCheckpointWrapper), (
                f"Layer {i} should be wrapped, got {type(layer)}"
            )
        else:
            assert not isinstance(layer, GradientCheckpointWrapper), (
                f"Layer {i} should NOT be wrapped"
            )


# ---------------------------------------------------------------------------
# 12. MemoryEfficientTrainer.train_step — returns loss and memory_mb
# ---------------------------------------------------------------------------


def test_memory_efficient_trainer_step_keys():
    model = AureliusTransformer(SMALL_CFG)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    config = OffloadConfig()

    trainer = MemoryEfficientTrainer(model, optimizer, config)

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, SMALL_CFG.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, SMALL_CFG.vocab_size, (batch_size, seq_len))

    result = trainer.train_step(input_ids, labels)

    assert "loss" in result, "result must contain 'loss'"
    assert "memory_mb" in result, "result must contain 'memory_mb'"
    assert isinstance(result["loss"], float)
    assert isinstance(result["memory_mb"], float)

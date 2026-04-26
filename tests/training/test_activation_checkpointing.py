"""Tests for activation checkpointing utilities.

Covers: CheckpointConfig, checkpoint_forward, CheckpointedSequential,
estimate_memory_savings, apply_activation_checkpointing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.activation_checkpointing import (
    CheckpointConfig,
    CheckpointedSequential,
    apply_activation_checkpointing,
    checkpoint_forward,
    estimate_memory_savings,
)

# Tiny dimensions used throughout
N_LAYERS = 4
D_MODEL = 16
SEQ_LEN = 8
BATCH = 2


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def make_linear_block(in_features=D_MODEL, out_features=D_MODEL):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())


def make_input(requires_grad=False):
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    if requires_grad:
        x.requires_grad_(True)
    return x


# ---------------------------------------------------------------------------
# 1. CheckpointConfig defaults correct
# ---------------------------------------------------------------------------


def test_checkpoint_config_defaults():
    cfg = CheckpointConfig()
    assert cfg.checkpoint_every_n_layers == 1
    assert cfg.offload_to_cpu is False
    assert cfg.use_reentrant is False


# ---------------------------------------------------------------------------
# 2. checkpoint_forward runs without error, output matches direct call
# ---------------------------------------------------------------------------


def test_checkpoint_forward_output_matches():
    torch.manual_seed(0)
    fn = nn.Linear(D_MODEL, D_MODEL)
    fn.eval()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        expected = fn(x)
        actual = checkpoint_forward(fn, x, use_reentrant=False)

    assert torch.allclose(expected, actual, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. checkpoint_forward gradient flows (backward completes)
# ---------------------------------------------------------------------------


def test_checkpoint_forward_gradient_flows():
    torch.manual_seed(1)
    fn = nn.Linear(D_MODEL, D_MODEL)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)

    out = checkpoint_forward(fn, x, use_reentrant=False)
    out.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 4. CheckpointedSequential forward output shape same as sequential
# ---------------------------------------------------------------------------


def test_checkpointed_sequential_output_shape():
    torch.manual_seed(2)
    modules = [make_linear_block() for _ in range(N_LAYERS)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential(modules, cfg)

    x = make_input()
    out = seq(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 5. CheckpointedSequential backward completes (no error)
# ---------------------------------------------------------------------------


def test_checkpointed_sequential_backward_completes():
    torch.manual_seed(3)
    modules = [make_linear_block() for _ in range(N_LAYERS)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential(modules, cfg)

    x = make_input(requires_grad=True)
    out = seq(x)
    out.sum().backward()

    assert x.grad is not None


# ---------------------------------------------------------------------------
# 6. CheckpointedSequential checkpoints correct fraction of modules
# ---------------------------------------------------------------------------


def test_checkpointed_sequential_correct_fraction():
    torch.manual_seed(4)
    modules = [make_linear_block() for _ in range(N_LAYERS)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=2)
    seq = CheckpointedSequential(modules, cfg)

    x = make_input()
    out = seq(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 7. estimate_memory_savings returns required keys
# ---------------------------------------------------------------------------


def test_estimate_memory_savings_keys():
    result = estimate_memory_savings(N_LAYERS, D_MODEL, SEQ_LEN, BATCH)
    assert "activation_bytes_no_checkpoint" in result
    assert "activation_bytes_with_checkpoint" in result
    assert "savings_fraction" in result


# ---------------------------------------------------------------------------
# 8. savings_fraction = 0.0 when checkpoint_every_n=1 and n_layers=1
# ---------------------------------------------------------------------------


def test_estimate_memory_savings_zero_savings_single_layer():
    result = estimate_memory_savings(
        n_layers=1, d_model=D_MODEL, seq_len=SEQ_LEN, batch_size=BATCH, checkpoint_every_n=1
    )
    assert result["savings_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 9. savings_fraction in [0, 1]
# ---------------------------------------------------------------------------


def test_estimate_memory_savings_fraction_in_range():
    result = estimate_memory_savings(N_LAYERS, D_MODEL, SEQ_LEN, BATCH, checkpoint_every_n=2)
    sf = result["savings_fraction"]
    assert 0.0 <= sf <= 1.0


# ---------------------------------------------------------------------------
# 10. no_checkpoint > with_checkpoint when n_layers>1 and every_n=2
# ---------------------------------------------------------------------------


def test_estimate_memory_savings_no_ckpt_greater():
    result = estimate_memory_savings(
        n_layers=4, d_model=D_MODEL, seq_len=SEQ_LEN, batch_size=BATCH, checkpoint_every_n=2
    )
    assert result["activation_bytes_no_checkpoint"] > result["activation_bytes_with_checkpoint"]


# ---------------------------------------------------------------------------
# 11. apply_activation_checkpointing returns correct count
# ---------------------------------------------------------------------------


def test_apply_activation_checkpointing_count():
    torch.manual_seed(5)
    model = nn.Sequential(*[nn.Linear(D_MODEL, D_MODEL) for _ in range(N_LAYERS)])
    cfg = CheckpointConfig()
    count = apply_activation_checkpointing(model, nn.Linear, cfg)
    assert count == N_LAYERS


# ---------------------------------------------------------------------------
# 12. apply_activation_checkpointing wrapped modules still produce output
# ---------------------------------------------------------------------------


def test_apply_activation_checkpointing_produces_output():
    torch.manual_seed(6)
    model = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
    cfg = CheckpointConfig()
    apply_activation_checkpointing(model, nn.Linear, cfg)

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = model(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 13. apply_activation_checkpointing backward completes after wrapping
# ---------------------------------------------------------------------------


def test_apply_activation_checkpointing_backward():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, D_MODEL))
    cfg = CheckpointConfig()
    apply_activation_checkpointing(model, nn.Linear, cfg)

    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert x.grad is not None


# ---------------------------------------------------------------------------
# 14. CheckpointedSequential single module list forward
# ---------------------------------------------------------------------------


def test_checkpointed_sequential_single_module():
    torch.manual_seed(8)
    module = make_linear_block()
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential([module], cfg)

    x = make_input()
    out = seq(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 15. estimate_memory_savings checkpoint_every_n=n_layers (only checkpoint 1)
# ---------------------------------------------------------------------------


def test_estimate_memory_savings_every_n_equals_n_layers():
    n = N_LAYERS
    result = estimate_memory_savings(
        n_layers=n, d_model=D_MODEL, seq_len=SEQ_LEN, batch_size=BATCH, checkpoint_every_n=n
    )
    expected_with = 4 * 1 * SEQ_LEN * BATCH * D_MODEL
    assert result["activation_bytes_with_checkpoint"] == expected_with
    expected_sf = 1.0 - (1.0 / n)
    assert abs(result["savings_fraction"] - expected_sf) < 1e-9


# ---------------------------------------------------------------------------
# 16. CheckpointedSequential forward output dtype preserved
# ---------------------------------------------------------------------------


def test_checkpointed_sequential_dtype_preserved():
    torch.manual_seed(9)
    modules = [make_linear_block() for _ in range(2)]
    cfg = CheckpointConfig(checkpoint_every_n_layers=1)
    seq = CheckpointedSequential(modules, cfg)

    x = make_input().float()
    out = seq(x)
    assert out.dtype == torch.float32

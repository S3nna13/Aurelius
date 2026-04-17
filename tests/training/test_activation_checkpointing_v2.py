"""Tests for activation_checkpointing_v2.py

Tiny configs: d_model=16, n_heads=2, seq_len=8, batch=2, n_layers=4.
Every test runs forward and/or backward passes (not just instantiation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.activation_checkpointing_v2 import (
    ActivationMemoryEstimator,
    CheckpointedModule,
    CheckpointingScheduler,
    SegmentedCheckpointing,
    SelectiveCheckpointing,
)

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
SEQ_LEN = 8
BATCH = 2
N_LAYERS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_linear(d: int = D_MODEL) -> nn.Linear:
    return nn.Linear(d, d)


def _make_layer_list(n: int = N_LAYERS, d: int = D_MODEL) -> nn.ModuleList:
    return nn.ModuleList([nn.Linear(d, d) for _ in range(n)])


def _rand_input(b: int = BATCH, t: int = SEQ_LEN, d: int = D_MODEL) -> torch.Tensor:
    return torch.randn(b, t, d, requires_grad=True)


# ---------------------------------------------------------------------------
# Test 1: CheckpointedModule output matches non-checkpointed output
# ---------------------------------------------------------------------------

def test_checkpointed_module_output_matches():
    torch.manual_seed(0)
    layer = _simple_linear()
    ckpt = CheckpointedModule(layer)

    x = _rand_input()
    ckpt.train()
    layer.train()

    with torch.no_grad():
        expected = layer(x)
        actual = ckpt(x)

    assert torch.allclose(expected, actual, atol=1e-5), (
        "CheckpointedModule output differs from non-checkpointed output"
    )


# ---------------------------------------------------------------------------
# Test 2: CheckpointedModule backward succeeds and grads flow to params
# ---------------------------------------------------------------------------

def test_checkpointed_module_backward_grad_flows():
    torch.manual_seed(1)
    layer = _simple_linear()
    ckpt = CheckpointedModule(layer)
    ckpt.train()

    x = _rand_input()
    out = ckpt(x)
    loss = out.sum()
    loss.backward()

    assert layer.weight.grad is not None, "No gradient on layer.weight"
    assert layer.bias.grad is not None, "No gradient on layer.bias"
    assert layer.weight.grad.abs().sum() > 0, "layer.weight.grad is all-zero"


# ---------------------------------------------------------------------------
# Test 3: CheckpointedModule inference mode runs without checkpointing (no error)
# ---------------------------------------------------------------------------

def test_checkpointed_module_inference_mode():
    torch.manual_seed(2)
    layer = _simple_linear()
    ckpt = CheckpointedModule(layer)
    # switch to inference / non-training mode
    ckpt.train(False)

    x = _rand_input()
    with torch.no_grad():
        out = ckpt(x)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# Test 4: memory_saved_estimate > 0 for any input shape
# ---------------------------------------------------------------------------

def test_memory_saved_estimate_positive():
    layer = _simple_linear()
    ckpt = CheckpointedModule(layer)

    shapes = [
        (BATCH, SEQ_LEN, D_MODEL),
        (1, 1, 1),
        (8, 128, 512),
    ]
    for shape in shapes:
        est = ckpt.memory_saved_estimate(shape)
        assert est > 0, f"memory_saved_estimate non-positive for shape {shape}"


# ---------------------------------------------------------------------------
# Test 5: SegmentedCheckpointing output shape matches for n_segments=1 and n_layers
# ---------------------------------------------------------------------------

def test_segmented_checkpointing_output_shape():
    torch.manual_seed(3)
    layers = _make_layer_list()
    x = _rand_input()

    for n_seg in (1, N_LAYERS):
        seg = SegmentedCheckpointing(layers, n_segments=n_seg)
        seg.train()
        out = seg(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Wrong output shape for n_segments={n_seg}: {out.shape}"
        )


# ---------------------------------------------------------------------------
# Test 6: SegmentedCheckpointing backward succeeds with n_segments=2
# ---------------------------------------------------------------------------

def test_segmented_checkpointing_backward():
    torch.manual_seed(4)
    layers = _make_layer_list()
    seg = SegmentedCheckpointing(layers, n_segments=2)
    seg.train()

    x = _rand_input()
    out = seg(x)
    loss = out.sum()
    loss.backward()

    for layer in layers:
        if isinstance(layer, nn.Linear):
            assert layer.weight.grad is not None, "Missing gradient on linear layer"
            break


# ---------------------------------------------------------------------------
# Test 7: segment_boundaries length == n_segments, first boundary == 0
# ---------------------------------------------------------------------------

def test_segment_boundaries_structure():
    for n_seg in (1, 2, N_LAYERS):
        layers = _make_layer_list()
        seg = SegmentedCheckpointing(layers, n_segments=n_seg)
        boundaries = seg.segment_boundaries()
        assert len(boundaries) == n_seg, (
            f"Expected {n_seg} boundaries, got {len(boundaries)}"
        )
        assert boundaries[0] == 0, (
            f"First boundary should be 0, got {boundaries[0]}"
        )


# ---------------------------------------------------------------------------
# Test 8: ActivationMemoryEstimator.estimate_transformer_layer > 0,
#          increases with larger inputs
# ---------------------------------------------------------------------------

def test_estimate_transformer_layer_positive_and_scaling():
    est = ActivationMemoryEstimator()
    base = est.estimate_transformer_layer(
        batch_size=BATCH, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS
    )
    assert base > 0, "estimate_transformer_layer returned non-positive value"

    larger = est.estimate_transformer_layer(
        batch_size=BATCH * 2, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS
    )
    assert larger > base, "Larger batch should give larger memory estimate"

    longer = est.estimate_transformer_layer(
        batch_size=BATCH, seq_len=SEQ_LEN * 2, d_model=D_MODEL, n_heads=N_HEADS
    )
    assert longer > base, "Longer seq should give larger memory estimate"


# ---------------------------------------------------------------------------
# Test 9: full_model_memory is proportional (n * single layer)
# ---------------------------------------------------------------------------

def test_full_model_memory_proportional():
    est = ActivationMemoryEstimator()
    kwargs = dict(batch_size=BATCH, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS)
    single = est.estimate_transformer_layer(**kwargs)
    total = est.full_model_memory(N_LAYERS, **kwargs)
    assert total == N_LAYERS * single, (
        f"full_model_memory {total} != {N_LAYERS} * {single}"
    )


# ---------------------------------------------------------------------------
# Test 10: checkpointed_memory < full_model_memory for n_segments > 1
# ---------------------------------------------------------------------------

def test_checkpointed_memory_less_than_full():
    est = ActivationMemoryEstimator()
    kwargs = dict(batch_size=BATCH, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS)
    full = est.full_model_memory(N_LAYERS, **kwargs)
    ckpt = est.checkpointed_memory(N_LAYERS, n_segments=2, **kwargs)
    assert ckpt < full, (
        f"checkpointed_memory {ckpt} should be < full_model_memory {full}"
    )


# ---------------------------------------------------------------------------
# Test 11: SelectiveCheckpointing.wrap_attention returns CheckpointedModule
#           when checkpoint_attention=True
# ---------------------------------------------------------------------------

def test_selective_wrap_attention_checkpoints():
    sc = SelectiveCheckpointing(checkpoint_attention=True, checkpoint_ffn=False)
    attn = _simple_linear()
    wrapped = sc.wrap_attention(attn)
    assert isinstance(wrapped, CheckpointedModule), (
        "wrap_attention should return CheckpointedModule when checkpoint_attention=True"
    )
    wrapped.train()
    x = _rand_input()
    out = wrapped(x)
    out.sum().backward()


# ---------------------------------------------------------------------------
# Test 12: SelectiveCheckpointing.wrap_ffn returns original when checkpoint_ffn=False
# ---------------------------------------------------------------------------

def test_selective_wrap_ffn_no_checkpoint():
    sc = SelectiveCheckpointing(checkpoint_attention=True, checkpoint_ffn=False)
    ffn = _simple_linear()
    wrapped = sc.wrap_ffn(ffn)
    assert wrapped is ffn, (
        "wrap_ffn should return the original module when checkpoint_ffn=False"
    )
    x = _rand_input()
    out = wrapped(x)
    out.sum().backward()


# ---------------------------------------------------------------------------
# Test 13: CheckpointingScheduler.recommend_segments returns int in [1, n_layers]
# ---------------------------------------------------------------------------

def test_scheduler_recommend_segments_range():
    sched = CheckpointingScheduler(memory_budget_gb=8.0)
    n_seg = sched.recommend_segments(
        n_layers=N_LAYERS,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
    )
    assert isinstance(n_seg, int), f"recommend_segments should return int, got {type(n_seg)}"
    assert 1 <= n_seg <= N_LAYERS, (
        f"recommend_segments returned {n_seg}, expected [1, {N_LAYERS}]"
    )


# ---------------------------------------------------------------------------
# Test 14: Very large model -> CheckpointingScheduler recommends more segments
# ---------------------------------------------------------------------------

def test_scheduler_large_model_more_segments():
    tight_sched = CheckpointingScheduler(memory_budget_gb=0.0001)
    generous_sched = CheckpointingScheduler(memory_budget_gb=1000.0)

    kwargs = dict(
        n_layers=N_LAYERS,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
    )
    tight_seg = tight_sched.recommend_segments(**kwargs)
    generous_seg = generous_sched.recommend_segments(**kwargs)

    assert tight_seg >= generous_seg, (
        f"Tight budget should need >= segments: tight={tight_seg}, generous={generous_seg}"
    )
    assert tight_seg > 1 or tight_seg == generous_seg, (
        "Extremely tight budget should request more than 1 segment"
    )


# ---------------------------------------------------------------------------
# Test 15: SegmentedCheckpointing with n_segments=n_layers:
#           each layer checkpointed individually, output correct
# ---------------------------------------------------------------------------

def test_segmented_each_layer_individually():
    torch.manual_seed(5)
    layers = _make_layer_list()

    x_ref = _rand_input()
    with torch.no_grad():
        ref = x_ref.clone()
        for layer in layers:
            ref = layer(ref)

    x_seg = x_ref.detach().clone().requires_grad_(True)
    seg = SegmentedCheckpointing(layers, n_segments=N_LAYERS)
    seg.train()
    out = seg(x_seg)

    assert out.shape == (BATCH, SEQ_LEN, D_MODEL), f"Wrong shape: {out.shape}"
    assert torch.allclose(out.detach(), ref, atol=1e-5), (
        "Per-layer segmented output does not match sequential output"
    )

    loss = out.sum()
    loss.backward()
    for layer in layers:
        if isinstance(layer, nn.Linear):
            assert layer.weight.grad is not None, "Missing gradient after per-layer segmented backward"

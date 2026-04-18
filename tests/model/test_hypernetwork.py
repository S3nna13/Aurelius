"""
Tests for src/model/hypernetwork.py
Covers: WeightGenerator, HyperLinear, HyperAttention,
        HyperTransformerBlock, TaskConditionedLM, HyperNetConfig
"""

import math
import torch
import pytest

from src.model.hypernetwork import (
    WeightGenerator,
    HyperLinear,
    HyperAttention,
    HyperTransformerBlock,
    TaskConditionedLM,
    HyperNetConfig,
)

# ---------------------------------------------------------------------------
# Tiny config constants
# ---------------------------------------------------------------------------
D_MODEL    = 16
VOCAB_SIZE = 16
N_LAYERS   = 2
N_HEADS    = 4
D_CONTEXT  = 8
N_TASKS    = 3
B          = 2
T          = 6
HIDDEN     = 16   # small hidden for speed


# ---------------------------------------------------------------------------
# WeightGenerator tests
# ---------------------------------------------------------------------------

def test_weight_generator_output_shape_1d():
    """WeightGenerator produces [B, out_features] for a 1-D target."""
    gen = WeightGenerator(D_CONTEXT, (D_MODEL,), hidden=HIDDEN)
    ctx = torch.randn(B, D_CONTEXT)
    out = gen(ctx)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


def test_weight_generator_output_shape_2d():
    """WeightGenerator produces [B, rows, cols] for a 2-D target."""
    rows, cols = 8, 4
    gen = WeightGenerator(D_CONTEXT, (rows, cols), hidden=HIDDEN)
    ctx = torch.randn(B, D_CONTEXT)
    out = gen(ctx)
    assert out.shape == (B, rows, cols), f"Expected ({B}, {rows}, {cols}), got {out.shape}"


def test_weight_generator_multi_dim_target_shape():
    """WeightGenerator handles a 3-D target shape."""
    shape = (4, 3, 2)
    gen = WeightGenerator(D_CONTEXT, shape, hidden=HIDDEN)
    ctx = torch.randn(B, D_CONTEXT)
    out = gen(ctx)
    assert out.shape == (B, *shape), f"Expected {(B,) + shape}, got {out.shape}"


def test_weight_generator_different_contexts_produce_different_weights():
    """Different context vectors should produce different weight tensors."""
    gen = WeightGenerator(D_CONTEXT, (D_MODEL, D_MODEL), hidden=HIDDEN)
    ctx_a = torch.randn(1, D_CONTEXT)
    ctx_b = torch.randn(1, D_CONTEXT)
    # Force them to be different
    while torch.allclose(ctx_a, ctx_b):
        ctx_b = torch.randn(1, D_CONTEXT)

    w_a = gen(ctx_a)
    w_b = gen(ctx_b)
    assert not torch.allclose(w_a, w_b), "Different contexts should give different weights"


# ---------------------------------------------------------------------------
# HyperLinear tests
# ---------------------------------------------------------------------------

def test_hyper_linear_output_shape():
    """HyperLinear forward produces [B, out_features]."""
    in_f, out_f = 8, 12
    layer = HyperLinear(D_CONTEXT, in_f, out_f, hidden=HIDDEN)
    x   = torch.randn(B, in_f)
    ctx = torch.randn(B, D_CONTEXT)
    out = layer(x, ctx)
    assert out.shape == (B, out_f), f"Expected ({B}, {out_f}), got {out.shape}"


def test_hyper_linear_different_contexts_different_outputs():
    """Same input x, different contexts → different HyperLinear outputs."""
    in_f, out_f = 8, 12
    layer = HyperLinear(D_CONTEXT, in_f, out_f, hidden=HIDDEN)
    x     = torch.randn(1, in_f)
    ctx_a = torch.randn(1, D_CONTEXT)
    ctx_b = torch.randn(1, D_CONTEXT)
    while torch.allclose(ctx_a, ctx_b):
        ctx_b = torch.randn(1, D_CONTEXT)

    out_a = layer(x, ctx_a)
    out_b = layer(x, ctx_b)
    assert not torch.allclose(out_a, out_b), "Different contexts should give different outputs"


def test_hyper_linear_gradient_flows_through_context():
    """Gradients should propagate back through the context input."""
    in_f, out_f = 8, 12
    layer = HyperLinear(D_CONTEXT, in_f, out_f, hidden=HIDDEN)
    x   = torch.randn(B, in_f)
    ctx = torch.randn(B, D_CONTEXT, requires_grad=True)
    out = layer(x, ctx)
    loss = out.sum()
    loss.backward()
    assert ctx.grad is not None, "Gradient must flow through context"
    assert ctx.grad.shape == ctx.shape


# ---------------------------------------------------------------------------
# HyperAttention tests
# ---------------------------------------------------------------------------

def test_hyper_attention_output_shape():
    """HyperAttention forward preserves [B, T, d_model] shape."""
    attn = HyperAttention(D_MODEL, D_CONTEXT, N_HEADS, hidden=HIDDEN)
    x   = torch.randn(B, T, D_MODEL)
    ctx = torch.randn(B, D_CONTEXT)
    out = attn(x, ctx)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_hyper_attention_different_contexts_different_outputs():
    """Different context vectors produce different attention outputs."""
    attn  = HyperAttention(D_MODEL, D_CONTEXT, N_HEADS, hidden=HIDDEN)
    x     = torch.randn(1, T, D_MODEL)
    ctx_a = torch.randn(1, D_CONTEXT)
    ctx_b = torch.randn(1, D_CONTEXT)
    while torch.allclose(ctx_a, ctx_b):
        ctx_b = torch.randn(1, D_CONTEXT)

    out_a = attn(x, ctx_a)
    out_b = attn(x, ctx_b)
    assert not torch.allclose(out_a, out_b), "Different contexts should produce different attention outputs"


# ---------------------------------------------------------------------------
# HyperTransformerBlock tests
# ---------------------------------------------------------------------------

def test_hyper_transformer_block_output_shape():
    """HyperTransformerBlock forward preserves [B, T, d_model] shape."""
    block = HyperTransformerBlock(D_MODEL, D_CONTEXT, N_HEADS, hidden=HIDDEN)
    x   = torch.randn(B, T, D_MODEL)
    ctx = torch.randn(B, D_CONTEXT)
    out = block(x, ctx)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_hyper_transformer_block_gradient_flows():
    """Gradients should flow back through all parameters of HyperTransformerBlock."""
    block = HyperTransformerBlock(D_MODEL, D_CONTEXT, N_HEADS, hidden=HIDDEN)
    x   = torch.randn(B, T, D_MODEL, requires_grad=True)
    ctx = torch.randn(B, D_CONTEXT, requires_grad=True)
    out = block(x, ctx)
    loss = out.sum()
    loss.backward()
    assert x.grad   is not None, "Gradient should flow to input x"
    assert ctx.grad is not None, "Gradient should flow to context"


# ---------------------------------------------------------------------------
# TaskConditionedLM tests
# ---------------------------------------------------------------------------

def _make_lm() -> TaskConditionedLM:
    return TaskConditionedLM(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_tasks=N_TASKS,
        d_context=D_CONTEXT,
        n_heads=N_HEADS,
        hidden=HIDDEN,
    )


def _random_input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T))


def _random_task_ids() -> torch.Tensor:
    return torch.randint(0, N_TASKS, (B,))


def test_task_conditioned_lm_forward_output_shape():
    """TaskConditionedLM forward produces [B, T, vocab_size] logits."""
    lm  = _make_lm()
    ids = _random_input_ids()
    tid = _random_task_ids()
    logits = lm(ids, tid)
    assert logits.shape == (B, T, VOCAB_SIZE), (
        f"Expected ({B}, {T}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_task_conditioned_lm_different_task_ids_different_logits():
    """Different task IDs should produce different logits for the same input."""
    lm  = _make_lm()
    ids = _random_input_ids()

    # Use task ids that are definitely different
    task_a = torch.zeros(B, dtype=torch.long)
    task_b = torch.ones(B, dtype=torch.long)

    logits_a = lm(ids, task_a)
    logits_b = lm(ids, task_b)
    assert not torch.allclose(logits_a, logits_b), (
        "Different task IDs must produce different logits"
    )


def test_task_conditioned_lm_compute_loss_finite_positive():
    """compute_loss should return a finite, positive scalar."""
    lm  = _make_lm()
    ids = _random_input_ids()
    tid = _random_task_ids()
    loss = lm.compute_loss(ids, tid)
    assert loss.ndim == 0, "Loss should be a scalar (0-dim tensor)"
    assert torch.isfinite(loss).item(), "Loss should be finite"
    assert loss.item() > 0.0, "Loss should be positive"


def test_task_conditioned_lm_compute_loss_backward():
    """Gradients should flow through compute_loss."""
    lm  = _make_lm()
    ids = _random_input_ids()
    tid = _random_task_ids()
    loss = lm.compute_loss(ids, tid)
    loss.backward()
    # Check at least one parameter received a gradient
    grads = [p.grad for p in lm.parameters() if p.grad is not None]
    assert len(grads) > 0, "At least one parameter should have a gradient after backward()"


def test_task_conditioned_lm_task_embeddings_learnable():
    """task_embeddings should be a registered learnable nn.Embedding."""
    lm = _make_lm()
    assert isinstance(lm.task_embeddings, torch.nn.Embedding), (
        "task_embeddings must be nn.Embedding"
    )
    # Must appear in named parameters
    param_names = [name for name, _ in lm.named_parameters()]
    assert any("task_embeddings" in n for n in param_names), (
        "task_embeddings.weight should be a learnable parameter"
    )
    # Verify shape
    assert lm.task_embeddings.weight.shape == (N_TASKS, D_CONTEXT), (
        f"Expected ({N_TASKS}, {D_CONTEXT}), got {lm.task_embeddings.weight.shape}"
    )


# ---------------------------------------------------------------------------
# HyperNetConfig tests
# ---------------------------------------------------------------------------

def test_hypernetconfig_defaults():
    """HyperNetConfig should have the documented default values."""
    cfg = HyperNetConfig()
    assert cfg.d_model    == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers   == 2
    assert cfg.n_heads    == 4
    assert cfg.d_context  == 16
    assert cfg.n_tasks    == 4
    assert cfg.hidden     == 32


def test_hypernetconfig_custom_values():
    """HyperNetConfig should accept and store custom values."""
    cfg = HyperNetConfig(d_model=64, vocab_size=128, n_layers=4, n_heads=8,
                         d_context=32, n_tasks=10, hidden=64)
    assert cfg.d_model    == 64
    assert cfg.vocab_size == 128
    assert cfg.n_layers   == 4
    assert cfg.n_heads    == 8
    assert cfg.d_context  == 32
    assert cfg.n_tasks    == 10
    assert cfg.hidden     == 64

"""
tests/interpretability/test_function_vectors.py

Tests for src/interpretability/function_vectors.py.

Tiny Aurelius config:
  n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
  d_ff=128, vocab_size=256, max_seq_len=64.

Pure PyTorch -- no HuggingFace, no scipy, no sklearn, no einops.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.interpretability.function_vectors import (
    FunctionVectorExtractor,
    FunctionVectorInjector,
    build_task_fv,
)


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
    tie_embeddings=True,
)

N_HEADS = TINY_CFG.n_heads       # 4
HEAD_DIM = TINY_CFG.head_dim     # 16
D_MODEL = TINY_CFG.d_model       # 64
VOCAB = TINY_CFG.vocab_size      # 256


@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    torch.manual_seed(42)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


def _make_input(n: int = 3, t: int = 8, seed: int = 0) -> Tensor:
    """Random int64 input_ids of shape (n, t)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, VOCAB, (n, t), generator=g)


# ---------------------------------------------------------------------------
# 1. Shape: extract() returns (n_heads, head_dim) for tiny config
# ---------------------------------------------------------------------------

def test_extract_shape(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demos = _make_input(n=3, t=8)
    fv = extractor.extract(demos)
    assert fv.shape == (N_HEADS, HEAD_DIM), (
        f"Expected ({N_HEADS}, {HEAD_DIM}), got {tuple(fv.shape)}"
    )


# ---------------------------------------------------------------------------
# 2. Extraction is deterministic: same input -> same FV
# ---------------------------------------------------------------------------

def test_extract_deterministic(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demos = _make_input(n=3, t=8, seed=7)
    fv1 = extractor.extract(demos)
    fv2 = extractor.extract(demos)
    assert torch.allclose(fv1, fv2, atol=1e-6), "Extraction is not deterministic."


# ---------------------------------------------------------------------------
# 3. FV norms: all head vectors have finite, non-zero norms
# ---------------------------------------------------------------------------

def test_extract_norms_finite_nonzero(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demos = _make_input(n=4, t=10)
    fv = extractor.extract(demos)                   # (n_heads, head_dim)
    norms = fv.norm(dim=-1)                         # (n_heads,)
    assert torch.all(torch.isfinite(norms)), "Some FV norms are not finite."
    assert torch.all(norms > 0), "Some FV norms are zero."


# ---------------------------------------------------------------------------
# 4. Importance scores shape: compute_head_importance returns (n_heads,)
# ---------------------------------------------------------------------------

def test_head_importance_shape(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demos = _make_input(n=3, t=8)
    zero_shot = _make_input(n=1, t=8, seed=99)

    def metric_fn(logits: Tensor) -> float:
        return logits[0, -1, :].mean().item()

    importance = extractor.compute_head_importance(demos, zero_shot, metric_fn)
    assert importance.shape == (N_HEADS,), (
        f"Expected ({N_HEADS},), got {tuple(importance.shape)}"
    )


# ---------------------------------------------------------------------------
# 5. Injection runs: inject() returns logits of correct shape
# ---------------------------------------------------------------------------

def test_inject_output_shape(tiny_model):
    T = 8
    zero_shot = _make_input(n=1, t=T)
    fv = torch.zeros(D_MODEL)
    injector = FunctionVectorInjector(tiny_model, layer=0)
    logits = injector.inject(zero_shot, fv)
    assert logits.shape == (1, T, VOCAB), (
        f"Expected (1, {T}, {VOCAB}), got {tuple(logits.shape)}"
    )


# ---------------------------------------------------------------------------
# 6. Injection changes output: logits with FV != logits without FV
# ---------------------------------------------------------------------------

def test_inject_changes_output(tiny_model):
    T = 8
    zero_shot = _make_input(n=1, t=T)

    with torch.no_grad():
        _, baseline_logits, _ = tiny_model(zero_shot)

    fv = torch.ones(D_MODEL) * 0.5
    injector = FunctionVectorInjector(tiny_model, layer=0)
    patched_logits = injector.inject(zero_shot, fv)

    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-6), (
        "Injecting a non-zero FV should change the output logits."
    )


# ---------------------------------------------------------------------------
# 7. Zero FV injection: injecting zero vector does not change output
# ---------------------------------------------------------------------------

def test_zero_fv_injection_no_change(tiny_model):
    T = 8
    zero_shot = _make_input(n=1, t=T)

    with torch.no_grad():
        _, baseline_logits, _ = tiny_model(zero_shot)

    fv_zero = torch.zeros(D_MODEL)
    injector = FunctionVectorInjector(tiny_model, layer=0)
    patched_logits = injector.inject(zero_shot, fv_zero)

    assert torch.allclose(baseline_logits, patched_logits, atol=1e-6), (
        "Injecting a zero FV should leave logits unchanged."
    )


# ---------------------------------------------------------------------------
# 8. Gradient flow: FV is differentiable w.r.t. model parameters
# ---------------------------------------------------------------------------

def test_gradient_flow():
    torch.manual_seed(0)
    model = AureliusTransformer(TINY_CFG)
    model.train()

    demos = _make_input(n=2, t=8)

    captured = []

    def _hook(_module, _inp, _out):
        captured.append(_inp[0])  # keep gradients -- no detach

    handle = model.layers[0].attn.o_proj.register_forward_hook(_hook)

    try:
        _, logits, _ = model(demos)
    finally:
        handle.remove()

    pre_proj = captured[0]   # (N, T, n_heads*head_dim)
    out = pre_proj.reshape(*pre_proj.shape[:-1], N_HEADS, HEAD_DIM)
    fv = out.mean(dim=(0, 1))   # (n_heads, head_dim)

    loss = fv.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to model parameters."


# ---------------------------------------------------------------------------
# 9. Multiple demonstrations: FV with N>1 is mean over individual FVs
# ---------------------------------------------------------------------------

def test_fv_is_mean_of_individual(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)

    demo1 = _make_input(n=1, t=8, seed=1)
    demo2 = _make_input(n=1, t=8, seed=2)
    demos_combined = torch.cat([demo1, demo2], dim=0)   # (2, 8)

    fv_combined = extractor.extract(demos_combined)

    individual_fvs = []
    for d in [demo1, demo2]:
        cap = []

        def _hook(_m, _inp, _out, _c=cap):
            _c.append(_inp[0].detach())

        handle = tiny_model.layers[0].attn.o_proj.register_forward_hook(_hook)
        with torch.no_grad():
            tiny_model(d)
        handle.remove()

        pre = cap[0]  # (1, T, ...)
        out = pre.reshape(*pre.shape[:-1], N_HEADS, HEAD_DIM)
        individual_fvs.append(out.mean(dim=(0, 1)))

    fv_manual_mean = torch.stack(individual_fvs).mean(dim=0)

    assert torch.allclose(fv_combined, fv_manual_mean, atol=1e-5), (
        "FV for N>1 demos should equal the mean of per-demo FVs."
    )


# ---------------------------------------------------------------------------
# 10. Different layers: extractor/injector at different layers work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", [0, 1])
def test_extract_at_different_layers(tiny_model, layer):
    extractor = FunctionVectorExtractor(tiny_model, layer=layer)
    demos = _make_input(n=2, t=6)
    fv = extractor.extract(demos)
    assert fv.shape == (N_HEADS, HEAD_DIM)
    assert torch.all(torch.isfinite(fv))


@pytest.mark.parametrize("layer", [0, 1])
def test_inject_at_different_layers(tiny_model, layer):
    T = 6
    zero_shot = _make_input(n=1, t=T)
    fv = torch.randn(D_MODEL) * 0.1
    injector = FunctionVectorInjector(tiny_model, layer=layer)
    logits = injector.inject(zero_shot, fv)
    assert logits.shape == (1, T, VOCAB)


# ---------------------------------------------------------------------------
# 11. Edge case: single demonstration (N=1)
# ---------------------------------------------------------------------------

def test_extract_single_demonstration(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demo = _make_input(n=1, t=8)
    fv = extractor.extract(demo)
    assert fv.shape == (N_HEADS, HEAD_DIM)
    assert torch.all(torch.isfinite(fv))


# ---------------------------------------------------------------------------
# 12. No NaN/Inf in extracted FVs or injected logits
# ---------------------------------------------------------------------------

def test_no_nan_inf_in_fv(tiny_model):
    extractor = FunctionVectorExtractor(tiny_model, layer=0)
    demos = _make_input(n=5, t=12)
    fv = extractor.extract(demos)
    assert not torch.any(torch.isnan(fv)), "NaN detected in extracted FV."
    assert not torch.any(torch.isinf(fv)), "Inf detected in extracted FV."


def test_no_nan_inf_in_injected_logits(tiny_model):
    T = 10
    zero_shot = _make_input(n=1, t=T)
    fv = torch.randn(D_MODEL)
    injector = FunctionVectorInjector(tiny_model, layer=0)
    logits = injector.inject(zero_shot, fv)
    assert not torch.any(torch.isnan(logits)), "NaN detected in injected logits."
    assert not torch.any(torch.isinf(logits)), "Inf detected in injected logits."


# ---------------------------------------------------------------------------
# Bonus: build_task_fv convenience function
# ---------------------------------------------------------------------------

def test_build_task_fv_shape(tiny_model):
    demos = _make_input(n=3, t=8)
    fv_task = build_task_fv(tiny_model, demos, layer=0)
    assert fv_task.shape == (D_MODEL,), (
        f"Expected ({D_MODEL},), got {tuple(fv_task.shape)}"
    )


def test_build_task_fv_top_k(tiny_model):
    demos = _make_input(n=3, t=8)
    fv_task = build_task_fv(tiny_model, demos, layer=0, top_k=2)
    assert fv_task.shape == (2 * HEAD_DIM,)   # top-k heads x head_dim
    assert torch.all(torch.isfinite(fv_task))

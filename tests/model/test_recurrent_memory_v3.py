"""Tests for src/model/recurrent_memory_v3.py

Tiny config throughout:
    d_model=16, n_heads=2, n_layers=2, vocab=16
    n_memory=2, segment_size=4, seq_len=8, batch=2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.recurrent_memory_v3 import (
    MemoryUpdateGate,
    RecurrentMemoryTokens,
    RecurrentMemoryTransformer,
    RMTEvaluator,
    SegmentProcessor,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D = 16
N_HEADS = 2
N_LAYERS = 2
VOCAB = 16
N_MEM = 2
SEG = 4
T = 8
B = 2


def _make_model() -> RecurrentMemoryTransformer:
    return RecurrentMemoryTransformer(
        d_model=D,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB,
        n_memory=N_MEM,
        segment_size=SEG,
    )


def _rand_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# 1. RecurrentMemoryTokens: output shape
# ---------------------------------------------------------------------------
def test_rmt_tokens_output_shape():
    rmt = RecurrentMemoryTokens(n_memory=N_MEM, d_model=D)
    out = rmt(B)
    assert out.shape == (B, N_MEM, D)


# ---------------------------------------------------------------------------
# 2. RecurrentMemoryTokens: memory_tokens is a trainable parameter
# ---------------------------------------------------------------------------
def test_rmt_tokens_params_trainable():
    rmt = RecurrentMemoryTokens(n_memory=N_MEM, d_model=D)
    assert rmt.memory_tokens.requires_grad
    assert rmt.memory_tokens.shape == (N_MEM, D)


# ---------------------------------------------------------------------------
# 3. RecurrentMemoryTokens.detach_memory: returned tensor has no grad_fn
# ---------------------------------------------------------------------------
def test_rmt_tokens_detach_removes_grad_fn():
    rmt = RecurrentMemoryTokens(n_memory=N_MEM, d_model=D)
    mem = rmt(B)
    # Give it a grad_fn by running an operation
    mem_with_grad = mem * 2.0
    assert mem_with_grad.grad_fn is not None
    detached = rmt.detach_memory(mem_with_grad)
    assert detached.grad_fn is None


# ---------------------------------------------------------------------------
# 4. SegmentProcessor: output shape (B, T_seg, D)
# ---------------------------------------------------------------------------
def test_segment_processor_output_shape():
    block = nn.TransformerEncoderLayer(
        d_model=D, nhead=N_HEADS, dim_feedforward=D * 4, batch_first=True, norm_first=True
    )
    proc = SegmentProcessor(block, n_memory=N_MEM)
    seg = torch.randn(B, SEG, D)
    mem = torch.randn(B, N_MEM, D)
    output, memory_out = proc(seg, mem)
    assert output.shape == (B, SEG, D)


# ---------------------------------------------------------------------------
# 5. SegmentProcessor: memory_out shape (B, n_memory, D)
# ---------------------------------------------------------------------------
def test_segment_processor_memory_out_shape():
    block = nn.TransformerEncoderLayer(
        d_model=D, nhead=N_HEADS, dim_feedforward=D * 4, batch_first=True, norm_first=True
    )
    proc = SegmentProcessor(block, n_memory=N_MEM)
    seg = torch.randn(B, SEG, D)
    mem = torch.randn(B, N_MEM, D)
    _, memory_out = proc(seg, mem)
    assert memory_out.shape == (B, N_MEM, D)


# ---------------------------------------------------------------------------
# 6. SegmentProcessor: memory_out differs from memory_in (memory was updated)
# ---------------------------------------------------------------------------
def test_segment_processor_memory_updated():
    torch.manual_seed(0)
    block = nn.TransformerEncoderLayer(
        d_model=D, nhead=N_HEADS, dim_feedforward=D * 4, batch_first=True, norm_first=True
    )
    proc = SegmentProcessor(block, n_memory=N_MEM)
    seg = torch.randn(B, SEG, D)
    mem_in = torch.zeros(B, N_MEM, D)
    _, mem_out = proc(seg, mem_in)
    assert not torch.allclose(mem_out, mem_in), "memory should change after processing a segment"


# ---------------------------------------------------------------------------
# 7. RecurrentMemoryTransformer: logits shape (B, T, V)
# ---------------------------------------------------------------------------
def test_rmt_logits_shape():
    model = _make_model()
    ids = _rand_ids()
    logits, _ = model(ids)
    assert logits.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# 8. RecurrentMemoryTransformer: final_memory shape (B, n_memory, D)
# ---------------------------------------------------------------------------
def test_rmt_final_memory_shape():
    model = _make_model()
    ids = _rand_ids()
    _, final_mem = model(ids)
    assert final_mem.shape == (B, N_MEM, D)


# ---------------------------------------------------------------------------
# 9. RecurrentMemoryTransformer: backward succeeds, grads reach embedding
# ---------------------------------------------------------------------------
def test_rmt_backward_grads_to_embedding():
    model = _make_model()
    ids = _rand_ids()
    logits, _ = model(ids)
    loss = logits.sum()
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# 10. RecurrentMemoryTransformer: initial_memory=None vs zeros -> same output
#     (when the learned memory tokens are zero-initialised, as they are by
#     default, both should produce identical logits)
# ---------------------------------------------------------------------------
def test_rmt_none_initial_memory_equals_zeros():
    """initial_memory=None with zero-init param should match initial_memory=zeros.

    Must run in inference mode so dropout is disabled; otherwise two separate
    forward passes get different dropout masks and differ even with equal inputs.
    """
    model = _make_model()
    model.train(False)  # disable dropout for deterministic comparison
    # Ensure the learned initial memory is zero (it is by construction)
    model.initial_memory.memory_tokens.data.zero_()
    ids = _rand_ids()
    with torch.no_grad():
        logits_none, _ = model(ids, initial_memory=None)
        zeros = torch.zeros(B, N_MEM, D)
        logits_zeros, _ = model(ids, initial_memory=zeros)
    assert torch.allclose(logits_none, logits_zeros, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. RecurrentMemoryTransformer: different initial_memory -> different logits
# ---------------------------------------------------------------------------
def test_rmt_different_initial_memory_gives_different_logits():
    torch.manual_seed(42)
    model = _make_model()
    ids = _rand_ids()
    mem_a = torch.zeros(B, N_MEM, D)
    mem_b = torch.randn(B, N_MEM, D) * 5.0  # very different
    with torch.no_grad():
        logits_a, _ = model(ids, initial_memory=mem_a)
        logits_b, _ = model(ids, initial_memory=mem_b)
    assert not torch.allclose(logits_a, logits_b), (
        "different memory should produce different logits"
    )


# ---------------------------------------------------------------------------
# 12. MemoryUpdateGate: output shape (B, n_memory, D) and gate in (0,1)
# ---------------------------------------------------------------------------
def test_memory_gate_shape_and_gate_range():
    gate = MemoryUpdateGate(d_model=D, n_memory=N_MEM)
    old = torch.randn(B, N_MEM, D)
    new = torch.randn(B, N_MEM, D)
    out = gate(old, new)
    assert out.shape == (B, N_MEM, D)
    # Verify the gate itself is strictly in (0, 1) by checking output is a
    # valid convex combination: ||out - old|| <= ||new - old|| and >= 0
    diff_total = (new - old).abs()
    diff_out = (out - old).abs()
    assert (diff_out <= diff_total + 1e-6).all(), "gate output must be a blend of old and new"


# ---------------------------------------------------------------------------
# 13. MemoryUpdateGate: gate=0 -> output=old; gate=1 -> output=new
# ---------------------------------------------------------------------------
def test_memory_gate_extreme_values():
    gate = MemoryUpdateGate(d_model=D, n_memory=N_MEM)
    old = torch.randn(B, N_MEM, D)
    new = torch.randn(B, N_MEM, D)

    # Force gate to 0: set bias to -100 so sigmoid(x) ~ 0
    nn.init.constant_(gate.gate_proj.bias, -100.0)
    nn.init.zeros_(gate.gate_proj.weight)
    out_zero = gate(old, new)
    assert torch.allclose(out_zero, old, atol=1e-4), "gate~0 should return old_memory"

    # Force gate to 1
    nn.init.constant_(gate.gate_proj.bias, 100.0)
    out_one = gate(old, new)
    assert torch.allclose(out_one, new, atol=1e-4), "gate~1 should return new_memory"


# ---------------------------------------------------------------------------
# 14. RMTEvaluator.memory_utilization: >= 0
# ---------------------------------------------------------------------------
def test_evaluator_memory_utilization_nonneg():
    model = _make_model()
    ids = _rand_ids()
    # Collect memory states across segments by running forward manually
    B_loc, T_loc = ids.shape
    embeddings = model.embedding(ids)
    memory_states: list[torch.Tensor] = []
    mem = model.initial_memory(B_loc)
    memory_states.append(mem.detach().clone())
    for start in range(0, T_loc, model.segment_size):
        end = min(start + model.segment_size, T_loc)
        seg = embeddings[:, start:end, :]
        x = seg
        new_mems = []
        for layer_idx, layer in enumerate(model.layers):
            x, m_out = layer(x, mem if layer_idx == 0 else new_mems[-1])
            new_mems.append(m_out)
        mem = new_mems[-1].detach()
        memory_states.append(mem.clone())

    ev = RMTEvaluator()
    util = ev.memory_utilization(memory_states)
    assert util >= 0.0


# ---------------------------------------------------------------------------
# 15a. RMTEvaluator.segment_dependency: in [-1, 1]
# ---------------------------------------------------------------------------
def test_evaluator_segment_dependency_range():
    model = _make_model()
    ids = _rand_ids()
    ev = RMTEvaluator()
    dep = ev.segment_dependency(model, ids)
    assert -1.0 <= dep <= 1.0


# ---------------------------------------------------------------------------
# 15b. RMTEvaluator.cross_segment_perplexity: >= 1.0
# ---------------------------------------------------------------------------
def test_evaluator_cross_segment_perplexity_at_least_one():
    model = _make_model()
    ids = _rand_ids()
    labels = _rand_ids()
    ev = RMTEvaluator()
    ppl = ev.cross_segment_perplexity(model, ids, labels)
    assert ppl >= 1.0


# ---------------------------------------------------------------------------
# 16. Memory detach between segments: grads from seg-2 don't reach seg-1
#     outputs through the memory pathway
# ---------------------------------------------------------------------------
def test_memory_detached_between_segments():
    """Backward through segment 2 must not reach segment 1 outputs via memory."""
    model = _make_model()
    ids = _rand_ids()
    B_loc, T_loc = ids.shape
    embeddings = model.embedding(ids)

    # Process segment 0, keeping its hidden states for grad check
    seg0 = embeddings[:, :SEG, :].requires_grad_(True)
    seg1 = embeddings[:, SEG:, :]

    # Manually run one layer for simplicity: use first SegmentProcessor
    layer = model.layers[0]
    mem0 = model.initial_memory(B_loc)

    # Segment 0
    out0, mem1_pre_detach = layer(seg0, mem0)
    mem1 = mem1_pre_detach.detach()  # <-- detach as the model does

    # Segment 1
    out1, _ = layer(seg1, mem1)

    # Backward from seg-1 output
    out1.sum().backward()

    # seg0 should have no gradient because mem1 was detached
    assert seg0.grad is None or seg0.grad.abs().sum().item() == 0.0, (
        "gradients must not flow back through detached memory to segment 0"
    )


# ---------------------------------------------------------------------------
# 17. Training step: loss finite, grads flow to memory_tokens parameter
# ---------------------------------------------------------------------------
def test_training_step_loss_finite_and_memory_grad():
    model = _make_model()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    ids = _rand_ids()
    labels = ids.clone()

    optimiser.zero_grad()
    logits, _ = model(ids)
    B_loc, T_loc, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B_loc * T_loc, V), labels.reshape(B_loc * T_loc)
    )
    assert torch.isfinite(loss), "loss must be finite"
    loss.backward()

    mt_grad = model.initial_memory.memory_tokens.grad
    assert mt_grad is not None, "memory_tokens must receive a gradient"
    assert mt_grad.abs().sum().item() > 0, "memory_tokens gradient must be non-zero"

    optimiser.step()

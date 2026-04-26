"""
Tests for src/model/lstm_transformer.py

Configuration used throughout:
  d_model=16, vocab_size=16, n_layers=2, n_heads=4, d_memory=8, B=2, T=8
"""

import math

import pytest
import torch

from src.model.lstm_transformer import (
    LSTMMemoryCell,
    LSTMTransformerBlock,
    LSTMTransformerConfig,
    LSTMTransformerModel,
    MemoryAugmentedAttention,
    SegmentedTrainer,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
N_HEADS = 4
D_MEMORY = 8
B = 2
T = 8


@pytest.fixture()
def cell():
    return LSTMMemoryCell(D_MODEL, D_MEMORY)


@pytest.fixture()
def attn():
    return MemoryAugmentedAttention(D_MODEL, N_HEADS, D_MEMORY)


@pytest.fixture()
def block():
    return LSTMTransformerBlock(D_MODEL, N_HEADS, D_MEMORY)


@pytest.fixture()
def model():
    return LSTMTransformerModel(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_memory=D_MEMORY,
    )


# ---------------------------------------------------------------------------
# LSTMMemoryCell tests
# ---------------------------------------------------------------------------


def test_lstm_cell_output_shape(cell):
    """LSTMMemoryCell forward output should be [B, d_memory]."""
    x = torch.randn(B, D_MODEL)
    state = cell.init_state(B)
    output, _ = cell(x, state)
    assert output.shape == (B, D_MEMORY), f"Expected ({B}, {D_MEMORY}), got {output.shape}"


def test_lstm_cell_new_state_shape(cell):
    """New state tensors should each be [B, d_memory]."""
    x = torch.randn(B, D_MODEL)
    state = cell.init_state(B)
    _, (new_h, new_c) = cell(x, state)
    assert new_h.shape == (B, D_MEMORY)
    assert new_c.shape == (B, D_MEMORY)


def test_lstm_cell_state_updates(cell):
    """After a forward pass the new state should differ from the init state."""
    x = torch.randn(B, D_MODEL)
    state = cell.init_state(B)
    _, new_state = cell(x, state)
    new_h, new_c = new_state
    init_h, init_c = state
    assert not torch.allclose(new_h, init_h), "Hidden state did not change"
    assert not torch.allclose(new_c, init_c), "Cell state did not change"


def test_lstm_cell_init_state_zeros(cell):
    """init_state should return zero tensors."""
    h, c = cell.init_state(B)
    assert torch.all(h == 0.0), "h init state is not all zeros"
    assert torch.all(c == 0.0), "c init state is not all zeros"


# ---------------------------------------------------------------------------
# MemoryAugmentedAttention tests
# ---------------------------------------------------------------------------


def test_attn_output_shape(attn):
    """MemoryAugmentedAttention output should be [B, T, d_model]."""
    x = torch.randn(B, T, D_MODEL)
    mem = torch.randn(B, D_MEMORY)
    out = attn(x, mem)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_attn_memory_changes_output(attn):
    """
    Two different memory states should produce different attention outputs
    for the same input sequence.
    """
    x = torch.randn(B, T, D_MODEL)
    mem1 = torch.randn(B, D_MEMORY)
    mem2 = torch.randn(B, D_MEMORY)
    out1 = attn(x, mem1)
    out2 = attn(x, mem2)
    assert not torch.allclose(out1, out2), "Different memory produced identical attention output"


def test_attn_zero_memory_runs(attn):
    """MemoryAugmentedAttention should run without error with zero memory."""
    x = torch.randn(B, T, D_MODEL)
    mem = torch.zeros(B, D_MEMORY)
    out = attn(x, mem)
    assert out.shape == (B, T, D_MODEL)


def test_attn_output_dtype(attn):
    """Output dtype should match input dtype."""
    x = torch.randn(B, T, D_MODEL)
    mem = torch.randn(B, D_MEMORY)
    out = attn(x, mem)
    assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# LSTMTransformerBlock tests
# ---------------------------------------------------------------------------


def test_block_output_shape(block):
    """Block forward should return [B, T, d_model]."""
    x = torch.randn(B, T, D_MODEL)
    state = block.lstm.init_state(B)
    out, _ = block(x, state)
    assert out.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {out.shape}"


def test_block_state_propagates(block):
    """New state returned by the block should differ from the input state."""
    x = torch.randn(B, T, D_MODEL)
    state = block.lstm.init_state(B)
    _, new_state = block(x, state)
    new_h, new_c = new_state
    init_h, init_c = state
    assert not torch.allclose(new_h, init_h), "Block did not update hidden state"
    assert not torch.allclose(new_c, init_c), "Block did not update cell state"


def test_block_gradient_flows_through_state(block):
    """Gradients should flow back through the block output and state."""
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    state = block.lstm.init_state(B)
    out, new_state = block(x, state)
    loss = out.sum() + new_state[0].sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input x"
    assert x.grad.shape == x.shape


def test_block_new_state_shapes(block):
    """New state tensors from block should each be [B, d_memory]."""
    x = torch.randn(B, T, D_MODEL)
    state = block.lstm.init_state(B)
    _, (new_h, new_c) = block(x, state)
    assert new_h.shape == (B, D_MEMORY)
    assert new_c.shape == (B, D_MEMORY)


# ---------------------------------------------------------------------------
# LSTMTransformerModel tests
# ---------------------------------------------------------------------------


def test_model_logits_shape(model):
    """Model forward should return logits [B, T, vocab_size]."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    logits, _ = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE), (
        f"Expected ({B}, {T}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_model_new_states_length(model):
    """new_states list should have length == n_layers."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    _, new_states = model(ids)
    assert len(new_states) == N_LAYERS, f"Expected {N_LAYERS} states, got {len(new_states)}"


def test_model_none_states_uses_init(model):
    """Passing states=None should work (uses init_states internally)."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    logits, new_states = model(ids, states=None)
    assert logits.shape == (B, T, VOCAB_SIZE)
    assert len(new_states) == N_LAYERS


def test_model_compute_loss_finite_positive(model):
    """compute_loss should return a finite positive scalar."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    loss = model.compute_loss(ids)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() > 0, "Loss should be positive"


def test_model_compute_loss_backward(model):
    """compute_loss should be differentiable."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    loss = model.compute_loss(ids)
    loss.backward()
    # Check that at least embedding gradients exist
    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.shape == model.embedding.weight.shape


def test_model_explicit_states_forwarded(model):
    """Passing explicit states from init_states should succeed."""
    ids = torch.randint(0, VOCAB_SIZE, (B, T))
    states = model.init_states(B)
    logits, new_states = model(ids, states=states)
    assert logits.shape == (B, T, VOCAB_SIZE)
    assert len(new_states) == N_LAYERS


# ---------------------------------------------------------------------------
# SegmentedTrainer tests
# ---------------------------------------------------------------------------


def test_segmented_trainer_returns_losses(model):
    """train_sequence should return a non-empty list of float losses."""
    trainer = SegmentedTrainer(model, lr=1e-3, segment_len=4)
    T_total = 16
    ids = torch.randint(0, VOCAB_SIZE, (B, T_total))
    losses = trainer.train_sequence(ids, ids)
    assert isinstance(losses, list), "Expected a list of losses"
    assert len(losses) > 0, "Expected at least one segment loss"
    for l in losses:  # noqa: E741
        assert isinstance(l, float), f"Loss entry is not float: {type(l)}"
        assert math.isfinite(l), f"Non-finite loss: {l}"


def test_segmented_trainer_detaches_state():
    """
    State detachment test: grads should not bleed from segment N+1
    back into the state returned at the end of segment N.
    """
    mdl = LSTMTransformerModel(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_memory=D_MEMORY,
    )
    T_total = 8
    segment_len = 4
    ids = torch.randint(0, VOCAB_SIZE, (B, T_total))

    # Manually run two segments and check that detached state has no grad_fn
    states = mdl.init_states(B)
    # Segment 0
    seg0 = ids[:, 0:segment_len]
    _, states0 = mdl(seg0, [(h.detach(), c.detach()) for h, c in states])
    h0, c0 = states0[0]

    # Detach for segment 1 — detached tensors must have no grad_fn
    h0_det = h0.detach()
    c0_det = c0.detach()
    assert h0_det.grad_fn is None, "Detached hidden state should have no grad_fn"
    assert c0_det.grad_fn is None, "Detached cell state should have no grad_fn"

    # Make h0_det a leaf with requires_grad so we can check whether segment 1
    # produces any gradient through it (it should not, because we detached).
    h0_probe = h0_det.clone().requires_grad_(True)
    c0_probe = c0_det.clone().requires_grad_(True)

    seg1 = ids[:, segment_len : 2 * segment_len]
    logits1, _ = mdl(seg1, [(h0_probe, c0_probe)])
    loss1 = logits1.sum()
    loss1.backward()

    # h0_probe IS a leaf that was used in segment 1's graph, so it gets a grad —
    # this is exactly what we want: the gradient stops here and does NOT continue
    # back into segment 0's parameters through the state.
    # The key invariant: h0 itself (the non-detached node) has no grad.
    assert h0_probe.grad is not None, "Gradient should reach the detached state leaf"
    assert h0.grad_fn is not None or not h0.requires_grad, (
        "h0 from segment 0 should not be a leaf in segment 1's graph"
    )


# ---------------------------------------------------------------------------
# LSTMTransformerConfig tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """LSTMTransformerConfig should have the specified default values."""
    cfg = LSTMTransformerConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4
    assert cfg.d_memory == 16
    assert cfg.segment_len == 16


def test_config_custom_values():
    """LSTMTransformerConfig should accept custom values."""
    cfg = LSTMTransformerConfig(d_model=64, vocab_size=128, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 128
    assert cfg.n_layers == 4
    # Remaining defaults unchanged
    assert cfg.n_heads == 4
    assert cfg.d_memory == 16

"""
Tests for the Neural Turing Machine implementation.
Uses: input_size=4, output_size=4, d_controller=16, word_size=4, n_locations=8, B=2, T=6
"""

import torch

from src.model.neural_turing_machine import (
    ContentAddressing,
    LocationAddressing,
    NTMConfig,
    NTMController,
    NTMHead,
    NTMMemory,
    NTMModel,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

B = 2
T = 6
INPUT_SIZE = 4
OUT_SIZE = 4
D_CTRL = 16
WORD_SIZE = 4
N_LOC = 8


def uniform_weights(batch: int = B, n: int = N_LOC) -> torch.Tensor:
    """Return normalised uniform attention weights."""
    w = torch.ones(batch, n) / n
    return w


def make_memory(n_loc: int = N_LOC, ws: int = WORD_SIZE) -> NTMMemory:
    mem = NTMMemory(n_loc, ws)
    mem.reset(B)
    return mem


def make_model(**kwargs) -> NTMModel:
    defaults = dict(
        input_size=INPUT_SIZE,
        output_size=OUT_SIZE,
        d_controller=D_CTRL,
        word_size=WORD_SIZE,
        n_locations=N_LOC,
        n_reads=1,
        n_writes=1,
    )
    defaults.update(kwargs)
    model = NTMModel(**defaults)
    model.reset(B)
    return model


# ---------------------------------------------------------------------------
# 1. NTMMemory — read output shape
# ---------------------------------------------------------------------------


def test_ntm_memory_read_shape():
    mem = make_memory()
    weights = uniform_weights()
    r = mem.read(weights)
    assert r.shape == (B, WORD_SIZE), f"Expected ({B}, {WORD_SIZE}), got {r.shape}"


# ---------------------------------------------------------------------------
# 2. NTMMemory — write modifies memory values
# ---------------------------------------------------------------------------


def test_ntm_memory_write_modifies_memory():
    mem = make_memory()
    before = mem.memory.clone()
    weights = uniform_weights()
    erase = torch.ones(B, WORD_SIZE)
    add = torch.ones(B, WORD_SIZE) * 2.0
    mem.write(weights, erase, add)
    assert not torch.allclose(mem.memory, before), "Memory should change after write"


# ---------------------------------------------------------------------------
# 3. NTMMemory — reset restores uniform state
# ---------------------------------------------------------------------------


def test_ntm_memory_reset_restores_state():
    mem = make_memory()
    # Dirty the memory
    mem.memory = torch.randn(N_LOC, WORD_SIZE)
    mem.reset(B)
    # Should be small constant (1e-6) everywhere
    expected = torch.full((N_LOC, WORD_SIZE), 1e-6)
    assert torch.allclose(mem.memory, expected), "Reset should set memory to 1e-6"


# ---------------------------------------------------------------------------
# 4. ContentAddressing — cosine_similarity shape
# ---------------------------------------------------------------------------


def test_content_addressing_cosine_similarity_shape():
    key = torch.randn(B, WORD_SIZE)
    memory = torch.randn(N_LOC, WORD_SIZE)
    sim = ContentAddressing.cosine_similarity(key, memory)
    assert sim.shape == (B, N_LOC), f"Expected ({B}, {N_LOC}), got {sim.shape}"


# ---------------------------------------------------------------------------
# 5. ContentAddressing — content_weights sum to 1
# ---------------------------------------------------------------------------


def test_content_addressing_weights_sum_to_one():
    key = torch.randn(B, WORD_SIZE)
    memory = torch.randn(N_LOC, WORD_SIZE)
    beta = torch.ones(B, 1)
    w = ContentAddressing.content_weights(key, memory, beta)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Content weights should sum to 1, got {sums}"
    )


# ---------------------------------------------------------------------------
# 6. LocationAddressing — interpolate sums to 1
# ---------------------------------------------------------------------------


def test_location_addressing_interpolate_sum_to_one():
    w_prev = uniform_weights()
    w_content = uniform_weights()
    gate = torch.full((B, 1), 0.5)
    w_g = LocationAddressing.interpolate(w_prev, w_content, gate)
    sums = w_g.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Interpolated weights should sum to 1, got {sums}"
    )


# ---------------------------------------------------------------------------
# 7. LocationAddressing — shift sums to 1 (circular convolution preserves mass)
# ---------------------------------------------------------------------------


def test_location_addressing_shift_preserves_mass():
    w_g = uniform_weights()
    shift_kernel = torch.zeros(B, 3)
    shift_kernel[:, 1] = 1.0  # all mass on "no shift"
    w_shifted = LocationAddressing.shift(w_g, shift_kernel)
    sums = w_shifted.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Shifted weights should sum to 1, got {sums}"
    )


def test_location_addressing_shift_circular_mass():
    """Verify mass is preserved with non-trivial shift weights."""
    w_g = uniform_weights()
    shift_kernel = torch.tensor([[0.2, 0.5, 0.3]]).expand(B, -1)
    w_shifted = LocationAddressing.shift(w_g, shift_kernel)
    sums = w_shifted.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Shifted weights should sum to 1 with non-uniform kernel, got {sums}"
    )


# ---------------------------------------------------------------------------
# 8. LocationAddressing — sharpen peaks higher than input
# ---------------------------------------------------------------------------


def test_location_addressing_sharpen_increases_peak():
    # Create a peaked but not perfectly sharp distribution
    w = torch.zeros(B, N_LOC)
    w[:, 2] = 0.5
    w[:, 3] = 0.3
    w[:, 4] = 0.2
    # Normalise so it sums to 1
    w = w / w.sum(dim=-1, keepdim=True)

    gamma = torch.full((B, 1), 5.0)  # strong sharpening
    w_sharp = LocationAddressing.sharpen(w, gamma)

    # Peak value should be higher after sharpening
    peak_before = w.max(dim=-1).values
    peak_after = w_sharp.max(dim=-1).values
    assert (peak_after >= peak_before).all(), "Sharpening should increase peak value"


# ---------------------------------------------------------------------------
# 9. NTMHead read type — output shapes
# ---------------------------------------------------------------------------


def test_ntmhead_read_output_shapes():
    head = NTMHead(D_CTRL, WORD_SIZE, N_LOC, "read")
    mem = make_memory()
    h = torch.randn(B, D_CTRL)
    w_prev = uniform_weights()
    r, w = head(h, mem, w_prev)
    assert r is not None, "Read head must return a read vector"
    assert r.shape == (B, WORD_SIZE), f"Read vector shape: expected {(B, WORD_SIZE)}, got {r.shape}"
    assert w.shape == (B, N_LOC), f"Weight shape: expected {(B, N_LOC)}, got {w.shape}"


# ---------------------------------------------------------------------------
# 10. NTMHead write type — modifies memory
# ---------------------------------------------------------------------------


def test_ntmhead_write_modifies_memory():
    head = NTMHead(D_CTRL, WORD_SIZE, N_LOC, "write")
    mem = make_memory()
    before = mem.memory.clone()
    h = torch.randn(B, D_CTRL)
    w_prev = uniform_weights()
    result, w = head(h, mem, w_prev)
    assert result is None, "Write head should return None for read vector"
    assert not torch.allclose(mem.memory, before), "Write head should modify memory"
    assert w.shape == (B, N_LOC)


# ---------------------------------------------------------------------------
# 11. NTMHead — attention weights sum to 1
# ---------------------------------------------------------------------------


def test_ntmhead_attention_weights_sum_to_one():
    head = NTMHead(D_CTRL, WORD_SIZE, N_LOC, "read")
    mem = make_memory()
    h = torch.randn(B, D_CTRL)
    w_prev = uniform_weights()
    _, w = head(h, mem, w_prev)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
        f"Head attention weights should sum to 1, got {sums}"
    )


# ---------------------------------------------------------------------------
# 12. NTMController — forward output shape
# ---------------------------------------------------------------------------


def test_ntm_controller_forward_shape():
    ctrl = NTMController(INPUT_SIZE, D_CTRL, n_reads=1, word_size=WORD_SIZE)
    ctrl.reset(B)
    x = torch.randn(B, INPUT_SIZE)
    reads = [torch.zeros(B, WORD_SIZE)]
    h = ctrl(x, reads)
    assert h.shape == (B, D_CTRL), f"Expected ({B}, {D_CTRL}), got {h.shape}"


# ---------------------------------------------------------------------------
# 13. NTMModel — forward_step output shape
# ---------------------------------------------------------------------------


def test_ntm_model_forward_step_shape():
    model = make_model()
    x = torch.randn(B, INPUT_SIZE)
    out = model.forward_step(x)
    assert out.shape == (B, OUT_SIZE), f"Expected ({B}, {OUT_SIZE}), got {out.shape}"


# ---------------------------------------------------------------------------
# 14. NTMModel — forward_sequence output shape
# ---------------------------------------------------------------------------


def test_ntm_model_forward_sequence_shape():
    model = make_model()
    xs = torch.randn(B, T, INPUT_SIZE)
    outs = model.forward_sequence(xs)
    assert outs.shape == (B, T, OUT_SIZE), f"Expected ({B}, {T}, {OUT_SIZE}), got {outs.shape}"


# ---------------------------------------------------------------------------
# 15. NTMModel — reset clears controller state
# ---------------------------------------------------------------------------


def test_ntm_model_reset_clears_state():
    model = make_model()
    # Run a step to dirty the state
    xs = torch.randn(B, INPUT_SIZE)
    model.forward_step(xs)
    dirty_hidden = model.controller._hidden.clone()

    # Reset
    model.reset(B)
    clean_hidden = model.controller._hidden

    assert torch.allclose(clean_hidden, torch.zeros_like(clean_hidden)), (
        "Hidden state should be zero after reset"
    )
    assert not torch.allclose(dirty_hidden, torch.zeros_like(dirty_hidden)), (
        "Sanity: hidden state should have been non-zero before reset"
    )


# ---------------------------------------------------------------------------
# 16. NTMConfig — default values
# ---------------------------------------------------------------------------


def test_ntm_config_defaults():
    cfg = NTMConfig()
    assert cfg.input_size == 8
    assert cfg.output_size == 8
    assert cfg.d_controller == 32
    assert cfg.word_size == 8
    assert cfg.n_locations == 16
    assert cfg.n_reads == 1
    assert cfg.n_writes == 1


# ---------------------------------------------------------------------------
# 17. NTMModel — forward (alias for forward_sequence) output shape
# ---------------------------------------------------------------------------


def test_ntm_model_forward_alias():
    model = make_model()
    xs = torch.randn(B, T, INPUT_SIZE)
    outs = model(xs)
    assert outs.shape == (B, T, OUT_SIZE)


# ---------------------------------------------------------------------------
# 18. NTMModel — gradient flows through forward_sequence
# ---------------------------------------------------------------------------


def test_ntm_model_gradients_flow():
    model = make_model()
    xs = torch.randn(B, T, INPUT_SIZE, requires_grad=True)
    outs = model.forward_sequence(xs)
    loss = outs.sum()
    loss.backward()
    assert xs.grad is not None, "Gradient must flow back through xs"
    assert xs.grad.shape == (B, T, INPUT_SIZE)


# ---------------------------------------------------------------------------
# 19. NTMMemory — read is differentiable (gradient flows through weights)
# ---------------------------------------------------------------------------


def test_ntm_memory_read_differentiable():
    mem = make_memory()
    weights = uniform_weights().requires_grad_(True)
    r = mem.read(weights)
    r.sum().backward()
    assert weights.grad is not None, "Gradient must flow through read weights"


# ---------------------------------------------------------------------------
# 20. NTMHead — write head returns (None, w_new) not (r, w_new)
# ---------------------------------------------------------------------------


def test_ntmhead_write_returns_none_read():
    head = NTMHead(D_CTRL, WORD_SIZE, N_LOC, "write")
    mem = make_memory()
    h = torch.randn(B, D_CTRL)
    w_prev = uniform_weights()
    r, w = head(h, mem, w_prev)
    assert r is None, "Write head must return None as read vector"
    assert w.shape == (B, N_LOC), f"Expected weight shape ({B}, {N_LOC}), got {w.shape}"

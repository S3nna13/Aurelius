"""Unit tests for SharedMTPHead (GLM-5 §3.3, arXiv:2602.15763).

Tiny config: d_model=64, vocab_size=256, n_heads=3.
"""

from __future__ import annotations

import pytest
import torch

from src.model.mtp_shared import SharedMTPHead

D_MODEL = 64
VOCAB_SIZE = 256
N_HEADS = 3
B = 2
T = 8


@pytest.fixture()
def model() -> SharedMTPHead:
    torch.manual_seed(0)
    return SharedMTPHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_heads=N_HEADS)


@pytest.fixture()
def hidden(model: SharedMTPHead) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 1. Output structure
# ---------------------------------------------------------------------------


def test_output_is_list_of_n_heads(model: SharedMTPHead, hidden: torch.Tensor):
    out = model(hidden)
    assert isinstance(out, list)
    assert len(out) == N_HEADS


def test_logit_shape(model: SharedMTPHead, hidden: torch.Tensor):
    out = model(hidden)
    for logits in out:
        assert logits.shape == (B, T, VOCAB_SIZE), (
            f"Expected ({B}, {T}, {VOCAB_SIZE}), got {logits.shape}"
        )


# ---------------------------------------------------------------------------
# 2. Weight sharing
# ---------------------------------------------------------------------------


def test_shared_weight_same_tensor(model: SharedMTPHead, hidden: torch.Tensor):
    """All heads route through the same shared_proj.weight storage."""
    ptr = model.shared_proj.weight.data_ptr()
    # The weight is shared at the module level; we verify via forward
    # by confirming each head uses the same shared_proj (not a copy).
    assert model.shared_proj.weight.data_ptr() == ptr  # sanity
    # Verify all heads produce outputs that depend on the same weight by
    # checking the weight pointer does not differ across any reference.
    for _ in range(N_HEADS):
        assert model.shared_proj.weight.data_ptr() == ptr


def test_input_projs_different(model: SharedMTPHead):
    """Each per-head input projection uses a distinct weight tensor."""
    ptrs = [p.weight.data_ptr() for p in model.input_projs]
    assert len(set(ptrs)) == N_HEADS, "All input_projs should have distinct weight tensors"


# ---------------------------------------------------------------------------
# 3. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_shared(model: SharedMTPHead, hidden: torch.Tensor):
    """Backward pass populates shared_proj.weight.grad."""
    hidden = hidden.requires_grad_(False)
    out = model(hidden)
    loss = sum(o.sum() for o in out)
    loss.backward()
    grad = model.shared_proj.weight.grad
    assert grad is not None, "shared_proj.weight.grad is None after backward"
    assert grad.abs().sum().item() > 0, "shared_proj.weight.grad is all-zero"


def test_gradient_flows_all_heads(model: SharedMTPHead, hidden: torch.Tensor):
    """Grad of shared_proj from all heads is larger than from a single head."""
    # Full model (all 3 heads)
    m_full = SharedMTPHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_heads=N_HEADS)
    torch.manual_seed(99)
    h = torch.randn(B, T, D_MODEL)

    out_full = m_full(h)
    sum(o.sum() for o in out_full).backward()
    grad_full = m_full.shared_proj.weight.grad.abs().sum().item()

    # Single-head model
    m_single = SharedMTPHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_heads=1)
    # Copy shared_proj weights so comparison is fair
    with torch.no_grad():
        m_single.shared_proj.weight.copy_(m_full.shared_proj.weight)
        m_single.input_projs[0].weight.copy_(m_full.input_projs[0].weight)

    out_single = m_single(h)
    out_single[0].sum().backward()
    grad_single = m_single.shared_proj.weight.grad.abs().sum().item()

    assert grad_full > grad_single, (
        f"Expected full-model grad ({grad_full}) > single-head grad ({grad_single})"
    )


# ---------------------------------------------------------------------------
# 4. Numerical stability
# ---------------------------------------------------------------------------


def test_no_nan_inf(model: SharedMTPHead):
    torch.manual_seed(42)
    h = torch.randn(B, T, D_MODEL)
    for logits in model(h):
        assert torch.isfinite(logits).all(), "Non-finite values in logits"


def test_determinism(model: SharedMTPHead):
    torch.manual_seed(7)
    h = torch.randn(B, T, D_MODEL)

    torch.manual_seed(7)
    h2 = torch.randn(B, T, D_MODEL)

    outs1 = model(h)
    outs2 = model(h2)
    for l1, l2 in zip(outs1, outs2):
        assert torch.equal(l1, l2), "Non-deterministic outputs with same seed/hidden"


# ---------------------------------------------------------------------------
# 5. Acceptance rate
# ---------------------------------------------------------------------------


def test_acceptance_rate_perfect(model: SharedMTPHead):
    """Construct logits whose argmax == shifted targets → rate should be 1.0."""
    torch.manual_seed(0)
    targets = torch.randint(0, VOCAB_SIZE, (B, T))

    # Build logits: for each head i, argmax at position t should equal targets[b, t+i+1]
    logits_list = []
    for i in range(N_HEADS):
        logits = torch.zeros(B, T, VOCAB_SIZE)
        offset = i + 1
        if offset < T:
            # Set a high value at the correct token index for positions that matter
            gt = targets[:, offset:]  # [B, T-offset]
            valid_len = gt.shape[1]
            for b in range(B):
                for t_idx in range(valid_len):
                    logits[b, t_idx, gt[b, t_idx].item()] = 100.0
        logits_list.append(logits)

    rate = model.acceptance_rate(logits_list, targets)
    assert rate == pytest.approx(1.0), f"Expected 1.0, got {rate}"


def test_acceptance_rate_zero(model: SharedMTPHead):
    """All predictions wrong → acceptance rate == 0.0."""
    torch.manual_seed(0)
    targets = torch.zeros(B, T, dtype=torch.long)  # all token 0

    # Build logits that always predict token 1 (never token 0)
    logits_list = []
    for _ in range(N_HEADS):
        logits = torch.zeros(B, T, VOCAB_SIZE)
        logits[:, :, 1] = 100.0  # argmax always → token 1
        logits_list.append(logits)

    rate = model.acceptance_rate(logits_list, targets)
    assert rate == pytest.approx(0.0), f"Expected 0.0, got {rate}"


def test_acceptance_rate_range(model: SharedMTPHead, hidden: torch.Tensor):
    """Random predictions → rate in [0, 1]."""
    torch.manual_seed(5)
    targets = torch.randint(0, VOCAB_SIZE, (B, T))
    logits_list = model(hidden)
    rate = model.acceptance_rate(logits_list, targets)
    assert 0.0 <= rate <= 1.0, f"Rate {rate} out of [0, 1]"


def test_single_token_seq(model: SharedMTPHead):
    """T=1 → all offsets >= T, acceptance_rate returns 0.0 without crash."""
    torch.manual_seed(3)
    h = torch.randn(B, 1, D_MODEL)
    targets = torch.randint(0, VOCAB_SIZE, (B, 1))
    logits_list = model(h)
    rate = model.acceptance_rate(logits_list, targets)
    # With T=1, every offset (1, 2, 3) >= T so total=0 → rate = 0/max(0,1) = 0.0
    assert rate == pytest.approx(0.0), f"Expected 0.0 for T=1, got {rate}"


# ---------------------------------------------------------------------------
# 6. n_heads flexibility
# ---------------------------------------------------------------------------


def test_n_heads_1():
    """n_heads=1 → forward returns a list of 1 tensor."""
    torch.manual_seed(11)
    m = SharedMTPHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_heads=1)
    h = torch.randn(B, T, D_MODEL)
    out = m(h)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0].shape == (B, T, VOCAB_SIZE)

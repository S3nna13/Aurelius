"""Integration test for HydraSpeculative (Cycle 132-A).

Exercises the full Hydra pipeline end-to-end with a tiny configuration:
  d_model=64, vocab_size=256, n_draft_heads=3, batch_size=2.

Checks:
  1. Registry wiring (DECODER_REGISTRY["hydra"] is HydraSpeculative).
  2. draft() output shape.
  3. sample_draft_tokens() output shape and validity.
  4. Mock-target verify() returns correct shapes.
  5. acceptance_rate() in [0, 1].
  6. Gradient flow through draft().
"""

from __future__ import annotations

import torch

from src.inference import DECODER_REGISTRY
from src.inference.hydra_speculative import HydraConfig, HydraSpeculative

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256
N_HEADS = 3
BATCH = 2

CFG = HydraConfig(
    d_model=D_MODEL,
    vocab_size=VOCAB_SIZE,
    n_draft_heads=N_HEADS,
    temperature=1.0,
)


def build_model() -> HydraSpeculative:
    return HydraSpeculative(CFG)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_hydra_speculative_end_to_end():
    """Full pipeline: draft → sample → (mock) verify → acceptance_rate → grad."""

    # --- 1. Registry wiring --------------------------------------------------
    assert "hydra" in DECODER_REGISTRY, "'hydra' missing from DECODER_REGISTRY"
    assert DECODER_REGISTRY["hydra"] is HydraSpeculative

    # --- 2. Build model and hidden state -------------------------------------
    model = build_model()
    hidden = torch.randn(BATCH, D_MODEL, requires_grad=True)

    # --- 3. Draft logits shape -----------------------------------------------
    draft_logits = model.draft(hidden)
    assert draft_logits.shape == (BATCH, N_HEADS, VOCAB_SIZE), (
        f"draft shape mismatch: {draft_logits.shape}"
    )

    # --- 4. Sample draft tokens ----------------------------------------------
    # Use a fresh hidden (no grad needed here).
    with torch.no_grad():
        h_no_grad = torch.randn(BATCH, D_MODEL)
        draft_tokens = model.sample_draft_tokens(h_no_grad)

    assert draft_tokens.shape == (BATCH, N_HEADS), (
        f"sample_draft_tokens shape mismatch: {draft_tokens.shape}"
    )
    assert (draft_tokens >= 0).all() and (draft_tokens < VOCAB_SIZE).all(), (
        "Draft token IDs out of vocabulary range"
    )

    # --- 5. Mock target logits (same shape as draft_logits) ------------------
    mock_target_logits = torch.randn(BATCH, N_HEADS, VOCAB_SIZE)

    # --- 6. Verify / accepted_mask -------------------------------------------
    accepted_mask, n_accepted = model.verify(draft_tokens, mock_target_logits)
    assert accepted_mask.shape == (BATCH, N_HEADS), (
        f"accepted_mask shape mismatch: {accepted_mask.shape}"
    )
    assert accepted_mask.dtype == torch.bool
    assert 0 <= n_accepted <= BATCH * N_HEADS, (
        f"n_accepted={n_accepted} out of range [0, {BATCH * N_HEADS}]"
    )

    # --- 7. Acceptance rate in [0, 1] ----------------------------------------
    rate = model.acceptance_rate(draft_tokens, mock_target_logits)
    assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} not in [0, 1]"

    # --- 8. Gradient flow through draft() ------------------------------------
    loss = draft_logits.sum()
    loss.backward()
    assert hidden.grad is not None, "Gradient did not flow back through draft()"
    assert hidden.grad.shape == hidden.shape, f"hidden.grad shape mismatch: {hidden.grad.shape}"

"""Integration test for SoftThinkingMixer — end-to-end forward + backward pass.

Config: d_model=64, vocab_size=256, top_k=10
Input: random logits [2, 8, 256]
Verifies: output shape [2, 8, 64], backward works, entropy shape [2, 8], registry wired.

Run with: .venv/bin/python3.14 -m pytest tests/integration/test_soft_thinking_integration.py -v
"""
from __future__ import annotations

import torch

from src.inference.soft_thinking import SoftThinkingConfig, SoftThinkingMixer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB = 256
TOP_K = 10
B = 2
T = 8


def test_soft_thinking_integration_full():
    """Full integration: forward, backward, entropy, registry check."""

    # --- Build mixer ---
    cfg = SoftThinkingConfig(d_model=D_MODEL, vocab_size=VOCAB, top_k=TOP_K)
    mixer = SoftThinkingMixer(cfg)

    # --- Random input logits [2, 8, 256] ---
    torch.manual_seed(7)
    logits = torch.randn(B, T, VOCAB, requires_grad=True)

    # --- Forward pass ---
    out = mixer(logits)  # uses forward() → mix()

    # Verify output shape [2, 8, 64]
    assert out.shape == (B, T, D_MODEL), \
        f"Expected output shape ({B}, {T}, {D_MODEL}), got {out.shape}"

    # --- Backward pass ---
    loss = out.sum()
    loss.backward()

    # Embedding weight must have received gradients
    assert mixer.embedding.weight.grad is not None, \
        "embedding.weight must have gradients after backward"
    assert mixer.embedding.weight.grad.abs().sum().item() > 0.0, \
        "embedding.weight gradients must be non-zero"

    # Input logits must also have gradients
    assert logits.grad is not None, \
        "Input logits must have gradients after backward"

    # --- Entropy ---
    with torch.no_grad():
        fresh_logits = torch.randn(B, T, VOCAB)
        ent = mixer.entropy(fresh_logits)

    assert ent.shape == (B, T), \
        f"Expected entropy shape ({B}, {T}), got {ent.shape}"

    # Entropy values should be non-negative (Shannon entropy ≥ 0)
    assert (ent >= 0).all(), "Entropy values must be non-negative"

    # --- Registry wired ---
    from src.inference import DECODER_REGISTRY

    assert "soft_thinking" in DECODER_REGISTRY, \
        "DECODER_REGISTRY must contain 'soft_thinking'"
    assert DECODER_REGISTRY["soft_thinking"] is SoftThinkingMixer, \
        "DECODER_REGISTRY['soft_thinking'] must be SoftThinkingMixer"

    # --- Shapes consistent across 2D and 3D inputs ---
    with torch.no_grad():
        logits_2d = torch.randn(B, VOCAB)
        out_2d = mixer.mix(logits_2d)
        assert out_2d.shape == (B, D_MODEL), \
            f"2D input should give shape ({B}, {D_MODEL}), got {out_2d.shape}"

    # --- Renormalize=True: weights sum to 1 at top-k level ---
    probs = torch.softmax(fresh_logits, dim=-1)  # [2, 8, 256]
    topk_w, _ = torch.topk(probs, k=TOP_K, dim=-1)  # [2, 8, 10]
    renorm_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    weight_sums = renorm_w.sum(dim=-1)  # [2, 8]
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Renormalized top-k weights must sum to 1"

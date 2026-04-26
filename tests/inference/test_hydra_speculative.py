"""Unit tests for src/inference/hydra_speculative.py.

Tests cover HydraConfig defaults, HydraHead shape, HydraSpeculative.draft,
sample_draft_tokens, verify, acceptance_rate, gradient flow, edge-case
batch sizes, and registry wiring.

Tiny config: d_model=64, vocab_size=256, n_draft_heads=3.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.hydra_speculative import (
    HydraConfig,
    HydraHead,
    HydraSpeculative,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY = HydraConfig(d_model=64, vocab_size=256, n_draft_heads=3)


def make_model(cfg: HydraConfig = TINY) -> HydraSpeculative:
    return HydraSpeculative(cfg)


def rand_hidden(B: int = 1, d: int = 64) -> torch.Tensor:
    return torch.randn(B, d)


def rand_target_logits(B: int = 1, n_heads: int = 3, vocab_size: int = 256) -> torch.Tensor:
    return torch.randn(B, n_heads, vocab_size)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = HydraConfig()
    assert cfg.n_draft_heads == 4
    assert cfg.temperature == 1.0
    assert cfg.d_model == 2048
    assert cfg.vocab_size == 128000
    assert cfg.head_hidden_dim is None


# ---------------------------------------------------------------------------
# 2. test_head_output_shape
# ---------------------------------------------------------------------------


def test_head_output_shape():
    head = HydraHead(d_model=64, hidden=64, vocab_size=256)
    h = rand_hidden(B=2)
    out = head(h)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_draft_shape
# ---------------------------------------------------------------------------


def test_draft_shape():
    model = make_model()
    h = rand_hidden(B=1)
    logits = model.draft(h)
    assert logits.shape == (1, 3, 256), f"Expected (1,3,256), got {logits.shape}"


# ---------------------------------------------------------------------------
# 4. test_draft_all_heads_run  — n_draft_heads=3 returns 3 predictions
# ---------------------------------------------------------------------------


def test_draft_all_heads_run():
    cfg = HydraConfig(d_model=64, vocab_size=256, n_draft_heads=3)
    model = HydraSpeculative(cfg)
    h = rand_hidden(B=1)
    logits = model.draft(h)
    assert logits.shape[1] == 3


# ---------------------------------------------------------------------------
# 5. test_sample_draft_tokens_shape
# ---------------------------------------------------------------------------


def test_sample_draft_tokens_shape():
    model = make_model()
    h = rand_hidden(B=2)
    tokens = model.sample_draft_tokens(h)
    assert tokens.shape == (2, 3), f"Expected (2,3), got {tokens.shape}"


# ---------------------------------------------------------------------------
# 6. test_sample_draft_tokens_valid
# ---------------------------------------------------------------------------


def test_sample_draft_tokens_valid():
    model = make_model()
    h = rand_hidden(B=4)
    tokens = model.sample_draft_tokens(h)
    assert tokens.dtype in (torch.int32, torch.int64, torch.long)
    assert (tokens >= 0).all(), "Token IDs must be non-negative"
    assert (tokens < 256).all(), "Token IDs must be < vocab_size"


# ---------------------------------------------------------------------------
# 7. test_verify_shape
# ---------------------------------------------------------------------------


def test_verify_shape():
    model = make_model()
    h = rand_hidden(B=2)
    draft_tokens = model.sample_draft_tokens(h)
    target_logits = rand_target_logits(B=2)
    accepted_mask, _ = model.verify(draft_tokens, target_logits)
    assert accepted_mask.shape == (2, 3), f"Expected (2,3), got {accepted_mask.shape}"
    assert accepted_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 8. test_verify_n_accepted_range
# ---------------------------------------------------------------------------


def test_verify_n_accepted_range():
    model = make_model()
    B = 3
    h = rand_hidden(B=B)
    draft_tokens = model.sample_draft_tokens(h)
    target_logits = rand_target_logits(B=B)
    _, n_accepted = model.verify(draft_tokens, target_logits)
    max_possible = B * 3
    assert 0 <= n_accepted <= max_possible, f"n_accepted={n_accepted} out of [0, {max_possible}]"


# ---------------------------------------------------------------------------
# 9. test_acceptance_rate_range
# ---------------------------------------------------------------------------


def test_acceptance_rate_range():
    model = make_model()
    h = rand_hidden(B=2)
    draft_tokens = model.sample_draft_tokens(h)
    target_logits = rand_target_logits(B=2)
    rate = model.acceptance_rate(draft_tokens, target_logits)
    assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} not in [0,1]"


# ---------------------------------------------------------------------------
# 10. test_acceptance_rate_perfect
#     target logits = one-hot mass on draft token → acceptance prob = 1.0
# ---------------------------------------------------------------------------


def test_acceptance_rate_perfect():
    torch.manual_seed(42)
    model = make_model()
    B, n_heads, V = 2, 3, 256
    draft_tokens = torch.randint(0, V, (B, n_heads))

    # Build target logits with near-infinite mass on the draft token.
    target_logits = torch.full((B, n_heads, V), -1e9)
    for b in range(B):
        for i in range(n_heads):
            target_logits[b, i, draft_tokens[b, i]] = 1e9

    rate = model.acceptance_rate(draft_tokens, target_logits)
    assert rate == pytest.approx(1.0, abs=1e-5), f"Expected ~1.0, got {rate}"


# ---------------------------------------------------------------------------
# 11. test_acceptance_rate_worst
#     target logits = one-hot mass on a DIFFERENT token → acceptance prob = 0.0
# ---------------------------------------------------------------------------


def test_acceptance_rate_worst():
    torch.manual_seed(0)
    model = make_model()
    B, n_heads, V = 2, 3, 256

    # Draft tokens are all 0; target puts all mass on token 1.
    draft_tokens = torch.zeros(B, n_heads, dtype=torch.long)
    target_logits = torch.full((B, n_heads, V), -1e9)
    target_logits[:, :, 1] = 1e9  # mass on token 1, NOT token 0

    rate = model.acceptance_rate(draft_tokens, target_logits)
    assert rate == pytest.approx(0.0, abs=1e-5), f"Expected ~0.0, got {rate}"


# ---------------------------------------------------------------------------
# 12. test_gradients_flow
# ---------------------------------------------------------------------------


def test_gradients_flow():
    model = make_model()
    h = torch.randn(1, 64, requires_grad=True)
    logits = model.draft(h)
    loss = logits.sum()
    loss.backward()
    assert h.grad is not None, "Gradients did not flow back to hidden input"
    assert h.grad.shape == h.shape


# ---------------------------------------------------------------------------
# 13. test_n_heads_1
# ---------------------------------------------------------------------------


def test_n_heads_1():
    cfg = HydraConfig(d_model=64, vocab_size=256, n_draft_heads=1)
    model = HydraSpeculative(cfg)
    h = rand_hidden(B=1)
    logits = model.draft(h)
    assert logits.shape == (1, 1, 256)
    tokens = model.sample_draft_tokens(h)
    assert tokens.shape == (1, 1)


# ---------------------------------------------------------------------------
# 14. test_batch_size_two
# ---------------------------------------------------------------------------


def test_batch_size_two():
    model = make_model()
    B = 2
    h = rand_hidden(B=B)
    logits = model.draft(h)
    assert logits.shape == (2, 3, 256)
    tokens = model.sample_draft_tokens(h)
    assert tokens.shape == (2, 3)
    tgt = rand_target_logits(B=B)
    mask, n = model.verify(tokens, tgt)
    assert mask.shape == (2, 3)
    assert 0 <= n <= 6


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry():
    from src.inference import DECODER_REGISTRY

    assert "hydra" in DECODER_REGISTRY, "'hydra' key missing from DECODER_REGISTRY"
    assert DECODER_REGISTRY["hydra"] is HydraSpeculative

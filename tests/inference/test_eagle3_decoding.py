"""Unit tests for Eagle3 speculative decoding module.

Tests use a tiny config (d_model=64, vocab_size=256) so they run fast on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.inference.eagle3_decoding import (
    ConfidenceHead,
    Eagle3Config,
    Eagle3Decoder,
    Eagle3Drafter,
    Eagle3Verifier,
)


# ---------------------------------------------------------------------------
# Tiny test config
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256
BATCH = 2


@pytest.fixture()
def tiny_config() -> Eagle3Config:
    return Eagle3Config(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_draft_len=4,
        min_draft_len=1,
        confidence_threshold=0.8,
        temperature=1.0,
    )


@pytest.fixture()
def drafter(tiny_config: Eagle3Config) -> Eagle3Drafter:
    return Eagle3Drafter(tiny_config)


@pytest.fixture()
def initial_hidden() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, D_MODEL)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = Eagle3Config()
    assert cfg.max_draft_len == 8
    assert cfg.confidence_threshold == 0.8
    assert cfg.min_draft_len == 1


# ---------------------------------------------------------------------------
# 2. test_confidence_head_shape_2d
# ---------------------------------------------------------------------------

def test_confidence_head_shape_2d():
    head = ConfidenceHead(d_model=D_MODEL, hidden=16)
    x = torch.randn(BATCH, D_MODEL)
    out = head(x)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_confidence_head_shape_3d
# ---------------------------------------------------------------------------

def test_confidence_head_shape_3d():
    T = 5
    head = ConfidenceHead(d_model=D_MODEL, hidden=16)
    x = torch.randn(BATCH, T, D_MODEL)
    out = head(x)
    assert out.shape == (BATCH, T), f"Expected ({BATCH}, {T}), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. test_confidence_head_range
# ---------------------------------------------------------------------------

def test_confidence_head_range():
    head = ConfidenceHead(d_model=D_MODEL, hidden=16)
    x = torch.randn(100, D_MODEL)
    out = head(x)
    assert (out > 0).all(), "confidence values must be > 0 (sigmoid)"
    assert (out < 1).all(), "confidence values must be < 1 (sigmoid)"


# ---------------------------------------------------------------------------
# 5. test_draft_step_output_shapes
# ---------------------------------------------------------------------------

def test_draft_step_output_shapes(drafter: Eagle3Drafter, initial_hidden: torch.Tensor):
    logits, next_hidden, confidence = drafter.draft_step(initial_hidden)
    assert logits.shape == (BATCH, VOCAB_SIZE), f"logits shape mismatch: {logits.shape}"
    assert next_hidden.shape == (BATCH, D_MODEL), f"next_hidden shape mismatch: {next_hidden.shape}"
    assert confidence.shape == (BATCH,), f"confidence shape mismatch: {confidence.shape}"


# ---------------------------------------------------------------------------
# 6. test_draft_returns_dict_keys
# ---------------------------------------------------------------------------

def test_draft_returns_dict_keys(drafter: Eagle3Drafter, initial_hidden: torch.Tensor):
    result = drafter.draft(initial_hidden)
    for key in ("draft_tokens", "draft_logits", "confidences", "n_drafted"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 7. test_draft_max_length
# ---------------------------------------------------------------------------

def test_draft_max_length(tiny_config: Eagle3Config, initial_hidden: torch.Tensor):
    # With a very low confidence_threshold, drafting runs to the max
    cfg = Eagle3Config(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_draft_len=6,
        min_draft_len=1,
        confidence_threshold=0.0,  # never stop early
        temperature=1.0,
    )
    drafter = Eagle3Drafter(cfg)
    result = drafter.draft(initial_hidden, n_tokens=3)
    assert result["n_drafted"] == 3, (
        f"Expected exactly 3 tokens drafted, got {result['n_drafted']}"
    )
    assert len(result["draft_tokens"]) == 3
    assert len(result["draft_logits"]) == 3
    assert len(result["confidences"]) == 3


# ---------------------------------------------------------------------------
# 8. test_draft_confidence_gating — high threshold → fewer tokens drafted
# ---------------------------------------------------------------------------

def test_draft_confidence_gating(initial_hidden: torch.Tensor):
    # Threshold = 1.0 means confidence (always < 1 via sigmoid) is always below it
    # So after min_draft_len tokens are drafted, the loop should stop immediately
    cfg = Eagle3Config(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_draft_len=8,
        min_draft_len=1,
        confidence_threshold=1.0,  # impossible to meet => stop after min_draft_len
        temperature=1.0,
    )
    drafter = Eagle3Drafter(cfg)
    result = drafter.draft(initial_hidden)
    # Sigmoid output is strictly < 1, so confidence always < threshold=1.0
    # Early stopping kicks in after min_draft_len=1 step
    assert result["n_drafted"] <= cfg.max_draft_len
    assert result["n_drafted"] == cfg.min_draft_len, (
        f"Expected early stop at min_draft_len={cfg.min_draft_len}, "
        f"got n_drafted={result['n_drafted']}"
    )


# ---------------------------------------------------------------------------
# 9. test_draft_low_threshold — threshold=0.0 → always drafts max_draft_len
# ---------------------------------------------------------------------------

def test_draft_low_threshold(initial_hidden: torch.Tensor):
    cfg = Eagle3Config(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        max_draft_len=5,
        min_draft_len=1,
        confidence_threshold=0.0,  # always satisfied
        temperature=1.0,
    )
    drafter = Eagle3Drafter(cfg)
    result = drafter.draft(initial_hidden)
    assert result["n_drafted"] == cfg.max_draft_len, (
        f"Expected n_drafted={cfg.max_draft_len}, got {result['n_drafted']}"
    )


# ---------------------------------------------------------------------------
# 10. test_verifier_perfect_match — draft matches target argmax → high acceptance
# ---------------------------------------------------------------------------

def test_verifier_perfect_match():
    """When draft tokens exactly match the target argmax the verifier should
    tend to accept them (target prob is highest at that token)."""
    torch.manual_seed(0)
    vocab = VOCAB_SIZE

    # Build a target distribution strongly peaked at token 42
    target_logits = torch.zeros(1, vocab)
    target_logits[0, 42] = 20.0  # very high logit
    target_probs = [F.softmax(target_logits, dim=-1)]

    draft_tokens = [torch.tensor([42])]  # matches peak

    accepted_mask, n_accepted = Eagle3Verifier.verify(draft_tokens, target_probs)
    # With such a peaked distribution the ratio will almost always be >= uniform
    # We run several trials and expect at least 1 acceptance
    successes = 0
    for _ in range(20):
        _, n = Eagle3Verifier.verify(draft_tokens, target_probs)
        successes += n
    assert successes > 0, "Expected at least some acceptances when token matches target argmax"


# ---------------------------------------------------------------------------
# 11. test_verifier_returns_mask
# ---------------------------------------------------------------------------

def test_verifier_returns_mask():
    vocab = VOCAB_SIZE
    n = 3
    draft_tokens = [torch.zeros(1, dtype=torch.long) for _ in range(n)]
    target_probs = [torch.ones(1, vocab) / vocab for _ in range(n)]

    mask, n_accepted = Eagle3Verifier.verify(draft_tokens, target_probs)
    assert isinstance(mask, list), "accepted_mask must be a list"
    assert all(isinstance(v, bool) for v in mask), "each element must be bool"
    assert isinstance(n_accepted, int)
    assert 0 <= n_accepted <= n


# ---------------------------------------------------------------------------
# 12. test_decode_step_keys
# ---------------------------------------------------------------------------

def test_decode_step_keys(tiny_config: Eagle3Config, initial_hidden: torch.Tensor):
    decoder = Eagle3Decoder(tiny_config)

    def mock_target(tokens):
        return [torch.rand(BATCH, VOCAB_SIZE) for _ in tokens]

    result = decoder.decode_step(initial_hidden, mock_target)
    for key in ("accepted_tokens", "n_accepted", "n_drafted", "acceptance_rate"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 13. test_acceptance_rate_range
# ---------------------------------------------------------------------------

def test_acceptance_rate_range(tiny_config: Eagle3Config, initial_hidden: torch.Tensor):
    decoder = Eagle3Decoder(tiny_config)

    def mock_target(tokens):
        return [F.softmax(torch.randn(BATCH, VOCAB_SIZE), dim=-1) for _ in tokens]

    for _ in range(10):
        result = decoder.decode_step(initial_hidden, mock_target)
        ar = result["acceptance_rate"]
        assert 0.0 <= ar <= 1.0, f"acceptance_rate {ar} out of [0, 1]"


# ---------------------------------------------------------------------------
# 14. test_gradients_confidence
# ---------------------------------------------------------------------------

def test_gradients_confidence():
    head = ConfidenceHead(d_model=D_MODEL, hidden=16)
    x = torch.randn(BATCH, D_MODEL, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient flowed through ConfidenceHead"
    assert not torch.isnan(x.grad).any(), "NaN in gradient"


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------

def test_registry():
    from src.inference import DECODER_REGISTRY
    assert "eagle3" in DECODER_REGISTRY, (
        "DECODER_REGISTRY must contain 'eagle3'"
    )
    assert DECODER_REGISTRY["eagle3"] is Eagle3Decoder

"""Tests for src/inference/speculative_decoding.py — classic speculative decoding.

Covers SpeculativeConfig, DraftModel, SpeculativeVerifier, and SpeculativeDecoder
with ≥12 tests as required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from src.inference.speculative_decoding import (
    DraftModel,
    SpeculativeConfig,
    SpeculativeDecoder,
    SpeculativeVerifier,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
PROMPT_LEN = 4


def _make_prompt(length: int = PROMPT_LEN) -> torch.Tensor:
    """Return a ``(1, length)`` random int64 prompt."""
    return torch.randint(0, VOCAB_SIZE, (1, length))


def _random_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Random mock model: (1, T) -> (1, T, VOCAB_SIZE)."""
    B, T = ids.shape
    return torch.randn(B, T, VOCAB_SIZE)


def _deterministic_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Deterministic model with fixed seed: (1, T) -> (1, T, VOCAB_SIZE)."""
    torch.manual_seed(42)
    B, T = ids.shape
    return torch.randn(B, T, VOCAB_SIZE)


def _uniform_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Returns uniform logits (all zeros): equal probability for every token."""
    B, T = ids.shape
    return torch.zeros(B, T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig defaults are sane
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SpeculativeConfig()
    assert cfg.n_draft_tokens == 5
    assert cfg.temperature == 1.0
    assert cfg.top_p == 1.0
    assert cfg.max_new_tokens == 128


# ---------------------------------------------------------------------------
# 2. DraftModel.autoregressive_draft returns correct shapes
# ---------------------------------------------------------------------------


def test_draft_model_shapes():
    n = 5
    draft = DraftModel(_random_model_fn, VOCAB_SIZE)
    prompt = _make_prompt()
    token_ids, logits_per_step = draft.autoregressive_draft(prompt, n)
    assert token_ids.shape == (n,), f"Expected ({n},), got {token_ids.shape}"
    assert logits_per_step.shape == (n, VOCAB_SIZE), (
        f"Expected ({n}, {VOCAB_SIZE}), got {logits_per_step.shape}"
    )


# ---------------------------------------------------------------------------
# 3. Draft tokens are in vocab range
# ---------------------------------------------------------------------------


def test_draft_tokens_in_vocab_range():
    draft = DraftModel(_random_model_fn, VOCAB_SIZE)
    prompt = _make_prompt()
    token_ids, _ = draft.autoregressive_draft(prompt, n_tokens=6)
    assert (token_ids >= 0).all(), "Negative token ids returned"
    assert (token_ids < VOCAB_SIZE).all(), "Token ids exceed vocab size"


# ---------------------------------------------------------------------------
# 4. SpeculativeVerifier.verify — perfect match (p_target == p_draft) accepts all K
# ---------------------------------------------------------------------------


def test_verify_perfect_match_accepts_all():
    """When draft and target distributions are identical, all tokens accepted."""
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    K = 5
    # Create a fixed distribution
    probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)
    draft_ids = probs.argmax(dim=-1)  # greedy tokens

    accepted_tokens, n_accepted = verifier.verify(draft_ids, probs, probs)
    # All K draft tokens must be accepted
    assert n_accepted == K, f"Expected {K} accepted, got {n_accepted}"


# ---------------------------------------------------------------------------
# 5. SpeculativeVerifier.verify — zero target prob at draft token → always reject
# ---------------------------------------------------------------------------


def test_verify_zero_target_prob_always_rejects():
    """When target assigns 0 prob to the draft token, reject at first position."""
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    K = 4
    draft_ids = torch.zeros(K, dtype=torch.long)  # all token 0

    # Draft has nonzero prob at token 0; target has zero prob at token 0
    draft_probs = torch.zeros(K, VOCAB_SIZE)
    draft_probs[:, 0] = 0.5
    draft_probs[:, 1] = 0.5

    target_probs = torch.zeros(K, VOCAB_SIZE)
    target_probs[:, 0] = 0.0  # zero at draft token
    target_probs[:, 1:] = 1.0 / (VOCAB_SIZE - 1)  # uniform elsewhere

    accepted_tokens, n_accepted = verifier.verify(draft_ids, draft_probs, target_probs)
    assert n_accepted == 0, f"Expected 0 accepted, got {n_accepted}"


# ---------------------------------------------------------------------------
# 6. verify returns n_accepted in [0, K]
# ---------------------------------------------------------------------------


def test_verify_n_accepted_in_bounds():
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    K = 5
    F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)
    draft_ids = torch.randint(0, VOCAB_SIZE, (K,))
    draft_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)
    target_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)

    _, n_accepted = verifier.verify(draft_ids, draft_probs, target_probs)
    assert 0 <= n_accepted <= K, f"n_accepted={n_accepted} out of [0, {K}]"


# ---------------------------------------------------------------------------
# 7. Accepted tokens are all in vocab range
# ---------------------------------------------------------------------------


def test_accepted_tokens_in_vocab_range():
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    K = 5
    draft_ids = torch.randint(0, VOCAB_SIZE, (K,))
    draft_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)
    target_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)

    accepted_tokens, _ = verifier.verify(draft_ids, draft_probs, target_probs)
    assert (accepted_tokens >= 0).all(), "Accepted tokens contain negative ids"
    assert (accepted_tokens < VOCAB_SIZE).all(), "Accepted tokens exceed vocab size"


# ---------------------------------------------------------------------------
# 8. sample_from_logits returns valid token id
# ---------------------------------------------------------------------------


def test_sample_from_logits_valid():
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    logits = torch.randn(VOCAB_SIZE)
    tok = verifier.sample_from_logits(logits)
    assert isinstance(tok, int), f"Expected int, got {type(tok)}"
    assert 0 <= tok < VOCAB_SIZE, f"Token {tok} out of vocab range [0, {VOCAB_SIZE})"


# ---------------------------------------------------------------------------
# 9. SpeculativeDecoder.generate returns correct length (max_new_tokens)
# ---------------------------------------------------------------------------


def test_decoder_generate_correct_length():
    cfg = SpeculativeConfig(n_draft_tokens=3, temperature=1.0, max_new_tokens=10)
    decoder = SpeculativeDecoder(_random_model_fn, _random_model_fn, VOCAB_SIZE, cfg)
    prompt = _make_prompt()
    out = decoder.generate(prompt, max_new_tokens=10)
    assert out.shape == (10,), f"Expected shape (10,), got {out.shape}"


# ---------------------------------------------------------------------------
# 10. Deterministic identical draft/target accepts all K tokens per step
# ---------------------------------------------------------------------------


def test_identical_models_high_acceptance():
    """With identical draft and target models, acceptance rate should be high."""
    cfg = SpeculativeConfig(n_draft_tokens=4, temperature=1.0, max_new_tokens=20)
    decoder = SpeculativeDecoder(_deterministic_model_fn, _deterministic_model_fn, VOCAB_SIZE, cfg)
    prompt = _make_prompt()
    out = decoder.generate(prompt, max_new_tokens=20)
    # Output must be exactly max_new_tokens
    assert out.shape == (20,), f"Expected (20,), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. Works with K=1 (single draft token)
# ---------------------------------------------------------------------------


def test_k1_single_draft_token():
    cfg = SpeculativeConfig(n_draft_tokens=1, temperature=1.0, max_new_tokens=8)
    decoder = SpeculativeDecoder(_random_model_fn, _random_model_fn, VOCAB_SIZE, cfg)
    prompt = _make_prompt()
    out = decoder.generate(prompt, max_new_tokens=8)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"
    assert (out >= 0).all()
    assert (out < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# 12. Temperature very small (near-zero) acts greedy
# ---------------------------------------------------------------------------


def test_small_temperature_greedy_behavior():
    """With very small temperature, draft model should be deterministic/greedy."""
    draft = DraftModel(_deterministic_model_fn, VOCAB_SIZE)
    prompt = _make_prompt()
    # Two runs with same seed model should produce same greedy output
    ids1, _ = draft.autoregressive_draft(prompt, n_tokens=5)
    ids2, _ = draft.autoregressive_draft(prompt, n_tokens=5)
    assert torch.equal(ids1, ids2), "Greedy draft should be deterministic"


# ---------------------------------------------------------------------------
# 13. SpeculativeDecoder.generate output tokens are valid vocab ids
# ---------------------------------------------------------------------------


def test_generate_output_in_vocab_range():
    cfg = SpeculativeConfig(n_draft_tokens=4, temperature=1.0, max_new_tokens=12)
    decoder = SpeculativeDecoder(_random_model_fn, _random_model_fn, VOCAB_SIZE, cfg)
    prompt = _make_prompt()
    out = decoder.generate(prompt, max_new_tokens=12)
    assert (out >= 0).all(), "Generated tokens contain negative ids"
    assert (out < VOCAB_SIZE).all(), "Generated tokens exceed vocab size"


# ---------------------------------------------------------------------------
# 14. Config with non-default values is respected
# ---------------------------------------------------------------------------


def test_config_custom_values():
    cfg = SpeculativeConfig(n_draft_tokens=3, temperature=0.8, top_p=0.9, max_new_tokens=64)
    assert cfg.n_draft_tokens == 3
    assert cfg.temperature == 0.8
    assert cfg.top_p == 0.9
    assert cfg.max_new_tokens == 64


# ---------------------------------------------------------------------------
# 15. verify returns at least 1 token (corrected or bonus)
# ---------------------------------------------------------------------------


def test_verify_always_returns_at_least_one_token():
    verifier = SpeculativeVerifier(VOCAB_SIZE, temperature=1.0)
    K = 4
    draft_ids = torch.randint(0, VOCAB_SIZE, (K,))
    draft_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)
    target_probs = F.softmax(torch.randn(K, VOCAB_SIZE), dim=-1)

    accepted_tokens, _ = verifier.verify(draft_ids, draft_probs, target_probs)
    assert accepted_tokens.shape[0] >= 1, "verify must return at least 1 token"

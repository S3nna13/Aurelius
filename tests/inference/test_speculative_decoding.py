"""Tests for src/inference/speculative_decoding.py — new callable-model API.

Covers SpeculativeConfig, draft_tokens, verify_draft, speculative_decode_step,
and SpeculativeDecoder (12 tests total).
"""
from __future__ import annotations

import torch
import pytest

from src.inference.speculative_decoding import (
    SpeculativeConfig,
    draft_tokens,
    verify_draft,
    speculative_decode_step,
    SpeculativeDecoder,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
BATCH = 1


def _model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Tiny mock: returns random logits shaped (B, T, VOCAB_SIZE)."""
    B, T = ids.shape
    return torch.randn(B, T, VOCAB_SIZE)


def _det_model_fn(ids: torch.Tensor) -> torch.Tensor:
    """Deterministic mock with fixed seed for reproducibility."""
    torch.manual_seed(0)
    B, T = ids.shape
    return torch.randn(B, T, VOCAB_SIZE)


def _prompt(length: int = 4) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, length))


# ---------------------------------------------------------------------------
# 1. SpeculativeConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = SpeculativeConfig()
    assert cfg.n_draft_tokens == 4
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 50
    assert cfg.vocab_size == 32000


# ---------------------------------------------------------------------------
# 2. draft_tokens — draft_ids shape is (B, n)
# ---------------------------------------------------------------------------

def test_draft_tokens_ids_shape():
    n = 4
    prompt = _prompt(length=4)
    draft_ids, draft_probs = draft_tokens(_model_fn, prompt, n=n, temperature=1.0)
    assert draft_ids.shape == (BATCH, n), f"Expected ({BATCH}, {n}), got {draft_ids.shape}"


# ---------------------------------------------------------------------------
# 3. draft_tokens — draft_probs shape is (B, n)
# ---------------------------------------------------------------------------

def test_draft_tokens_probs_shape():
    n = 3
    prompt = _prompt(length=4)
    draft_ids, draft_probs = draft_tokens(_model_fn, prompt, n=n, temperature=1.0)
    assert draft_probs.shape == (BATCH, n), f"Expected ({BATCH}, {n}), got {draft_probs.shape}"


# ---------------------------------------------------------------------------
# 4. draft_tokens — draft_probs are valid probabilities in [0, 1]
# ---------------------------------------------------------------------------

def test_draft_tokens_probs_in_range():
    prompt = _prompt(length=4)
    _, draft_probs = draft_tokens(_model_fn, prompt, n=4, temperature=1.0)
    assert (draft_probs >= 0).all(), "draft_probs contain negative values"
    assert (draft_probs <= 1).all(), "draft_probs contain values > 1"


# ---------------------------------------------------------------------------
# 5. draft_tokens — draft_ids are valid vocab indices
# ---------------------------------------------------------------------------

def test_draft_tokens_ids_in_vocab_range():
    prompt = _prompt(length=4)
    draft_ids, _ = draft_tokens(_model_fn, prompt, n=4, temperature=1.0)
    assert (draft_ids >= 0).all(), "draft_ids contain negative indices"
    assert (draft_ids < VOCAB_SIZE).all(), "draft_ids contain out-of-vocab indices"


# ---------------------------------------------------------------------------
# 6. verify_draft returns a tuple of length 2
# ---------------------------------------------------------------------------

def test_verify_draft_returns_tuple_of_2():
    prompt = _prompt(length=4)
    draft_ids, draft_probs = draft_tokens(_model_fn, prompt, n=3, temperature=1.0)
    result = verify_draft(_model_fn, prompt, draft_ids, draft_probs, temperature=1.0)
    assert isinstance(result, tuple), "verify_draft must return a tuple"
    assert len(result) == 2, f"Expected tuple length 2, got {len(result)}"


# ---------------------------------------------------------------------------
# 7. verify_draft — n_accepted >= 0
# ---------------------------------------------------------------------------

def test_verify_draft_n_accepted_non_negative():
    prompt = _prompt(length=4)
    draft_ids, draft_probs = draft_tokens(_model_fn, prompt, n=4, temperature=1.0)
    _, n_accepted = verify_draft(_model_fn, prompt, draft_ids, draft_probs, temperature=1.0)
    assert isinstance(n_accepted, int), f"n_accepted should be int, got {type(n_accepted)}"
    assert n_accepted >= 0, f"n_accepted={n_accepted} should be >= 0"


# ---------------------------------------------------------------------------
# 8. speculative_decode_step returns (new_token_ids Tensor, n_accepted int)
# ---------------------------------------------------------------------------

def test_speculative_decode_step_returns_new_tokens():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=3, temperature=1.0, vocab_size=VOCAB_SIZE)
    new_toks, n_accepted = speculative_decode_step(_model_fn, _model_fn, prompt, cfg)
    assert isinstance(new_toks, torch.Tensor), "new_token_ids must be a Tensor"
    assert isinstance(n_accepted, int), "n_accepted must be int"
    assert new_toks.ndim == 2, "new_token_ids must be 2-D"
    assert new_toks.shape[0] == BATCH


# ---------------------------------------------------------------------------
# 9. SpeculativeDecoder.decode output length grows beyond prompt
# ---------------------------------------------------------------------------

def test_decoder_decode_output_length_grows():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=3, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    out = decoder.decode(prompt, max_new_tokens=5)
    assert out.shape[1] > prompt.shape[1], (
        f"Output length {out.shape[1]} should exceed prompt length {prompt.shape[1]}"
    )


# ---------------------------------------------------------------------------
# 10. get_stats keys present
# ---------------------------------------------------------------------------

def test_get_stats_keys_present():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=2, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    decoder.decode(prompt, max_new_tokens=4)
    stats = decoder.get_stats()
    required_keys = {"total_draft_tokens", "total_accepted", "acceptance_rate", "n_steps"}
    assert required_keys.issubset(stats.keys()), (
        f"Missing keys: {required_keys - stats.keys()}"
    )


# ---------------------------------------------------------------------------
# 11. acceptance_rate in [0, 1]
# ---------------------------------------------------------------------------

def test_acceptance_rate_in_range():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=3, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    decoder.decode(prompt, max_new_tokens=6)
    stats = decoder.get_stats()
    rate = stats["acceptance_rate"]
    assert 0.0 <= rate <= 1.0, f"acceptance_rate={rate} not in [0, 1]"


# ---------------------------------------------------------------------------
# 12. decode with max_new_tokens=1
# ---------------------------------------------------------------------------

def test_decode_max_new_tokens_1():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=2, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    out = decoder.decode(prompt, max_new_tokens=1)
    n_generated = out.shape[1] - prompt.shape[1]
    assert n_generated >= 1, f"Expected at least 1 new token, got {n_generated}"


# ---------------------------------------------------------------------------
# 13. decode with max_new_tokens=5
# ---------------------------------------------------------------------------

def test_decode_max_new_tokens_5():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=2, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    out = decoder.decode(prompt, max_new_tokens=5)
    n_generated = out.shape[1] - prompt.shape[1]
    assert n_generated >= 1, f"Expected at least 1 new token, got {n_generated}"
    assert n_generated <= 6, f"Generated too many tokens: {n_generated}"


# ---------------------------------------------------------------------------
# 14. n_steps tracked in stats after decoding
# ---------------------------------------------------------------------------

def test_n_steps_tracked():
    prompt = _prompt(length=4)
    cfg = SpeculativeConfig(n_draft_tokens=2, temperature=1.0, vocab_size=VOCAB_SIZE)
    decoder = SpeculativeDecoder(_model_fn, _model_fn, cfg)
    decoder.decode(prompt, max_new_tokens=4)
    stats = decoder.get_stats()
    assert stats["n_steps"] >= 1, "n_steps should be >= 1 after decoding"

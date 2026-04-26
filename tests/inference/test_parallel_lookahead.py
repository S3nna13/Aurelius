"""Tests for parallel_lookahead.py — n-gram caching and Jacobi decoding.

Covers LookaheadConfig defaults, NGramCache behaviour, draft generation,
draft verification, JacobiDecoder end-to-end, and the speedup estimator.
Tests use a tiny AureliusTransformer so they run fast on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.inference.parallel_lookahead import (
    JacobiDecoder,
    LookaheadConfig,
    NGramCache,
    estimate_speedup,
    generate_lookahead_branch,
    verify_draft,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 256


@pytest.fixture(scope="module")
def small_model() -> AureliusTransformer:
    """Tiny model that satisfies the AureliusConfig constraints and runs fast."""
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def dummy_encode(text: str) -> list[int]:
    """Encode text as byte values modulo VOCAB."""
    return [b % VOCAB for b in text.encode("utf-8")] or [0]


def dummy_decode(token_ids: list[int]) -> str:
    """Decode token ids back to a string (best-effort utf-8)."""
    return bytes(t % 256 for t in token_ids).decode("utf-8", errors="replace")


@pytest.fixture
def jacobi_decoder(small_model) -> JacobiDecoder:
    cfg = LookaheadConfig(
        lookahead_window=2,
        guess_set_size=2,
        n_gram_size=3,
        max_new_tokens=4,
        verification_steps=1,
    )
    return JacobiDecoder(small_model, cfg, dummy_encode, dummy_decode)


# ---------------------------------------------------------------------------
# 1. LookaheadConfig defaults
# ---------------------------------------------------------------------------


def test_lookahead_config_defaults():
    cfg = LookaheadConfig()
    assert cfg.lookahead_window == 7
    assert cfg.guess_set_size == 7
    assert cfg.n_gram_size == 3
    assert cfg.max_new_tokens == 256
    assert cfg.verification_steps == 1


# ---------------------------------------------------------------------------
# 2. NGramCache starts empty
# ---------------------------------------------------------------------------


def test_ngram_cache_starts_empty():
    cache = NGramCache(n=3)
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# 3. NGramCache.update adds n-grams
# ---------------------------------------------------------------------------


def test_ngram_cache_update_adds_ngrams():
    cache = NGramCache(n=3)
    cache.update([10, 20, 30, 40])
    # keys: (10,20)->30 and (20,30)->40
    assert len(cache) >= 2


# ---------------------------------------------------------------------------
# 4. NGramCache.lookup finds correct continuations
# ---------------------------------------------------------------------------


def test_ngram_cache_lookup_finds_continuation():
    cache = NGramCache(n=3)
    cache.update([1, 2, 3, 4, 5])
    # key (1, 2) -> 3
    assert 3 in cache.lookup((1, 2))
    # key (2, 3) -> 4
    assert 4 in cache.lookup((2, 3))


# ---------------------------------------------------------------------------
# 5. NGramCache.lookup returns empty for unknown context
# ---------------------------------------------------------------------------


def test_ngram_cache_lookup_empty_on_miss():
    cache = NGramCache(n=3)
    cache.update([1, 2, 3])
    result = cache.lookup((99, 99))
    assert result == []


# ---------------------------------------------------------------------------
# 6. NGramCache.__len__ correct
# ---------------------------------------------------------------------------


def test_ngram_cache_len_correct():
    cache = NGramCache(n=3)
    assert len(cache) == 0
    cache.update([1, 2, 3])
    # One unique key: (1, 2)
    assert len(cache) == 1
    cache.update([4, 5, 6])
    # Adds (4, 5); total >= 2
    assert len(cache) >= 2


# ---------------------------------------------------------------------------
# 7. generate_lookahead_branch returns list of correct length
# ---------------------------------------------------------------------------


def test_generate_lookahead_branch_length(small_model):
    prefix = [10, 20, 30]
    branch_len = 4
    draft = generate_lookahead_branch(small_model, prefix, branch_len)
    assert isinstance(draft, list)
    assert len(draft) == branch_len


# ---------------------------------------------------------------------------
# 8. generate_lookahead_branch tokens are valid vocab ids
# ---------------------------------------------------------------------------


def test_generate_lookahead_branch_valid_tokens(small_model):
    prefix = [0, 1, 2]
    draft = generate_lookahead_branch(small_model, prefix, branch_len=4)
    for tok in draft:
        assert isinstance(tok, int)
        assert 0 <= tok < VOCAB


# ---------------------------------------------------------------------------
# 9. verify_draft returns (list, int)
# ---------------------------------------------------------------------------


def test_verify_draft_return_types(small_model):
    prefix = [5, 10, 15]
    draft = [20, 25]
    result = verify_draft(small_model, prefix, draft)
    assert isinstance(result, tuple)
    assert len(result) == 2
    accepted, n_accepted = result
    assert isinstance(accepted, list)
    assert isinstance(n_accepted, int)


# ---------------------------------------------------------------------------
# 10. verify_draft n_accepted <= len(draft)
# ---------------------------------------------------------------------------


def test_verify_draft_n_accepted_bounded(small_model):
    prefix = [1, 2, 3]
    draft = [4, 5, 6]
    _, n_accepted = verify_draft(small_model, prefix, draft)
    assert 0 <= n_accepted <= len(draft)


# ---------------------------------------------------------------------------
# 11. verify_draft correct draft tokens are accepted
# ---------------------------------------------------------------------------


def test_verify_draft_accepted_tokens_match(small_model):
    """Accepted tokens must be a prefix of the draft list."""
    prefix = [7, 8, 9]
    draft = generate_lookahead_branch(small_model, prefix, branch_len=3)
    accepted, n_accepted = verify_draft(small_model, prefix, draft)
    assert accepted == draft[:n_accepted]
    assert len(accepted) == n_accepted


# ---------------------------------------------------------------------------
# 12. JacobiDecoder.decode returns (str, dict)
# ---------------------------------------------------------------------------


def test_jacobi_decoder_decode_return_types(jacobi_decoder):
    result = jacobi_decoder.decode("hello")
    assert isinstance(result, tuple)
    assert len(result) == 2
    text, stats = result
    assert isinstance(text, str)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# 13. JacobiDecoder.decode stats has required keys
# ---------------------------------------------------------------------------


def test_jacobi_decoder_stats_keys(jacobi_decoder):
    _, stats = jacobi_decoder.decode("test")
    assert "tokens_generated" in stats
    assert "cache_hits" in stats
    assert "mean_tokens_per_step" in stats


# ---------------------------------------------------------------------------
# 14. estimate_speedup returns float >= 1.0
# ---------------------------------------------------------------------------


def test_estimate_speedup_returns_float():
    # With n_accepted_per_step >= 1 and small draft_cost, speedup > 1
    speedup = estimate_speedup(n_accepted_per_step=3.0, draft_cost=0.1)
    assert isinstance(speedup, float)
    assert speedup >= 1.0


def test_estimate_speedup_formula():
    """Verify the formula: n / (1 + draft_cost * n)."""
    n = 4.0
    c = 0.1
    expected = n / (1.0 + c * n)
    assert abs(estimate_speedup(n, c) - expected) < 1e-9


# ---------------------------------------------------------------------------
# 15. NGramCache round-trip: update then lookup
# ---------------------------------------------------------------------------


def test_ngram_cache_roundtrip():
    cache = NGramCache(n=3)
    tokens = [100, 200, 50, 75, 200]
    cache.update(tokens)
    # key (100, 200) -> 50
    assert 50 in cache.lookup((100, 200))
    # key (200, 50) -> 75
    assert 75 in cache.lookup((200, 50))
    # key (50, 75) -> 200
    assert 200 in cache.lookup((50, 75))

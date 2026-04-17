"""Tests for src/inference/lookahead_decoding_v2.py.

13+ tests covering:
  - LookaheadConfig defaults
  - NGramPool: add, query, size, clear, pool_size limit
  - LookaheadVerifier: verify_ngram (all-match, first-mismatch), select_best_candidate
  - LookaheadDecoder: generate output shape & token range, generate_step >= 1 token

Mock model: nn.Embedding + nn.Linear, deterministic via fixed seed.
  vocab_size=32, d_model=16
"""
from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn as nn
from torch import LongTensor

from src.inference.lookahead_decoding_v2 import (
    LookaheadConfig,
    LookaheadDecoder,
    LookaheadVerifier,
    NGramPool,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16


class MockModel(nn.Module):
    """Tiny embedding + linear head — returns (1, T, VOCAB) logits."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: LongTensor) -> LongTensor:  # (1, T) -> (1, T, V)
        x = self.embed(input_ids)  # (1, T, D)
        return self.head(x)        # (1, T, V)


@pytest.fixture()
def model_fn():
    """Return a callable model_fn(ids) -> logits."""
    m = MockModel()
    m.eval()
    def _fn(ids: LongTensor) -> LongTensor:
        with torch.no_grad():
            return m(ids)
    return _fn


@pytest.fixture()
def default_config() -> LookaheadConfig:
    return LookaheadConfig()


@pytest.fixture()
def pool() -> NGramPool:
    return NGramPool(ngram_size=3, pool_size=8)


@pytest.fixture()
def verifier() -> LookaheadVerifier:
    return LookaheadVerifier(vocab_size=VOCAB)


@pytest.fixture()
def decoder(model_fn) -> LookaheadDecoder:
    cfg = LookaheadConfig(window_size=3, ngram_size=3, pool_size=16, guess_set_size=4)
    return LookaheadDecoder(model_fn=model_fn, vocab_size=VOCAB, config=cfg)


# ---------------------------------------------------------------------------
# 1. LookaheadConfig defaults
# ---------------------------------------------------------------------------

class TestLookaheadConfig:
    def test_default_window_size(self, default_config):
        assert default_config.window_size == 5

    def test_default_ngram_size(self, default_config):
        assert default_config.ngram_size == 3

    def test_default_pool_size(self, default_config):
        assert default_config.pool_size == 64

    def test_default_guess_set_size(self, default_config):
        assert default_config.guess_set_size == 5


# ---------------------------------------------------------------------------
# 2. NGramPool
# ---------------------------------------------------------------------------

class TestNGramPool:
    def test_add_stores_ngrams(self, pool: NGramPool):
        """Adding a 5-token sequence should store 3 tri-grams (5-3+1=3)."""
        pool.add([1, 2, 3, 4, 5])
        assert pool.size() == 3

    def test_query_returns_matching_ngrams(self, pool: NGramPool):
        """Query with context ending in [1,2] should return n-grams starting with (1,2)."""
        pool.add([1, 2, 3, 1, 2, 4])
        results = pool.query([0, 1, 2], k=5)
        assert len(results) > 0
        for gram in results:
            assert gram[:2] == [1, 2], f"Prefix mismatch: {gram}"

    def test_query_empty_pool_returns_empty(self, pool: NGramPool):
        results = pool.query([1, 2, 3], k=5)
        assert results == []

    def test_query_short_context_returns_empty(self, pool: NGramPool):
        """Context shorter than ngram_size - 1 should return empty result."""
        pool.add([1, 2, 3, 4, 5])
        # ngram_size=3, need prefix_len=2; context length 1 is too short
        results = pool.query([1], k=5)
        assert results == []

    def test_size_after_add(self, pool: NGramPool):
        assert pool.size() == 0
        pool.add([10, 20, 30])
        assert pool.size() == 1

    def test_clear_resets_to_zero(self, pool: NGramPool):
        pool.add([1, 2, 3, 4, 5])
        pool.clear()
        assert pool.size() == 0

    def test_pool_size_limit(self):
        """Pool should not exceed pool_size; LRU eviction fires."""
        small_pool = NGramPool(ngram_size=3, pool_size=3)
        # Add 10 unique tri-grams
        for i in range(10):
            small_pool.add([i, i + 1, i + 2])
        assert small_pool.size() <= 3

    def test_query_returns_at_most_k(self, pool: NGramPool):
        # Add many n-grams with the same prefix [1,2]
        for suffix in range(20):
            pool.add([1, 2, suffix + 3])
        results = pool.query([0, 1, 2], k=3)
        assert len(results) <= 3

    def test_query_no_prefix_match_returns_empty(self, pool: NGramPool):
        pool.add([5, 6, 7, 8, 9])
        # Context tail [1,2] does not match any stored gram prefix [5,6],[6,7],[7,8]
        results = pool.query([0, 1, 2], k=5)
        assert results == []


# ---------------------------------------------------------------------------
# 3. LookaheadVerifier
# ---------------------------------------------------------------------------

class TestLookaheadVerifier:
    def test_verify_ngram_all_match(self, verifier: LookaheadVerifier):
        assert verifier.verify_ngram([1, 2, 3], [1, 2, 3]) == 3

    def test_verify_ngram_first_mismatch_returns_zero(self, verifier: LookaheadVerifier):
        assert verifier.verify_ngram([9, 2, 3], [1, 2, 3]) == 0

    def test_verify_ngram_partial_match(self, verifier: LookaheadVerifier):
        assert verifier.verify_ngram([1, 2, 9], [1, 2, 3]) == 2

    def test_select_best_candidate_returns_best(self, verifier: LookaheadVerifier):
        candidates = [
            [9, 2, 3],   # 0 matches
            [1, 2, 9],   # 2 matches
            [1, 9, 3],   # 1 match
        ]
        best, n = verifier.select_best_candidate(candidates, ground_truth=[1, 2, 3])
        assert n == 2
        assert best == [1, 2, 9]

    def test_select_best_candidate_empty_returns_zero(self, verifier: LookaheadVerifier):
        best, n = verifier.select_best_candidate([], ground_truth=[1, 2, 3])
        assert n == 0
        assert best == []


# ---------------------------------------------------------------------------
# 4. LookaheadDecoder
# ---------------------------------------------------------------------------

class TestLookaheadDecoder:
    def test_generate_returns_correct_length(self, decoder: LookaheadDecoder):
        """Output should have prompt_len + max_new_tokens columns."""
        prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        max_new = 6
        out = decoder.generate(prompt, max_new_tokens=max_new)
        assert out.shape == (1, prompt.shape[1] + max_new)

    def test_generated_tokens_in_valid_range(self, decoder: LookaheadDecoder):
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = decoder.generate(prompt, max_new_tokens=8)
        new_tokens = out[0, prompt.shape[1]:].tolist()
        assert all(0 <= t < VOCAB for t in new_tokens), f"Out-of-range tokens: {new_tokens}"

    def test_generate_step_returns_at_least_one_token(self, decoder: LookaheadDecoder):
        p = NGramPool(ngram_size=3, pool_size=16)
        context = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        accepted, n_accepted = decoder.generate_step(context, p)
        assert n_accepted >= 1
        assert len(accepted) >= 1

    def test_generate_output_is_long_tensor(self, decoder: LookaheadDecoder):
        prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
        out = decoder.generate(prompt, max_new_tokens=4)
        assert out.dtype == torch.long

    def test_generate_preserves_prompt(self, decoder: LookaheadDecoder):
        """Prompt tokens should be unchanged in the output."""
        prompt_tokens = [5, 10, 15, 20]
        prompt = torch.tensor([prompt_tokens], dtype=torch.long)
        out = decoder.generate(prompt, max_new_tokens=5)
        assert out[0, : len(prompt_tokens)].tolist() == prompt_tokens

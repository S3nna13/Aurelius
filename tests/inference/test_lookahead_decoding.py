"""Tests for lookahead_decoding.py — 16 tests using a lightweight mock model.

Mock model: nn.Embedding + nn.Linear, forward(ids) -> (None, logits, None)
vocab=32, d_model=16, seq_len=4, batch=1, max_new_tokens=6
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.inference.lookahead_decoding import (
    LookaheadConfig,
    LookaheadDecoder,
    NGramPool,
    lookahead_decode_step,
    verify_candidates,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16
SEQ_LEN = 4
BATCH = 1
MAX_NEW_TOKENS = 6
N_GRAM = 3


# ---------------------------------------------------------------------------
# Mock model fixture
# ---------------------------------------------------------------------------


class MockLM(nn.Module):
    """Minimal LM: Embedding -> Linear, returns (None, logits, None)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: Tensor):
        x = self.embed(input_ids)  # (B, L, d_model)
        logits = self.head(x)  # (B, L, vocab)
        return None, logits, None


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MockLM(vocab=VOCAB, d_model=D_MODEL)
    m.eval()
    return m


@pytest.fixture
def config():
    return LookaheadConfig(
        window_size=5,
        n_gram_size=N_GRAM,
        guess_set_size=5,
        max_new_tokens=MAX_NEW_TOKENS,
    )


@pytest.fixture
def pool():
    return NGramPool(n=N_GRAM)


@pytest.fixture
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. NGramPool starts empty
# ---------------------------------------------------------------------------


def test_ngram_pool_starts_empty():
    p = NGramPool(n=3)
    assert len(p) == 0


# ---------------------------------------------------------------------------
# 2. NGramPool update fills it
# ---------------------------------------------------------------------------


def test_ngram_pool_update_fills():
    p = NGramPool(n=3)
    p.update([1, 2, 3, 4, 5])
    # 3-grams: (1,2,3), (2,3,4), (3,4,5)
    assert len(p) == 3


# ---------------------------------------------------------------------------
# 3. NGramPool lookup finds prefix matches
# ---------------------------------------------------------------------------


def test_ngram_pool_lookup_finds_prefix():
    p = NGramPool(n=3)
    p.update([10, 20, 30, 40])
    # prefix for n=3 is length 2: (10, 20) -> (10,20,30)
    results = p.lookup((10, 20))
    assert len(results) > 0
    assert (10, 20, 30) in results


# ---------------------------------------------------------------------------
# 4. NGramPool len correct after updates
# ---------------------------------------------------------------------------


def test_ngram_pool_len_correct():
    p = NGramPool(n=2)
    p.update([1, 2, 3])  # bigrams: (1,2), (2,3) -> 2
    p.update([4, 5, 6])  # bigrams: (4,5), (5,6) -> 2 more
    assert len(p) == 4


# ---------------------------------------------------------------------------
# 5. lookup returns empty list for unknown prefix
# ---------------------------------------------------------------------------


def test_ngram_pool_lookup_unknown_prefix():
    p = NGramPool(n=3)
    p.update([1, 2, 3])
    result = p.lookup((99, 88))
    assert result == []


# ---------------------------------------------------------------------------
# 6. NGramPool update with short sequence handles gracefully
# ---------------------------------------------------------------------------


def test_ngram_pool_update_short_sequence():
    p = NGramPool(n=5)
    p.update([1, 2])  # length 2 < n=5, no n-grams extracted
    assert len(p) == 0


# ---------------------------------------------------------------------------
# 7. verify_candidates returns tuple of (Tensor, int)
# ---------------------------------------------------------------------------


def test_verify_candidates_return_type(model, input_ids):
    cand = torch.randint(0, VOCAB, (3,))
    result = verify_candidates(model, input_ids, [cand])
    assert isinstance(result, tuple)
    assert len(result) == 2
    tokens, n = result
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(n, int)


# ---------------------------------------------------------------------------
# 8. verify_candidates n_accepted >= 1
# ---------------------------------------------------------------------------


def test_verify_candidates_at_least_one(model, input_ids):
    cand = torch.randint(0, VOCAB, (3,))
    _, n = verify_candidates(model, input_ids, [cand])
    assert n >= 1


# ---------------------------------------------------------------------------
# 9. lookahead_decode_step returns 1-D tensor and positive int
# ---------------------------------------------------------------------------


def test_lookahead_decode_step_return_types(model, input_ids, pool, config):
    result = lookahead_decode_step(model, input_ids, pool, config)
    assert isinstance(result, tuple)
    tokens, n = result
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dim() == 1
    assert isinstance(n, int)
    assert n > 0


# ---------------------------------------------------------------------------
# 10. lookahead_decode_step updates pool
# ---------------------------------------------------------------------------


def test_lookahead_decode_step_updates_pool(model, input_ids, config):
    p = NGramPool(n=N_GRAM)
    before = len(p)
    lookahead_decode_step(model, input_ids, p, config)
    after = len(p)
    assert after > before


# ---------------------------------------------------------------------------
# 11. LookaheadDecoder decode output correct shape
# ---------------------------------------------------------------------------


def test_lookahead_decoder_decode_shape(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    out = decoder.decode(ids, MAX_NEW_TOKENS)
    assert out.dim() == 1
    assert out.shape[0] == MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# 12. decode generates max_new_tokens tokens
# ---------------------------------------------------------------------------


def test_decode_generates_max_new_tokens(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    out = decoder.decode(ids, MAX_NEW_TOKENS)
    assert out.shape[0] == MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# 13. decode_with_stats has required keys
# ---------------------------------------------------------------------------


def test_decode_with_stats_required_keys(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    stats = decoder.decode_with_stats(ids, MAX_NEW_TOKENS)
    assert "output" in stats
    assert "total_steps" in stats
    assert "total_tokens" in stats
    assert "mean_tokens_per_step" in stats


# ---------------------------------------------------------------------------
# 14. mean_tokens_per_step >= 1.0
# ---------------------------------------------------------------------------


def test_decode_with_stats_mean_tokens(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    stats = decoder.decode_with_stats(ids, MAX_NEW_TOKENS)
    assert stats["mean_tokens_per_step"] >= 1.0


# ---------------------------------------------------------------------------
# 15. total_tokens == max_new_tokens
# ---------------------------------------------------------------------------


def test_decode_with_stats_total_tokens(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    stats = decoder.decode_with_stats(ids, MAX_NEW_TOKENS)
    assert stats["total_tokens"] == MAX_NEW_TOKENS


# ---------------------------------------------------------------------------
# 16. decode output dtype is torch.long
# ---------------------------------------------------------------------------


def test_decode_output_dtype(model, config):
    decoder = LookaheadDecoder(model, config)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    out = decoder.decode(ids, MAX_NEW_TOKENS)
    assert out.dtype == torch.long


# ---------------------------------------------------------------------------
# Bonus: NGramPool n-grams correctly formed (3-gram from [1,2,3] -> (1,2,3))
# ---------------------------------------------------------------------------


def test_ngram_pool_ngrams_correctly_formed():
    p = NGramPool(n=3)
    p.update([1, 2, 3])
    results = p.lookup((1, 2))
    assert (1, 2, 3) in results


# ---------------------------------------------------------------------------
# Bonus: verify_candidates n_accepted <= candidate length
# ---------------------------------------------------------------------------


def test_verify_candidates_n_accepted_le_candidate_len(model, input_ids):
    cand = torch.randint(0, VOCAB, (4,))
    _, n = verify_candidates(model, input_ids, [cand])
    assert n <= 4

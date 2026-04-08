"""Tests for simplified lookahead decoding."""
import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.lookahead import LookaheadConfig, NGramCache, lookahead_generate


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=64, max_seq_len=64,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def test_ngram_cache_update_and_lookup():
    cache = NGramCache(n=3)
    cache.update([1, 2, 3, 4, 5])
    # Prefix (1,2) -> 3
    assert 3 in cache.candidates([1, 2])
    # Prefix (2,3) -> 4
    assert 4 in cache.candidates([2, 3])


def test_ngram_cache_empty_on_miss():
    cache = NGramCache(n=3)
    cache.update([1, 2, 3])
    assert cache.candidates([9, 9]) == []


def test_ngram_cache_short_context():
    cache = NGramCache(n=3)
    cache.update([1, 2, 3])
    assert cache.candidates([1]) == []  # context too short for n=3


def test_lookahead_generates_tokens(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = LookaheadConfig(max_new_tokens=10, ngram_size=3)
    result = lookahead_generate(small_model, prompt, cfg)
    assert result.shape[0] == 1
    assert result.shape[1] > 4  # generated some tokens


def test_lookahead_respects_max_tokens(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = LookaheadConfig(max_new_tokens=5, ngram_size=3)
    result = lookahead_generate(small_model, prompt, cfg)
    assert result.shape[1] <= 4 + 5 + 1  # prompt + max_new + possible 1 extra from lookahead


def test_lookahead_eos_stops(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = LookaheadConfig(max_new_tokens=50, eos_token_id=1, ngram_size=3)
    result = lookahead_generate(small_model, prompt, cfg)
    # If EOS was generated, it should be the last token
    tokens = result[0].tolist()
    if 1 in tokens[4:]:
        eos_pos = tokens.index(1, 4)
        assert eos_pos == len(tokens) - 1


def test_lookahead_no_cache_fallback(small_model):
    """With no n-gram cache hits, should fall back to normal sampling."""
    prompt = torch.randint(0, 64, (1, 4))
    cfg = LookaheadConfig(max_new_tokens=8, ngram_size=10)  # large n = no cache hits
    result = lookahead_generate(small_model, prompt, cfg)
    assert result.shape[1] == 4 + 8  # exactly max_new_tokens (all fallback)

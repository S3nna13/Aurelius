"""Tests for lookahead_decoding.py — at least 15 tests."""
import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.lookahead_decoding import (
    LookaheadConfig,
    NGramPool,
    LookaheadDecoder,
    lookahead_step,
    verify_ngram,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    torch.manual_seed(42)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def pool():
    return NGramPool(n_gram_size=3)


@pytest.fixture
def config():
    return LookaheadConfig(
        window_size=4,
        n_gram_size=3,
        n_candidates=5,
        temperature=1.0,
        max_new_tokens=8,
    )


# ---------------------------------------------------------------------------
# 1. LookaheadConfig defaults
# ---------------------------------------------------------------------------

def test_lookahead_config_defaults():
    cfg = LookaheadConfig()
    assert cfg.window_size == 4
    assert cfg.n_gram_size == 3
    assert cfg.n_candidates == 5
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 32


# ---------------------------------------------------------------------------
# 2. NGramPool starts empty
# ---------------------------------------------------------------------------

def test_ngram_pool_starts_empty():
    p = NGramPool(n_gram_size=3)
    assert len(p) == 0


# ---------------------------------------------------------------------------
# 3. NGramPool.add increases size
# ---------------------------------------------------------------------------

def test_ngram_pool_add_increases_size(pool):
    tokens = torch.tensor([1, 2, 3, 4, 5])
    pool.add(tokens)
    # 5 tokens, n=3 -> 3 n-grams: [1,2,3], [2,3,4], [3,4,5]
    assert len(pool) == 3


# ---------------------------------------------------------------------------
# 4. NGramPool.add with short sequence (<n_gram_size) doesn't crash
# ---------------------------------------------------------------------------

def test_ngram_pool_add_short_sequence_no_crash():
    p = NGramPool(n_gram_size=5)
    tokens = torch.tensor([1, 2])  # length 2 < n_gram_size 5
    p.add(tokens)  # should not raise
    assert len(p) == 0


# ---------------------------------------------------------------------------
# 5. NGramPool.get_candidates returns list of tensors of shape (n_gram_size,)
# ---------------------------------------------------------------------------

def test_ngram_pool_get_candidates_shape():
    p = NGramPool(n_gram_size=3)
    tokens = torch.tensor([10, 20, 30, 40, 50])
    p.add(tokens)
    prefix = torch.tensor([10, 20])
    candidates = p.get_candidates(prefix, n_candidates=5)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    for cand in candidates:
        assert isinstance(cand, torch.Tensor)
        assert cand.shape == (3,)


# ---------------------------------------------------------------------------
# 6. NGramPool.get_candidates on empty pool returns []
# ---------------------------------------------------------------------------

def test_ngram_pool_get_candidates_empty_pool():
    p = NGramPool(n_gram_size=3)
    prefix = torch.tensor([1, 2, 3])
    result = p.get_candidates(prefix, n_candidates=5)
    assert result == []


# ---------------------------------------------------------------------------
# 7. NGramPool.get_candidates respects n_candidates limit
# ---------------------------------------------------------------------------

def test_ngram_pool_get_candidates_respects_limit():
    p = NGramPool(n_gram_size=3)
    # Build many n-grams that all start with token 5
    # e.g., [5, 0, x] for x in 0..9
    for i in range(10):
        # Each is a 3-gram starting with 5
        p._pool.append(torch.tensor([5, i, i + 1]))
    prefix = torch.tensor([5])
    candidates = p.get_candidates(prefix, n_candidates=3)
    assert len(candidates) <= 3


# ---------------------------------------------------------------------------
# 8. verify_ngram returns (int, Tensor)
# ---------------------------------------------------------------------------

def test_verify_ngram_return_types(small_model):
    context = torch.randint(0, 256, (1, 8))
    candidate = torch.randint(0, 256, (3,))
    result = verify_ngram(small_model, context, candidate)
    assert isinstance(result, tuple)
    assert len(result) == 2
    n_accepted, accepted_tokens = result
    assert isinstance(n_accepted, int)
    assert isinstance(accepted_tokens, torch.Tensor)


# ---------------------------------------------------------------------------
# 9. verify_ngram n_accepted in [0, n_gram_size]
# ---------------------------------------------------------------------------

def test_verify_ngram_n_accepted_range(small_model):
    context = torch.randint(0, 256, (1, 8))
    candidate = torch.randint(0, 256, (4,))
    n_accepted, _ = verify_ngram(small_model, context, candidate)
    assert 0 <= n_accepted <= 4


# ---------------------------------------------------------------------------
# 10. verify_ngram accepted_tokens shape is (n_accepted,)
# ---------------------------------------------------------------------------

def test_verify_ngram_accepted_tokens_shape(small_model):
    context = torch.randint(0, 256, (1, 6))
    candidate = torch.randint(0, 256, (3,))
    n_accepted, accepted_tokens = verify_ngram(small_model, context, candidate)
    assert accepted_tokens.shape == (n_accepted,)


# ---------------------------------------------------------------------------
# 11. lookahead_step returns (Tensor, int)
# ---------------------------------------------------------------------------

def test_lookahead_step_return_types(small_model, pool, config):
    context = torch.randint(0, 256, (1, 6))
    result = lookahead_step(small_model, context, pool, config)
    assert isinstance(result, tuple)
    assert len(result) == 2
    accepted_ids, n_accepted = result
    assert isinstance(accepted_ids, torch.Tensor)
    assert isinstance(n_accepted, int)


# ---------------------------------------------------------------------------
# 12. lookahead_step n_accepted >= 1 (at least greedy fallback)
# ---------------------------------------------------------------------------

def test_lookahead_step_at_least_one_token(small_model, pool, config):
    context = torch.randint(0, 256, (1, 6))
    _, n_accepted = lookahead_step(small_model, context, pool, config)
    assert n_accepted >= 1


# ---------------------------------------------------------------------------
# 13. LookaheadDecoder instantiates
# ---------------------------------------------------------------------------

def test_lookahead_decoder_instantiates(small_model, config):
    decoder = LookaheadDecoder(small_model, config)
    assert decoder.model is small_model
    assert decoder.config is config
    assert isinstance(decoder.pool, NGramPool)


# ---------------------------------------------------------------------------
# 14. LookaheadDecoder.generate returns tensor longer than input
# ---------------------------------------------------------------------------

def test_lookahead_decoder_generate_longer(small_model, config):
    decoder = LookaheadDecoder(small_model, config)
    input_ids = torch.randint(0, 256, (1, 5))
    output_ids, _ = decoder.generate(input_ids)
    assert output_ids.shape[1] > input_ids.shape[1]


# ---------------------------------------------------------------------------
# 15. LookaheadDecoder.generate stats dict has correct keys
# ---------------------------------------------------------------------------

def test_lookahead_decoder_generate_stats_keys(small_model, config):
    decoder = LookaheadDecoder(small_model, config)
    input_ids = torch.randint(0, 256, (1, 5))
    _, stats = decoder.generate(input_ids)
    assert "n_steps" in stats
    assert "n_tokens" in stats
    assert "mean_accept_len" in stats


# ---------------------------------------------------------------------------
# Bonus tests
# ---------------------------------------------------------------------------

def test_lookahead_decoder_generate_correct_n_tokens(small_model):
    cfg = LookaheadConfig(max_new_tokens=10)
    decoder = LookaheadDecoder(small_model, cfg)
    input_ids = torch.randint(0, 256, (1, 4))
    output_ids, stats = decoder.generate(input_ids)
    assert stats["n_tokens"] == cfg.max_new_tokens
    assert output_ids.shape[1] == input_ids.shape[1] + cfg.max_new_tokens


def test_lookahead_decoder_stats_mean_accept_len(small_model):
    cfg = LookaheadConfig(max_new_tokens=8)
    decoder = LookaheadDecoder(small_model, cfg)
    input_ids = torch.randint(0, 256, (1, 4))
    _, stats = decoder.generate(input_ids)
    assert stats["mean_accept_len"] >= 1.0


def test_update_pool_adds_to_pool(small_model, config):
    decoder = LookaheadDecoder(small_model, config)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6])
    before = len(decoder.pool)
    decoder.update_pool(tokens)
    after = len(decoder.pool)
    assert after > before


def test_verify_ngram_with_forced_match(small_model):
    """Force a candidate that exactly matches what the model would predict."""
    context = torch.randint(0, 256, (1, 6))
    # Run model once to get what it predicts
    with torch.no_grad():
        _, logits, _ = small_model(context)
    predicted_first = logits[0, -1, :].argmax().item()
    # Build a candidate whose first token matches
    candidate = torch.tensor([predicted_first, 0, 0])
    n_accepted, accepted_tokens = verify_ngram(small_model, context, candidate)
    assert n_accepted >= 1  # at least first token accepted
    assert accepted_tokens[0].item() == predicted_first


def test_ngram_pool_max_pool_size():
    """Pool should not grow beyond max_pool_size."""
    p = NGramPool(n_gram_size=2, max_pool_size=5)
    tokens = torch.arange(20)  # 19 bigrams
    p.add(tokens)
    assert len(p) <= 5


def test_lookahead_step_with_pool_hit(small_model, config):
    """When pool has a matching candidate, step may accept > 1 token."""
    context = torch.randint(0, 256, (1, 6))
    # Find what the model predicts and plant it in pool
    with torch.no_grad():
        _, logits, _ = small_model(context)
    first_predicted = logits[0, -1, :].argmax().item()
    # Plant a 3-gram starting with last context token, continuing with predicted
    last_ctx_token = context[0, -1].item()
    pool = NGramPool(n_gram_size=config.n_gram_size)
    candidate = torch.tensor([last_ctx_token, first_predicted, first_predicted])
    pool._pool.append(candidate)
    accepted_ids, n_accepted = lookahead_step(small_model, context, pool, config)
    assert n_accepted >= 1

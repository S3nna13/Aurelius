"""Tests for Contrastive Decoding (Li et al., 2022)."""
from __future__ import annotations

import math

import torch
import pytest

from src.inference.contrastive_decoding import (
    CDConfig,
    compute_contrastive_logits,
    nucleus_sample,
    ContrastiveDecoder,
    VocabProjectionAmateur,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _small_config() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def expert_model():
    torch.manual_seed(0)
    return AureliusTransformer(_small_config())


@pytest.fixture
def amateur_model():
    torch.manual_seed(1)
    return AureliusTransformer(_small_config())


@pytest.fixture
def cd_decoder(expert_model, amateur_model):
    cfg = CDConfig(max_new_tokens=3)
    return ContrastiveDecoder(expert_model, amateur_model, cfg)


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 4))


# ---------------------------------------------------------------------------
# 1. CDConfig defaults
# ---------------------------------------------------------------------------

def test_cd_config_defaults():
    cfg = CDConfig()
    assert cfg.alpha == 0.1
    assert cfg.beta == 0.5
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 50
    assert cfg.top_p == 0.95


# ---------------------------------------------------------------------------
# 2. compute_contrastive_logits shape
# ---------------------------------------------------------------------------

def test_compute_contrastive_logits_shape():
    V = 256
    expert_logits = torch.randn(V)
    amateur_logits = torch.randn(V)
    score = compute_contrastive_logits(expert_logits, amateur_logits, alpha=0.1)
    assert score.shape == (V,)


# ---------------------------------------------------------------------------
# 3. compute_contrastive_logits plausibility mask
# ---------------------------------------------------------------------------

def test_compute_contrastive_logits_plausibility_mask():
    V = 16
    # Expert strongly peaks at token 0; all others far below threshold
    expert_logits = torch.full((V,), -100.0)
    expert_logits[0] = 10.0
    amateur_logits = torch.zeros(V)

    score = compute_contrastive_logits(expert_logits, amateur_logits, alpha=0.1)

    # Tokens 1..V-1 are below threshold → -inf
    assert score[0] != float("-inf"), "Token 0 should be in plausibility set"
    for i in range(1, V):
        assert score[i] == float("-inf"), f"Token {i} should be masked to -inf"


# ---------------------------------------------------------------------------
# 4. compute_contrastive_logits top token
# ---------------------------------------------------------------------------

def test_compute_contrastive_logits_top_token():
    V = 32
    # Expert strongly prefers token 0; amateur is uniform
    expert_logits = torch.full((V,), -1.0)
    expert_logits[0] = 10.0
    amateur_logits = torch.zeros(V)  # uniform amateur

    score = compute_contrastive_logits(expert_logits, amateur_logits, alpha=0.1)

    # Token 0 should have the highest (non -inf) score
    assert score.argmax().item() == 0


# ---------------------------------------------------------------------------
# 5. nucleus_sample returns valid token id
# ---------------------------------------------------------------------------

def test_nucleus_sample_returns_valid_token():
    V = 256
    torch.manual_seed(0)
    logits = torch.randn(V)
    token_id = nucleus_sample(logits, top_p=0.9, temperature=1.0)
    assert isinstance(token_id, int)
    assert 0 <= token_id < V


# ---------------------------------------------------------------------------
# 6. nucleus_sample top_p=1.0 allows all tokens
# ---------------------------------------------------------------------------

def test_nucleus_sample_top_p_one():
    V = 64
    torch.manual_seed(42)
    # Flat logits → all tokens equally likely
    logits = torch.zeros(V)
    seen = set()
    for seed in range(200):
        torch.manual_seed(seed)
        seen.add(nucleus_sample(logits, top_p=1.0, temperature=1.0))
    # With top_p=1.0 and enough seeds we should see more than 1 distinct token
    assert len(seen) > 1


# ---------------------------------------------------------------------------
# 7. ContrastiveDecoder generate shape
# ---------------------------------------------------------------------------

def test_contrastive_decoder_generate_shape(cd_decoder, input_ids):
    output = cd_decoder.generate(input_ids, max_new_tokens=3)
    assert output.shape[0] == 1
    T = input_ids.shape[1]
    assert output.shape[1] == T + 3


# ---------------------------------------------------------------------------
# 8. ContrastiveDecoder generate_greedy shape
# ---------------------------------------------------------------------------

def test_contrastive_decoder_greedy_shape(cd_decoder, input_ids):
    output = cd_decoder.generate_greedy(input_ids, max_new_tokens=3)
    assert output.shape[0] == 1
    T = input_ids.shape[1]
    assert output.shape[1] == T + 3


# ---------------------------------------------------------------------------
# 9. ContrastiveDecoder generate_greedy is deterministic
# ---------------------------------------------------------------------------

def test_contrastive_decoder_greedy_deterministic(cd_decoder, input_ids):
    out1 = cd_decoder.generate_greedy(input_ids, max_new_tokens=3)
    out2 = cd_decoder.generate_greedy(input_ids, max_new_tokens=3)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# 10. VocabProjectionAmateur — no projection when vocab sizes match
# ---------------------------------------------------------------------------

def test_vocab_projection_same_vocab(expert_model):
    """When amateur and expert share vocab size, no projection layer is created."""
    wrapper = VocabProjectionAmateur(expert_model, expert_vocab_size=expert_model.config.vocab_size)
    assert wrapper.projection is None


# ---------------------------------------------------------------------------
# 11. VocabProjectionAmateur — output shape matches expert vocab
# ---------------------------------------------------------------------------

def test_vocab_projection_output_shape():
    """Output logits shape must be (B, T, expert_vocab_size) even when vocabs differ."""
    torch.manual_seed(0)
    amateur_cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=128, max_seq_len=512,
    )
    amateur = AureliusTransformer(amateur_cfg)
    expert_vocab_size = 256

    wrapper = VocabProjectionAmateur(amateur, expert_vocab_size=expert_vocab_size)
    assert wrapper.projection is not None

    # Input ids must be valid for the *amateur* vocab (< 128)
    input_ids = torch.randint(0, 128, (1, 4))
    logits = wrapper(input_ids)
    assert logits.shape == (1, 4, expert_vocab_size)

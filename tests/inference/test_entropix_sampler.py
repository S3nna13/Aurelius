"""Tests for src/inference/entropix_sampler.py.

Covers all 12 required cases:
 1.  token_entropy shape (B,)
 2.  Uniform logits → entropy = log(V)
 3.  One-hot logits → entropy ≈ 0
 4.  attn_varentropy shape (B,)
 5.  Identical heads → varentropy = 0
 6.  sample returns (B,) ints in [0, V)
 7.  Very peaked logits → always argmax
 8.  High entropy + high varentropy → varied samples (not all same token)
 9.  No NaN/Inf in entropy with -inf logits
10.  No NaN/Inf in varentropy
11.  Determinism with manual_seed generator
12.  Batch size 1 works
"""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.entropix_sampler import EntropixConfig, EntropixSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH = 4
VOCAB = 128
HEADS = 8
SEQ = 16


def make_sampler(**kwargs) -> EntropixSampler:
    cfg = EntropixConfig(**kwargs)
    return EntropixSampler(cfg)


def uniform_logits(B: int = BATCH, V: int = VOCAB) -> torch.Tensor:
    """All-zeros logits → uniform distribution."""
    return torch.zeros(B, V)


def peaked_logits(B: int = BATCH, V: int = VOCAB, peak_token: int = 7) -> torch.Tensor:
    """Very confident logits: one token has +1000, rest have -1000."""
    l = torch.full((B, V), -1000.0)
    l[:, peak_token] = 1000.0
    return l


def random_attn(B: int = BATCH, H: int = HEADS, T: int = SEQ) -> torch.Tensor:
    """Random attention weights (softmax-normalised)."""
    torch.manual_seed(42)
    raw = torch.rand(B, H, T, T)
    return torch.softmax(raw, dim=-1)


def uniform_attn(B: int = BATCH, H: int = HEADS, T: int = SEQ) -> torch.Tensor:
    """Uniform attention weights (all heads identical)."""
    w = torch.full((B, H, T, T), 1.0 / T)
    return w


# ---------------------------------------------------------------------------
# Test 1 – token_entropy shape (B,)
# ---------------------------------------------------------------------------

def test_token_entropy_shape():
    sampler = make_sampler()
    logits = uniform_logits()
    ent = sampler.token_entropy(logits)
    assert ent.shape == (BATCH,), f"Expected ({BATCH},) got {ent.shape}"


# ---------------------------------------------------------------------------
# Test 2 – Uniform logits → entropy = log(V)
# ---------------------------------------------------------------------------

def test_token_entropy_uniform_equals_log_vocab():
    sampler = make_sampler()
    logits = uniform_logits()
    ent = sampler.token_entropy(logits)
    expected = math.log(VOCAB)
    assert torch.allclose(ent, torch.full_like(ent, expected), atol=1e-5), (
        f"Uniform entropy should be log({VOCAB})={expected:.4f}, got {ent}"
    )


# ---------------------------------------------------------------------------
# Test 3 – One-hot logits → entropy ≈ 0
# ---------------------------------------------------------------------------

def test_token_entropy_one_hot_near_zero():
    sampler = make_sampler()
    logits = peaked_logits()
    ent = sampler.token_entropy(logits)
    assert (ent < 1e-3).all(), f"One-hot entropy should be near 0, got {ent}"


# ---------------------------------------------------------------------------
# Test 4 – attn_varentropy shape (B,)
# ---------------------------------------------------------------------------

def test_attn_varentropy_shape():
    sampler = make_sampler()
    attn = random_attn()
    ve = sampler.attn_varentropy(attn)
    assert ve.shape == (BATCH,), f"Expected ({BATCH},) got {ve.shape}"


# ---------------------------------------------------------------------------
# Test 5 – Identical heads → varentropy = 0
# ---------------------------------------------------------------------------

def test_attn_varentropy_identical_heads_zero():
    sampler = make_sampler()
    attn = uniform_attn()   # every head is the same uniform distribution
    ve = sampler.attn_varentropy(attn)
    assert torch.allclose(ve, torch.zeros_like(ve), atol=1e-6), (
        f"Identical heads should yield varentropy=0, got {ve}"
    )


# ---------------------------------------------------------------------------
# Test 6 – sample returns (B,) ints in [0, V)
# ---------------------------------------------------------------------------

def test_sample_output_shape_and_range():
    sampler = make_sampler()
    logits = uniform_logits()
    attn = random_attn()
    tokens = sampler.sample(logits, attn)
    assert tokens.shape == (BATCH,), f"Expected ({BATCH},) got {tokens.shape}"
    assert tokens.dtype in (torch.long, torch.int64)
    assert (tokens >= 0).all() and (tokens < VOCAB).all(), (
        f"Tokens out of range [0, {VOCAB}): {tokens}"
    )


# ---------------------------------------------------------------------------
# Test 7 – Very peaked logits → always argmax
# ---------------------------------------------------------------------------

def test_sample_peaked_logits_always_argmax():
    # Force low-H + low-VH path (identical heads, very confident logits)
    cfg = EntropixConfig(
        high_ent_thresh=10.0,   # ensure low-H branch
        high_vent_thresh=10.0,  # ensure low-VH branch
    )
    sampler = EntropixSampler(cfg)
    peak_tok = 7
    logits = peaked_logits(peak_token=peak_tok)
    attn = uniform_attn()
    tokens = sampler.sample(logits, attn)
    assert (tokens == peak_tok).all(), (
        f"Peaked logits should always select token {peak_tok}, got {tokens}"
    )


# ---------------------------------------------------------------------------
# Test 8 – High entropy + high varentropy → varied samples
# ---------------------------------------------------------------------------

def test_sample_high_ent_high_vent_varied():
    # Force all samples through the creative (high-H + high-VH) branch
    cfg = EntropixConfig(
        high_ent_thresh=0.0,    # everything is "high entropy"
        high_vent_thresh=0.0,   # everything is "high varentropy"
        creative_temp=1.5,
    )
    sampler = EntropixSampler(cfg)
    logits = uniform_logits(B=64, V=VOCAB)
    attn = random_attn(B=64)
    tokens = sampler.sample(logits, attn)
    # With 64 samples from 128 tokens at high temp, expect > 1 unique token
    unique = tokens.unique()
    assert unique.numel() > 1, (
        f"Creative branch on uniform logits should produce varied tokens, "
        f"got {unique.numel()} unique tokens"
    )


# ---------------------------------------------------------------------------
# Test 9 – No NaN/Inf in entropy with -inf logits
# ---------------------------------------------------------------------------

def test_token_entropy_no_nan_with_neg_inf_logits():
    sampler = make_sampler()
    logits = torch.full((BATCH, VOCAB), float("-inf"))
    logits[:, 0] = 0.0   # one valid token per row
    ent = sampler.token_entropy(logits)
    assert not torch.isnan(ent).any(), "NaN in entropy with -inf logits"
    assert not torch.isinf(ent).any(), "Inf in entropy with -inf logits"


# ---------------------------------------------------------------------------
# Test 10 – No NaN/Inf in varentropy
# ---------------------------------------------------------------------------

def test_attn_varentropy_no_nan_or_inf():
    sampler = make_sampler()
    # Attempt with random weights including near-zero values
    torch.manual_seed(99)
    raw = torch.rand(BATCH, HEADS, SEQ, SEQ) * 0.01
    attn = torch.softmax(raw, dim=-1)
    ve = sampler.attn_varentropy(attn)
    assert not torch.isnan(ve).any(), "NaN in varentropy"
    assert not torch.isinf(ve).any(), "Inf in varentropy"


# ---------------------------------------------------------------------------
# Test 11 – Determinism with manual_seed generator
# ---------------------------------------------------------------------------

def test_sample_deterministic_with_generator():
    sampler = make_sampler()
    logits = uniform_logits()
    attn = random_attn()

    gen1 = torch.Generator()
    gen1.manual_seed(1234)
    tokens1 = sampler.sample(logits, attn, generator=gen1)

    gen2 = torch.Generator()
    gen2.manual_seed(1234)
    tokens2 = sampler.sample(logits, attn, generator=gen2)

    assert torch.equal(tokens1, tokens2), (
        f"Same seed should produce identical tokens:\n{tokens1}\nvs\n{tokens2}"
    )


# ---------------------------------------------------------------------------
# Test 12 – Batch size 1 works
# ---------------------------------------------------------------------------

def test_sample_batch_size_one():
    sampler = make_sampler()
    logits = uniform_logits(B=1)
    attn = random_attn(B=1)
    tokens = sampler.sample(logits, attn)
    assert tokens.shape == (1,), f"Expected (1,) got {tokens.shape}"
    assert 0 <= tokens[0].item() < VOCAB, f"Token {tokens[0]} out of vocab range"

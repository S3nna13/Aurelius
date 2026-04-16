"""Tests for src/inference/arithmetic_coding.py

All tests pass pre-computed probability tensors so no real LM is required.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.inference.arithmetic_coding import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    LMArithmeticCoder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(vocab: int) -> torch.Tensor:
    return torch.ones(vocab) / vocab


def _peaked(vocab: int, hot_tok: int = 0, peak: float = 0.95) -> torch.Tensor:
    p = torch.full((vocab,), (1.0 - peak) / (vocab - 1))
    p[hot_tok] = peak
    return p


def _random_seq(n: int, vocab: int, seed: int = 42) -> list[int]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=rng).tolist()


def _uniform_probs(n: int, vocab: int) -> list[torch.Tensor]:
    p = _uniform(vocab)
    return [p.clone() for _ in range(n)]


def _peaked_probs(n: int, vocab: int, **kw) -> list[torch.Tensor]:
    p = _peaked(vocab, **kw)
    return [p.clone() for _ in range(n)]


def _make_probs_fn(probs: list[torch.Tensor]):
    """Return a stateful probs_fn that yields tensors in order."""
    def fn(decoded: list[int]) -> torch.Tensor:
        idx = len(decoded)
        if idx < len(probs):
            return probs[idx]
        # fallback: uniform
        return _uniform(probs[0].shape[0])
    return fn


# ---------------------------------------------------------------------------
# Test 1 — round-trip: single token
# ---------------------------------------------------------------------------

def test_roundtrip_single_token():
    vocab = 256
    tok = [7]
    probs = _uniform_probs(1, vocab)

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), 1)

    assert recovered == tok, f"Expected {tok}, got {recovered}"


# ---------------------------------------------------------------------------
# Test 2 — round-trip: 10-token sequence, vocab=256
# ---------------------------------------------------------------------------

def test_roundtrip_10_tokens():
    vocab = 256
    tok = _random_seq(10, vocab)
    probs = _uniform_probs(10, vocab)

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), 10)

    assert recovered == tok


# ---------------------------------------------------------------------------
# Test 3 — round-trip: 50-token sequence
# ---------------------------------------------------------------------------

def test_roundtrip_50_tokens():
    vocab = 256
    tok = _random_seq(50, vocab, seed=7)
    probs = _uniform_probs(50, vocab)

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), 50)

    assert recovered == tok


# ---------------------------------------------------------------------------
# Test 4 — determinism: same input → same bitstream
# ---------------------------------------------------------------------------

def test_determinism():
    vocab = 64
    tok = _random_seq(20, vocab)
    probs = _uniform_probs(20, vocab)

    enc = ArithmeticEncoder()
    bs1 = enc.encode(tok, probs)
    bs2 = enc.encode(tok, probs)

    assert bs1 == bs2, "Encoding is not deterministic"


# ---------------------------------------------------------------------------
# Test 5 — different inputs → different bitstreams
# ---------------------------------------------------------------------------

def test_different_inputs_different_bitstreams():
    vocab = 256
    tok_a = _random_seq(10, vocab, seed=1)
    tok_b = _random_seq(10, vocab, seed=2)
    assert tok_a != tok_b

    probs = _uniform_probs(10, vocab)
    enc = ArithmeticEncoder()
    bs_a = enc.encode(tok_a, probs)
    bs_b = enc.encode(tok_b, probs)

    assert bs_a != bs_b, "Different inputs produced identical bitstreams"


# ---------------------------------------------------------------------------
# Test 6 — compression ratio: bits ≤ ceil(H * n / 8) + overhead
# ---------------------------------------------------------------------------

def test_compression_ratio():
    vocab = 16
    n = 40
    tok = _random_seq(n, vocab)
    probs = _uniform_probs(n, vocab)  # H = log2(16) = 4 bits/token

    enc = ArithmeticEncoder()
    bs = enc.encode(tok, probs)

    H_bits_per_tok = math.log2(vocab)  # 4
    max_bytes = math.ceil(H_bits_per_tok * n / 8) + 8  # generous overhead
    assert len(bs) <= max_bytes, (
        f"Compressed size {len(bs)} B > expected max {max_bytes} B"
    )


# ---------------------------------------------------------------------------
# Test 7 — bits_per_token ≈ log2(vocab) for uniform distribution
# ---------------------------------------------------------------------------

def test_bits_per_token_uniform():
    vocab = 256
    n = 100
    tok = _random_seq(n, vocab)
    probs = _uniform_probs(n, vocab)

    coder = LMArithmeticCoder()
    bpt = coder.bits_per_token(tok, probs)

    expected = math.log2(vocab)  # 8.0
    assert abs(bpt - expected) <= 1.0, (
        f"bits_per_token={bpt:.2f}, expected ~{expected:.2f} ± 1"
    )


# ---------------------------------------------------------------------------
# Test 8 — peaked distribution compresses better than uniform
# ---------------------------------------------------------------------------

def test_peaked_compresses_better():
    vocab = 256
    n = 50
    hot_tok = 3

    tok_uniform = [hot_tok] * n  # same tokens
    probs_uniform = _uniform_probs(n, vocab)
    probs_peaked = _peaked_probs(n, vocab, hot_tok=hot_tok)

    coder = LMArithmeticCoder()
    bpt_uniform = coder.bits_per_token(tok_uniform, probs_uniform)
    bpt_peaked = coder.bits_per_token(tok_uniform, probs_peaked)

    assert bpt_peaked < bpt_uniform, (
        f"Peaked ({bpt_peaked:.2f} bpt) not better than uniform ({bpt_uniform:.2f} bpt)"
    )


# ---------------------------------------------------------------------------
# Test 9 — single-token vocabulary (trivially compressible)
# ---------------------------------------------------------------------------

def test_single_token_vocabulary():
    vocab = 1
    tok = [0, 0, 0, 0, 0]
    # one-hot distribution
    probs = [torch.tensor([1.0]) for _ in range(5)]

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), 5)

    assert recovered == tok
    # should be very short (only 1-byte header + minimal data)
    assert len(bs) <= 4, f"Expected tiny bitstream, got {len(bs)} bytes"


# ---------------------------------------------------------------------------
# Test 10 — binary alphabet, uniform → ~1 bit/token
# ---------------------------------------------------------------------------

def test_binary_alphabet_uniform():
    vocab = 2
    n = 64
    tok = _random_seq(n, vocab, seed=99)
    probs = _uniform_probs(n, vocab)

    coder = LMArithmeticCoder()
    bpt = coder.bits_per_token(tok, probs)

    # Should be close to 1.0 bit per token
    assert abs(bpt - 1.0) <= 1.0, f"Expected ~1 bit/token, got {bpt:.2f}"

    # Ensure round-trip works
    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), n)
    assert recovered == tok


# ---------------------------------------------------------------------------
# Test 11 — numerical stability: near-zero probabilities
# ---------------------------------------------------------------------------

def test_near_zero_probabilities():
    vocab = 8
    n = 10
    # Construct a near-degenerate distribution with one near-zero prob
    p_raw = torch.tensor([0.0, 1e-38, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    p_raw = p_raw / p_raw.sum()  # normalise (still near-zero at idx 1)
    probs = [p_raw.clone() for _ in range(n)]
    tok = [2, 3, 4, 5, 6, 7, 2, 3, 4, 5]

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), n)

    assert recovered == tok, f"Expected {tok}, got {recovered}"


# ---------------------------------------------------------------------------
# Test 12 — empty sequence
# ---------------------------------------------------------------------------

def test_empty_sequence():
    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()

    bs = enc.encode([], [])
    assert bs == b"", f"Empty encode should return b'', got {bs!r}"

    recovered = dec.decode(bs, lambda _: _uniform(256), 0)
    assert recovered == []


# ---------------------------------------------------------------------------
# Test 13 — compressed result is valid bytes object
# ---------------------------------------------------------------------------

def test_compressed_is_bytes():
    vocab = 64
    tok = _random_seq(5, vocab)
    probs = _uniform_probs(5, vocab)

    enc = ArithmeticEncoder()
    bs = enc.encode(tok, probs)

    assert isinstance(bs, bytes), f"Expected bytes, got {type(bs)}"


# ---------------------------------------------------------------------------
# Test 14 — large sequence (200 tokens) round-trips correctly
# ---------------------------------------------------------------------------

def test_roundtrip_200_tokens():
    vocab = 128
    n = 200
    tok = _random_seq(n, vocab, seed=314)
    probs = _uniform_probs(n, vocab)

    enc = ArithmeticEncoder()
    dec = ArithmeticDecoder()
    bs = enc.encode(tok, probs)
    recovered = dec.decode(bs, _make_probs_fn(probs), n)

    assert recovered == tok, (
        f"Round-trip failed: first mismatch at "
        f"index {next(i for i,(a,b) in enumerate(zip(tok,recovered)) if a!=b)}"
        if recovered != tok else ""
    )


# ---------------------------------------------------------------------------
# Test 15 — LMArithmeticCoder compress/decompress round-trip
# ---------------------------------------------------------------------------

def test_lm_coder_compress_decompress():
    vocab = 32
    n = 25
    tok = _random_seq(n, vocab, seed=77)
    probs = _uniform_probs(n, vocab)

    coder = LMArithmeticCoder()
    bs = coder.compress(tok, probs)
    recovered = coder.decompress(bs, _make_probs_fn(probs), n)

    assert recovered == tok


# ---------------------------------------------------------------------------
# Test 16 — bits_per_token returns 0 for empty sequence
# ---------------------------------------------------------------------------

def test_bits_per_token_empty():
    coder = LMArithmeticCoder()
    assert coder.bits_per_token([], []) == 0.0

"""
Tests for hierarchical_softmax.py

Tests cover:
  1-5  : HuffmanTree correctness
  6-11 : HierarchicalSoftmax behaviour & gradients
  12-15: AdaptiveSoftmax behaviour & gradients
  16   : SoftmaxConfig defaults
"""

import math
import torch
import pytest

from src.model.hierarchical_softmax import (
    HuffmanTree,
    HierarchicalSoftmax,
    AdaptiveSoftmax,
    SoftmaxConfig,
)

# ---------------------------------------------------------------------------
# Tiny config used throughout
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB_SIZE = 32
CUTOFFS = [8, 16]
BATCH = 4


def make_freqs(vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Uniform frequencies (ones)."""
    return torch.ones(vocab_size)


def make_varied_freqs(vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Exponentially decreasing frequencies so word 0 is most frequent."""
    return torch.exp(-torch.arange(vocab_size, dtype=torch.float))


# ---------------------------------------------------------------------------
# HuffmanTree tests  (1-5)
# ---------------------------------------------------------------------------

def test_huffman_all_words_reachable():
    """All vocab words must appear in the encoding."""
    tree = HuffmanTree(make_freqs())
    enc = tree.encode_vocab()
    assert len(enc) == VOCAB_SIZE
    for wid in range(VOCAB_SIZE):
        assert wid in enc


def test_huffman_max_depth_positive():
    """A tree with more than 1 word must have depth > 0."""
    tree = HuffmanTree(make_freqs())
    assert tree.max_depth() > 0


def test_huffman_get_path_binary_codes():
    """Every code in every path must be 0 or 1."""
    tree = HuffmanTree(make_freqs())
    for wid in range(VOCAB_SIZE):
        codes, inner = tree.get_path(wid)
        for c in codes:
            assert c in (0, 1), f"Non-binary code {c} for word {wid}"


def test_huffman_frequent_words_shorter_paths():
    """Word 0 (most frequent) must have a path no longer than word -1 (least frequent)."""
    tree = HuffmanTree(make_varied_freqs())
    codes_0, _ = tree.get_path(0)
    codes_last, _ = tree.get_path(VOCAB_SIZE - 1)
    assert len(codes_0) <= len(codes_last)


def test_huffman_encode_vocab_covers_all():
    """encode_vocab returns exactly vocab_size entries."""
    tree = HuffmanTree(make_freqs())
    enc = tree.encode_vocab()
    assert set(enc.keys()) == set(range(VOCAB_SIZE))


# ---------------------------------------------------------------------------
# HierarchicalSoftmax tests  (6-11)
# ---------------------------------------------------------------------------

def make_hsoftmax() -> HierarchicalSoftmax:
    return HierarchicalSoftmax(D_MODEL, VOCAB_SIZE, make_freqs())


def make_input() -> tuple:
    inp = torch.randn(BATCH, D_MODEL)
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH,))
    return inp, tgt


def test_hsoftmax_log_prob_shape():
    """log_prob must return a tensor of shape [B]."""
    model = make_hsoftmax()
    inp, tgt = make_input()
    lp = model.log_prob(inp, tgt)
    assert lp.shape == (BATCH,), f"Expected ({BATCH},), got {lp.shape}"


def test_hsoftmax_log_prob_non_positive():
    """All log-probabilities must be ≤ 0."""
    model = make_hsoftmax()
    inp, tgt = make_input()
    lp = model.log_prob(inp, tgt)
    assert (lp <= 1e-9).all(), f"Some log-probs are positive: {lp}"


def test_hsoftmax_loss_finite_positive():
    """Loss must be a finite positive scalar."""
    model = make_hsoftmax()
    inp, tgt = make_input()
    l = model.loss(inp, tgt)
    assert l.ndim == 0, "Loss must be scalar"
    assert math.isfinite(l.item()), "Loss must be finite"
    assert l.item() > 0, "Loss must be positive"


def test_hsoftmax_sample_shape():
    """sample must return a tensor of shape [B]."""
    model = make_hsoftmax()
    inp, _ = make_input()
    out = model.sample(inp)
    assert out.shape == (BATCH,), f"Expected ({BATCH},), got {out.shape}"


def test_hsoftmax_sample_valid_vocab():
    """Every sampled word must be in [0, vocab_size)."""
    model = make_hsoftmax()
    inp, _ = make_input()
    out = model.sample(inp)
    assert (out >= 0).all() and (out < VOCAB_SIZE).all(), \
        f"Out-of-range sample: {out}"


def test_hsoftmax_loss_backward():
    """Gradients must flow through the loss to inner_weights."""
    model = make_hsoftmax()
    inp, tgt = make_input()
    inp = inp.requires_grad_(True)
    l = model.loss(inp, tgt)
    l.backward()
    assert model.inner_weights.grad is not None, "inner_weights has no grad"
    assert inp.grad is not None, "input has no grad"
    assert not torch.isnan(model.inner_weights.grad).any(), "NaN in inner_weights grad"


# ---------------------------------------------------------------------------
# AdaptiveSoftmax tests  (12-15)
# ---------------------------------------------------------------------------

def make_adaptive() -> AdaptiveSoftmax:
    return AdaptiveSoftmax(D_MODEL, VOCAB_SIZE, CUTOFFS, factor=2.0)


def test_adaptive_log_prob_shape():
    """log_prob must return a tensor of shape [B]."""
    model = make_adaptive()
    inp, tgt = make_input()
    lp = model.log_prob(inp, tgt)
    assert lp.shape == (BATCH,), f"Expected ({BATCH},), got {lp.shape}"


def test_adaptive_log_prob_non_positive():
    """All adaptive log-probabilities must be ≤ 0."""
    model = make_adaptive()
    inp, tgt = make_input()
    lp = model.log_prob(inp, tgt)
    assert (lp <= 1e-9).all(), f"Some log-probs are positive: {lp}"


def test_adaptive_loss_finite_positive():
    """Adaptive loss must be a finite positive scalar."""
    model = make_adaptive()
    inp, tgt = make_input()
    l = model.loss(inp, tgt)
    assert l.ndim == 0, "Loss must be scalar"
    assert math.isfinite(l.item()), "Loss must be finite"
    assert l.item() > 0, "Loss must be positive"


def test_adaptive_loss_backward():
    """Gradients must flow through the adaptive softmax loss."""
    model = make_adaptive()
    inp = torch.randn(BATCH, D_MODEL, requires_grad=True)
    tgt = torch.randint(0, VOCAB_SIZE, (BATCH,))
    l = model.loss(inp, tgt)
    l.backward()
    assert inp.grad is not None, "input has no grad"
    assert not torch.isnan(inp.grad).any(), "NaN in input grad"
    # At least head weights must have gradients
    assert model.head.weight.grad is not None, "head.weight has no grad"


# ---------------------------------------------------------------------------
# SoftmaxConfig  (16)
# ---------------------------------------------------------------------------

def test_softmax_config_defaults():
    """SoftmaxConfig must have the specified default values."""
    cfg = SoftmaxConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.cutoffs == [16, 32]
    assert cfg.factor == 2.0

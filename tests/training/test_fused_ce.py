"""Tests for chunked cross-entropy loss (src/training/fused_ce.py)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.training.fused_ce import chunked_cross_entropy, ChunkedCrossEntropyLoss


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, S, V = 2, 64, 512  # small but realistic shape for fast tests
ATOL = 1e-5


def _make_logits_labels(seed: int = 0, b: int = B, s: int = S, v: int = V):
    torch.manual_seed(seed)
    logits = torch.randn(b, s, v, requires_grad=True)
    labels = torch.randint(0, v, (b, s))
    return logits, labels


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_chunked_ce_matches_standard():
    """Chunked result must match F.cross_entropy within float32 tolerance."""
    logits, labels = _make_logits_labels()
    expected = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    actual = chunked_cross_entropy(logits, labels, chunk_size=16)
    assert torch.allclose(actual, expected, atol=ATOL), (
        f"Expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_chunked_ce_with_ignore_index():
    """-100 labels must be excluded, matching standard ignore_index behaviour."""
    logits, labels = _make_logits_labels(seed=1)
    # mask out roughly half the tokens
    mask = torch.rand(B, S) < 0.5
    labels[mask] = -100

    expected = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
    actual = chunked_cross_entropy(logits, labels, chunk_size=16, ignore_index=-100)
    assert torch.allclose(actual, expected, atol=ATOL), (
        f"Expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_chunked_ce_gradient_flows():
    """loss.backward() must succeed and produce a non-None gradient on logits."""
    logits, labels = _make_logits_labels(seed=2)
    loss = chunked_cross_entropy(logits, labels, chunk_size=32)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_chunked_ce_gradient_matches_standard():
    """Gradients must match standard cross-entropy within tolerance."""
    logits, labels = _make_logits_labels(seed=3)

    # Standard path
    logits_std = logits.detach().clone().requires_grad_(True)
    loss_std = F.cross_entropy(logits_std.view(-1, V), labels.view(-1))
    loss_std.backward()

    # Chunked path
    logits_chk = logits.detach().clone().requires_grad_(True)
    loss_chk = chunked_cross_entropy(logits_chk, labels, chunk_size=16)
    loss_chk.backward()

    assert torch.allclose(logits_chk.grad, logits_std.grad, atol=ATOL), (
        f"Max grad diff: {(logits_chk.grad - logits_std.grad).abs().max().item():.2e}"
    )


def test_chunked_ce_sum_reduction():
    """reduction='sum' must match F.cross_entropy with reduction='sum'."""
    logits, labels = _make_logits_labels(seed=4)
    expected = F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction="sum")
    actual = chunked_cross_entropy(logits, labels, chunk_size=16, reduction="sum")
    assert torch.allclose(actual, expected, atol=ATOL), (
        f"Expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_chunked_ce_chunk_size_1():
    """chunk_size=1 (extreme case) must still produce the correct result."""
    logits, labels = _make_logits_labels(seed=5)
    expected = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    actual = chunked_cross_entropy(logits, labels, chunk_size=1)
    assert torch.allclose(actual, expected, atol=ATOL), (
        f"Expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_chunked_ce_chunk_larger_than_seq():
    """chunk_size larger than total tokens must still work correctly."""
    logits, labels = _make_logits_labels(seed=6)
    N = B * S
    expected = F.cross_entropy(logits.view(-1, V), labels.view(-1))
    actual = chunked_cross_entropy(logits, labels, chunk_size=N * 2)
    assert torch.allclose(actual, expected, atol=ATOL), (
        f"Expected {expected.item():.6f}, got {actual.item():.6f}"
    )


def test_chunked_ce_module_forward():
    """ChunkedCrossEntropyLoss module must produce the same result as the function."""
    logits, labels = _make_logits_labels(seed=7)
    module = ChunkedCrossEntropyLoss(chunk_size=32)
    loss_fn = chunked_cross_entropy(logits, labels, chunk_size=32)
    loss_mod = module(logits, labels)
    assert torch.allclose(loss_fn, loss_mod, atol=1e-9)


def test_memory_usage_chunked_less_than_full():
    """chunked_bytes must be strictly less than full_bytes for typical shapes."""
    module = ChunkedCrossEntropyLoss(chunk_size=128)
    info = module.memory_usage_bytes(batch=4, seq_len=2048, vocab=128_000)
    assert info["chunked_bytes"] < info["full_bytes"], (
        f"chunked ({info['chunked_bytes']}) should be < full ({info['full_bytes']})"
    )
    assert info["ratio"] > 1, "Expected chunked to save memory"


def test_chunked_ce_all_ignored():
    """When all labels are -100, loss should be zero (no NaN/inf)."""
    logits, _ = _make_logits_labels(seed=8)
    labels = torch.full((B, S), -100, dtype=torch.long)
    loss = chunked_cross_entropy(logits, labels, chunk_size=16)
    assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"
    assert loss.item() == 0.0, f"Expected 0.0 loss, got {loss.item()}"

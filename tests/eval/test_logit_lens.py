"""Tests for src/eval/logit_lens.py (Logit Lens — nostalgebraist 2020).

Import path: from aurelius.eval.logit_lens import ...
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn as nn

from aurelius.eval.logit_lens import (
    LogitLens,
    LogitLensAnalyzer,
    LogitLensConfig,
    LogitLensTracker,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

B, T, D, V = 2, 6, 32, 64  # batch, seq-len, d_model, vocab_size


def make_unembed(vocab_size: int = V, d_model: int = D) -> nn.Linear:
    """Create a deterministic unembedding linear layer (no bias)."""
    torch.manual_seed(0)
    lin = nn.Linear(d_model, vocab_size, bias=False)
    return lin


def make_layer_norm(d_model: int = D) -> nn.LayerNorm:
    return nn.LayerNorm(d_model)


def make_hidden(batch: int = B, seq: int = T, d: int = D) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, seq, d)


def make_lens(with_ln: bool = True) -> LogitLens:
    unembed = make_unembed()
    ln = make_layer_norm() if with_ln else None
    return LogitLens(unembed, layer_norm=ln)


# ---------------------------------------------------------------------------
# 1. LogitLensConfig – fields and defaults
# ---------------------------------------------------------------------------


def test_logit_lens_config_fields():
    cfg = LogitLensConfig(n_layers=12, d_model=768, vocab_size=50257)
    assert cfg.n_layers == 12
    assert cfg.d_model == 768
    assert cfg.vocab_size == 50257
    assert cfg.use_layer_norm is True  # default


def test_logit_lens_config_use_layer_norm_false():
    cfg = LogitLensConfig(n_layers=4, d_model=64, vocab_size=128, use_layer_norm=False)
    assert cfg.use_layer_norm is False


# ---------------------------------------------------------------------------
# 2. LogitLens.project — shape
# ---------------------------------------------------------------------------


def test_logit_lens_project_shape():
    lens = make_lens(with_ln=True)
    hidden = make_hidden()
    logits = lens.project(hidden)
    assert logits.shape == (B, T, V), f"Expected ({B}, {T}, {V}), got {logits.shape}"


# ---------------------------------------------------------------------------
# 3. LogitLens.project — equivalent to unembed(ln(h)) when layer_norm given
# ---------------------------------------------------------------------------


def test_logit_lens_project_with_layer_norm():
    """project(h) == linear(ln(h)) when a layer_norm is set."""
    unembed = make_unembed()
    ln = make_layer_norm()
    lens = LogitLens(unembed, layer_norm=ln)

    hidden = make_hidden()
    logits = lens.project(hidden)

    # Manual computation
    normed = ln(hidden)
    expected = nn.functional.linear(normed, unembed.weight, unembed.bias)

    assert torch.allclose(logits, expected, atol=1e-5), (
        "project() result does not match manual ln → unembed computation"
    )


def test_logit_lens_project_without_layer_norm():
    """project(h) == unembed(h) when no layer_norm is provided."""
    unembed = make_unembed()
    lens = LogitLens(unembed, layer_norm=None)

    hidden = make_hidden()
    logits = lens.project(hidden)
    expected = nn.functional.linear(hidden, unembed.weight, unembed.bias)

    assert torch.allclose(logits, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. LogitLens.top_k_tokens — shape
# ---------------------------------------------------------------------------


def test_top_k_tokens_shape():
    k = 5
    lens = make_lens()
    hidden = make_hidden()
    top_k = lens.top_k_tokens(hidden, k=k)
    assert top_k.shape == (B, T, k), f"Expected ({B}, {T}, {k}), got {top_k.shape}"


# ---------------------------------------------------------------------------
# 5. LogitLens.top_k_tokens — indices in valid vocab range
# ---------------------------------------------------------------------------


def test_top_k_tokens_vocab_range():
    k = 5
    lens = make_lens()
    hidden = make_hidden()
    top_k = lens.top_k_tokens(hidden, k=k)
    assert (top_k >= 0).all(), "Negative token index found"
    assert (top_k < V).all(), f"Token index >= vocab_size ({V}) found"


# ---------------------------------------------------------------------------
# 6. LogitLens.entropy — shape and non-negative
# ---------------------------------------------------------------------------


def test_entropy_shape():
    lens = make_lens()
    hidden = make_hidden()
    ent = lens.entropy(hidden)
    assert ent.shape == (B, T), f"Expected ({B}, {T}), got {ent.shape}"


def test_entropy_non_negative():
    lens = make_lens()
    hidden = make_hidden()
    ent = lens.entropy(hidden)
    assert (ent >= -1e-6).all(), "Entropy contains negative values"


# ---------------------------------------------------------------------------
# 7. LogitLens.entropy — zero for a perfectly peaked distribution
# ---------------------------------------------------------------------------


def test_entropy_zero_for_peaked_distribution():
    """If the projected logits are a one-hot (very large at one position),
    the softmax entropy should be near zero."""
    unembed = nn.Linear(D, V, bias=False)
    # Zero out the weight, then force a single huge logit via construction:
    # We'll craft a hidden state such that W @ h^T = e_k * 1e9.
    with torch.no_grad():
        unembed.weight.zero_()
        unembed.weight[0, 0] = 1.0  # only token 0 gets a non-zero logit

    lens = LogitLens(unembed, layer_norm=None)

    # hidden with only the first feature active → logit for token 0 is huge
    hidden = torch.zeros(1, 1, D)
    hidden[0, 0, 0] = 1e6  # makes logit[0] = 1e6, all others = 0

    ent = lens.entropy(hidden)
    assert ent.item() < 1e-3, f"Expected near-zero entropy, got {ent.item()}"


# ---------------------------------------------------------------------------
# 8. LogitLensAnalyzer.analyze — correct output keys
# ---------------------------------------------------------------------------


def test_analyzer_analyze_keys():
    n_layers = 3
    lenses = [make_lens() for _ in range(n_layers)]
    analyzer = LogitLensAnalyzer(lenses)

    hiddens = [make_hidden() for _ in range(n_layers)]
    result = analyzer.analyze(hiddens)

    assert "logits" in result
    assert "top1" in result
    assert "entropy" in result


# ---------------------------------------------------------------------------
# 9. LogitLensAnalyzer.analyze — correct number of layer entries
# ---------------------------------------------------------------------------


def test_analyzer_analyze_num_layers():
    n_layers = 4
    lenses = [make_lens() for _ in range(n_layers)]
    analyzer = LogitLensAnalyzer(lenses)

    hiddens = [make_hidden() for _ in range(n_layers)]
    result = analyzer.analyze(hiddens)

    assert len(result["logits"]) == n_layers
    assert len(result["top1"]) == n_layers
    assert len(result["entropy"]) == n_layers


# ---------------------------------------------------------------------------
# 10. LogitLensAnalyzer.rank_of_true_token — correct return shapes
# ---------------------------------------------------------------------------


def test_rank_of_true_token_shapes():
    n_layers = 3
    lenses = [make_lens() for _ in range(n_layers)]
    analyzer = LogitLensAnalyzer(lenses)

    hiddens = [make_hidden() for _ in range(n_layers)]
    true_tokens = torch.randint(0, V, (B, T))

    ranks = analyzer.rank_of_true_token(hiddens, true_tokens)

    assert len(ranks) == n_layers
    for r in ranks:
        assert r.shape == (B, T), f"Expected ({B}, {T}), got {r.shape}"


# ---------------------------------------------------------------------------
# 11. rank_of_true_token — rank 0 when true token is top-predicted
# ---------------------------------------------------------------------------


def test_rank_zero_when_true_token_is_top():
    """Construct a situation where the true token is always rank 0."""
    # Use bias-only linear: project gives a fixed vector regardless of input.
    # Set bias so token 7 always has the highest logit.
    unembed = nn.Linear(D, V, bias=True)
    with torch.no_grad():
        unembed.weight.zero_()
        unembed.bias.zero_()
        unembed.bias[7] = 1e6  # token 7 always top

    lens = LogitLens(unembed, layer_norm=None)
    analyzer = LogitLensAnalyzer([lens])

    hidden = make_hidden(batch=1, seq=3)
    true_tokens = torch.full((1, 3), fill_value=7, dtype=torch.long)

    ranks = analyzer.rank_of_true_token([hidden], true_tokens)
    assert len(ranks) == 1
    assert (ranks[0] == 0).all(), f"Expected all-zero ranks, got {ranks[0]}"


# ---------------------------------------------------------------------------
# 12. LogitLensTracker — context manager installs and removes hooks
# ---------------------------------------------------------------------------


class _SimpleLayer(nn.Module):
    """Trivial layer that passes input through unchanged."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_tracker_installs_and_removes_hooks():
    layers = [_SimpleLayer(), _SimpleLayer()]
    tracker = LogitLensTracker(layers)

    with tracker:
        # Inside: hooks should be registered
        for layer in layers:
            assert len(layer._forward_hooks) > 0, "Hook not installed"

    # Outside: hooks should be removed
    for layer in layers:
        assert len(layer._forward_hooks) == 0, "Hook not removed after __exit__"


# ---------------------------------------------------------------------------
# 13. LogitLensTracker — collects hidden states after forward pass
# ---------------------------------------------------------------------------


def test_tracker_collects_hiddens():
    layers = [_SimpleLayer(), _SimpleLayer()]
    tracker = LogitLensTracker(layers)
    x = torch.randn(1, 4, D)

    with tracker:
        out = x
        for layer in layers:
            out = layer(out)

    hiddens = tracker.get_hiddens()
    assert len(hiddens) == len(layers), (
        f"Expected {len(layers)} hiddens, got {len(hiddens)}"
    )
    for h in hiddens:
        assert isinstance(h, torch.Tensor)
        assert h.shape == x.shape


# ---------------------------------------------------------------------------
# 14. Works with B=1, T=1 (minimal shape)
# ---------------------------------------------------------------------------


def test_minimal_shape_b1_t1():
    lens = make_lens(with_ln=True)

    hidden = torch.randn(1, 1, D)
    logits = lens.project(hidden)
    assert logits.shape == (1, 1, V)

    top_k = lens.top_k_tokens(hidden, k=3)
    assert top_k.shape == (1, 1, 3)

    ent = lens.entropy(hidden)
    assert ent.shape == (1, 1)
    assert ent.item() >= 0.0

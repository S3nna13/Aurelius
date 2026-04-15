"""
Tests for src/interpretability/logit_lens.py

Tiny configuration:
    VOCAB=32, D=16, N_LAYERS=3, B=2, T=4
"""

import pytest
import torch

from src.interpretability.logit_lens import (
    LogitLensConfig,
    LogitLens,
    LayerwiseEntropyTracker,
    compute_kl_divergence,
    get_top_tokens,
    project_hidden_state,
)

# ---------------------------------------------------------------------------
# Tiny constants used throughout all tests
# ---------------------------------------------------------------------------
VOCAB = 32
D = 16
N_LAYERS = 3
B = 2
T = 4
K = 5  # top-k for get_top_tokens tests

torch.manual_seed(42)


def _make_config(apply_ln: bool = True) -> LogitLensConfig:
    return LogitLensConfig(vocab_size=VOCAB, d_model=D, n_layers=N_LAYERS, apply_ln=apply_ln)


def _make_unembed() -> torch.Tensor:
    return torch.randn(VOCAB, D)


def _make_ln_params() -> tuple:
    """Return (ln_weight, ln_bias) matching D."""
    weight = torch.ones(D)
    bias = torch.zeros(D)
    return weight, bias


def _make_hidden() -> torch.Tensor:
    return torch.randn(B, T, D)


def _make_hiddens() -> list:
    return [torch.randn(B, T, D) for _ in range(N_LAYERS)]


def _make_lens(apply_ln: bool = True) -> LogitLens:
    config = _make_config(apply_ln=apply_ln)
    unembed = _make_unembed()
    ln_weight, ln_bias = _make_ln_params()
    return LogitLens(config, unembed, ln_weight, ln_bias)


# ---------------------------------------------------------------------------
# 1. LogitLensConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = LogitLensConfig()
    assert cfg.vocab_size == 50257
    assert cfg.d_model == 512
    assert cfg.n_layers == 12
    assert cfg.apply_ln is True


# ---------------------------------------------------------------------------
# 2. project_hidden_state output shape
# ---------------------------------------------------------------------------

def test_project_hidden_state_shape():
    hidden = _make_hidden()
    unembed = _make_unembed()
    logits = project_hidden_state(hidden, unembed)
    assert logits.shape == (B, T, VOCAB), f"Expected ({B},{T},{VOCAB}), got {logits.shape}"


# ---------------------------------------------------------------------------
# 3. project_hidden_state shape with LayerNorm
# ---------------------------------------------------------------------------

def test_project_hidden_state_with_ln_shape():
    hidden = _make_hidden()
    unembed = _make_unembed()
    ln_weight, ln_bias = _make_ln_params()
    logits = project_hidden_state(hidden, unembed, ln_weight, ln_bias)
    assert logits.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# 4. get_top_tokens output shape
# ---------------------------------------------------------------------------

def test_get_top_tokens_shape():
    logits = torch.randn(B, T, VOCAB)
    indices = get_top_tokens(logits, k=K)
    assert indices.shape == (B, T, K), f"Expected ({B},{T},{K}), got {indices.shape}"


# ---------------------------------------------------------------------------
# 5. get_top_tokens returns valid token indices
# ---------------------------------------------------------------------------

def test_get_top_tokens_valid_range():
    logits = torch.randn(B, T, VOCAB)
    indices = get_top_tokens(logits, k=K)
    assert indices.min().item() >= 0
    assert indices.max().item() < VOCAB


# ---------------------------------------------------------------------------
# 6. KL divergence is non-negative
# ---------------------------------------------------------------------------

def test_kl_divergence_non_negative():
    p = torch.randn(B, T, VOCAB)
    q = torch.randn(B, T, VOCAB)
    kl = compute_kl_divergence(p, q)
    assert kl.shape == (B, T)
    assert (kl >= -1e-6).all(), "KL divergence should be non-negative"


# ---------------------------------------------------------------------------
# 7. KL divergence of identical logits is ~0
# ---------------------------------------------------------------------------

def test_kl_divergence_identical_is_zero():
    p = torch.randn(B, T, VOCAB)
    kl = compute_kl_divergence(p, p)
    assert kl.abs().max().item() < 1e-5, "KL(P || P) should be ~0"


# ---------------------------------------------------------------------------
# 8. LogitLens.analyze_layer shape
# ---------------------------------------------------------------------------

def test_analyze_layer_shape():
    lens = _make_lens()
    hidden = _make_hidden()
    logits = lens.analyze_layer(hidden)
    assert logits.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# 9. LogitLens.analyze_all_layers shape
# ---------------------------------------------------------------------------

def test_analyze_all_layers_shape():
    lens = _make_lens()
    hiddens = _make_hiddens()
    all_logits = lens.analyze_all_layers(hiddens)
    assert all_logits.shape == (N_LAYERS, B, T, VOCAB), (
        f"Expected ({N_LAYERS},{B},{T},{VOCAB}), got {all_logits.shape}"
    )


# ---------------------------------------------------------------------------
# 10. LogitLens.get_prediction_depth shape
# ---------------------------------------------------------------------------

def test_get_prediction_depth_shape():
    lens = _make_lens()
    hiddens = _make_hiddens()
    target_ids = torch.randint(0, VOCAB, (B, T))
    depth = lens.get_prediction_depth(hiddens, target_ids, k=1)
    assert depth.shape == (B, T), f"Expected ({B},{T}), got {depth.shape}"


# ---------------------------------------------------------------------------
# 11. get_prediction_depth values are in [-1, n_layers-1]
# ---------------------------------------------------------------------------

def test_get_prediction_depth_values_in_range():
    lens = _make_lens()
    hiddens = _make_hiddens()
    target_ids = torch.randint(0, VOCAB, (B, T))
    depth = lens.get_prediction_depth(hiddens, target_ids, k=1)
    assert depth.min().item() >= -1
    assert depth.max().item() <= N_LAYERS - 1


# ---------------------------------------------------------------------------
# 12. get_prediction_depth: forced match at layer 0
# ---------------------------------------------------------------------------

def test_get_prediction_depth_forced_match():
    """When the top-1 token at layer 0 is always the target, depth should be 0."""
    unembed = _make_unembed()
    ln_weight, ln_bias = _make_ln_params()
    config = _make_config()
    lens = LogitLens(config, unembed, ln_weight, ln_bias)

    # Build hiddens such that layer-0 logits have a dominant token per position
    hidden_0 = torch.zeros(B, T, D)
    # Force unembed[target_id] to be the highest-logit direction
    target_ids = torch.zeros(B, T, dtype=torch.long)
    # Set each hidden vector to the unembed row for token 0
    for b in range(B):
        for t in range(T):
            hidden_0[b, t] = unembed[0] * 100.0  # very large scale

    hiddens = [hidden_0] + [torch.randn(B, T, D) for _ in range(N_LAYERS - 1)]
    depth = lens.get_prediction_depth(hiddens, target_ids, k=1)
    assert (depth == 0).all(), "All positions should be matched at layer 0"


# ---------------------------------------------------------------------------
# 13. LayerwiseEntropyTracker.compute_entropy shape
# ---------------------------------------------------------------------------

def test_compute_entropy_shape():
    tracker = LayerwiseEntropyTracker()
    logits = torch.randn(B, T, VOCAB)
    entropy = tracker.compute_entropy(logits)
    assert entropy.shape == (B, T), f"Expected ({B},{T}), got {entropy.shape}"


# ---------------------------------------------------------------------------
# 14. Entropy is non-negative
# ---------------------------------------------------------------------------

def test_compute_entropy_non_negative():
    tracker = LayerwiseEntropyTracker()
    logits = torch.randn(B, T, VOCAB)
    entropy = tracker.compute_entropy(logits)
    assert (entropy >= -1e-6).all(), "Entropy must be non-negative"


# ---------------------------------------------------------------------------
# 15. LayerwiseEntropyTracker.track shape
# ---------------------------------------------------------------------------

def test_entropy_track_shape():
    lens = _make_lens()
    hiddens = _make_hiddens()
    tracker = LayerwiseEntropyTracker()
    result = tracker.track(lens, hiddens)
    assert result.shape == (N_LAYERS, B, T), (
        f"Expected ({N_LAYERS},{B},{T}), got {result.shape}"
    )


# ---------------------------------------------------------------------------
# 16. Entropy is bounded by log(vocab_size)
# ---------------------------------------------------------------------------

def test_compute_entropy_upper_bound():
    """Entropy H <= log(V) for any distribution over V tokens."""
    tracker = LayerwiseEntropyTracker()
    logits = torch.zeros(B, T, VOCAB)  # uniform distribution
    entropy = tracker.compute_entropy(logits)
    max_entropy = torch.tensor(VOCAB, dtype=torch.float).log()
    assert (entropy <= max_entropy + 1e-5).all()


# ---------------------------------------------------------------------------
# 17. LogitLens without LayerNorm (apply_ln=False)
# ---------------------------------------------------------------------------

def test_analyze_layer_no_ln_shape():
    lens = _make_lens(apply_ln=False)
    hidden = _make_hidden()
    logits = lens.analyze_layer(hidden)
    assert logits.shape == (B, T, VOCAB)


# ---------------------------------------------------------------------------
# 18. project_hidden_state: no ln_weight/ln_bias path is consistent
# ---------------------------------------------------------------------------

def test_project_hidden_state_no_ln_consistent():
    """Without LN, projecting scaled hidden should scale logits proportionally."""
    hidden = torch.randn(1, 1, D)
    unembed = torch.randn(VOCAB, D)
    logits_1 = project_hidden_state(hidden, unembed)
    logits_2 = project_hidden_state(hidden * 2, unembed)
    # logits_2 should be exactly 2x logits_1
    assert torch.allclose(logits_2, logits_1 * 2, atol=1e-5)


# ---------------------------------------------------------------------------
# 19. get_prediction_depth: no match returns -1
# ---------------------------------------------------------------------------

def test_get_prediction_depth_no_match_returns_minus_one():
    """When target token is set to an impossible id that never appears in top-k=1,
    we force the scenario by using k=1 and a target_id that equals the argmin."""
    torch.manual_seed(99)
    lens = _make_lens()
    hiddens = _make_hiddens()

    # Get the actual top-1 predictions for all layers
    all_logits = lens.analyze_all_layers(hiddens)  # (L, B, T, V)
    top1 = all_logits.argmax(dim=-1)  # (L, B, T)

    # For each (b, t), pick a token id that is never the argmax across any layer
    target_ids = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            used = set(top1[:, b, t].tolist())
            # Find a token not in used
            for candidate in range(VOCAB):
                if candidate not in used:
                    target_ids[b, t] = candidate
                    break

    depth = lens.get_prediction_depth(hiddens, target_ids, k=1)
    assert (depth == -1).all(), "All should be -1 when target never appears in top-k"


# ---------------------------------------------------------------------------
# 20. analyze_all_layers consistency: each layer matches analyze_layer
# ---------------------------------------------------------------------------

def test_analyze_all_layers_consistent_with_analyze_layer():
    lens = _make_lens()
    hiddens = _make_hiddens()
    all_logits = lens.analyze_all_layers(hiddens)  # (L, B, T, V)
    for l, h in enumerate(hiddens):
        layer_logits = lens.analyze_layer(h)
        assert torch.allclose(all_logits[l], layer_logits, atol=1e-6), (
            f"Layer {l} mismatch between analyze_all_layers and analyze_layer"
        )

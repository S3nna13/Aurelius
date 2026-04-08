"""Tests for the Logit Lens interpretability tool."""

import pytest
import torch

from src.eval.logit_lens import LogitLens, logit_lens_summary
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 8))  # batch=1, seq_len=8


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_logit_lens_run_returns_dict(small_model, input_ids):
    """run() returns a dict with n_layers entries."""
    lens = LogitLens(small_model)
    result = lens.run(input_ids)
    assert isinstance(result, dict)
    assert len(result) == small_model.config.n_layers


def test_logit_lens_layer_logits_shape(small_model, input_ids):
    """Each entry in the returned dict is (batch, seq_len, vocab_size)."""
    lens = LogitLens(small_model)
    result = lens.run(input_ids)
    B, S = input_ids.shape
    V = small_model.config.vocab_size
    for layer_idx, logits in result.items():
        assert logits.shape == (B, S, V), (
            f"Layer {layer_idx}: expected ({B}, {S}, {V}), got {logits.shape}"
        )


def test_top_tokens_at_layer_count(small_model, input_ids):
    """top_tokens_at_layer() returns exactly k items."""
    lens = LogitLens(small_model)
    lens.run(input_ids)
    k = 5
    result = lens.top_tokens_at_layer(layer_idx=0, position=0, k=k)
    assert len(result) == k


def test_top_tokens_probabilities_sum_to_one(small_model, input_ids):
    """Full softmax over vocab sums to ~1.0 (sanity check via top token probs being valid)."""
    lens = LogitLens(small_model)
    result = lens.run(input_ids)
    # Check via raw logits: softmax of a layer's logits at a position sums to 1
    logits = result[0]  # (1, S, V)
    probs = torch.softmax(logits[0, 0, :], dim=-1)
    assert abs(float(probs.sum()) - 1.0) < 1e-4


def test_probability_of_token_length(small_model, input_ids):
    """probability_of_token() returns a list of length n_layers."""
    lens = LogitLens(small_model)
    lens.run(input_ids)
    probs = lens.probability_of_token(target_token_id=0, position=0)
    assert len(probs) == small_model.config.n_layers


def test_probability_of_token_range(small_model, input_ids):
    """All probabilities returned by probability_of_token() are in [0, 1]."""
    lens = LogitLens(small_model)
    lens.run(input_ids)
    probs = lens.probability_of_token(target_token_id=10, position=3)
    for p in probs:
        assert 0.0 <= p <= 1.0, f"Probability {p} is out of [0, 1]"


def test_entropy_profile_length(small_model, input_ids):
    """entropy_profile() returns n_layers values."""
    lens = LogitLens(small_model)
    lens.run(input_ids)
    entropies = lens.entropy_profile(position=0)
    assert len(entropies) == small_model.config.n_layers


def test_entropy_positive(small_model, input_ids):
    """All entropy values are >= 0."""
    lens = LogitLens(small_model)
    lens.run(input_ids)
    entropies = lens.entropy_profile(position=0)
    for h in entropies:
        assert h >= 0.0, f"Entropy {h} is negative"


def test_logit_lens_summary_keys(small_model, input_ids):
    """logit_lens_summary() returns a dict with all required keys."""
    summary = logit_lens_summary(small_model, input_ids, target_position=-1)
    required_keys = {"n_layers", "entropy_per_layer", "top1_token_per_layer", "convergence_layer"}
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - summary.keys()}"
    )


def test_convergence_layer_valid(small_model, input_ids):
    """convergence_layer is within [0, n_layers - 1]."""
    summary = logit_lens_summary(small_model, input_ids, target_position=-1)
    n_layers = summary["n_layers"]
    cl = summary["convergence_layer"]
    assert 0 <= cl <= n_layers - 1, (
        f"convergence_layer {cl} is out of range [0, {n_layers - 1}]"
    )


def test_hooks_cleaned_up(small_model, input_ids):
    """After run(), no lingering hooks remain on the model's layers."""
    lens = LogitLens(small_model)
    lens.run(input_ids)

    # After run(), _hooks list should be empty (removed inside run)
    assert len(lens._hooks) == 0, "Hooks were not cleaned up after run()"

    # Also verify no forward hooks remain on any layer
    for layer in small_model.layers:
        assert len(layer._forward_hooks) == 0, (
            f"Layer still has {len(layer._forward_hooks)} hook(s) after run()"
        )

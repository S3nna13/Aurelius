"""Tests for the logit lens module (Nostalgebraist 2020)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.eval.logit_lens import (
    HiddenStateCollector,
    LogitLens,
    LogitLensConfig,
    compute_answer_rank,
    compute_layer_entropy,
    get_top_tokens,
    plot_logit_lens_text,
    project_hidden_to_logits,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
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


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (1, 8))


@pytest.fixture
def default_config():
    return LogitLensConfig()


# ---------------------------------------------------------------------------
# 1. test_logit_lens_config_defaults
# ---------------------------------------------------------------------------

def test_logit_lens_config_defaults():
    cfg = LogitLensConfig()
    assert cfg.n_top_tokens == 10
    assert cfg.normalize_hidden is True
    assert cfg.track_entropy is True
    assert cfg.track_rank is True


# ---------------------------------------------------------------------------
# 2. test_hidden_state_collector_captures
# ---------------------------------------------------------------------------

def test_hidden_state_collector_captures(small_model, input_ids):
    collector = HiddenStateCollector(small_model)
    layers = list(small_model.layers)
    collector.register(layers)
    try:
        with torch.no_grad():
            small_model(input_ids)
    finally:
        collector.remove_hooks()

    assert len(collector.hiddens) == len(layers), (
        f"Expected {len(layers)} hiddens, got {len(collector.hiddens)}"
    )
    # Each hidden should be a tensor of shape (1, 8, 64)
    for h in collector.hiddens:
        assert isinstance(h, torch.Tensor)
        assert h.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# 3. test_hidden_state_collector_context_manager
# ---------------------------------------------------------------------------

def test_hidden_state_collector_context_manager(small_model, input_ids):
    collector = HiddenStateCollector(small_model)
    layers = list(small_model.layers)

    with collector:
        collector.register(layers)
        # Hooks are registered inside
        assert len(collector._hooks) == len(layers)

    # After exiting context manager, hooks must be removed
    assert len(collector._hooks) == 0, "Hooks not removed after context manager exit"
    for layer in layers:
        assert len(layer._forward_hooks) == 0, "Layer still has forward hooks"


# ---------------------------------------------------------------------------
# 4. test_project_hidden_to_logits_shape
# ---------------------------------------------------------------------------

def test_project_hidden_to_logits_shape():
    B, T, D, V = 2, 5, 64, 256
    hidden = torch.randn(B, T, D)
    unembed = torch.randn(V, D)
    logits = project_hidden_to_logits(hidden, unembed, layer_norm=None)
    assert logits.shape == (B, T, V), f"Expected ({B}, {T}, {V}), got {logits.shape}"


# ---------------------------------------------------------------------------
# 5. test_project_hidden_with_layernorm
# ---------------------------------------------------------------------------

def test_project_hidden_with_layernorm():
    B, T, D, V = 1, 4, 64, 256
    hidden = torch.randn(B, T, D)
    unembed = torch.randn(V, D)
    layer_norm = nn.LayerNorm(D)
    logits = project_hidden_to_logits(hidden, unembed, layer_norm=layer_norm)
    assert logits.shape == (B, T, V)

    # Verify that layer norm was applied: compute expected result manually
    normed = layer_norm(hidden)
    expected = normed @ unembed.T
    assert torch.allclose(logits, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. test_get_top_tokens_shapes
# ---------------------------------------------------------------------------

def test_get_top_tokens_shapes():
    B, T, V = 3, 6, 256
    n_top = 10
    logits = torch.randn(B, T, V)
    top_ids, top_logits = get_top_tokens(logits, n_top=n_top)
    assert top_ids.shape == (B, n_top), f"top_ids shape: {top_ids.shape}"
    assert top_logits.shape == (B, n_top), f"top_logits shape: {top_logits.shape}"


# ---------------------------------------------------------------------------
# 7. test_compute_layer_entropy_shape
# ---------------------------------------------------------------------------

def test_compute_layer_entropy_shape():
    B, T, V = 2, 8, 256
    logits = torch.randn(B, T, V)
    entropy = compute_layer_entropy(logits)
    assert entropy.shape == (B, T), f"Expected ({B}, {T}), got {entropy.shape}"


# ---------------------------------------------------------------------------
# 8. test_compute_layer_entropy_positive
# ---------------------------------------------------------------------------

def test_compute_layer_entropy_positive():
    B, T, V = 2, 8, 256
    logits = torch.randn(B, T, V)
    entropy = compute_layer_entropy(logits)
    assert (entropy >= 0).all(), "Some entropy values are negative"


# ---------------------------------------------------------------------------
# 9. test_compute_answer_rank_shape
# ---------------------------------------------------------------------------

def test_compute_answer_rank_shape():
    B, T, V = 3, 6, 256
    logits = torch.randn(B, T, V)
    answer_token_id = 42
    ranks = compute_answer_rank(logits, answer_token_id)
    assert ranks.shape == (B,), f"Expected ({B},), got {ranks.shape}"


# ---------------------------------------------------------------------------
# 10. test_logit_lens_analyze_keys
# ---------------------------------------------------------------------------

def test_logit_lens_analyze_keys(small_model, input_ids, default_config):
    lens = LogitLens(small_model, default_config)
    results = lens.analyze(input_ids, answer_token_id=5)

    required_keys = {"n_layers", "layer_entropies", "layer_top_tokens", "answer_ranks"}
    assert required_keys.issubset(results.keys()), (
        f"Missing keys: {required_keys - results.keys()}"
    )
    assert results["n_layers"] == 2
    assert len(results["layer_entropies"]) == 2
    assert len(results["layer_top_tokens"]) == 2
    assert results["answer_ranks"] is not None
    assert len(results["answer_ranks"]) == 2


# ---------------------------------------------------------------------------
# 11. test_logit_lens_layer_prediction_agreement_range
# ---------------------------------------------------------------------------

def test_logit_lens_layer_prediction_agreement_range(small_model, input_ids, default_config):
    lens = LogitLens(small_model, default_config)
    results = lens.analyze(input_ids)
    agreement = lens.layer_prediction_agreement(results)
    assert 0.0 <= agreement <= 1.0, f"Agreement {agreement} out of [0, 1]"


# ---------------------------------------------------------------------------
# 12. test_plot_logit_lens_text_is_string
# ---------------------------------------------------------------------------

def test_plot_logit_lens_text_is_string(small_model, input_ids, default_config):
    lens = LogitLens(small_model, default_config)
    results = lens.analyze(input_ids)

    def decode(token_id: int) -> str:
        return f"<{token_id}>"

    text = plot_logit_lens_text(results, tokenizer_decode=decode)
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    assert len(text) > 0, "plot_logit_lens_text returned empty string"

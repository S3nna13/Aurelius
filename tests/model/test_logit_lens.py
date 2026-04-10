"""Tests for logit_lens.py — Logit Lens Analysis."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.logit_lens import (
    LogitLens,
    LogitLensConfig,
    compute_layer_entropy,
    extract_layer_hidden_states,
    get_top_tokens,
    project_to_vocab,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
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
def cfg():
    return TINY_CFG


@pytest.fixture
def model(cfg):
    m = AureliusTransformer(cfg)
    m.eval()
    return m


@pytest.fixture
def input_ids():
    return torch.randint(0, 256, (2, 8))


# ---------------------------------------------------------------------------
# LogitLensConfig tests
# ---------------------------------------------------------------------------

class TestLogitLensConfig:
    def test_defaults(self):
        c = LogitLensConfig()
        assert c.layers is None
        assert c.normalize is True
        assert c.top_k == 5

    def test_custom_values(self):
        c = LogitLensConfig(layers=[0, 1], normalize=False, top_k=10)
        assert c.layers == [0, 1]
        assert c.normalize is False
        assert c.top_k == 10


# ---------------------------------------------------------------------------
# extract_layer_hidden_states tests
# ---------------------------------------------------------------------------

class TestExtractLayerHiddenStates:
    def test_returns_list_per_layer(self, model, input_ids):
        states = extract_layer_hidden_states(model, input_ids)
        assert len(states) == model.config.n_layers

    def test_hidden_state_shapes(self, model, input_ids):
        states = extract_layer_hidden_states(model, input_ids)
        B, T = input_ids.shape
        for s in states:
            assert s.shape == (B, T, model.config.d_model)

    def test_hidden_states_are_detached(self, model, input_ids):
        states = extract_layer_hidden_states(model, input_ids)
        for s in states:
            assert not s.requires_grad

    def test_hooks_are_removed(self, model, input_ids):
        hooks_before = len(model.layers[0]._forward_hooks)
        extract_layer_hidden_states(model, input_ids)
        hooks_after = len(model.layers[0]._forward_hooks)
        assert hooks_before == hooks_after


# ---------------------------------------------------------------------------
# project_to_vocab tests
# ---------------------------------------------------------------------------

class TestProjectToVocab:
    def test_output_shape(self, model):
        hidden = torch.randn(2, 8, model.config.d_model)
        logits = project_to_vocab(hidden, model)
        assert logits.shape == (2, 8, model.config.vocab_size)

    def test_output_dtype(self, model):
        hidden = torch.randn(1, 4, model.config.d_model)
        logits = project_to_vocab(hidden, model)
        assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# get_top_tokens tests
# ---------------------------------------------------------------------------

class TestGetTopTokens:
    def test_shapes(self):
        logits = torch.randn(2, 8, 256)
        indices, probs = get_top_tokens(logits, k=5)
        assert indices.shape == (2, 8, 5)
        assert probs.shape == (2, 8, 5)

    def test_probs_sum_leq_one(self):
        logits = torch.randn(2, 8, 256)
        _, probs = get_top_tokens(logits, k=5)
        assert (probs.sum(dim=-1) <= 1.0 + 1e-5).all()

    def test_probs_non_negative(self):
        logits = torch.randn(2, 8, 256)
        _, probs = get_top_tokens(logits, k=3)
        assert (probs >= 0).all()

    def test_indices_in_range(self):
        V = 256
        logits = torch.randn(2, 8, V)
        indices, _ = get_top_tokens(logits, k=5)
        assert (indices >= 0).all()
        assert (indices < V).all()

    def test_top1_matches_argmax(self):
        logits = torch.randn(2, 8, 256)
        indices, _ = get_top_tokens(logits, k=1)
        expected = logits.argmax(dim=-1, keepdim=True)
        assert torch.equal(indices, expected)


# ---------------------------------------------------------------------------
# compute_layer_entropy tests
# ---------------------------------------------------------------------------

class TestComputeLayerEntropy:
    def test_output_shape(self):
        logits = torch.randn(2, 8, 256)
        ent = compute_layer_entropy(logits)
        assert ent.shape == (2, 8)

    def test_non_negative(self):
        logits = torch.randn(3, 4, 256)
        ent = compute_layer_entropy(logits)
        assert (ent >= -1e-6).all()

    def test_uniform_has_max_entropy(self):
        """Uniform distribution should have near-maximum entropy = log(V)."""
        import math
        V = 64
        logits = torch.zeros(1, 1, V)
        ent = compute_layer_entropy(logits)
        expected = math.log(V)
        assert abs(ent.item() - expected) < 1e-4

    def test_peaked_has_low_entropy(self):
        """A very peaked distribution should have entropy near 0."""
        logits = torch.full((1, 1, 256), -1e6)
        logits[0, 0, 42] = 100.0
        ent = compute_layer_entropy(logits)
        assert ent.item() < 0.01


# ---------------------------------------------------------------------------
# LogitLens class tests
# ---------------------------------------------------------------------------

class TestLogitLens:
    def test_analyze_keys(self, model, input_ids):
        lens = LogitLens(model)
        result = lens.analyze(input_ids)
        assert set(result.keys()) == {
            "layer_logits",
            "layer_entropies",
            "layer_top_tokens",
            "convergence",
        }

    def test_analyze_all_layers(self, model, input_ids):
        lens = LogitLens(model)
        result = lens.analyze(input_ids)
        n = model.config.n_layers
        assert len(result["layer_logits"]) == n
        assert len(result["layer_entropies"]) == n
        assert len(result["layer_top_tokens"]) == n

    def test_analyze_selected_layers(self, model, input_ids):
        cfg = LogitLensConfig(layers=[0])
        lens = LogitLens(model, cfg)
        result = lens.analyze(input_ids)
        assert len(result["layer_logits"]) == 1

    def test_convergence_shape(self, model, input_ids):
        lens = LogitLens(model)
        result = lens.analyze(input_ids)
        # n_layers=2 => convergence has 1 element
        assert result["convergence"].shape == (1,)

    def test_convergence_bounded(self, model, input_ids):
        lens = LogitLens(model)
        result = lens.analyze(input_ids)
        conv = result["convergence"]
        assert (conv >= -1.0 - 1e-5).all()
        assert (conv <= 1.0 + 1e-5).all()

    def test_no_normalize(self, model, input_ids):
        cfg = LogitLensConfig(normalize=False)
        lens = LogitLens(model, cfg)
        result = lens.analyze(input_ids)
        # Should still produce valid output
        B, T = input_ids.shape
        assert result["layer_logits"][0].shape == (B, T, model.config.vocab_size)

    def test_custom_top_k(self, model, input_ids):
        cfg = LogitLensConfig(top_k=3)
        lens = LogitLens(model, cfg)
        result = lens.analyze(input_ids)
        indices, probs = result["layer_top_tokens"][0]
        assert indices.shape[-1] == 3
        assert probs.shape[-1] == 3

    def test_single_layer_convergence_empty(self, model, input_ids):
        cfg = LogitLensConfig(layers=[0])
        lens = LogitLens(model, cfg)
        result = lens.analyze(input_ids)
        assert result["convergence"].numel() == 0

    def test_entropy_shapes_in_result(self, model, input_ids):
        lens = LogitLens(model)
        result = lens.analyze(input_ids)
        B, T = input_ids.shape
        for ent in result["layer_entropies"]:
            assert ent.shape == (B, T)

    def test_out_of_range_layers_ignored(self, model, input_ids):
        cfg = LogitLensConfig(layers=[0, 99])
        lens = LogitLens(model, cfg)
        result = lens.analyze(input_ids)
        assert len(result["layer_logits"]) == 1

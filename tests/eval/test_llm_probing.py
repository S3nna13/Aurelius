"""
Tests for src/eval/llm_probing.py

Uses a tiny inline 2-layer transformer (d_model=16, vocab_size=16, n_layers=2)
to avoid importing other Aurelius files.

Constants:
    D_MODEL    = 16
    VOCAB_SIZE = 16
    N_LAYERS   = 2
    SEQ_LEN    = 8
    BATCH      = 4
    N_CLASSES  = 3
    N_SAMPLES  = 20
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math

import pytest
import torch
import torch.nn as nn

from src.eval.llm_probing import (
    ActivationExtractor,
    LinearProbe,
    NonlinearProbe,
    ProbeEvaluationSuite,
    ProbingConfig,
    RepresentationSimilarityAnalysis,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
SEQ_LEN = 8
BATCH = 4
N_CLASSES = 3
N_SAMPLES = 20

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Tiny inline transformer (no Aurelius imports)
# ---------------------------------------------------------------------------


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, n_heads, head_dim]
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = _SelfAttention(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """2-layer transformer with accessible named sub-modules."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([_TransformerBlock(d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> TinyTransformer:
    model = TinyTransformer()
    model.eval()
    return model


@pytest.fixture(scope="module")
def layer_names() -> list:
    return ["blocks.0", "blocks.1"]


@pytest.fixture(scope="module")
def batch_input() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture(scope="module")
def sample_input() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN))


@pytest.fixture(scope="module")
def sample_labels() -> torch.Tensor:
    return torch.randint(0, N_CLASSES, (N_SAMPLES,))


# ---------------------------------------------------------------------------
# ActivationExtractor tests
# ---------------------------------------------------------------------------


def test_extractor_captures_activations(tiny_model, layer_names, batch_input):
    """ActivationExtractor populates activations dict after forward."""
    extractor = ActivationExtractor(tiny_model, layer_names)
    acts = extractor.forward(batch_input)
    assert len(acts) > 0, "activations dict should be non-empty after forward"


def test_extractor_keys_match_layer_names(tiny_model, layer_names, batch_input):
    """Keys in activations match the requested layer_names."""
    extractor = ActivationExtractor(tiny_model, layer_names)
    acts = extractor.forward(batch_input)
    assert set(acts.keys()) == set(layer_names)


def test_extractor_activation_shape(tiny_model, layer_names, batch_input):
    """Each activation has shape [B, T, d_model]."""
    extractor = ActivationExtractor(tiny_model, layer_names)
    acts = extractor.forward(batch_input)
    for name, tensor in acts.items():
        assert tensor.shape == (BATCH, SEQ_LEN, D_MODEL), (
            f"Layer '{name}' activation shape mismatch: {tensor.shape}"
        )


def test_extractor_clear_empties_dict(tiny_model, layer_names, batch_input):
    """clear() removes all stored activations."""
    extractor = ActivationExtractor(tiny_model, layer_names)
    extractor.forward(batch_input)
    assert len(extractor.activations) > 0
    extractor.clear()
    assert len(extractor.activations) == 0, "activations should be empty after clear()"


def test_extractor_context_manager(tiny_model, layer_names, batch_input):
    """Context-manager form registers hooks and works correctly."""
    with ActivationExtractor(tiny_model, layer_names) as ext:
        # Run a raw forward through the model (not via extractor.forward)
        with torch.no_grad():
            tiny_model(batch_input)
        acts = ext.activations
    assert set(acts.keys()) == set(layer_names), (
        "Context-manager activations keys should match layer_names"
    )


def test_extractor_context_manager_removes_hooks(tiny_model, layer_names, batch_input):
    """After __exit__, hooks are cleaned up (no duplicate accumulation)."""
    with ActivationExtractor(tiny_model, layer_names) as ext:
        pass  # immediately exit
    assert len(ext._hooks) == 0, "Hooks should be removed after __exit__"


# ---------------------------------------------------------------------------
# LinearProbe tests
# ---------------------------------------------------------------------------


def test_linear_probe_forward_shape():
    """LinearProbe.forward outputs [B, n_classes]."""
    probe = LinearProbe(D_MODEL, N_CLASSES)
    x = torch.randn(BATCH, D_MODEL)
    logits = probe(x)
    assert logits.shape == (BATCH, N_CLASSES)


def test_linear_probe_fit_returns_loss_history():
    """LinearProbe.fit returns a list of floats with length == n_epochs."""
    probe = LinearProbe(D_MODEL, N_CLASSES)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    history = probe.fit(features, labels, n_epochs=3, lr=1e-3)
    assert isinstance(history, list), "fit() should return a list"
    assert len(history) == 3, "loss history length should equal n_epochs"
    assert all(isinstance(v, float) for v in history), "each entry should be a float"


def test_linear_probe_evaluate_returns_tuple():
    """LinearProbe.evaluate returns a 2-tuple."""
    probe = LinearProbe(D_MODEL, N_CLASSES)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    result = probe.evaluate(features, labels)
    assert isinstance(result, tuple) and len(result) == 2


def test_linear_probe_accuracy_in_range():
    """LinearProbe accuracy is in [0, 1]."""
    probe = LinearProbe(D_MODEL, N_CLASSES)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    acc, f1 = probe.evaluate(features, labels)
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of range"


def test_linear_probe_f1_in_range():
    """LinearProbe macro F1 is in [0, 1]."""
    probe = LinearProbe(D_MODEL, N_CLASSES)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    acc, f1 = probe.evaluate(features, labels)
    assert 0.0 <= f1 <= 1.0, f"f1 {f1} out of range"


def test_linear_probe_loss_decreases():
    """LinearProbe loss should generally decrease when classes are separable."""
    torch.manual_seed(0)
    # Build linearly separable data: class i -> feature direction i
    features = torch.zeros(30, D_MODEL)
    labels = torch.zeros(30, dtype=torch.long)
    for i in range(3):
        features[i * 10 : (i + 1) * 10, i] = 5.0
        labels[i * 10 : (i + 1) * 10] = i

    probe = LinearProbe(D_MODEL, N_CLASSES)
    history = probe.fit(features, labels, n_epochs=50, lr=0.1)
    assert history[-1] < history[0], "loss should decrease on separable data"


# ---------------------------------------------------------------------------
# NonlinearProbe tests
# ---------------------------------------------------------------------------


def test_nonlinear_probe_forward_shape():
    """NonlinearProbe.forward outputs [B, n_classes]."""
    probe = NonlinearProbe(D_MODEL, N_CLASSES, hidden=64)
    x = torch.randn(BATCH, D_MODEL)
    logits = probe(x)
    assert logits.shape == (BATCH, N_CLASSES)


def test_nonlinear_probe_evaluate_accuracy_in_range():
    """NonlinearProbe accuracy is in [0, 1]."""
    probe = NonlinearProbe(D_MODEL, N_CLASSES, hidden=64)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    acc, f1 = probe.evaluate(features, labels)
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} out of range"


def test_nonlinear_probe_fit_returns_loss_history():
    """NonlinearProbe.fit returns list of floats."""
    probe = NonlinearProbe(D_MODEL, N_CLASSES, hidden=64)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    history = probe.fit(features, labels, n_epochs=4, lr=1e-3)
    assert isinstance(history, list) and len(history) == 4


def test_nonlinear_probe_f1_in_range():
    """NonlinearProbe macro F1 is in [0, 1]."""
    probe = NonlinearProbe(D_MODEL, N_CLASSES, hidden=64)
    features = torch.randn(N_SAMPLES, D_MODEL)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    _, f1 = probe.evaluate(features, labels)
    assert 0.0 <= f1 <= 1.0, f"f1 {f1} out of range"


# ---------------------------------------------------------------------------
# RepresentationSimilarityAnalysis tests
# ---------------------------------------------------------------------------


def test_rsa_cka_value_in_range():
    """CKA of two random matrices is in [0, 1]."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    Y = torch.randn(N_SAMPLES, D_MODEL)
    val = rsa.cka(X, Y)
    assert 0.0 <= val <= 1.0, f"CKA value {val} out of [0,1]"


def test_rsa_cka_self_equals_one():
    """CKA of X with itself should be 1.0."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    val = rsa.cka(X, X)
    assert abs(val - 1.0) < 1e-4, f"CKA(X,X) should be ~1.0, got {val}"


def test_rsa_procrustes_distance_in_range():
    """Procrustes distance of two random matrices is in [0, 1]."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    Y = torch.randn(N_SAMPLES, D_MODEL)
    dist = rsa.procrustes_distance(X, Y)
    assert 0.0 <= dist <= 1.0, f"Procrustes distance {dist} out of [0,1]"


def test_rsa_procrustes_distance_self_near_zero():
    """Procrustes distance of X with itself should be ~0."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    dist = rsa.procrustes_distance(X, X)
    assert dist < 1e-4, f"Procrustes distance(X,X) should be ~0, got {dist}"


def test_rsa_mutual_knn_in_range():
    """mutual_knn of two random matrices is in [0, 1]."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    Y = torch.randn(N_SAMPLES, D_MODEL)
    val = rsa.mutual_knn(X, Y, k=5)
    assert 0.0 <= val <= 1.0, f"mutual_knn value {val} out of [0,1]"


def test_rsa_mutual_knn_self_equals_one():
    """mutual_knn of X with itself should be 1.0."""
    rsa = RepresentationSimilarityAnalysis()
    X = torch.randn(N_SAMPLES, D_MODEL)
    val = rsa.mutual_knn(X, X, k=5)
    assert abs(val - 1.0) < 1e-6, f"mutual_knn(X,X) should be 1.0, got {val}"


# ---------------------------------------------------------------------------
# ProbeEvaluationSuite tests
# ---------------------------------------------------------------------------


def test_suite_run_probe_returns_dict_with_layer_keys(
    tiny_model, layer_names, sample_input, sample_labels
):
    """run_probe returns a dict whose keys match requested layer_names."""
    suite = ProbeEvaluationSuite(tiny_model, layer_names)
    results = suite.run_probe(sample_input, sample_labels, probe_type="linear")
    assert set(results.keys()) == set(layer_names), (
        f"Expected keys {set(layer_names)}, got {set(results.keys())}"
    )


def test_suite_run_probe_result_structure(tiny_model, layer_names, sample_input, sample_labels):
    """Each layer result has acc, f1, loss_curve keys."""
    suite = ProbeEvaluationSuite(tiny_model, layer_names)
    results = suite.run_probe(sample_input, sample_labels, probe_type="linear")
    for layer, info in results.items():
        assert "acc" in info, f"Missing 'acc' for layer {layer}"
        assert "f1" in info, f"Missing 'f1' for layer {layer}"
        assert "loss_curve" in info, f"Missing 'loss_curve' for layer {layer}"


def test_suite_run_probe_nonlinear(tiny_model, layer_names, sample_input, sample_labels):
    """run_probe works with probe_type='nonlinear'."""
    suite = ProbeEvaluationSuite(tiny_model, layer_names)
    results = suite.run_probe(sample_input, sample_labels, probe_type="nonlinear")
    assert set(results.keys()) == set(layer_names)


def test_suite_compare_layers_returns_nested_dict(tiny_model, layer_names, sample_input):
    """compare_layers returns a nested dict of CKA values."""
    suite = ProbeEvaluationSuite(tiny_model, layer_names)
    cka_matrix = suite.compare_layers(sample_input)
    assert set(cka_matrix.keys()) == set(layer_names)
    for name_a in layer_names:
        for name_b in layer_names:
            val = cka_matrix[name_a][name_b]
            assert 0.0 <= val <= 1.0, f"CKA[{name_a}][{name_b}] = {val} out of range"


# ---------------------------------------------------------------------------
# ProbingConfig tests
# ---------------------------------------------------------------------------


def test_probing_config_defaults():
    """ProbingConfig has expected default values."""
    cfg = ProbingConfig()
    assert cfg.n_epochs == 5
    assert cfg.lr == 1e-3
    assert cfg.hidden == 64
    assert cfg.probe_type == "linear"
    assert cfg.k_knn == 5


def test_probing_config_custom():
    """ProbingConfig accepts custom values."""
    cfg = ProbingConfig(n_epochs=10, lr=5e-4, hidden=128, probe_type="nonlinear", k_knn=3)
    assert cfg.n_epochs == 10
    assert cfg.lr == 5e-4
    assert cfg.hidden == 128
    assert cfg.probe_type == "nonlinear"
    assert cfg.k_knn == 3

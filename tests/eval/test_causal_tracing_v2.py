"""Tests for causal tracing v2 — knowledge localization (src/eval/causal_tracing_v2.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from aurelius.eval.causal_tracing_v2 import (
    ActivationStore,
    HookManager,
    CorruptionModel,
    CausalTracer,
    TracingAnalyzer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 32
T = 6
N_LAYERS = 3
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Tiny test model
# ---------------------------------------------------------------------------

class _LinearBlock(nn.Module):
    """A simple (B, T, d_model) -> (B, T, d_model) block."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyModel(nn.Module):
    """3 hidden layers + a projection to vocab_size. Returns (B, T, V) logits."""

    def __init__(
        self,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
        vocab_size: int = VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_LinearBlock(d_model) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        x = self.embed(input_ids)          # (B, T, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)               # (B, T, V)


@pytest.fixture(scope="module")
def model() -> _TinyModel:
    torch.manual_seed(0)
    m = _TinyModel(N_LAYERS, D_MODEL, VOCAB_SIZE)
    m.eval()
    return m


@pytest.fixture(scope="module")
def layer_names() -> list[str]:
    return [f"layers.{i}" for i in range(N_LAYERS)]


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, T))


@pytest.fixture(scope="module")
def target_token_id() -> int:
    return 5


# ---------------------------------------------------------------------------
# 1-3: ActivationStore
# ---------------------------------------------------------------------------

def test_activation_store_store_and_get():
    """store then get returns the same tensor."""
    store = ActivationStore()
    t = torch.randn(2, 4)
    store.store("layer0", t)
    retrieved = store.get("layer0")
    assert retrieved is not None
    assert torch.equal(retrieved, t)


def test_activation_store_names():
    """names() returns all stored keys."""
    store = ActivationStore()
    store.store("a", torch.zeros(1))
    store.store("b", torch.zeros(1))
    assert set(store.names()) == {"a", "b"}


def test_activation_store_clear():
    """clear() empties the store."""
    store = ActivationStore()
    store.store("x", torch.zeros(1))
    store.clear()
    assert store.names() == []
    assert store.get("x") is None


# ---------------------------------------------------------------------------
# 4: HookManager
# ---------------------------------------------------------------------------

def test_hook_manager_register_and_remove():
    """Capture hook records an activation; remove_all_hooks cleans up."""
    linear = nn.Linear(8, 8, bias=False)
    model_wrapper = nn.Sequential(linear)

    # Access the linear by submodule name '0'
    hook_manager = HookManager(model_wrapper)
    store = ActivationStore()
    hook_manager.register_capture_hook("0", store)

    x = torch.randn(1, 8)
    with torch.no_grad():
        model_wrapper(x)

    assert store.get("0") is not None

    # After removing all hooks a second forward should NOT update the store
    hook_manager.remove_all_hooks()
    store.clear()
    with torch.no_grad():
        model_wrapper(x)
    assert store.get("0") is None


# ---------------------------------------------------------------------------
# 5-7: CorruptionModel
# ---------------------------------------------------------------------------

def test_corruption_model_corrupt_changes_values():
    """Corrupted embeddings must differ from originals."""
    cm = CorruptionModel(noise_scale=1.0, seed=0)
    emb = torch.zeros(1, T, D_MODEL)
    corrupted = cm.corrupt(emb)
    assert not torch.equal(corrupted, emb)


def test_corruption_model_corrupt_same_shape():
    """Output shape equals input shape."""
    cm = CorruptionModel()
    emb = torch.randn(BATCH_SIZE, T, D_MODEL)
    corrupted = cm.corrupt(emb)
    assert corrupted.shape == emb.shape


def test_corruption_model_corrupt_positions_only_changes_specified():
    """corrupt_positions only alters the listed positions."""
    cm = CorruptionModel(noise_scale=1.0, seed=0)
    emb = torch.zeros(1, T, D_MODEL)
    corrupt_pos = [1, 3]
    result = cm.corrupt_positions(emb, corrupt_pos)

    for pos in range(T):
        row_orig = emb[0, pos, :]
        row_result = result[0, pos, :]
        if pos in corrupt_pos:
            assert not torch.equal(row_result, row_orig), (
                f"Position {pos} should be corrupted"
            )
        else:
            assert torch.equal(row_result, row_orig), (
                f"Position {pos} should be unchanged"
            )


# ---------------------------------------------------------------------------
# 8: CausalTracer initialization
# ---------------------------------------------------------------------------

def test_causal_tracer_init(model, layer_names):
    """CausalTracer stores model and layer_names correctly."""
    tracer = CausalTracer(model, layer_names)
    assert tracer.model is model
    assert tracer.layer_names == layer_names
    assert isinstance(tracer.hook_manager, HookManager)
    assert isinstance(tracer.clean_store, ActivationStore)
    assert isinstance(tracer.corruption_model, CorruptionModel)


# ---------------------------------------------------------------------------
# 9: run_clean
# ---------------------------------------------------------------------------

def test_run_clean_shape(model, layer_names, input_ids):
    """run_clean returns (B, T, V) logits and populates clean_store."""
    tracer = CausalTracer(model, layer_names)
    logits = tracer.run_clean(input_ids)
    assert logits.shape == (BATCH_SIZE, T, VOCAB_SIZE)
    # All layer activations should be captured
    for name in layer_names:
        assert tracer.clean_store.get(name) is not None, (
            f"Expected activation for {name} in clean_store"
        )


# ---------------------------------------------------------------------------
# 10: run_corrupted
# ---------------------------------------------------------------------------

def test_run_corrupted_differs_from_clean(model, layer_names, input_ids):
    """run_corrupted should return different values than run_clean."""
    tracer = CausalTracer(model, layer_names)
    clean_logits = tracer.run_clean(input_ids)
    corrupted_logits = tracer.run_corrupted(input_ids)
    assert not torch.allclose(clean_logits, corrupted_logits), (
        "Corrupted logits should differ from clean logits"
    )


# ---------------------------------------------------------------------------
# 11: patch_and_score
# ---------------------------------------------------------------------------

def test_patch_and_score_returns_valid_float(model, layer_names, input_ids, target_token_id):
    """patch_and_score returns a float in [0, 1]."""
    tracer = CausalTracer(model, layer_names)
    tracer.run_clean(input_ids)
    score = tracer.patch_and_score(input_ids, layer_names[0], position=0,
                                   target_token_id=target_token_id)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Score {score} not in [0, 1]"


# ---------------------------------------------------------------------------
# 12: trace_all_layers - dict keys
# ---------------------------------------------------------------------------

def test_trace_all_layers_returns_all_layer_names(model, layer_names, input_ids, target_token_id):
    """trace_all_layers result must contain every layer_name as a key."""
    tracer = CausalTracer(model, layer_names)
    results = tracer.trace_all_layers(input_ids, target_token_id)
    assert isinstance(results, dict)
    for name in layer_names:
        assert name in results, f"Missing key '{name}' in tracing results"


# ---------------------------------------------------------------------------
# 13: trace_all_layers - score list length
# ---------------------------------------------------------------------------

def test_trace_all_layers_scores_length(model, layer_names, input_ids, target_token_id):
    """Each layer's score list must have length == T (number of token positions)."""
    tracer = CausalTracer(model, layer_names)
    results = tracer.trace_all_layers(input_ids, target_token_id)
    for name, scores in results.items():
        assert len(scores) == T, (
            f"Layer '{name}': expected {T} scores, got {len(scores)}"
        )


# ---------------------------------------------------------------------------
# 14: TracingAnalyzer.peak_effect_layer
# ---------------------------------------------------------------------------

def test_peak_effect_layer_returns_valid_key():
    """peak_effect_layer returns a key that exists in tracing_results."""
    analyzer = TracingAnalyzer()
    results = {
        "layer.0": [0.1, 0.9, 0.2],
        "layer.1": [0.3, 0.4, 0.5],
        "layer.2": [0.7, 0.6, 0.8],
    }
    peak = analyzer.peak_effect_layer(results)
    assert peak in results, f"'{peak}' not found in tracing results"
    # layer.0 has max 0.9 which is the global max
    assert peak == "layer.0"


# ---------------------------------------------------------------------------
# 15: TracingAnalyzer.effect_matrix shape
# ---------------------------------------------------------------------------

def test_effect_matrix_shape():
    """effect_matrix returns tensor of shape (n_layers, T)."""
    analyzer = TracingAnalyzer()
    n = N_LAYERS
    t = T
    results = {f"layer.{i}": [float(i + j) for j in range(t)] for i in range(n)}
    matrix = analyzer.effect_matrix(results)
    assert isinstance(matrix, torch.Tensor)
    assert matrix.shape == (n, t), f"Expected ({n}, {t}), got {matrix.shape}"

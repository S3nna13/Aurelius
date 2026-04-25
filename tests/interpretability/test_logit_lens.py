"""
Tests for src/interpretability/logit_lens.py

Covers: LogitLensResult, LogitLensAnalyzer, and LOGIT_LENS_REGISTRY.
Minimum 28 tests.
"""

from __future__ import annotations

import math
import pytest

from src.interpretability.logit_lens import (
    LogitLensResult,
    LogitLensAnalyzer,
    LOGIT_LENS_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 10
HIDDEN_DIM = 4


def _make_analyzer(vocab_size: int = VOCAB) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(vocab_size=vocab_size)


def _identity_unembedding(vocab: int, dim: int) -> list[list[float]]:
    """vocab x dim matrix where row v has 1.0 at position v%dim and 0 elsewhere."""
    mat = []
    for v in range(vocab):
        row = [0.0] * dim
        row[v % dim] = 1.0
        mat.append(row)
    return mat


def _uniform_unembedding(vocab: int, dim: int, val: float = 1.0) -> list[list[float]]:
    return [[val] * dim for _ in range(vocab)]


def _make_hidden(dim: int = HIDDEN_DIM, seed_val: float = 0.1) -> list[float]:
    return [seed_val * (i + 1) for i in range(dim)]


def _project_manual(hidden: list[float], unembedding: list[list[float]]) -> list[float]:
    """Reference matrix-vector product."""
    return [sum(w * h for w, h in zip(row, hidden)) for row in unembedding]


# ---------------------------------------------------------------------------
# 1. LogitLensResult dataclass — frozen
# ---------------------------------------------------------------------------

class TestLogitLensResultDataclass:
    def test_fields_stored(self):
        r = LogitLensResult(
            layer=2, position=3,
            top_token_ids=[0, 1, 2, 3, 4],
            top_logits=[5.0, 4.0, 3.0, 2.0, 1.0],
            entropy=1.5,
        )
        assert r.layer == 2
        assert r.position == 3
        assert r.entropy == 1.5

    def test_frozen_layer(self):
        r = LogitLensResult(layer=0, position=0, top_token_ids=[0], top_logits=[1.0], entropy=0.0)
        with pytest.raises((AttributeError, TypeError)):
            r.layer = 99  # type: ignore[misc]

    def test_frozen_entropy(self):
        r = LogitLensResult(layer=0, position=0, top_token_ids=[0], top_logits=[1.0], entropy=0.5)
        with pytest.raises((AttributeError, TypeError)):
            r.entropy = 9.9  # type: ignore[misc]

    def test_equality(self):
        r1 = LogitLensResult(layer=1, position=2, top_token_ids=[0, 1], top_logits=[2.0, 1.0], entropy=0.3)
        r2 = LogitLensResult(layer=1, position=2, top_token_ids=[0, 1], top_logits=[2.0, 1.0], entropy=0.3)
        assert r1 == r2


# ---------------------------------------------------------------------------
# 2. LogitLensAnalyzer — constructor
# ---------------------------------------------------------------------------

class TestAnalyzerConstructor:
    def test_vocab_size_stored(self):
        a = _make_analyzer(vocab_size=50)
        assert a.vocab_size == 50

    def test_instantiation_default(self):
        a = LogitLensAnalyzer(vocab_size=100)
        assert isinstance(a, LogitLensAnalyzer)


# ---------------------------------------------------------------------------
# 3. _softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_softmax_sums_to_one(self):
        a = _make_analyzer()
        logits = [1.0, 2.0, 3.0, 4.0]
        probs = a._softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_softmax_all_non_negative(self):
        a = _make_analyzer()
        logits = [-100.0, 0.0, 100.0]
        probs = a._softmax(logits)
        assert all(p >= 0.0 for p in probs)

    def test_softmax_stable_large_logits(self):
        """Large logit values must not overflow."""
        a = _make_analyzer()
        logits = [1000.0, 999.0, 998.0]
        probs = a._softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(math.isfinite(p) for p in probs)

    def test_softmax_stable_negative_large(self):
        a = _make_analyzer()
        logits = [-1000.0, -999.0]
        probs = a._softmax(logits)
        assert abs(sum(probs) - 1.0) < 1e-9

    def test_softmax_uniform_logits(self):
        """Equal logits -> uniform probabilities."""
        a = _make_analyzer()
        logits = [3.0, 3.0, 3.0, 3.0]
        probs = a._softmax(logits)
        for p in probs:
            assert abs(p - 0.25) < 1e-9

    def test_softmax_single_element(self):
        a = _make_analyzer()
        probs = a._softmax([5.0])
        assert abs(probs[0] - 1.0) < 1e-9

    def test_softmax_length_preserved(self):
        a = _make_analyzer()
        logits = [0.1, 0.2, 0.3, 0.4, 0.5]
        probs = a._softmax(logits)
        assert len(probs) == 5

    def test_softmax_highest_logit_has_highest_prob(self):
        a = _make_analyzer()
        logits = [1.0, 5.0, 2.0]
        probs = a._softmax(logits)
        assert probs[1] == max(probs)


# ---------------------------------------------------------------------------
# 4. project — result type and structure
# ---------------------------------------------------------------------------

class TestProject:
    def test_project_returns_logit_lens_result(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _uniform_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        assert isinstance(result, LogitLensResult)

    def test_project_layer_and_position_stored(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _uniform_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(3, 7, hidden, unembed)
        assert result.layer == 3
        assert result.position == 7

    def test_project_top_token_ids_length_5(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _identity_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        assert len(result.top_token_ids) == 5

    def test_project_top_logits_length_5(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _identity_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        assert len(result.top_logits) == 5

    def test_project_top_logits_sorted_descending(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _identity_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        for i in range(len(result.top_logits) - 1):
            assert result.top_logits[i] >= result.top_logits[i + 1]

    def test_project_entropy_non_negative(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _uniform_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        assert result.entropy >= 0.0

    def test_project_entropy_finite(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _identity_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        assert math.isfinite(result.entropy)

    def test_project_matrix_vector_product_correct(self):
        """Verify logits match manual computation."""
        vocab, dim = 5, 3
        a = LogitLensAnalyzer(vocab_size=vocab)
        unembed = [[float(v * dim + d) for d in range(dim)] for v in range(vocab)]
        hidden = [1.0, 2.0, 3.0]
        result = a.project(0, 0, hidden, unembed)
        expected = _project_manual(hidden, unembed)
        # Top 5 = all 5 tokens; check that the top token id matches the argmax
        top_id = result.top_token_ids[0]
        assert expected[top_id] == max(expected)

    def test_project_top_token_ids_are_valid_vocab_indices(self):
        a = _make_analyzer(vocab_size=VOCAB)
        unembed = _uniform_unembedding(VOCAB, HIDDEN_DIM)
        hidden = _make_hidden()
        result = a.project(0, 0, hidden, unembed)
        for tid in result.top_token_ids:
            assert 0 <= tid < VOCAB

    def test_project_uniform_unembedding_entropy_is_max(self):
        """Uniform logits -> entropy ~ log(vocab_size)."""
        vocab = 8
        a = LogitLensAnalyzer(vocab_size=vocab)
        unembed = [[1.0] * 1 for _ in range(vocab)]
        hidden = [1.0]  # all logits equal
        result = a.project(0, 0, hidden, unembed)
        expected_max = math.log(vocab)
        # entropy should be close to log(vocab) (uniform distribution)
        assert result.entropy >= expected_max - 0.01


# ---------------------------------------------------------------------------
# 5. layer_trajectories
# ---------------------------------------------------------------------------

class TestLayerTrajectories:
    def _make_results(self) -> list[LogitLensResult]:
        return [
            LogitLensResult(layer=0, position=0, top_token_ids=[2, 1, 0, 3, 4], top_logits=[5.0, 4.0, 3.0, 2.0, 1.0], entropy=1.0),
            LogitLensResult(layer=1, position=0, top_token_ids=[1, 2, 0, 3, 4], top_logits=[6.0, 5.0, 4.0, 3.0, 2.0], entropy=0.8),
            LogitLensResult(layer=2, position=0, top_token_ids=[0, 1, 2, 3, 4], top_logits=[7.0, 6.0, 5.0, 4.0, 3.0], entropy=0.6),
        ]

    def test_trajectory_length_matches_results(self):
        a = _make_analyzer()
        results = self._make_results()
        traj = a.layer_trajectories(results, token_id=2)
        assert len(traj) == 3

    def test_trajectory_sorted_by_layer(self):
        a = _make_analyzer()
        # Provide results in reverse layer order
        results = list(reversed(self._make_results()))
        traj = a.layer_trajectories(results, token_id=2)
        # Layer 0: token_id=2 -> logit 5.0
        assert abs(traj[0] - 5.0) < 1e-9

    def test_trajectory_returns_zero_for_absent_token(self):
        a = _make_analyzer()
        results = self._make_results()
        traj = a.layer_trajectories(results, token_id=9)  # not in any top-5
        assert traj == [0.0, 0.0, 0.0]

    def test_trajectory_token_in_some_layers(self):
        a = _make_analyzer()
        results = self._make_results()
        # token_id=0 appears in layer 0 (logit 3.0), layer 1 (logit 4.0), layer 2 (logit 7.0)
        traj = a.layer_trajectories(results, token_id=0)
        assert abs(traj[0] - 3.0) < 1e-9
        assert abs(traj[1] - 4.0) < 1e-9
        assert abs(traj[2] - 7.0) < 1e-9

    def test_trajectory_empty_results(self):
        a = _make_analyzer()
        traj = a.layer_trajectories([], token_id=0)
        assert traj == []


# ---------------------------------------------------------------------------
# 6. entropy_by_layer
# ---------------------------------------------------------------------------

class TestEntropyByLayer:
    def _make_results(self) -> list[LogitLensResult]:
        return [
            LogitLensResult(layer=0, position=0, top_token_ids=[0, 1, 2, 3, 4], top_logits=[5.0, 4.0, 3.0, 2.0, 1.0], entropy=1.0),
            LogitLensResult(layer=0, position=1, top_token_ids=[0, 1, 2, 3, 4], top_logits=[5.0, 4.0, 3.0, 2.0, 1.0], entropy=2.0),
            LogitLensResult(layer=1, position=0, top_token_ids=[0, 1, 2, 3, 4], top_logits=[5.0, 4.0, 3.0, 2.0, 1.0], entropy=0.5),
        ]

    def test_entropy_by_layer_returns_dict(self):
        a = _make_analyzer()
        results = self._make_results()
        out = a.entropy_by_layer(results)
        assert isinstance(out, dict)

    def test_entropy_by_layer_keys(self):
        a = _make_analyzer()
        results = self._make_results()
        out = a.entropy_by_layer(results)
        assert 0 in out
        assert 1 in out

    def test_entropy_by_layer_mean_value_layer0(self):
        a = _make_analyzer()
        results = self._make_results()
        out = a.entropy_by_layer(results)
        assert abs(out[0] - 1.5) < 1e-9  # mean(1.0, 2.0) = 1.5

    def test_entropy_by_layer_single_result_layer1(self):
        a = _make_analyzer()
        results = self._make_results()
        out = a.entropy_by_layer(results)
        assert abs(out[1] - 0.5) < 1e-9

    def test_entropy_by_layer_empty(self):
        a = _make_analyzer()
        out = a.entropy_by_layer([])
        assert out == {}


# ---------------------------------------------------------------------------
# 7. REGISTRY
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_default_key(self):
        assert "default" in LOGIT_LENS_REGISTRY

    def test_registry_default_is_class(self):
        assert LOGIT_LENS_REGISTRY["default"] is LogitLensAnalyzer

    def test_registry_default_is_instantiable(self):
        cls = LOGIT_LENS_REGISTRY["default"]
        instance = cls(vocab_size=100)
        assert isinstance(instance, LogitLensAnalyzer)

"""Tests for gradient_attribution.py."""

import math
import pytest
import torch

from src.interpretability.gradient_attribution import (
    Attribution,
    AttributionMethod,
    GradientAttribution,
)

SEQ_LEN = 4
D_MODEL = 8


def make_embeddings(seed=0):
    torch.manual_seed(seed)
    return torch.rand(SEQ_LEN, D_MODEL)


def make_gradient(seed=1):
    torch.manual_seed(seed)
    return torch.rand(SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# AttributionMethod enum
# ---------------------------------------------------------------------------

class TestAttributionMethodEnum:
    def test_saliency_value(self):
        assert AttributionMethod.SALIENCY == "saliency"

    def test_integrated_gradients_value(self):
        assert AttributionMethod.INTEGRATED_GRADIENTS == "integrated_gradients"

    def test_gradient_x_input_value(self):
        assert AttributionMethod.GRADIENT_X_INPUT == "gradient_x_input"

    def test_enum_has_3_members(self):
        assert len(AttributionMethod) == 3

    def test_is_str_subclass(self):
        assert isinstance(AttributionMethod.SALIENCY, str)

    def test_saliency_name(self):
        assert AttributionMethod.SALIENCY.name == "SALIENCY"

    def test_integrated_gradients_name(self):
        assert AttributionMethod.INTEGRATED_GRADIENTS.name == "INTEGRATED_GRADIENTS"

    def test_gradient_x_input_name(self):
        assert AttributionMethod.GRADIENT_X_INPUT.name == "GRADIENT_X_INPUT"


# ---------------------------------------------------------------------------
# Attribution dataclass
# ---------------------------------------------------------------------------

class TestAttributionDataclass:
    def test_token_ids_stored(self):
        a = Attribution(
            token_ids=[1, 2, 3],
            scores=[0.1, 0.2, 0.3],
            method=AttributionMethod.SALIENCY,
        )
        assert a.token_ids == [1, 2, 3]

    def test_scores_stored(self):
        a = Attribution(
            token_ids=[1, 2],
            scores=[0.5, 0.5],
            method=AttributionMethod.SALIENCY,
        )
        assert a.scores == [0.5, 0.5]

    def test_method_stored(self):
        a = Attribution(
            token_ids=[1],
            scores=[1.0],
            method=AttributionMethod.INTEGRATED_GRADIENTS,
        )
        assert a.method == AttributionMethod.INTEGRATED_GRADIENTS

    def test_normalized_default_false(self):
        a = Attribution(token_ids=[], scores=[], method=AttributionMethod.SALIENCY)
        assert a.normalized is False

    def test_normalized_true(self):
        a = Attribution(
            token_ids=[],
            scores=[],
            method=AttributionMethod.SALIENCY,
            normalized=True,
        )
        assert a.normalized is True

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(Attribution)


# ---------------------------------------------------------------------------
# GradientAttribution.normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def setup_method(self):
        self.ga = GradientAttribution()

    def test_l2_norm_approx_1(self):
        scores = [3.0, 4.0]
        normalized = self.ga.normalize(scores)
        norm = math.sqrt(sum(s ** 2 for s in normalized))
        assert abs(norm - 1.0) < 1e-4

    def test_all_zero_input_returns_zeros_not_nan(self):
        scores = [0.0, 0.0, 0.0]
        normalized = self.ga.normalize(scores)
        for v in normalized:
            assert not math.isnan(v)
            assert v < 1.0  # should be near zero

    def test_output_length_matches_input(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        normalized = self.ga.normalize(scores)
        assert len(normalized) == len(scores)

    def test_single_element(self):
        scores = [5.0]
        normalized = self.ga.normalize(scores)
        assert len(normalized) == 1
        norm = math.sqrt(sum(s ** 2 for s in normalized))
        assert abs(norm - 1.0) < 1e-4

    def test_returns_list(self):
        result = self.ga.normalize([1.0, 2.0])
        assert isinstance(result, list)

    def test_negative_values_preserved_in_l2(self):
        scores = [-3.0, 4.0]
        normalized = self.ga.normalize(scores)
        norm = math.sqrt(sum(s ** 2 for s in normalized))
        assert abs(norm - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# GradientAttribution.saliency
# ---------------------------------------------------------------------------

class TestSaliency:
    def setup_method(self):
        self.ga = GradientAttribution(method=AttributionMethod.SALIENCY)
        self.emb = make_embeddings()
        self.grad = make_gradient()

    def test_returns_list(self):
        result = self.ga.saliency(self.emb, self.grad)
        assert isinstance(result, list)

    def test_length_equals_seq_len(self):
        result = self.ga.saliency(self.emb, self.grad)
        assert len(result) == SEQ_LEN

    def test_values_non_negative(self):
        result = self.ga.saliency(self.emb, self.grad)
        for v in result:
            assert v >= 0

    def test_zero_gradient_gives_zeros(self):
        zero_grad = torch.zeros(SEQ_LEN, D_MODEL)
        result = self.ga.saliency(self.emb, zero_grad)
        for v in result:
            assert v == pytest.approx(0.0)

    def test_values_are_floats(self):
        result = self.ga.saliency(self.emb, self.grad)
        for v in result:
            assert isinstance(v, float)

    def test_matches_manual_computation(self):
        result = self.ga.saliency(self.emb, self.grad)
        expected = self.grad.abs().sum(-1).tolist()
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# GradientAttribution.gradient_x_input
# ---------------------------------------------------------------------------

class TestGradientXInput:
    def setup_method(self):
        self.ga = GradientAttribution(method=AttributionMethod.GRADIENT_X_INPUT)
        self.emb = make_embeddings()
        self.grad = make_gradient()

    def test_returns_list(self):
        result = self.ga.gradient_x_input(self.emb, self.grad)
        assert isinstance(result, list)

    def test_length_equals_seq_len(self):
        result = self.ga.gradient_x_input(self.emb, self.grad)
        assert len(result) == SEQ_LEN

    def test_values_non_negative(self):
        result = self.ga.gradient_x_input(self.emb, self.grad)
        for v in result:
            assert v >= 0

    def test_zero_embeddings_gives_zeros(self):
        zero_emb = torch.zeros(SEQ_LEN, D_MODEL)
        result = self.ga.gradient_x_input(zero_emb, self.grad)
        for v in result:
            assert v == pytest.approx(0.0)

    def test_values_are_floats(self):
        result = self.ga.gradient_x_input(self.emb, self.grad)
        for v in result:
            assert isinstance(v, float)

    def test_matches_manual_computation(self):
        result = self.ga.gradient_x_input(self.emb, self.grad)
        expected = (self.grad * self.emb).abs().sum(-1).tolist()
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# GradientAttribution.integrated_gradients_scores
# ---------------------------------------------------------------------------

class TestIntegratedGradientsScores:
    def setup_method(self):
        self.ga = GradientAttribution(method=AttributionMethod.INTEGRATED_GRADIENTS, n_steps=3)

    def test_returns_list(self):
        baseline_grads = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]
        input_grads = [1.0, 1.0, 1.0, 1.0]
        result = self.ga.integrated_gradients_scores(baseline_grads, input_grads)
        assert isinstance(result, list)

    def test_length_matches_input_grads(self):
        baseline_grads = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        input_grads = [1.0, 2.0, 3.0, 4.0]
        result = self.ga.integrated_gradients_scores(baseline_grads, input_grads)
        assert len(result) == len(input_grads)

    def test_values_are_floats(self):
        baseline_grads = [[0.1, 0.2], [0.3, 0.4]]
        input_grads = [1.0, 1.0]
        result = self.ga.integrated_gradients_scores(baseline_grads, input_grads)
        for v in result:
            assert isinstance(v, float)

    def test_manual_computation(self):
        baseline_grads = [[1.0, 2.0], [3.0, 4.0]]
        input_grads = [2.0, 2.0]
        result = self.ga.integrated_gradients_scores(baseline_grads, input_grads)
        # mean: [2.0, 3.0], * input: [4.0, 6.0]
        assert result == pytest.approx([4.0, 6.0], abs=1e-6)

    def test_single_step(self):
        baseline_grads = [[0.5, 0.5, 0.5, 0.5]]
        input_grads = [2.0, 2.0, 2.0, 2.0]
        result = self.ga.integrated_gradients_scores(baseline_grads, input_grads)
        assert result == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# GradientAttribution.create_attribution
# ---------------------------------------------------------------------------

class TestCreateAttribution:
    def setup_method(self):
        self.ga = GradientAttribution()
        self.token_ids = [10, 20, 30, 40]
        self.scores = [1.0, 2.0, 3.0, 4.0]

    def test_returns_attribution(self):
        result = self.ga.create_attribution(self.token_ids, self.scores)
        assert isinstance(result, Attribution)

    def test_token_ids_correct(self):
        result = self.ga.create_attribution(self.token_ids, self.scores)
        assert result.token_ids == self.token_ids

    def test_method_set(self):
        result = self.ga.create_attribution(self.token_ids, self.scores)
        assert result.method == self.ga.method

    def test_normalize_true_sets_flag(self):
        result = self.ga.create_attribution(self.token_ids, self.scores, normalize=True)
        assert result.normalized is True

    def test_normalize_false_sets_flag(self):
        result = self.ga.create_attribution(self.token_ids, self.scores, normalize=False)
        assert result.normalized is False

    def test_normalize_true_scores_l2_norm_approx_1(self):
        result = self.ga.create_attribution(self.token_ids, self.scores, normalize=True)
        norm = math.sqrt(sum(s ** 2 for s in result.scores))
        assert abs(norm - 1.0) < 1e-4

    def test_normalize_false_scores_unchanged(self):
        result = self.ga.create_attribution(self.token_ids, self.scores, normalize=False)
        assert result.scores == self.scores

    def test_scores_length(self):
        result = self.ga.create_attribution(self.token_ids, self.scores)
        assert len(result.scores) == len(self.scores)

    def test_default_normalize_is_true(self):
        result = self.ga.create_attribution(self.token_ids, self.scores)
        assert result.normalized is True

    def test_integrated_gradients_method(self):
        ga = GradientAttribution(method=AttributionMethod.INTEGRATED_GRADIENTS)
        result = ga.create_attribution(self.token_ids, self.scores)
        assert result.method == AttributionMethod.INTEGRATED_GRADIENTS

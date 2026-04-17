"""Tests for src/inference/calm.py — CALM: Confident Adaptive Language Modeling.

Test configuration: n_layers=4, d_model=64, vocab_size=256  (tiny, fast).
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.inference.calm import (
    CALMCalibrator,
    CALMConfidenceScorer,
    CALMDecoder,
    CALMEarlyExitDecoder,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

N_LAYERS = 4
D_MODEL = 64
VOCAB_SIZE = 256
B = 2   # batch size
T = 5   # sequence length


def _make_lm_head(d_model: int = D_MODEL, vocab_size: int = VOCAB_SIZE) -> nn.Linear:
    torch.manual_seed(0)
    head = nn.Linear(d_model, vocab_size, bias=False)
    return head


def _make_layer_outputs(
    n_layers: int = N_LAYERS,
    batch: int = B,
    seq_len: int = T,
    d_model: int = D_MODEL,
    seed: int = 42,
) -> List[Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(batch, seq_len, d_model) for _ in range(n_layers)]


def _peaked_logits(vocab_size: int = VOCAB_SIZE) -> Tensor:
    """Logits concentrated on one token → high confidence."""
    logits = torch.full((vocab_size,), -10.0)
    logits[0] = 100.0
    return logits


def _uniform_logits(vocab_size: int = VOCAB_SIZE) -> Tensor:
    """Logits all equal → uniform distribution → low confidence."""
    return torch.zeros(vocab_size)


# ---------------------------------------------------------------------------
# CALMConfidenceScorer tests (1-4)
# ---------------------------------------------------------------------------

class TestCALMConfidenceScorer:

    @pytest.mark.parametrize("method", ["softmax_max", "softmax_entropy"])
    def test_output_in_unit_interval(self, method: str) -> None:
        """Test 1 & 2: Both methods return score ∈ [0, 1]."""
        scorer = CALMConfidenceScorer(method=method)
        torch.manual_seed(1)
        logits = torch.randn(B, T, VOCAB_SIZE)
        c = scorer.score(logits)
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0, f"Score {c} not in [0, 1] for method '{method}'"

    def test_peaked_distribution_high_confidence_softmax_max(self) -> None:
        """Test 3: Peaked distribution → confidence near 1.0 (softmax_max)."""
        scorer = CALMConfidenceScorer(method="softmax_max")
        logits = _peaked_logits()
        c = scorer.score(logits)
        assert c > 0.99, f"Expected confidence near 1.0, got {c}"

    def test_peaked_distribution_high_confidence_softmax_entropy(self) -> None:
        """Test 3 (entropy): Peaked distribution → confidence near 1.0."""
        scorer = CALMConfidenceScorer(method="softmax_entropy")
        logits = _peaked_logits()
        c = scorer.score(logits)
        assert c > 0.95, f"Expected confidence near 1.0, got {c}"

    def test_uniform_distribution_low_confidence_softmax_max(self) -> None:
        """Test 4: Uniform distribution → low confidence (softmax_max)."""
        scorer = CALMConfidenceScorer(method="softmax_max")
        logits = _uniform_logits()
        c = scorer.score(logits)
        expected = 1.0 / VOCAB_SIZE
        assert abs(c - expected) < 1e-5, f"Expected {expected}, got {c}"

    def test_uniform_distribution_low_confidence_softmax_entropy(self) -> None:
        """Test 4 (entropy): Uniform distribution → confidence near 0.0."""
        scorer = CALMConfidenceScorer(method="softmax_entropy")
        logits = _uniform_logits()
        c = scorer.score(logits)
        assert c < 0.05, f"Expected confidence near 0.0, got {c}"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown confidence method"):
            CALMConfidenceScorer(method="invalid_method")


# ---------------------------------------------------------------------------
# CALMEarlyExitDecoder tests (5-12)
# ---------------------------------------------------------------------------

class TestCALMEarlyExitDecoder:

    def test_exits_at_correct_layer_when_threshold_met(self) -> None:
        """Test 5: Decoder exits at the first layer that meets the threshold."""
        lm_head = _make_lm_head()
        # Build layers where only layer index 2 produces high confidence.
        # Strategy: make layer 2 a peaked hidden state that, after lm_head,
        # has a dominant logit. We achieve this by pointing h toward the
        # first row of lm_head.weight (the weight for token 0).
        layer_outputs = _make_layer_outputs()  # normal random (low conf)
        # Craft a very "aligned" hidden state for layer index 2
        with torch.no_grad():
            # h such that lm_head(h)[0] >> rest → softmax_max ≈ 1
            w0 = lm_head.weight[0]  # (d_model,)
            peaked_h = w0.unsqueeze(0).unsqueeze(0).expand(B, T, -1) * 200.0
            layer_outputs[2] = peaked_h

        # Set threshold low enough that the peaked layer should trigger
        decoder = CALMEarlyExitDecoder(
            n_layers=N_LAYERS, threshold=0.5, confidence_method="softmax_max"
        )
        _, exit_layer, _ = decoder.forward(layer_outputs, lm_head)
        assert exit_layer == 2, f"Expected exit at layer 2, got {exit_layer}"

    def test_exits_at_last_layer_when_no_layer_meets_threshold(self) -> None:
        """Test 6: Decoder exits at last layer when threshold is never met."""
        lm_head = _make_lm_head()
        layer_outputs = _make_layer_outputs()
        # threshold=1.01 is never achievable → must reach last layer
        decoder = CALMEarlyExitDecoder(
            n_layers=N_LAYERS, threshold=1.01, confidence_method="softmax_max"
        )
        _, exit_layer, _ = decoder.forward(layer_outputs, lm_head)
        assert exit_layer == N_LAYERS - 1, (
            f"Expected exit at layer {N_LAYERS-1}, got {exit_layer}"
        )

    def test_exit_layer_in_valid_range(self) -> None:
        """Test 7: exit_layer is always in [0, n_layers - 1]."""
        lm_head = _make_lm_head()
        for threshold in [0.0, 0.5, 0.9, 1.01]:
            layer_outputs = _make_layer_outputs(seed=threshold * 100)
            decoder = CALMEarlyExitDecoder(
                n_layers=N_LAYERS, threshold=threshold
            )
            _, exit_layer, _ = decoder.forward(layer_outputs, lm_head)
            assert 0 <= exit_layer <= N_LAYERS - 1, (
                f"exit_layer {exit_layer} out of range for threshold={threshold}"
            )

    def test_determinism_under_manual_seed(self) -> None:
        """Test 8: Deterministic outputs under torch.manual_seed."""
        lm_head = _make_lm_head()

        def run() -> tuple:
            torch.manual_seed(99)
            lo = _make_layer_outputs(seed=99)
            decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.7)
            logits, el, conf = decoder.forward(lo, lm_head)
            return el, round(conf, 6)

        r1, r2 = run(), run()
        assert r1 == r2, f"Non-deterministic results: {r1} vs {r2}"

    def test_batch_size_one(self) -> None:
        """Test 9: Works with batch_size=1."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs(batch=1)
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.8)
        logits, exit_layer, conf = decoder.forward(lo, lm_head)
        assert logits.shape == (1, T, VOCAB_SIZE)
        assert 0 <= exit_layer < N_LAYERS
        assert 0.0 <= conf <= 1.0

    def test_single_token_sequence(self) -> None:
        """Test 10: Works with T=1 (single token)."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs(seq_len=1)
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.8)
        logits, exit_layer, conf = decoder.forward(lo, lm_head)
        assert logits.shape == (B, 1, VOCAB_SIZE)
        assert 0 <= exit_layer < N_LAYERS

    def test_no_nan_or_inf_on_normal_inputs(self) -> None:
        """Test 11: No NaN or Inf in output logits."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs()
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.8)
        logits, _, _ = decoder.forward(lo, lm_head)
        assert not torch.isnan(logits).any(), "NaN in output logits"
        assert not torch.isinf(logits).any(), "Inf in output logits"

    def test_threshold_zero_exits_at_first_layer(self) -> None:
        """Test 12: threshold=0.0 → always exit at layer 0."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs()
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.0)
        _, exit_layer, _ = decoder.forward(lo, lm_head)
        assert exit_layer == 0, f"Expected exit at layer 0, got {exit_layer}"

    def test_threshold_above_one_exits_at_last_layer(self) -> None:
        """Test 13: threshold=1.01 → always exit at last layer."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs()
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=1.01)
        _, exit_layer, _ = decoder.forward(lo, lm_head)
        assert exit_layer == N_LAYERS - 1, (
            f"Expected exit at layer {N_LAYERS-1}, got {exit_layer}"
        )

    def test_wrong_number_of_layer_outputs_raises(self) -> None:
        """Defensive test: mismatched layer count raises ValueError."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs(n_layers=3)  # wrong count
        decoder = CALMEarlyExitDecoder(n_layers=N_LAYERS, threshold=0.5)
        with pytest.raises(ValueError, match="Expected"):
            decoder.forward(lo, lm_head)


# ---------------------------------------------------------------------------
# CALMCalibrator tests (14-15)
# ---------------------------------------------------------------------------

class TestCALMCalibrator:

    def _make_calibration_set(
        self, n_examples: int = 20, seed: int = 7
    ) -> List[List[Tensor]]:
        torch.manual_seed(seed)
        return [
            _make_layer_outputs(seed=seed + i) for i in range(n_examples)
        ]

    def test_returned_threshold_in_unit_interval(self) -> None:
        """Test 14: Calibrated λ is in [0, 1]."""
        lm_head = _make_lm_head()
        cal_set = self._make_calibration_set()
        calibrator = CALMCalibrator()
        λ = calibrator.calibrate(cal_set, lm_head, target_coverage=0.95)
        assert isinstance(λ, float)
        assert 0.0 <= λ <= 1.0, f"Threshold {λ} not in [0, 1]"

    def test_lower_coverage_gives_lower_threshold(self) -> None:
        """Test 15: lower target_coverage → lower threshold (earlier exit)."""
        lm_head = _make_lm_head()
        cal_set = self._make_calibration_set()
        calibrator = CALMCalibrator()
        λ_high = calibrator.calibrate(cal_set, lm_head, target_coverage=0.99)
        λ_low  = calibrator.calibrate(cal_set, lm_head, target_coverage=0.50)
        assert λ_low <= λ_high, (
            f"Expected λ_low ({λ_low:.4f}) <= λ_high ({λ_high:.4f})"
        )


# ---------------------------------------------------------------------------
# CALMDecoder tests (16-17)
# ---------------------------------------------------------------------------

class TestCALMDecoder:

    def test_easy_inputs_exit_early(self) -> None:
        """Test 16: On easy (peaked) inputs, avg exit layer < n_layers - 1."""
        lm_head = _make_lm_head()
        # Create peaked hidden states: all layers point to token 0
        with torch.no_grad():
            w0 = lm_head.weight[0]  # (d_model,)
            peaked_h = w0.unsqueeze(0).unsqueeze(0).expand(B, T, -1) * 500.0
            layer_outputs = [peaked_h.clone() for _ in range(N_LAYERS)]

        decoder = CALMDecoder(
            n_layers=N_LAYERS,
            lm_head=lm_head,
            threshold=0.5,
            confidence_method="softmax_max",
        )
        all_exit_layers: List[int] = []
        # Simulate multiple decoding steps (each with same easy inputs)
        for _ in range(10):
            _, exit_layers, confidences = decoder.decode_step(layer_outputs)
            all_exit_layers.extend(exit_layers)
            for c in confidences:
                assert 0.0 <= c <= 1.0

        avg = CALMDecoder.average_exit_layer(all_exit_layers)
        assert avg < N_LAYERS - 1, (
            f"Expected average exit layer < {N_LAYERS - 1}, got {avg}"
        )

    def test_average_exit_layer_empty_raises(self) -> None:
        """Defensive: empty exit_layers list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            CALMDecoder.average_exit_layer([])

    def test_decode_step_output_shapes(self) -> None:
        """Test 17: decode_step returns correct output shapes."""
        lm_head = _make_lm_head()
        lo = _make_layer_outputs()
        decoder = CALMDecoder(
            n_layers=N_LAYERS, lm_head=lm_head, threshold=0.8
        )
        token_ids, exit_layers, confidences = decoder.decode_step(lo)
        assert token_ids.shape == (B, T)
        assert token_ids.dtype == torch.long
        assert len(exit_layers) == 1
        assert len(confidences) == 1
        assert 0 <= exit_layers[0] < N_LAYERS
        assert 0.0 <= confidences[0] <= 1.0

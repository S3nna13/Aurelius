"""Tests for conformal prediction sets for language model token prediction.

~15 tests covering CalibrationSet, RAPSCalibrationSet, ConformalTokenSet,
and ConformalLMDecoder.
"""

import pytest
import torch
import torch.nn.functional as F
from src.inference.conformal_lm import (
    CalibrationSet,
    ConformalLMDecoder,
    ConformalTokenSet,
    RAPSCalibrationSet,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

V = 32  # small vocabulary size used throughout


def _uniform_probs(v: int = V) -> torch.Tensor:
    """Return uniform probability distribution over v tokens."""
    return torch.ones(v) / v


def _peaked_probs(peak_token: int = 0, v: int = V) -> torch.Tensor:
    """Return probabilities strongly peaked at peak_token."""
    logits = torch.zeros(v)
    logits[peak_token] = 10.0
    return F.softmax(logits, dim=0)


def _random_probs(v: int = V, seed: int = 42) -> torch.Tensor:
    """Return random softmax probabilities."""
    g = torch.Generator()
    g.manual_seed(seed)
    return F.softmax(torch.rand(v, generator=g), dim=0)


def _make_calibration_data(n: int = 50, v: int = V, seed: int = 0):
    """Generate n random (probs, true_token) calibration pairs."""
    g = torch.Generator()
    g.manual_seed(seed)
    probs_list = [F.softmax(torch.rand(v, generator=g), dim=0) for _ in range(n)]
    true_tokens = [int(torch.randint(v, (1,), generator=g).item()) for _ in range(n)]
    return probs_list, true_tokens


# =========================================================================== #
# 1. CalibrationSet.add increments n
# =========================================================================== #
def test_calibration_set_add_increments_n():
    cs = CalibrationSet()
    assert cs.n == 0
    cs.add(_random_probs(), true_token_id=3)
    assert cs.n == 1
    cs.add(_random_probs(seed=1), true_token_id=7)
    assert cs.n == 2


# =========================================================================== #
# 2. CalibrationSet scores are in [0, 1]
# =========================================================================== #
def test_calibration_set_scores_in_unit_interval():
    cs = CalibrationSet()
    for i in range(20):
        probs = _random_probs(seed=i)
        cs.add(probs, true_token_id=i % V)
    for score in cs.scores:
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0,1]"


# =========================================================================== #
# 3. CalibrationSet.quantile returns 1.0 for empty set
# =========================================================================== #
def test_calibration_set_quantile_empty():
    cs = CalibrationSet()
    assert cs.quantile(alpha=0.1) == 1.0


# =========================================================================== #
# 4. CalibrationSet.quantile returns valid float
# =========================================================================== #
def test_calibration_set_quantile_valid_float():
    cs = CalibrationSet()
    probs_list, true_tokens = _make_calibration_data(n=50)
    for p, t in zip(probs_list, true_tokens):
        cs.add(p, t)
    tau = cs.quantile(alpha=0.1)
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0


# =========================================================================== #
# 5. RAPSCalibrationSet.add computes positive score
# =========================================================================== #
def test_raps_calibration_set_add_positive_score():
    rcs = RAPSCalibrationSet(k_reg=5, lambda_reg=0.01)
    probs = _random_probs(seed=99)
    rcs.add(probs, true_token_id=10)
    assert len(rcs.scores) == 1
    assert rcs.scores[0] > 0.0


# =========================================================================== #
# 6. RAPSCalibrationSet.quantile with data returns float
# =========================================================================== #
def test_raps_calibration_set_quantile_returns_float():
    rcs = RAPSCalibrationSet()
    probs_list, true_tokens = _make_calibration_data(n=40)
    for p, t in zip(probs_list, true_tokens):
        rcs.add(p, t)
    tau = rcs.quantile(alpha=0.1)
    assert isinstance(tau, float)
    assert tau > 0.0


# =========================================================================== #
# 7. ConformalTokenSet.predict_set returns list of ints
# =========================================================================== #
def test_conformal_token_set_predict_set_returns_list_of_ints():
    cts = ConformalTokenSet(tau=0.5)
    result = cts.predict_set(_random_probs())
    assert isinstance(result, list)
    assert len(result) > 0
    for token_id in result:
        assert isinstance(token_id, int)


# =========================================================================== #
# 8. predict_set with tau=1.0 returns all tokens (V)
# =========================================================================== #
def test_conformal_token_set_tau_one_returns_all_tokens():
    cts = ConformalTokenSet(tau=1.0)
    result = cts.predict_set(_uniform_probs())
    # With tau=1.0 the cumsum reaches 1.0 exactly at the last token,
    # but since we include the token that *crosses* tau, all V tokens are included.
    assert len(result) == V


# =========================================================================== #
# 9. predict_set with tau=0.0 returns at least 1 token (the most probable)
# =========================================================================== #
def test_conformal_token_set_tau_zero_returns_at_least_one():
    cts = ConformalTokenSet(tau=0.0)
    result = cts.predict_set(_peaked_probs(peak_token=5))
    # The first token already pushes cumsum above 0, so we must get >= 1 token.
    assert len(result) >= 1


# =========================================================================== #
# 10. Set sizes are positive integers
# =========================================================================== #
def test_set_sizes_are_positive():
    cts = ConformalTokenSet(tau=0.5)
    probs_batch = torch.stack([_random_probs(seed=i) for i in range(10)])
    stats = cts.set_size_stats(probs_batch)
    assert stats["min_set_size"] >= 1
    assert stats["max_set_size"] >= stats["min_set_size"]


# =========================================================================== #
# 11. set_size_stats returns expected keys
# =========================================================================== #
def test_set_size_stats_keys():
    cts = ConformalTokenSet(tau=0.5)
    probs_batch = torch.stack([_random_probs(seed=i) for i in range(5)])
    stats = cts.set_size_stats(probs_batch)
    assert "mean_set_size" in stats
    assert "max_set_size" in stats
    assert "min_set_size" in stats


# =========================================================================== #
# 12. ConformalLMDecoder.calibrate returns float tau
# =========================================================================== #
def test_conformal_lm_decoder_calibrate_returns_float():
    decoder = ConformalLMDecoder(alpha=0.1, use_raps=False)
    probs_list, true_tokens = _make_calibration_data(n=50)
    tau = decoder.calibrate(probs_list, true_tokens)
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0


# =========================================================================== #
# 13. ConformalLMDecoder.predict returns list
# =========================================================================== #
def test_conformal_lm_decoder_predict_returns_list():
    decoder = ConformalLMDecoder(alpha=0.1)
    probs_list, true_tokens = _make_calibration_data(n=50)
    decoder.calibrate(probs_list, true_tokens)
    result = decoder.predict(_random_probs(seed=77))
    assert isinstance(result, list)
    assert len(result) >= 1
    for tid in result:
        assert isinstance(tid, int)
        assert 0 <= tid < V


# =========================================================================== #
# 14. Coverage estimate is in [0, 1]
# =========================================================================== #
def test_coverage_estimate_in_unit_interval():
    decoder = ConformalLMDecoder(alpha=0.1)
    cal_probs, cal_tokens = _make_calibration_data(n=100, seed=0)
    decoder.calibrate(cal_probs, cal_tokens)
    test_probs, test_tokens = _make_calibration_data(n=50, seed=1)
    cov = decoder.coverage_estimate(test_probs, test_tokens)
    assert 0.0 <= cov <= 1.0


# =========================================================================== #
# 15. ConformalLMDecoder.predict raises RuntimeError if not calibrated
# =========================================================================== #
def test_conformal_lm_decoder_predict_raises_if_not_calibrated():
    decoder = ConformalLMDecoder(alpha=0.1)
    with pytest.raises(RuntimeError, match="Not calibrated"):
        decoder.predict(_random_probs())

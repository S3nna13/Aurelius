"""Tests for robustness_evaluator.py."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.robustness_evaluator import RobustnessEvaluator, RobustnessReport

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

TINY_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

B = 1
S = 6
N_SAMPLES = 3


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(0)
    m = AureliusTransformer(TINY_CONFIG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def evaluator(model: AureliusTransformer) -> RobustnessEvaluator:
    return RobustnessEvaluator(model)


@pytest.fixture(scope="module")
def input_ids_list() -> list:
    torch.manual_seed(1)
    return [torch.randint(0, TINY_CONFIG.vocab_size, (B, S)) for _ in range(N_SAMPLES)]


@pytest.fixture(scope="module")
def perturbed_ids_list() -> list:
    torch.manual_seed(2)
    return [torch.randint(0, TINY_CONFIG.vocab_size, (B, S)) for _ in range(N_SAMPLES)]


# ---------------------------------------------------------------------------
# Helper: get actual model predictions so we can construct exact label sets
# ---------------------------------------------------------------------------


def get_predictions(model: AureliusTransformer, ids_list: list) -> list:
    preds = []
    with torch.no_grad():
        for ids in ids_list:
            _, logits, _ = model(ids)
            preds.append(int(logits[:, -1, :].argmax(dim=-1).item()))
    return preds


# ---------------------------------------------------------------------------
# Test 1: RobustnessEvaluator instantiates
# ---------------------------------------------------------------------------


def test_instantiation(evaluator: RobustnessEvaluator) -> None:
    assert isinstance(evaluator, RobustnessEvaluator)


# ---------------------------------------------------------------------------
# Test 2: clean_accuracy returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_clean_accuracy_range(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    acc = evaluator.clean_accuracy(input_ids_list, label_ids)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Test 3: All-matching labels -> accuracy = 1.0
# ---------------------------------------------------------------------------


def test_clean_accuracy_all_correct(
    evaluator: RobustnessEvaluator,
    model: AureliusTransformer,
    input_ids_list: list,
) -> None:
    label_ids = get_predictions(model, input_ids_list)
    acc = evaluator.clean_accuracy(input_ids_list, label_ids)
    assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 4: All-mismatched labels -> accuracy = 0.0
# ---------------------------------------------------------------------------


def test_clean_accuracy_all_wrong(
    evaluator: RobustnessEvaluator,
    model: AureliusTransformer,
    input_ids_list: list,
) -> None:
    preds = get_predictions(model, input_ids_list)
    bad_labels = [(p + 1) % TINY_CONFIG.vocab_size for p in preds]
    acc = evaluator.clean_accuracy(input_ids_list, bad_labels)
    assert acc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: attack_success_rate returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_attack_success_rate_range(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    asr = evaluator.attack_success_rate(input_ids_list, perturbed_ids_list, label_ids)
    assert isinstance(asr, float)
    assert 0.0 <= asr <= 1.0


# ---------------------------------------------------------------------------
# Test 6: Identical clean and perturbed -> attack_success_rate = 0.0
# ---------------------------------------------------------------------------


def test_attack_success_rate_identical(
    evaluator: RobustnessEvaluator,
    model: AureliusTransformer,
    input_ids_list: list,
) -> None:
    label_ids = get_predictions(model, input_ids_list)
    asr = evaluator.attack_success_rate(input_ids_list, input_ids_list, label_ids)
    assert asr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 7: semantic_preservation returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_semantic_preservation_range(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    sp = evaluator.semantic_preservation(input_ids_list, perturbed_ids_list)
    assert isinstance(sp, float)
    assert 0.0 <= sp <= 1.0


# ---------------------------------------------------------------------------
# Test 8: Identical sequences -> preservation = 1.0
# ---------------------------------------------------------------------------


def test_semantic_preservation_identical(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
) -> None:
    sp = evaluator.semantic_preservation(input_ids_list, input_ids_list)
    assert sp == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 9: certified_lower_bound returns non-negative float
# ---------------------------------------------------------------------------


def test_certified_lower_bound_nonnegative(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
) -> None:
    radius = evaluator.certified_lower_bound(input_ids_list[0], n_samples=N_SAMPLES, sigma=0.1)
    assert isinstance(radius, float)
    assert radius >= 0.0


# ---------------------------------------------------------------------------
# Test 10: evaluate returns RobustnessReport
# ---------------------------------------------------------------------------


def test_evaluate_returns_report(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    report = evaluator.evaluate(input_ids_list, perturbed_ids_list, label_ids)
    assert isinstance(report, RobustnessReport)


# ---------------------------------------------------------------------------
# Test 11: RobustnessReport has all required fields
# ---------------------------------------------------------------------------


def test_report_has_required_fields(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    report = evaluator.evaluate(input_ids_list, perturbed_ids_list, label_ids)
    assert hasattr(report, "clean_accuracy")
    assert hasattr(report, "attack_success_rate")
    assert hasattr(report, "certified_radius")
    assert hasattr(report, "semantic_preservation")
    assert hasattr(report, "n_samples")


# ---------------------------------------------------------------------------
# Test 12: report.n_samples matches input list length
# ---------------------------------------------------------------------------


def test_report_n_samples(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    report = evaluator.evaluate(input_ids_list, perturbed_ids_list, label_ids)
    assert report.n_samples == len(input_ids_list)


# ---------------------------------------------------------------------------
# Test 13: report.semantic_preservation in [0, 1]
# ---------------------------------------------------------------------------


def test_report_semantic_preservation_range(
    evaluator: RobustnessEvaluator,
    input_ids_list: list,
    perturbed_ids_list: list,
) -> None:
    label_ids = [0] * N_SAMPLES
    report = evaluator.evaluate(input_ids_list, perturbed_ids_list, label_ids)
    assert 0.0 <= report.semantic_preservation <= 1.0

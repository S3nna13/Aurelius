"""Tests for cross-lingual transfer evaluation."""
import pytest
import torch
import torch.nn as nn

from src.eval.crosslingual_transfer import (
    LanguageDetector,
    LanguageResult,
    CrossLingualTransferResult,
    MultilingualConsistencyChecker,
    CrossLingualEvaluator,
    aggregate_transfer_results,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return (None, logits, None)


VOCAB_SIZE = 128

def encode_fn(text: str):
    return [ord(c) % VOCAB_SIZE for c in text] or [0]

def decode_fn(token_id: int) -> str:
    return chr(token_id % 128)

@pytest.fixture
def detector():
    return LanguageDetector()

@pytest.fixture
def model():
    torch.manual_seed(42)
    return MockModel(vocab_size=VOCAB_SIZE, d_model=16)

@pytest.fixture
def evaluator(model):
    return CrossLingualEvaluator(model, encode_fn, decode_fn, source_language="en")

@pytest.fixture
def consistency_checker(model):
    return MultilingualConsistencyChecker(model, encode_fn, decode_fn)

# Simple MC data: 3 prompts, each with 2 choices
def make_mc_data(lang="en", n=3):
    prompts = [f"Question {i} in {lang}?" for i in range(n)]
    labels = [0] * n
    choices = [["Answer A", "Answer B"]] * n
    return prompts, labels, choices


# ---------------------------------------------------------------------------
# LanguageDetector tests
# ---------------------------------------------------------------------------

def test_detect_script_chinese(detector):
    # CJK Unified Ideographs block
    text = "\u4e2d\u6587"  # "中文"
    assert detector.detect_script(text) == "chinese"

def test_detect_script_latin(detector):
    assert detector.detect_script("Hello world") == "latin"

def test_detect_script_empty(detector):
    assert detector.detect_script("") == "unknown"

def test_detect_script_arabic(detector):
    text = "\u0645\u0631\u062d\u0628\u0627"  # "مرحبا"
    assert detector.detect_script(text) == "arabic"

def test_is_target_language_chinese_correct(detector):
    text = "\u4e2d\u6587\u6587\u5b57"
    assert detector.is_target_language(text, "zh") is True

def test_is_target_language_chinese_incorrect(detector):
    assert detector.is_target_language("Hello world", "zh") is False

def test_is_target_language_latin_correct(detector):
    assert detector.is_target_language("Hello world", "en") is True

def test_is_target_language_unknown_lang(detector):
    # Unknown expected language → False
    assert detector.is_target_language("Hello", "xx") is False

def test_detect_language_chinese(detector):
    text = "\u4e2d\u6587"
    assert detector.detect_language(text) == "zh"

def test_detect_language_latin(detector):
    assert detector.detect_language("Hello") == "en"


# ---------------------------------------------------------------------------
# CrossLingualEvaluator tests
# ---------------------------------------------------------------------------

def test_evaluator_instantiates(evaluator):
    assert evaluator.source_language == "en"

def test_evaluate_language_accuracy_in_range(evaluator):
    prompts, labels, choices = make_mc_data("en", n=4)
    result = evaluator.evaluate_language(prompts, labels, choices, language="en")
    assert isinstance(result, LanguageResult)
    assert 0.0 <= result.accuracy <= 1.0

def test_evaluate_language_n_samples(evaluator):
    n = 5
    prompts, labels, choices = make_mc_data("en", n=n)
    result = evaluator.evaluate_language(prompts, labels, choices, language="en")
    assert result.n_samples == n

def test_evaluate_language_confidence_in_range(evaluator):
    prompts, labels, choices = make_mc_data("en", n=3)
    result = evaluator.evaluate_language(prompts, labels, choices, language="en")
    assert 0.0 <= result.avg_confidence <= 1.0

def test_evaluate_transfer_returns_result(evaluator):
    src = make_mc_data("en", n=3)
    tgt = {"fr": make_mc_data("fr", n=3), "de": make_mc_data("de", n=3)}
    result = evaluator.evaluate_transfer(src, tgt)
    assert isinstance(result, CrossLingualTransferResult)

def test_evaluate_transfer_source_language(evaluator):
    src = make_mc_data("en", n=3)
    tgt = {"fr": make_mc_data("fr", n=3)}
    result = evaluator.evaluate_transfer(src, tgt)
    assert result.source_language == "en"

def test_evaluate_transfer_gap_keys(evaluator):
    src = make_mc_data("en", n=3)
    tgt = {"fr": make_mc_data("fr", n=3), "de": make_mc_data("de", n=3)}
    result = evaluator.evaluate_transfer(src, tgt)
    assert set(result.transfer_gaps.keys()) == {"fr", "de"}

def test_evaluate_transfer_avg_gap_is_mean(evaluator):
    src = make_mc_data("en", n=3)
    tgt = {"fr": make_mc_data("fr", n=3), "de": make_mc_data("de", n=3)}
    result = evaluator.evaluate_transfer(src, tgt)
    expected_avg = sum(result.transfer_gaps.values()) / len(result.transfer_gaps)
    assert abs(result.avg_transfer_gap - expected_avg) < 1e-6

def test_compute_transfer_gap(evaluator):
    src = LanguageResult("en", accuracy=0.8, n_samples=10, avg_confidence=0.7, language_confusion_rate=0.0)
    tgt = LanguageResult("fr", accuracy=0.6, n_samples=10, avg_confidence=0.6, language_confusion_rate=0.0)
    gap = evaluator.compute_transfer_gap(src, tgt)
    assert abs(gap - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# MultilingualConsistencyChecker tests
# ---------------------------------------------------------------------------

def test_consistency_checker_returns_responses_key(consistency_checker):
    prompts = {"en": "Hello how are you?", "fr": "Bonjour comment allez-vous?"}
    result = consistency_checker.check_consistency(prompts, max_new_tokens=5)
    assert "responses" in result

def test_consistency_checker_responses_has_all_langs(consistency_checker):
    langs = {"en": "Hello", "fr": "Bonjour", "de": "Hallo"}
    result = consistency_checker.check_consistency(langs, max_new_tokens=3)
    assert set(result["responses"].keys()) == set(langs.keys())

def test_consistency_checker_language_scores_key(consistency_checker):
    prompts = {"en": "Hello world"}
    result = consistency_checker.check_consistency(prompts, max_new_tokens=3)
    assert "language_scores" in result

def test_consistency_checker_semantic_consistency_in_range(consistency_checker):
    prompts = {"en": "Hello", "fr": "Bonjour"}
    result = consistency_checker.check_consistency(prompts, max_new_tokens=5)
    assert 0.0 <= result["semantic_consistency"] <= 1.0


# ---------------------------------------------------------------------------
# aggregate_transfer_results tests
# ---------------------------------------------------------------------------

def test_aggregate_returns_mean_transfer_gap_key():
    r1 = CrossLingualTransferResult(
        source_language="en",
        source_accuracy=0.8,
        target_results={},
        transfer_gaps={"fr": 0.2, "de": 0.1},
        avg_transfer_gap=0.15,
        consistency_scores={"fr": 0.9},
    )
    r2 = CrossLingualTransferResult(
        source_language="en",
        source_accuracy=0.7,
        target_results={},
        transfer_gaps={"fr": 0.3, "de": 0.2},
        avg_transfer_gap=0.25,
        consistency_scores={"fr": 0.8},
    )
    agg = aggregate_transfer_results([r1, r2])
    assert "mean_transfer_gap" in agg

def test_aggregate_mean_transfer_gap_correct():
    r1 = CrossLingualTransferResult("en", 0.8, {}, {}, avg_transfer_gap=0.2, consistency_scores={})
    r2 = CrossLingualTransferResult("en", 0.7, {}, {}, avg_transfer_gap=0.4, consistency_scores={})
    agg = aggregate_transfer_results([r1, r2])
    assert abs(agg["mean_transfer_gap"] - 0.3) < 1e-6

def test_aggregate_empty_list():
    agg = aggregate_transfer_results([])
    assert "mean_transfer_gap" in agg
    assert agg["mean_transfer_gap"] == 0.0

def test_aggregate_has_std_key():
    r = CrossLingualTransferResult("en", 0.8, {}, {}, avg_transfer_gap=0.1, consistency_scores={})
    agg = aggregate_transfer_results([r])
    assert "std_transfer_gap" in agg

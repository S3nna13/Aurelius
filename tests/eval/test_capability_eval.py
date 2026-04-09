"""Tests for the capability evaluation harness."""

import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.capability_eval import (
    CapabilityConfig,
    EvalExample,
    TaskResult,
    GreedyGenerator,
    MultipleChoiceEvaluator,
    ExactMatchScorer,
    CapabilityEvaluator,
    create_synthetic_examples,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer_encode():
    return lambda t: list(t.encode("utf-8")[:32])


@pytest.fixture(scope="module")
def tokenizer_decode():
    return lambda ids: bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. CapabilityConfig defaults
# ---------------------------------------------------------------------------

def test_capability_config_defaults():
    cfg = CapabilityConfig()
    assert cfg.n_shots == 0
    assert cfg.max_new_tokens == 32
    assert cfg.temperature == 0.0
    assert cfg.batch_size == 4
    assert cfg.eval_tasks == ["reasoning", "knowledge", "generation", "math"]


# ---------------------------------------------------------------------------
# 2. EvalExample fields
# ---------------------------------------------------------------------------

def test_eval_example_fields():
    ex = EvalExample(
        task="reasoning",
        prompt="Does A imply C?",
        gold_answer="Yes",
        choices=["Yes", "No"],
        metadata={"difficulty": "easy"},
    )
    assert ex.task == "reasoning"
    assert ex.prompt == "Does A imply C?"
    assert ex.gold_answer == "Yes"
    assert ex.choices == ["Yes", "No"]
    assert ex.metadata == {"difficulty": "easy"}

    # Open-ended example
    ex2 = EvalExample(task="generation", prompt="The sky is", gold_answer="blue")
    assert ex2.choices is None
    assert ex2.metadata is None


# ---------------------------------------------------------------------------
# 3. TaskResult fields
# ---------------------------------------------------------------------------

def test_task_result_fields():
    result = TaskResult(
        task="math",
        n_examples=10,
        accuracy=0.8,
        mean_score=0.8,
        scores=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    )
    assert result.task == "math"
    assert result.n_examples == 10
    assert result.accuracy == 0.8
    assert result.mean_score == 0.8
    assert len(result.scores) == 10


# ---------------------------------------------------------------------------
# 4. GreedyGenerator returns string
# ---------------------------------------------------------------------------

def test_greedy_generator_returns_string(small_model, tokenizer_encode, tokenizer_decode):
    gen = GreedyGenerator(small_model, tokenizer_encode, tokenizer_decode)
    result = gen.generate("hello", max_new_tokens=5)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 5. GreedyGenerator batch length
# ---------------------------------------------------------------------------

def test_greedy_generator_batch_length(small_model, tokenizer_encode, tokenizer_decode):
    gen = GreedyGenerator(small_model, tokenizer_encode, tokenizer_decode)
    prompts = ["hello", "world", "foo"]
    results = gen.generate_batch(prompts, max_new_tokens=4)
    assert isinstance(results, list)
    assert len(results) == len(prompts)
    for r in results:
        assert isinstance(r, str)


# ---------------------------------------------------------------------------
# 6. MultipleChoiceEvaluator score is float
# ---------------------------------------------------------------------------

def test_mc_evaluator_score_is_float(small_model, tokenizer_encode):
    ev = MultipleChoiceEvaluator(small_model, tokenizer_encode)
    score = ev.score_choice("The capital of France is", "Paris")
    assert isinstance(score, float)
    import math
    assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 7. MultipleChoiceEvaluator predict returns valid index
# ---------------------------------------------------------------------------

def test_mc_evaluator_predict_valid_index(small_model, tokenizer_encode):
    ev = MultipleChoiceEvaluator(small_model, tokenizer_encode)
    choices = ["London", "Paris", "Berlin"]
    idx = ev.predict("The capital of France is", choices)
    assert isinstance(idx, int)
    assert 0 <= idx < len(choices)


# ---------------------------------------------------------------------------
# 8. ExactMatchScorer identical -> 1.0
# ---------------------------------------------------------------------------

def test_exact_match_scorer_identical():
    scorer = ExactMatchScorer()
    assert scorer.score("Paris", "Paris") == 1.0
    assert scorer.score("paris!", "Paris") == 1.0
    assert scorer.score("  yes  ", "Yes") == 1.0


# ---------------------------------------------------------------------------
# 9. ExactMatchScorer different -> 0.0
# ---------------------------------------------------------------------------

def test_exact_match_scorer_different():
    scorer = ExactMatchScorer()
    assert scorer.score("London", "Paris") == 0.0
    assert scorer.score("no", "yes") == 0.0


# ---------------------------------------------------------------------------
# 10. CapabilityEvaluator.evaluate_example
# ---------------------------------------------------------------------------

def test_capability_evaluator_evaluate_example(small_model, tokenizer_encode, tokenizer_decode):
    cfg = CapabilityConfig(max_new_tokens=4)
    evaluator = CapabilityEvaluator(small_model, cfg, tokenizer_encode, tokenizer_decode)

    # Multiple-choice example
    mc_example = EvalExample(
        task="math",
        prompt="2+2=?",
        gold_answer="4",
        choices=["3", "4", "5"],
    )
    score = evaluator.evaluate_example(mc_example)
    assert score in (0.0, 1.0)

    # Open-ended example
    oe_example = EvalExample(
        task="generation",
        prompt="The sky is",
        gold_answer="blue",
    )
    score_oe = evaluator.evaluate_example(oe_example)
    assert score_oe in (0.0, 1.0)


# ---------------------------------------------------------------------------
# 11. CapabilityEvaluator.summary_report keys
# ---------------------------------------------------------------------------

def test_capability_evaluator_summary_report_keys(small_model, tokenizer_encode, tokenizer_decode):
    cfg = CapabilityConfig(max_new_tokens=4)
    evaluator = CapabilityEvaluator(small_model, cfg, tokenizer_encode, tokenizer_decode)

    examples = [
        EvalExample(task="math", prompt="2+2=?", gold_answer="4", choices=["3", "4", "5"]),
        EvalExample(task="reasoning", prompt="A implies B and B implies C, so A implies C?", gold_answer="Yes", choices=["Yes", "No"]),
    ]
    results = evaluator.evaluate_all(examples)
    report = evaluator.summary_report(results)

    assert "overall_accuracy" in report
    assert "n_tasks" in report
    assert report["n_tasks"] == 2
    for task in results:
        assert task in report
    assert 0.0 <= report["overall_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 12. create_synthetic_examples count
# ---------------------------------------------------------------------------

def test_create_synthetic_examples_count():
    n = 5
    examples = create_synthetic_examples(n_per_task=n, seed=42)
    assert len(examples) == 4 * n

    tasks = [ex.task for ex in examples]
    for task in ["reasoning", "knowledge", "generation", "math"]:
        assert tasks.count(task) == n

"""Tests for src/eval/few_shot_eval.py.

Uses only stdlib + PyTorch; no HuggingFace, scipy, or sklearn.
All model/tokenizer fixtures are tiny inline stubs.
"""
from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn.functional as F

from src.eval.few_shot_eval import (
    FewShotConfig,
    FewShotEvaluator,
    build_few_shot_prompt,
    compute_accuracy,
    compute_per_class_accuracy,
    format_example,
    score_multiple_choice,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32  # small vocabulary for tests


def _make_examples(n: int = 6, n_choices: int = 4) -> List[dict]:
    """Return a list of toy multiple-choice examples."""
    choices = [f"Choice_{c}" for c in range(n_choices)]
    return [
        {
            "question": f"What is item {i}?",
            "choices": choices[:],
            "answer_idx": i % n_choices,
        }
        for i in range(n)
    ]


def _dummy_tokenizer(text: str) -> List[int]:
    """Character-based tokenizer: each char → its ASCII value (mod VOCAB_SIZE)."""
    return [ord(c) % VOCAB_SIZE for c in text] or [0]


def _dummy_model(token_ids: torch.Tensor) -> torch.Tensor:
    """Deterministic model: returns uniform logits shaped (T, VOCAB_SIZE)."""
    T = token_ids.shape[0]
    return torch.zeros(T, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 1. FewShotConfig defaults
# ---------------------------------------------------------------------------

def test_fewshotconfig_defaults():
    cfg = FewShotConfig()
    assert cfg.n_shots == 5
    assert cfg.max_seq_len == 2048
    assert cfg.answer_prefix == "Answer:"
    assert cfg.separator == "\n\n"
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2. format_example with answer includes letter
# ---------------------------------------------------------------------------

def test_format_example_with_answer_includes_letter():
    cfg = FewShotConfig()
    result = format_example("What is 2+2?", ["3", "4", "5"], answer_idx=1, config=cfg)
    # Correct answer is index 1 → label 'B'
    assert "Answer: B" in result


# ---------------------------------------------------------------------------
# 3. format_example without answer omits letter
# ---------------------------------------------------------------------------

def test_format_example_without_answer_omits_letter():
    cfg = FewShotConfig()
    result = format_example("What is 2+2?", ["3", "4", "5"], answer_idx=None, config=cfg)
    # Prefix present but no letter after it
    assert "Answer:" in result
    # The line containing "Answer:" should not end with a letter
    for line in result.splitlines():
        if line.startswith("Answer:"):
            stripped = line[len("Answer:"):].strip()
            assert stripped == "", f"Expected no letter after prefix, got: {stripped!r}"


# ---------------------------------------------------------------------------
# 4. format_example choices use A/B/C/D labels
# ---------------------------------------------------------------------------

def test_format_example_choice_labels():
    choices = ["Alpha", "Beta", "Gamma", "Delta"]
    result = format_example("Pick one.", choices, config=FewShotConfig())
    assert "A. Alpha" in result
    assert "B. Beta" in result
    assert "C. Gamma" in result
    assert "D. Delta" in result


# ---------------------------------------------------------------------------
# 5. build_few_shot_prompt contains separator between examples
# ---------------------------------------------------------------------------

def test_build_few_shot_prompt_contains_separator():
    cfg = FewShotConfig(n_shots=2, separator="\n\n")
    pool = _make_examples(5)
    test_ex = pool[-1]
    prompt = build_few_shot_prompt(pool[:4], test_ex, cfg)
    assert "\n\n" in prompt


# ---------------------------------------------------------------------------
# 6. build_few_shot_prompt last example has no answer letter
# ---------------------------------------------------------------------------

def test_build_few_shot_prompt_last_example_no_answer():
    cfg = FewShotConfig(n_shots=2, separator="\n\n")
    pool = _make_examples(5)
    test_ex = {"question": "Final?", "choices": ["Yes", "No"], "answer_idx": 0}
    prompt = build_few_shot_prompt(pool[:3], test_ex, cfg)
    # Split on separator; the last block must end with "Answer:" (no letter)
    last_block = prompt.split(cfg.separator)[-1]
    answer_lines = [l for l in last_block.splitlines() if l.startswith("Answer:")]
    assert len(answer_lines) == 1
    letter_part = answer_lines[0][len("Answer:"):].strip()
    assert letter_part == ""


# ---------------------------------------------------------------------------
# 7. build_few_shot_prompt uses n_shots examples (+ 1 test block)
# ---------------------------------------------------------------------------

def test_build_few_shot_prompt_uses_n_shots():
    n_shots = 3
    cfg = FewShotConfig(n_shots=n_shots, separator="\n\n")
    pool = _make_examples(8)
    test_ex = pool[-1]
    prompt = build_few_shot_prompt(pool[:-1], test_ex, cfg)
    # Total blocks = n_shots + 1 (test)
    blocks = prompt.split(cfg.separator)
    assert len(blocks) == n_shots + 1


# ---------------------------------------------------------------------------
# 8. score_multiple_choice returns index of max logprob choice
# ---------------------------------------------------------------------------

def test_score_multiple_choice_returns_max():
    vocab = 16
    logprobs = torch.full((vocab,), -10.0)
    choice_indices = [2, 5, 9]
    # Make index 5 the winner
    logprobs[5] = 0.0
    result = score_multiple_choice(logprobs, choice_indices)
    # Index of 5 in choice_indices is 1
    assert result == 1


# ---------------------------------------------------------------------------
# 9. score_multiple_choice returns valid index (in range)
# ---------------------------------------------------------------------------

def test_score_multiple_choice_valid_index():
    vocab = 20
    logprobs = torch.randn(vocab)
    choice_indices = [1, 3, 7, 11]
    result = score_multiple_choice(logprobs, choice_indices)
    assert 0 <= result < len(choice_indices)


# ---------------------------------------------------------------------------
# 10. compute_accuracy = 1.0 when all correct
# ---------------------------------------------------------------------------

def test_compute_accuracy_all_correct():
    preds = [0, 1, 2, 3]
    targets = [0, 1, 2, 3]
    assert compute_accuracy(preds, targets) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. compute_accuracy = 0.0 when all wrong
# ---------------------------------------------------------------------------

def test_compute_accuracy_all_wrong():
    preds = [1, 2, 3, 0]
    targets = [0, 1, 2, 3]
    assert compute_accuracy(preds, targets) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 12. compute_per_class_accuracy length == n_classes
# ---------------------------------------------------------------------------

def test_compute_per_class_accuracy_length():
    preds = [0, 1, 2, 0, 1]
    targets = [0, 1, 2, 1, 0]
    n_classes = 4
    result = compute_per_class_accuracy(preds, targets, n_classes)
    assert len(result) == n_classes


# ---------------------------------------------------------------------------
# 13. compute_per_class_accuracy correctness
# ---------------------------------------------------------------------------

def test_compute_per_class_accuracy_values():
    # Class 0: pred=0, target=0 → correct; Class 1: pred=0, target=1 → wrong
    preds = [0, 0]
    targets = [0, 1]
    result = compute_per_class_accuracy(preds, targets, n_classes=2)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 14. FewShotEvaluator.evaluate_example returns valid choice index
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_example_valid_index():
    cfg = FewShotConfig(n_shots=2)
    evaluator = FewShotEvaluator(_dummy_model, _dummy_tokenizer, cfg)
    choices = ["A choice", "B choice", "C choice"]
    prompt = "Question: Foo?\nChoices:\nA. A choice\nB. B choice\nC. C choice\nAnswer:"
    result = evaluator.evaluate_example(prompt, choices)
    assert 0 <= result < len(choices)


# ---------------------------------------------------------------------------
# 15. FewShotEvaluator.evaluate_example: biased model picks correct class
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_example_biased_model():
    """A model that strongly favors token id for 'B' should predict index 1."""
    cfg = FewShotConfig(n_shots=1)

    # Find token id for 'B' under our dummy tokenizer
    b_token = ord("B") % VOCAB_SIZE

    def biased_model(token_ids: torch.Tensor) -> torch.Tensor:
        T = token_ids.shape[0]
        logits = torch.full((T, VOCAB_SIZE), -100.0)
        logits[:, b_token] = 100.0
        return logits

    evaluator = FewShotEvaluator(biased_model, _dummy_tokenizer, cfg)
    choices = ["Alpha", "Beta", "Gamma"]
    # first token of "Alpha" → ord('A') % 32 = 65 % 32 = 1
    # first token of "Beta"  → ord('B') % 32 = 66 % 32 = 2
    # first token of "Gamma" → ord('G') % 32 = 71 % 32 = 7
    prompt = "Question: Test?\nChoices:\nA. Alpha\nB. Beta\nC. Gamma\nAnswer:"
    result = evaluator.evaluate_example(prompt, choices)
    # biased_model strongly favors b_token = 2, which maps to "Beta" → index 1
    assert result == 1


# ---------------------------------------------------------------------------
# 16. FewShotEvaluator.evaluate_dataset returns dict with required keys
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_dataset_keys():
    cfg = FewShotConfig(n_shots=2)
    evaluator = FewShotEvaluator(_dummy_model, _dummy_tokenizer, cfg)
    examples = _make_examples(4)
    pool = _make_examples(3)
    result = evaluator.evaluate_dataset(examples, pool)
    assert "accuracy" in result
    assert "n_correct" in result
    assert "n_total" in result
    assert "per_class" in result


# ---------------------------------------------------------------------------
# 17. FewShotEvaluator.evaluate_dataset n_total matches input length
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_dataset_n_total():
    cfg = FewShotConfig(n_shots=2)
    evaluator = FewShotEvaluator(_dummy_model, _dummy_tokenizer, cfg)
    examples = _make_examples(5)
    pool = _make_examples(4)
    result = evaluator.evaluate_dataset(examples, pool)
    assert result["n_total"] == 5


# ---------------------------------------------------------------------------
# 18. FewShotEvaluator.evaluate_dataset accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_evaluator_evaluate_dataset_accuracy_range():
    cfg = FewShotConfig(n_shots=1)
    evaluator = FewShotEvaluator(_dummy_model, _dummy_tokenizer, cfg)
    examples = _make_examples(4)
    pool = _make_examples(2)
    result = evaluator.evaluate_dataset(examples, pool)
    assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# 19. compute_per_class_accuracy with zero-example class returns 0.0
# ---------------------------------------------------------------------------

def test_compute_per_class_accuracy_missing_class():
    # Class 2 has no examples
    preds = [0, 1, 0, 1]
    targets = [0, 1, 0, 1]
    result = compute_per_class_accuracy(preds, targets, n_classes=3)
    assert result[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 20. build_few_shot_prompt: question text appears in prompt
# ---------------------------------------------------------------------------

def test_build_few_shot_prompt_question_text():
    cfg = FewShotConfig(n_shots=1)
    pool = [{"question": "Capital of France?", "choices": ["Paris", "London"], "answer_idx": 0}]
    test_ex = {"question": "Capital of Germany?", "choices": ["Berlin", "Rome"], "answer_idx": 0}
    prompt = build_few_shot_prompt(pool, test_ex, cfg)
    assert "Capital of France?" in prompt
    assert "Capital of Germany?" in prompt

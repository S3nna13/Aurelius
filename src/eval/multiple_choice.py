"""Native multiple-choice benchmark via log-probability scoring.

Evaluates LLMs on multiple-choice benchmarks (ARC, HellaSwag style) by
scoring each answer option with the model's log-probability and picking
the highest-scoring option.

Scoring: log_prob(prompt + answer) normalized by answer token count.
Normalization prevents bias toward short answer choices.

No external benchmark dependencies required — just pass question/choice dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultipleChoiceItem:
    """A single multiple-choice benchmark item."""

    question: str  # The question/context/premise
    choices: list[str]  # List of answer choices (A, B, C, D...)
    correct_idx: int  # Index of the correct answer (0-based)
    metadata: dict = field(default_factory=dict)  # Optional metadata


@dataclass
class EvalResult:
    """Result of scoring one multiple-choice item."""

    item: MultipleChoiceItem
    predicted_idx: int
    correct: bool
    scores: list[float]  # Log-prob score for each choice

    @property
    def predicted_choice(self) -> str:
        return self.item.choices[self.predicted_idx]

    @property
    def correct_choice(self) -> str:
        return self.item.choices[self.item.correct_idx]


@dataclass
class BenchmarkResult:
    """Aggregated results for a benchmark."""

    name: str
    accuracy: float
    n_correct: int
    n_total: int
    results: list[EvalResult]

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(name={self.name!r}, "
            f"accuracy={self.accuracy:.1%}, "
            f"{self.n_correct}/{self.n_total})"
        )


@torch.no_grad()
def score_completion(
    model: nn.Module,
    tokenizer,
    prefix: str,
    completion: str,
    normalize: bool = True,
) -> float:
    """Score a completion given a prefix using log-probability.

    Computes the total log-probability of `completion` tokens given `prefix`.
    Normalizes by the number of completion tokens to prevent length bias.

    Args:
        model: AureliusTransformer.
        tokenizer: Tokenizer with encode(str) -> list[int].
        prefix: The question/context text.
        completion: The answer choice text.
        normalize: If True, divide total log-prob by completion token count.

    Returns:
        Log-probability score (higher = more likely under the model).
    """
    model.eval()
    device = next(model.parameters()).device
    max_seq = model.config.max_seq_len

    # Encode full sequence
    prefix_ids = tokenizer.encode(prefix)
    completion_ids = tokenizer.encode(completion)

    if not completion_ids:
        return float("-inf")

    full_ids = prefix_ids + completion_ids

    # Truncate from the left if too long
    if len(full_ids) > max_seq:
        full_ids = full_ids[-(max_seq):]
        # Recalculate how many completion tokens remain
        total = len(full_ids)
        n_completion = min(len(completion_ids), total - 1)
    else:
        n_completion = len(completion_ids)

    if n_completion == 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Forward pass — get logits
    _, logits, _ = model(input_ids)

    # Score only the completion tokens
    # logits[:, i] predicts full_ids[i+1]
    # Completion starts at index (total_len - n_completion) in full_ids
    total_len = len(full_ids)
    completion_start = total_len - n_completion

    log_probs = F.log_softmax(logits[0, :-1], dim=-1)  # (total_len-1, vocab)
    targets = input_ids[0, 1:]  # (total_len-1,)

    # Gather log-probs for each target token
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (total_len-1,)

    # Sum log-probs for completion tokens (indices completion_start-1 to total_len-2)
    completion_log_probs = token_log_probs[completion_start - 1 :]  # n_completion values
    total_log_prob = completion_log_probs.sum().item()

    if normalize:
        return total_log_prob / n_completion
    return total_log_prob


def score_item(
    model: nn.Module,
    tokenizer,
    item: MultipleChoiceItem,
    normalize: bool = True,
) -> EvalResult:
    """Score one multiple-choice item.

    Args:
        model: AureliusTransformer.
        tokenizer: Tokenizer.
        item: MultipleChoiceItem with question and choices.
        normalize: Normalize log-prob by answer length.

    Returns:
        EvalResult with predicted answer and correctness.
    """
    scores = []
    for choice in item.choices:
        score = score_completion(model, tokenizer, item.question, choice, normalize)
        scores.append(score)

    predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
    correct = predicted_idx == item.correct_idx

    return EvalResult(
        item=item,
        predicted_idx=predicted_idx,
        correct=correct,
        scores=scores,
    )


# Public alias used by tests
evaluate_item = score_item


def run_benchmark(
    model: nn.Module,
    tokenizer,
    items: list[MultipleChoiceItem],
    name: str = "benchmark",
    normalize: bool = True,
) -> BenchmarkResult:
    """Run a full multiple-choice benchmark.

    Args:
        model: AureliusTransformer.
        tokenizer: Tokenizer.
        items: List of MultipleChoiceItem.
        name: Benchmark name for the result.
        normalize: Normalize log-prob by answer length.

    Returns:
        BenchmarkResult with accuracy and per-item results.
    """
    results = []
    for i, item in enumerate(items):
        result = score_item(model, tokenizer, item, normalize)
        results.append(result)

        if (i + 1) % 10 == 0:
            current_acc = sum(r.correct for r in results) / len(results)
            logger.info("Progress: %d/%d  acc=%.1f%%", i + 1, len(items), current_acc * 100)

    n_correct = sum(r.correct for r in results)
    accuracy = n_correct / max(1, len(results))

    logger.info("%s: %.1f%% (%d/%d)", name, accuracy * 100, n_correct, len(results))

    return BenchmarkResult(
        name=name,
        accuracy=accuracy,
        n_correct=n_correct,
        n_total=len(results),
        results=results,
    )


# ---------------------------------------------------------------------------
# Built-in toy benchmark for smoke testing
# ---------------------------------------------------------------------------

AURELIUS_SANITY_BENCHMARK: list[MultipleChoiceItem] = [
    MultipleChoiceItem(
        question="The capital of France is",
        choices=["London", "Paris", "Berlin", "Rome"],
        correct_idx=1,
    ),
    MultipleChoiceItem(
        question="Water freezes at",
        choices=[
            "100 degrees Celsius",
            "0 degrees Celsius",
            "50 degrees Celsius",
            "-100 degrees Celsius",
        ],
        correct_idx=1,
    ),
    MultipleChoiceItem(
        question="The sun rises in the",
        choices=["West", "North", "East", "South"],
        correct_idx=2,
    ),
    MultipleChoiceItem(
        question="Two plus two equals",
        choices=["3", "4", "5", "6"],
        correct_idx=1,
    ),
]

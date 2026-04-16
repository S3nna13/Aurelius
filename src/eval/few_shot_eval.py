"""Few-shot evaluation for classification / multiple-choice tasks.

Formats prompts with k in-context examples and evaluates model accuracy by
scoring the first token of each answer choice via log-probabilities at the
final prompt position.

No external dependencies beyond PyTorch.
"""
from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FewShotConfig:
    """Configuration for few-shot prompt formatting and evaluation."""
    n_shots: int = 5
    max_seq_len: int = 2048
    answer_prefix: str = "Answer:"
    separator: str = "\n\n"
    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------

def _choice_label(idx: int) -> str:
    """Return alphabetic label for a choice index (0 → 'A', 1 → 'B', …)."""
    return string.ascii_uppercase[idx]


def format_example(
    question: str,
    choices: List[str],
    answer_idx: Optional[int] = None,
    config: Optional[FewShotConfig] = None,
) -> str:
    """Format a single question-choices block.

    If *answer_idx* is given the answer letter is appended after the prefix;
    otherwise the prefix is included but the letter is omitted (leaving the
    model to complete the prompt).

    Args:
        question:   The question text.
        choices:    List of answer-choice strings.
        answer_idx: 0-based index of the correct choice, or None for the test
                    example where the answer is unknown.
        config:     FewShotConfig (uses default if not supplied).

    Returns:
        Formatted string block.
    """
    if config is None:
        config = FewShotConfig()

    lines: List[str] = [f"Question: {question}", "Choices:"]
    for i, choice in enumerate(choices):
        lines.append(f"{_choice_label(i)}. {choice}")

    if answer_idx is not None:
        lines.append(f"{config.answer_prefix} {_choice_label(answer_idx)}")
    else:
        lines.append(config.answer_prefix)

    return "\n".join(lines)


def build_few_shot_prompt(
    examples: List[Dict],
    test_example: Dict,
    config: FewShotConfig,
) -> str:
    """Build a complete few-shot prompt.

    Selects the first *config.n_shots* items from *examples* as in-context
    demonstrations (each dict must have "question", "choices", "answer_idx"),
    then appends *test_example* without an answer letter.

    Args:
        examples:     Pool of labeled examples (dicts with question/choices/answer_idx).
        test_example: The example to evaluate (same keys, answer_idx unused).
        config:       FewShotConfig controlling separator and n_shots.

    Returns:
        Full prompt string.
    """
    shots = examples[: config.n_shots]
    parts: List[str] = []
    for ex in shots:
        parts.append(
            format_example(
                ex["question"],
                ex["choices"],
                answer_idx=ex["answer_idx"],
                config=config,
            )
        )

    # Append test example without the answer letter
    parts.append(
        format_example(
            test_example["question"],
            test_example["choices"],
            answer_idx=None,
            config=config,
        )
    )

    return config.separator.join(parts)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def score_multiple_choice(logprobs: torch.Tensor, choice_indices: List[int]) -> int:
    """Choose the answer option whose first token has the highest log-prob.

    Args:
        logprobs:       1-D tensor of shape (vocab_size,) — log-probs at the
                        answer position.
        choice_indices: Vocabulary indices corresponding to the first token of
                        each answer choice (e.g. token id for 'A', 'B', …).

    Returns:
        0-based index into *choice_indices* (not the vocab index) of the
        highest-scoring choice.
    """
    scores = logprobs[torch.tensor(choice_indices, dtype=torch.long)]
    return int(scores.argmax().item())


def compute_accuracy(predictions: List[int], targets: List[int]) -> float:
    """Fraction of predictions that match their target.

    Args:
        predictions: Predicted class indices.
        targets:     Ground-truth class indices.

    Returns:
        Float in [0, 1].
    """
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions)


def compute_per_class_accuracy(
    predictions: List[int],
    targets: List[int],
    n_classes: int,
) -> List[float]:
    """Per-class accuracy.

    Args:
        predictions: Predicted class indices.
        targets:     Ground-truth class indices.
        n_classes:   Total number of classes (determines output list length).

    Returns:
        List of length *n_classes*; each entry is the accuracy for that class
        (0.0 if the class has no examples).
    """
    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    for p, t in zip(predictions, targets):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    return [
        class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        for c in range(n_classes)
    ]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FewShotEvaluator:
    """Evaluates a language model on multiple-choice tasks using few-shot prompts.

    Args:
        model_fn:      Callable that takes a 1-D LongTensor of token ids and
                       returns a 2-D FloatTensor of shape (T, vocab_size).
        tokenizer_fn:  Callable that takes a string and returns a List[int] of
                       token ids.
        config:        FewShotConfig.
    """

    def __init__(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        tokenizer_fn: Callable[[str], List[int]],
        config: FewShotConfig,
    ) -> None:
        self.model_fn = model_fn
        self.tokenizer_fn = tokenizer_fn
        self.config = config

    # ------------------------------------------------------------------

    def evaluate_example(self, prompt: str, choices: List[str]) -> int:
        """Predict the answer choice for a single formatted prompt.

        Tokenizes *prompt*, truncates to *max_seq_len*, runs the model, takes
        the log-softmax over the final token position, then scores the first
        token of each choice string.

        Args:
            prompt:  The full few-shot prompt (ending with the answer prefix).
            choices: List of raw choice strings (e.g. ["Paris", "London", …]).

        Returns:
            0-based predicted choice index.
        """
        token_ids = self.tokenizer_fn(prompt)
        if len(token_ids) > self.config.max_seq_len:
            token_ids = token_ids[-self.config.max_seq_len :]

        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        with torch.no_grad():
            logits = self.model_fn(input_tensor)  # (T, vocab)

        logprobs = F.log_softmax(logits[-1], dim=-1)  # (vocab,)

        # Obtain the first token id of each choice string
        choice_indices: List[int] = []
        for choice in choices:
            tokens = self.tokenizer_fn(choice)
            choice_indices.append(tokens[0] if tokens else 0)

        return score_multiple_choice(logprobs, choice_indices)

    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        examples: List[Dict],
        few_shot_pool: List[Dict],
    ) -> Dict:
        """Evaluate every example in *examples* using *few_shot_pool* for context.

        Each item in *examples* must have "question", "choices", and "answer_idx".
        The few-shot demonstrations are drawn from *few_shot_pool*.

        Args:
            examples:      Test examples to evaluate.
            few_shot_pool: Pool of labeled examples used as in-context shots.

        Returns:
            Dict with keys:
              - "accuracy"  (float)
              - "n_correct" (int)
              - "n_total"   (int)
              - "per_class" (List[float])
        """
        predictions: List[int] = []
        targets: List[int] = []

        n_classes = 0

        for ex in examples:
            prompt = build_few_shot_prompt(few_shot_pool, ex, self.config)
            pred = self.evaluate_example(prompt, ex["choices"])
            predictions.append(pred)
            targets.append(ex["answer_idx"])
            n_classes = max(n_classes, len(ex["choices"]))

        accuracy = compute_accuracy(predictions, targets)
        n_correct = sum(p == t for p, t in zip(predictions, targets))
        per_class = compute_per_class_accuracy(predictions, targets, n_classes)

        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": len(examples),
            "per_class": per_class,
        }

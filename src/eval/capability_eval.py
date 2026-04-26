"""Self-evaluation harness: measure model capabilities across reasoning, knowledge, and generation tasks."""  # noqa: E501

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CapabilityConfig:
    """Configuration for the capability evaluation harness."""

    n_shots: int = 0
    max_new_tokens: int = 32
    temperature: float = 0.0  # 0 = greedy
    batch_size: int = 4
    eval_tasks: list[str] = field(
        default_factory=lambda: ["reasoning", "knowledge", "generation", "math"]
    )


@dataclass
class EvalExample:
    """A single evaluation example."""

    task: str
    prompt: str
    gold_answer: str
    choices: list[str] | None = None  # None = open-ended
    metadata: dict | None = None


@dataclass
class TaskResult:
    """Aggregated result for one task."""

    task: str
    n_examples: int
    accuracy: float
    mean_score: float
    scores: list[float]


class GreedyGenerator:
    """Simple greedy token generator."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Greedy decode up to max_new_tokens. Returns generated text (excluding prompt)."""
        input_ids = self.tokenizer_encode(prompt)
        ids = list(input_ids)
        generated = []

        for _ in range(max_new_tokens):
            x = torch.tensor([ids], dtype=torch.long)
            _, logits, _ = self.model(x)
            # logits: (1, seq_len, vocab_size) — pick last position
            next_token = int(logits[0, -1, :].argmax().item())
            generated.append(next_token)
            ids.append(next_token)

        return self.tokenizer_decode(generated)

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 32) -> list[str]:
        """Sequential generation for each prompt."""
        return [self.generate(p, max_new_tokens) for p in prompts]


class MultipleChoiceEvaluator:
    """Score answers for multiple-choice questions using log-likelihood."""

    def __init__(self, model: nn.Module, tokenizer_encode: Callable) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode

    @torch.no_grad()
    def score_choice(self, prompt: str, choice: str) -> float:
        """Compute mean log-likelihood of choice tokens given prompt. Higher = more likely."""
        prompt_ids = self.tokenizer_encode(prompt)
        choice_ids = self.tokenizer_encode(choice)

        if not choice_ids:
            return float("-inf")

        # Full sequence: prompt + choice
        full_ids = list(prompt_ids) + list(choice_ids)
        input_tensor = torch.tensor([full_ids], dtype=torch.long)

        _, logits, _ = self.model(input_tensor)
        # logits: (1, seq_len, vocab_size)
        # For each choice token at position p+i, the logit at position p+i-1 predicts it
        prompt_len = len(prompt_ids)

        log_probs = F.log_softmax(logits[0], dim=-1)  # (seq_len, vocab_size)

        total_log_prob = 0.0
        for i, tok in enumerate(choice_ids):
            pos = prompt_len + i - 1  # logit position that predicts this token
            if pos < 0:
                # No context for the first choice token if prompt is empty
                pos = 0
            total_log_prob += log_probs[pos, tok].item()

        return total_log_prob / len(choice_ids)

    def predict(self, prompt: str, choices: list[str]) -> int:
        """Returns index of highest-scoring choice."""
        scores = [self.score_choice(prompt, c) for c in choices]
        return int(max(range(len(scores)), key=lambda i: scores[i]))


class ExactMatchScorer:
    """Normalize and compare strings for exact match."""

    def normalize(self, text: str) -> str:
        """Lowercase, strip punctuation and whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.strip()
        return text

    def score(self, prediction: str, gold: str) -> float:
        """Returns 1.0 if normalized strings match, else 0.0."""
        return 1.0 if self.normalize(prediction) == self.normalize(gold) else 0.0

    def score_batch(self, predictions: list[str], golds: list[str]) -> list[float]:
        return [self.score(p, g) for p, g in zip(predictions, golds)]


class CapabilityEvaluator:
    """Main evaluation coordinator."""

    def __init__(
        self,
        model: nn.Module,
        config: CapabilityConfig,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
    ) -> None:
        self.model = model
        self.config = config
        self.generator = GreedyGenerator(model, tokenizer_encode, tokenizer_decode)
        self.mc_evaluator = MultipleChoiceEvaluator(model, tokenizer_encode)
        self.em_scorer = ExactMatchScorer()

    def evaluate_example(self, example: EvalExample) -> float:
        """Score a single example. Returns 1.0 if correct, else 0.0."""
        if example.choices is not None:
            predicted_idx = self.mc_evaluator.predict(example.prompt, example.choices)
            predicted_answer = example.choices[predicted_idx]
            return (
                1.0
                if self.em_scorer.normalize(predicted_answer)
                == self.em_scorer.normalize(example.gold_answer)
                else 0.0
            )
        else:
            prediction = self.generator.generate(example.prompt, self.config.max_new_tokens)
            return self.em_scorer.score(prediction, example.gold_answer)

    def evaluate_task(self, examples: list[EvalExample]) -> TaskResult:
        """Evaluate all examples in a task, compute accuracy + mean_score."""
        if not examples:
            task = ""
            return TaskResult(task=task, n_examples=0, accuracy=0.0, mean_score=0.0, scores=[])

        task = examples[0].task
        scores = [self.evaluate_example(ex) for ex in examples]
        accuracy = sum(scores) / len(scores)
        mean_score = accuracy  # same for binary scores

        return TaskResult(
            task=task,
            n_examples=len(examples),
            accuracy=accuracy,
            mean_score=mean_score,
            scores=scores,
        )

    def evaluate_all(self, examples: list[EvalExample]) -> dict[str, TaskResult]:
        """Group by task, evaluate each, return {task: TaskResult}."""
        grouped: dict[str, list[EvalExample]] = {}
        for ex in examples:
            grouped.setdefault(ex.task, []).append(ex)

        return {task: self.evaluate_task(task_examples) for task, task_examples in grouped.items()}

    def summary_report(self, results: dict[str, TaskResult]) -> dict[str, float]:
        """Returns overall_accuracy, n_tasks, and per-task accuracies."""
        if not results:
            return {"overall_accuracy": 0.0, "n_tasks": 0}

        all_accuracies = [r.accuracy for r in results.values()]
        overall = sum(all_accuracies) / len(all_accuracies)

        report: dict[str, float] = {
            "overall_accuracy": overall,
            "n_tasks": len(results),
        }
        for task, res in results.items():
            report[task] = res.accuracy

        return report


def create_synthetic_examples(n_per_task: int = 5, seed: int = 42) -> list[EvalExample]:
    """Create simple synthetic eval examples for each task."""
    random.Random(seed)
    examples: list[EvalExample] = []

    # --- reasoning ---
    for _ in range(n_per_task):
        examples.append(
            EvalExample(
                task="reasoning",
                prompt="If A implies B and B implies C, does A imply C?",
                gold_answer="Yes",
                choices=["Yes", "No"],
            )
        )

    # --- knowledge ---
    knowledge_items = [
        ("What is the capital of France?", "Paris", ["London", "Berlin", "Paris", "Rome"]),
        ("What planet is closest to the Sun?", "Mercury", ["Venus", "Mercury", "Earth", "Mars"]),
        ("How many sides does a triangle have?", "3", ["2", "3", "4", "5"]),
        ("What is the chemical symbol for water?", "H2O", ["CO2", "H2O", "O2", "NaCl"]),
        ("What is the speed of light in m/s (approx)?", "3e8", ["3e6", "3e8", "3e10", "3e12"]),
    ]
    for i in range(n_per_task):
        item = knowledge_items[i % len(knowledge_items)]
        examples.append(
            EvalExample(
                task="knowledge",
                prompt=item[0],
                gold_answer=item[1],
                choices=item[2],
            )
        )

    # --- generation ---
    gen_items = [
        ("Complete the sentence: The sky is", "blue"),
        ("What color is grass?", "green"),
        ("Name a primary color:", "red"),
        ("The opposite of hot is", "cold"),
        ("Water freezes at 0 degrees", "celsius"),
    ]
    for i in range(n_per_task):
        item = gen_items[i % len(gen_items)]
        examples.append(
            EvalExample(
                task="generation",
                prompt=item[0],
                gold_answer=item[1],
                choices=None,
            )
        )

    # --- math ---
    math_items = [
        ("2+2=?", "4", ["3", "4", "5", "6"]),
        ("3*3=?", "9", ["6", "8", "9", "12"]),
        ("10-4=?", "6", ["4", "5", "6", "7"]),
        ("8/2=?", "4", ["2", "3", "4", "5"]),
        ("5+7=?", "12", ["10", "11", "12", "13"]),
    ]
    for i in range(n_per_task):
        item = math_items[i % len(math_items)]
        examples.append(
            EvalExample(
                task="math",
                prompt=item[0],
                gold_answer=item[1],
                choices=item[2],
            )
        )

    return examples

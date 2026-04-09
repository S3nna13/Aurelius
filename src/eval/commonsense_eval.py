"""Commonsense reasoning evaluation: HellaSwag-style, Winogrande-style, and ARC-style tasks."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class CommonsenseTask:
    """A single commonsense reasoning task."""
    task_type: str       # "hellaswag" | "winogrande" | "arc"
    context: str
    choices: list[str]
    correct_idx: int


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

_HELLASWAG_TEMPLATES = [
    {
        "activity": "cooking a meal",
        "correct": "stirs the ingredients together in the pan",
        "wrong": [
            "drives to the nearest gas station",
            "begins reading a book on quantum physics",
            "waters the plants in the garden",
        ],
    },
    {
        "activity": "riding a bicycle",
        "correct": "pedals forward and steers around a corner",
        "wrong": [
            "opens a laptop and checks email",
            "fills a glass of water from the tap",
            "adjusts the thermostat on the wall",
        ],
    },
    {
        "activity": "washing the dishes",
        "correct": "scrubs the plates with soap and rinses them",
        "wrong": [
            "plays a video game on the couch",
            "ties shoelaces before going for a run",
            "mows the lawn outside",
        ],
    },
    {
        "activity": "writing a letter",
        "correct": "picks up a pen and begins forming words on paper",
        "wrong": [
            "starts the car engine and reverses out of the driveway",
            "feeds the cat its evening meal",
            "hammers a nail into the wooden frame",
        ],
    },
    {
        "activity": "planting a garden",
        "correct": "digs a small hole and places a seed inside",
        "wrong": [
            "logs into a social media account",
            "wraps a present with decorative paper",
            "tunes a guitar before a concert",
        ],
    },
]

_WINOGRANDE_TEMPLATES = [
    {
        "context": "Sarah gave her jacket to Emma because _ was cold.",
        "choices": ["Emma", "Sarah"],
        "correct_idx": 0,
    },
    {
        "context": "The trophy did not fit in the suitcase because _ was too large.",
        "choices": ["the trophy", "the suitcase"],
        "correct_idx": 0,
    },
    {
        "context": "Tom lent his bicycle to Mark because _ needed to get to work.",
        "choices": ["Mark", "Tom"],
        "correct_idx": 0,
    },
    {
        "context": "The manager fired the assistant because _ made too many mistakes.",
        "choices": ["the assistant", "the manager"],
        "correct_idx": 0,
    },
    {
        "context": "Anna told Maria that _ had won the award.",
        "choices": ["Maria", "Anna"],
        "correct_idx": 0,
    },
    {
        "context": "Jake poured juice into the glass until _ was full.",
        "choices": ["the glass", "the pitcher"],
        "correct_idx": 0,
    },
    {
        "context": "The dog chased the cat until _ was exhausted.",
        "choices": ["the dog", "the cat"],
        "correct_idx": 0,
    },
]

_ARC_TEMPLATES = [
    {
        "context": "What is the boiling point of water at standard atmospheric pressure?",
        "choices": ["100 degrees Celsius", "0 degrees Celsius", "50 degrees Celsius", "200 degrees Celsius"],
        "correct_idx": 0,
    },
    {
        "context": "Which planet is closest to the Sun?",
        "choices": ["Mercury", "Venus", "Earth", "Mars"],
        "correct_idx": 0,
    },
    {
        "context": "What gas do plants primarily absorb during photosynthesis?",
        "choices": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"],
        "correct_idx": 0,
    },
    {
        "context": "How many sides does a triangle have?",
        "choices": ["3", "4", "5", "6"],
        "correct_idx": 0,
    },
    {
        "context": "What is the chemical symbol for water?",
        "choices": ["H2O", "CO2", "NaCl", "O2"],
        "correct_idx": 0,
    },
    {
        "context": "Which organ pumps blood through the human body?",
        "choices": ["Heart", "Liver", "Lungs", "Kidney"],
        "correct_idx": 0,
    },
    {
        "context": "What force keeps objects on the surface of the Earth?",
        "choices": ["Gravity", "Magnetism", "Friction", "Buoyancy"],
        "correct_idx": 0,
    },
    {
        "context": "What is the freezing point of water at standard atmospheric pressure?",
        "choices": ["0 degrees Celsius", "100 degrees Celsius", "-50 degrees Celsius", "37 degrees Celsius"],
        "correct_idx": 0,
    },
    {
        "context": "Which of the following is a primary color of light?",
        "choices": ["Red", "Purple", "Orange", "Brown"],
        "correct_idx": 0,
    },
    {
        "context": "What is the largest planet in our solar system?",
        "choices": ["Jupiter", "Saturn", "Neptune", "Earth"],
        "correct_idx": 0,
    },
]


def generate_hellaswag_tasks(n: int, seed: int = 42) -> list[CommonsenseTask]:
    """Generate n synthetic HellaSwag-style activity-completion tasks."""
    rng = random.Random(seed)
    tasks: list[CommonsenseTask] = []

    for i in range(n):
        template = _HELLASWAG_TEMPLATES[i % len(_HELLASWAG_TEMPLATES)]
        context = f"A person is {template['activity']}."
        correct_text = template["correct"]
        wrong_texts = template["wrong"][:3]

        all_choices = wrong_texts[:]
        correct_pos = rng.randint(0, 3)
        all_choices.insert(correct_pos, correct_text)

        tasks.append(CommonsenseTask(
            task_type="hellaswag",
            context=context,
            choices=all_choices,
            correct_idx=correct_pos,
        ))

    return tasks


def generate_winogrande_tasks(n: int, seed: int = 42) -> list[CommonsenseTask]:
    """Generate n Winogrande-style pronoun-resolution tasks."""
    rng = random.Random(seed)
    tasks: list[CommonsenseTask] = []

    for i in range(n):
        template = _WINOGRANDE_TEMPLATES[i % len(_WINOGRANDE_TEMPLATES)]

        choices = template["choices"][:]
        correct_idx = template["correct_idx"]

        if rng.random() < 0.5:
            choices = list(reversed(choices))
            correct_idx = len(choices) - 1 - correct_idx

        tasks.append(CommonsenseTask(
            task_type="winogrande",
            context=template["context"],
            choices=choices,
            correct_idx=correct_idx,
        ))

    return tasks


def generate_arc_tasks(n: int, seed: int = 42) -> list[CommonsenseTask]:
    """Generate n ARC-style 4-choice science question tasks."""
    rng = random.Random(seed)
    tasks: list[CommonsenseTask] = []

    for i in range(n):
        template = _ARC_TEMPLATES[i % len(_ARC_TEMPLATES)]

        choices = template["choices"][:]
        correct_text = choices[template["correct_idx"]]

        rng.shuffle(choices)
        correct_idx = choices.index(correct_text)

        tasks.append(CommonsenseTask(
            task_type="arc",
            context=template["context"],
            choices=choices,
            correct_idx=correct_idx,
        ))

    return tasks


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_task_by_likelihood(
    model,
    tokenizer_encode: Callable[[str], list[int]],
    task: CommonsenseTask,
) -> int:
    """Score a CommonsenseTask by log-likelihood for each choice.

    Returns the index of the highest log-likelihood choice.
    """
    model.eval()
    device = next(model.parameters()).device

    scores: list[float] = []
    context_ids = tokenizer_encode(task.context)

    for choice in task.choices:
        choice_ids = tokenizer_encode(choice)

        if not choice_ids:
            scores.append(float("-inf"))
            continue

        full_ids = context_ids + choice_ids

        max_seq = getattr(model.config, "max_seq_len", 512)
        if len(full_ids) > max_seq:
            full_ids = full_ids[-max_seq:]
            n_choice = min(len(choice_ids), len(full_ids) - 1)
        else:
            n_choice = len(choice_ids)

        if n_choice == 0:
            scores.append(float("-inf"))
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        _, logits, _ = model(input_ids)

        total_len = len(full_ids)
        completion_start = total_len - n_choice

        log_probs = F.log_softmax(logits[0, :-1], dim=-1)
        targets = input_ids[0, 1:]

        token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        choice_log_probs = token_log_probs[completion_start - 1:]
        score = choice_log_probs.sum().item()
        scores.append(score)

    return int(max(range(len(scores)), key=lambda i: scores[i]))


def compute_accuracy(predictions: list[int], tasks: list[CommonsenseTask]) -> float:
    """Compute fraction of correct predictions."""
    if not tasks:
        return 0.0
    n_correct = sum(
        pred == task.correct_idx
        for pred, task in zip(predictions, tasks)
    )
    return n_correct / len(tasks)


def length_normalize_scores(scores: list[float], choices: list[str]) -> list[float]:
    """Divide each score by the character length of the corresponding choice."""
    normalized: list[float] = []
    for score, choice in zip(scores, choices):
        length = max(len(choice), 1)
        normalized.append(score / length)
    return normalized


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class CommonsenseEvaluator:
    """Evaluates an AureliusTransformer on commonsense reasoning tasks."""

    def __init__(
        self,
        model,
        encode_fn: Callable[[str], list[int]],
        config: dict | None = None,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.config = config or {}

    def evaluate(self, tasks: list[CommonsenseTask]) -> dict:
        """Score all tasks and return aggregated metrics.

        Returns dict with keys: accuracy, n_correct, n_total, by_task_type.
        """
        predictions: list[int] = []
        for task in tasks:
            pred = score_task_by_likelihood(self.model, self.encode_fn, task)
            predictions.append(pred)

        accuracy = compute_accuracy(predictions, tasks)
        n_total = len(tasks)
        n_correct = sum(
            pred == task.correct_idx
            for pred, task in zip(predictions, tasks)
        )

        type_groups: dict[str, list[tuple[int, CommonsenseTask]]] = {}
        for pred, task in zip(predictions, tasks):
            type_groups.setdefault(task.task_type, []).append((pred, task))

        by_task_type: dict[str, dict] = {}
        for task_type, pairs in type_groups.items():
            preds_t = [p for p, _ in pairs]
            tasks_t = [t for _, t in pairs]
            acc_t = compute_accuracy(preds_t, tasks_t)
            nc_t = sum(p == t.correct_idx for p, t in zip(preds_t, tasks_t))
            by_task_type[task_type] = {
                "accuracy": acc_t,
                "n_correct": nc_t,
                "n_total": len(tasks_t),
            }

        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "by_task_type": by_task_type,
        }

    def evaluate_suite(self, n_per_task: int = 10) -> dict:
        """Generate and evaluate all 3 task types.

        Returns dict with keys: hellaswag, winogrande, arc, overall.
        """
        hellaswag_tasks = generate_hellaswag_tasks(n_per_task)
        winogrande_tasks = generate_winogrande_tasks(n_per_task)
        arc_tasks = generate_arc_tasks(n_per_task)

        all_tasks = hellaswag_tasks + winogrande_tasks + arc_tasks

        hellaswag_result = self.evaluate(hellaswag_tasks)
        winogrande_result = self.evaluate(winogrande_tasks)
        arc_result = self.evaluate(arc_tasks)
        overall_result = self.evaluate(all_tasks)

        return {
            "hellaswag": hellaswag_result,
            "winogrande": winogrande_result,
            "arc": arc_result,
            "overall": overall_result,
        }

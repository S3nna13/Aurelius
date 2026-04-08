"""Automatic step labeling for process supervision data generation.

Parses reasoning chains into steps, labels each step by checking if
model completions from that prefix reach the correct answer.

Used to generate PRM training data without human annotation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import torch
import torch.nn as nn


class StepLabel(IntEnum):
    INCORRECT = -1
    NEUTRAL = 0
    CORRECT = 1


@dataclass
class Step:
    """A single reasoning step."""
    text: str                  # step text (between delimiters)
    token_ids: list[int]       # tokenized step
    position: int              # step index in chain
    label: StepLabel = StepLabel.NEUTRAL


@dataclass
class LabeledChain:
    """A reasoning chain with step-level labels."""
    prompt_ids: list[int]      # original prompt token IDs
    steps: list[Step]          # parsed steps with labels
    final_answer: str          # extracted final answer
    correct_answer: str        # ground truth answer
    is_correct: bool           # does final_answer match correct_answer?

    @property
    def step_labels(self) -> list[int]:
        return [s.label.value for s in self.steps]

    @property
    def n_correct_steps(self) -> int:
        return sum(1 for s in self.steps if s.label == StepLabel.CORRECT)


@dataclass
class StepLabelerConfig:
    step_delimiter: int = -1        # token ID for step separator (e.g., newline token)
    answer_delimiter: str = "####"  # string that precedes final answer
    n_completions: int = 4          # samples per step for labeling
    max_completion_tokens: int = 64
    temperature: float = 0.7
    correct_threshold: float = 0.5  # fraction of completions that must be correct
    normalize_answers: bool = True  # strip punctuation/whitespace for comparison


def parse_steps(
    chain_ids: list[int],
    step_delimiter_id: int,
) -> list[list[int]]:
    """Split a token sequence into steps at delimiter positions.

    Returns list of token ID lists, one per step (excluding delimiter tokens).
    """
    steps: list[list[int]] = []
    current: list[int] = []
    for tok in chain_ids:
        if tok == step_delimiter_id:
            steps.append(current)
            current = []
        else:
            current.append(tok)
    steps.append(current)
    return steps


def extract_answer(text: str, answer_delimiter: str = "####") -> str | None:
    """Extract final answer after the delimiter.

    Returns None if delimiter not found.
    """
    idx = text.find(answer_delimiter)
    if idx == -1:
        return None
    return text[idx + len(answer_delimiter):].strip()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def answers_match(predicted: str, reference: str, normalize: bool = True) -> bool:
    """Compare two answer strings."""
    if normalize:
        return normalize_answer(predicted) == normalize_answer(reference)
    return predicted.strip() == reference.strip()


class StepLabeler:
    """Automatically labels reasoning steps using model completions.

    For each step prefix, sample n_completions continuations.
    Label the step as CORRECT if >= correct_threshold fraction of
    completions reach the correct answer.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: StepLabelerConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or StepLabelerConfig()

    def _generate_completion(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Generate a single completion from prefix_ids using the model.

        Attempts to call model.generate if available, otherwise falls back
        to a simple greedy token-by-token generation loop.

        Returns full sequence tensor (prefix + generated tokens), shape (1, S).
        """
        prefix = prefix_ids.unsqueeze(0) if prefix_ids.dim() == 1 else prefix_ids
        max_new = self.cfg.max_completion_tokens
        temp = self.cfg.temperature

        if hasattr(self.model, "generate"):
            try:
                return self.model.generate(
                    prefix,
                    max_new_tokens=max_new,
                    temperature=temp,
                )
            except Exception:
                pass  # fall through to manual loop

        # Manual greedy/sampled generation loop
        ids = prefix
        with torch.no_grad():
            for _ in range(max_new):
                output = self.model(ids)
                # Support (loss, logits, kv) tuple or plain logits tensor
                if isinstance(output, tuple):
                    logits = output[1]
                else:
                    logits = output
                next_logits = logits[:, -1, :]  # (1, vocab)
                if temp != 1.0 and temp > 0:
                    next_logits = next_logits / temp
                probs = next_logits.softmax(dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                ids = torch.cat([ids, next_token], dim=1)
        return ids

    def label_step(
        self,
        prefix_ids: torch.Tensor,   # (S_prefix,) prompt + steps so far
        correct_answer: str,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> tuple[StepLabel, float]:
        """Label a single step prefix.

        Generates n_completions from prefix, checks each for correct answer.
        Returns (label, fraction_correct).
        """
        n = self.cfg.n_completions
        n_correct = 0

        for _ in range(n):
            generated = self._generate_completion(prefix_ids)
            # generated is (1, S) — take first row and convert to list
            gen_ids = generated[0].tolist()

            if decode_fn is not None:
                text = decode_fn(gen_ids)
            else:
                # Fallback: join token IDs as space-separated strings
                text = " ".join(str(t) for t in gen_ids)

            predicted = extract_answer(text, self.cfg.answer_delimiter)
            if predicted is not None and answers_match(
                predicted, correct_answer, normalize=self.cfg.normalize_answers
            ):
                n_correct += 1

        fraction = n_correct / n if n > 0 else 0.0

        if fraction >= self.cfg.correct_threshold:
            label = StepLabel.CORRECT
        elif fraction == 0.0:
            label = StepLabel.INCORRECT
        else:
            label = StepLabel.NEUTRAL

        return label, fraction

    def label_chain(
        self,
        prompt_ids: list[int],
        chain_ids: list[int],       # full reasoning chain token IDs
        correct_answer: str,
        decode_fn: Callable[[list[int]], str] | None = None,
    ) -> LabeledChain:
        """Label all steps in a reasoning chain.

        Parses into steps, labels each, extracts final answer.
        """
        delimiter_id = self.cfg.step_delimiter
        step_token_lists = parse_steps(chain_ids, delimiter_id)

        # Decode the full chain to extract final answer
        all_ids = prompt_ids + chain_ids
        if decode_fn is not None:
            full_text = decode_fn(all_ids)
        else:
            full_text = " ".join(str(t) for t in all_ids)

        raw_answer = extract_answer(full_text, self.cfg.answer_delimiter)
        final_answer = raw_answer if raw_answer is not None else ""
        is_correct = (raw_answer is not None) and answers_match(
            final_answer, correct_answer, normalize=self.cfg.normalize_answers
        )

        steps: list[Step] = []
        # Build prefix incrementally: prompt_ids + steps so far
        prefix_ids_list: list[int] = list(prompt_ids)

        for i, tok_list in enumerate(step_token_lists):
            # Add current step tokens (and delimiter) to running prefix
            # First, create the Step object (label determined next)
            step_text = decode_fn(tok_list) if decode_fn is not None else " ".join(str(t) for t in tok_list)
            step = Step(
                text=step_text,
                token_ids=tok_list,
                position=i,
                label=StepLabel.NEUTRAL,
            )

            # Build prefix up to and including this step
            current_prefix = prefix_ids_list + tok_list
            prefix_tensor = torch.tensor(current_prefix, dtype=torch.long)

            label, _ = self.label_step(prefix_tensor, correct_answer, decode_fn=decode_fn)
            step.label = label
            steps.append(step)

            # Update running prefix: add step tokens + delimiter (except after last step)
            prefix_ids_list = prefix_ids_list + tok_list
            if i < len(step_token_lists) - 1:
                prefix_ids_list = prefix_ids_list + [delimiter_id]

        return LabeledChain(
            prompt_ids=prompt_ids,
            steps=steps,
            final_answer=final_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
        )

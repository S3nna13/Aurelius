"""Self-reflection and self-correction: model reflects on outputs and iteratively refines them."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReflectionConfig:
    """Configuration for the self-reflection loop."""

    max_reflection_rounds: int = 3
    reflection_temperature: float = 0.7
    improvement_threshold: float = 0.01  # stop if score improves by less than this
    use_critic: bool = True             # whether to use a critic model
    critique_prompt_template: str = "Review this response: {response}\n\nIdentify issues:"
    revision_prompt_template: str = "Original: {response}\nCritique: {critique}\n\nRevised:"


@dataclass
class ReflectionStep:
    """Record of a single reflection round."""

    round: int
    response: str
    critique: str
    score: float
    improved: bool


class ResponseScorer:
    """Scores response quality using perplexity (no external oracle needed).

    Lower perplexity = higher quality, so score = -log(perplexity).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    @torch.no_grad()
    def score_response(self, prompt: str, response: str) -> float:
        """Compute -log(perplexity) of the response given the prompt.

        Higher score = better quality.
        """
        combined = prompt + response
        ids = self.tokenizer_encode(combined)
        if len(ids) < 2:
            return 0.0

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Run forward pass: (loss, logits, past_kv)
        loss, logits, _ = self.model(input_ids)

        if loss is not None:
            # Cross-entropy loss IS mean NLL per token
            nll = loss.item()
        else:
            # Compute NLL manually from logits
            shift_logits = logits[:, :-1, :].contiguous()  # (1, S-1, V)
            shift_labels = input_ids[:, 1:].contiguous()   # (1, S-1)
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).item()

        log_perplexity = nll  # NLL = log(perplexity)
        return -log_perplexity  # higher is better

    def score_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        """Score a batch of (prompt, response) pairs."""
        return [self.score_response(p, r) for p, r in zip(prompts, responses)]


class SelfCritiqueGenerator:
    """Generates critiques and revisions by prompting the model greedily."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        config: ReflectionConfig,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    @torch.no_grad()
    def _greedy_decode(self, prompt: str, max_tokens: int) -> str:
        """Greedy decode up to max_tokens new tokens given a prompt string."""
        ids = self.tokenizer_encode(prompt)
        if not ids:
            return ""

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        generated: list[int] = []
        past_key_values = None
        cur_ids = input_ids

        for _ in range(max_tokens):
            _, logits, past_key_values = self.model(cur_ids, past_key_values=past_key_values)
            next_logit = logits[:, -1, :]          # (1, vocab)
            next_token = next_logit.argmax(dim=-1)  # (1,)
            token_id = next_token.item()
            generated.append(token_id)
            cur_ids = next_token.unsqueeze(1)       # (1, 1)

        return self.tokenizer_decode(generated)

    def generate_critique(self, response: str, max_tokens: int = 64) -> str:
        """Generate a critique of the response using the configured template."""
        prompt = self.config.critique_prompt_template.format(response=response)
        return self._greedy_decode(prompt, max_tokens)

    def generate_revision(self, response: str, critique: str, max_tokens: int = 128) -> str:
        """Generate a revised response given the original response and its critique."""
        prompt = self.config.revision_prompt_template.format(
            response=response, critique=critique
        )
        return self._greedy_decode(prompt, max_tokens)


class SelfReflectionLoop:
    """Orchestrates multi-round self-reflection: critique → revise → score → repeat."""

    def __init__(
        self,
        model: nn.Module,
        scorer: ResponseScorer,
        critic: SelfCritiqueGenerator,
        config: ReflectionConfig,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.critic = critic
        self.config = config

    def reflect(
        self,
        initial_response: str,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[ReflectionStep]]:
        """Run the reflection loop, returning (best_response, steps).

        Args:
            initial_response: The starting response text.
            max_new_tokens: Token budget for critique/revision generation.

        Returns:
            (best_response, steps) where steps records each reflection round.
        """
        # Use an empty prompt for scoring — only the response content matters here.
        current_response = initial_response
        current_score = self.scorer.score_response("", current_response)
        best_response = current_response
        best_score = current_score
        steps: list[ReflectionStep] = []

        for round_idx in range(1, self.config.max_reflection_rounds + 1):
            critique = self.critic.generate_critique(current_response, max_tokens=max_new_tokens)
            revision = self.critic.generate_revision(current_response, critique, max_tokens=max_new_tokens)
            new_score = self.scorer.score_response("", revision)

            improvement = new_score - current_score
            improved = improvement >= self.config.improvement_threshold

            step = ReflectionStep(
                round=round_idx,
                response=revision,
                critique=critique,
                score=new_score,
                improved=improved,
            )
            steps.append(step)

            if new_score > best_score:
                best_score = new_score
                best_response = revision

            current_response = revision
            current_score = new_score

            # Early stopping: improvement below threshold
            if improvement < self.config.improvement_threshold:
                break

        return best_response, steps

    def best_of_reflection(self, responses: list[str]) -> str:
        """Score all responses and return the one with the highest score."""
        if not responses:
            return ""
        scores = [self.scorer.score_response("", r) for r in responses]
        best_idx = scores.index(max(scores))
        return responses[best_idx]


def compute_self_consistency_score(
    responses: list[str],
    tokenizer_encode: Callable[[str], list[int]],
) -> float:
    """Measure agreement among responses using mean pairwise Jaccard similarity.

    Args:
        responses: List of response strings.
        tokenizer_encode: Callable that maps a string to a list of token ids.

    Returns:
        Float in [0, 1]. 1.0 = all responses identical, 0.0 = all disjoint.
    """
    if len(responses) < 2:
        return 1.0

    token_sets = [set(tokenizer_encode(r)) for r in responses]

    similarities: list[float] = []
    n = len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            intersection = token_sets[i] & token_sets[j]
            union = token_sets[i] | token_sets[j]
            jaccard = len(intersection) / len(union) if union else 1.0
            similarities.append(jaccard)

    return sum(similarities) / len(similarities)


class ReflectionDataCollector:
    """Collects (prompt, initial, critique, revision) tuples for SFT/DPO training."""

    def __init__(self) -> None:
        self.data: list[dict] = []

    def add_step(self, prompt: str, step: ReflectionStep) -> None:
        """Record a single reflection step associated with a prompt."""
        self.data.append(
            {
                "prompt": prompt,
                "response": step.response,
                "critique": step.critique,
                "score": step.score,
                "round": step.round,
                "improved": step.improved,
            }
        )

    def get_training_pairs(self) -> list[dict]:
        """Return chosen/rejected pairs per prompt for DPO/preference training.

        For each prompt, chosen = step with highest score, rejected = step with
        lowest score.  Prompts with only one step are skipped.
        """
        # Group by prompt
        by_prompt: dict[str, list[dict]] = {}
        for record in self.data:
            by_prompt.setdefault(record["prompt"], []).append(record)

        pairs: list[dict] = []
        for prompt, records in by_prompt.items():
            if len(records) < 2:
                continue
            chosen = max(records, key=lambda r: r["score"])
            rejected = min(records, key=lambda r: r["score"])
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                }
            )
        return pairs

    def clear(self) -> None:
        """Remove all collected data."""
        self.data = []

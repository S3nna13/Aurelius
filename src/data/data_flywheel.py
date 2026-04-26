"""Autonomous data generation and quality filtering pipeline for Aurelius LLM.

DataFlywheelGenerator uses the model to generate responses from seed prompts,
filters them by quality/diversity, and accumulates a growing buffer of
high-quality (prompt, response) pairs for downstream training.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FlywheelConfig:
    """Configuration for the data flywheel pipeline."""

    min_quality_score: float = 0.5
    max_buffer_size: int = 10000
    diversity_threshold: float = 0.3  # minimum pairwise diversity required
    n_generate_per_step: int = 16
    temperature: float = 0.8


@dataclass
class GeneratedSample:
    """A single generated (prompt, response) pair with quality metadata."""

    prompt: str
    response: str
    quality_score: float = 0.0
    diversity_score: float = 0.0
    accepted: bool = False
    generation_step: int = 0


# ---------------------------------------------------------------------------
# QualityFilter
# ---------------------------------------------------------------------------


class QualityFilter:
    """Heuristic quality and diversity filter for generated samples."""

    def __init__(self, config: FlywheelConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Individual scoring methods
    # ------------------------------------------------------------------

    def score_length(self, text: str) -> float:
        """Return a [0, 1] score based on text length.

        - Empty / very short (<10 chars): 0
        - Optimal range (50–500 chars): 1
        - Too long: sigmoid-like decay toward 0
        """
        n = len(text)
        if n < 10:
            return 0.0
        if n < 50:
            # Linear ramp from 0 → 1 in [10, 50)
            return (n - 10) / 40.0
        if n <= 500:
            return 1.0
        # Decay beyond 500: sigmoid-like
        excess = n - 500
        score = 1.0 / (1.0 + math.log1p(excess / 500.0))
        return float(max(0.0, min(1.0, score)))

    def score_diversity(self, text: str, existing: list[str]) -> float:
        """Return [0, 1] diversity of *text* relative to *existing* samples.

        Uses character trigram Jaccard distance. Returns 1.0 if no existing
        samples. Returns 1 − max_similarity (distance from closest neighbour).
        """
        if not existing:
            return 1.0

        def trigrams(s: str) -> set[str]:
            return {s[i : i + 3] for i in range(max(0, len(s) - 2))} if len(s) >= 3 else {s}

        tg_text = trigrams(text)
        if not tg_text:
            return 0.0

        max_sim = 0.0
        for ex in existing:
            tg_ex = trigrams(ex)
            intersection = len(tg_text & tg_ex)
            union = len(tg_text | tg_ex)
            sim = intersection / union if union > 0 else 0.0
            if sim > max_sim:
                max_sim = sim

        return float(1.0 - max_sim)

    def score_coherence(self, prompt: str, response: str) -> float:
        """Heuristic coherence score [0, 1].

        Checks:
          - response is non-empty (required)
          - at least one word from the prompt appears in the response (overlap)
          - response ends with punctuation
        Each check contributes ~1/3 of the score.
        """
        if not response.strip():
            return 0.0

        score = 0.0
        # Non-empty: base contribution
        score += 1 / 3

        # Word overlap
        prompt_words = set(prompt.lower().split())
        response_lower = response.lower()
        if prompt_words and any(w in response_lower for w in prompt_words):
            score += 1 / 3

        # Ends with punctuation
        if response.rstrip() and response.rstrip()[-1] in ".!?,;:":
            score += 1 / 3

        return float(min(1.0, score))

    # ------------------------------------------------------------------
    # Batch filter
    # ------------------------------------------------------------------

    def filter(
        self,
        samples: list[GeneratedSample],
        existing_texts: list[str],
    ) -> list[GeneratedSample]:
        """Score all samples, set quality_score / diversity_score / accepted.

        Returns all samples; accepted=True when overall quality >= threshold.
        """
        cfg = self.config
        seen_responses: list[str] = list(existing_texts)

        for sample in samples:
            # Hard reject empty responses before scoring
            if not sample.response.strip():
                sample.quality_score = 0.0
                sample.diversity_score = 0.0
                sample.accepted = False
                continue

            len_score = self.score_length(sample.response)
            coh_score = self.score_coherence(sample.prompt, sample.response)
            div_score = self.score_diversity(sample.response, seen_responses)

            # Combined quality: average of length and coherence
            quality = (len_score + coh_score) / 2.0

            sample.quality_score = float(quality)
            sample.diversity_score = float(div_score)

            accepted = quality >= cfg.min_quality_score and div_score >= cfg.diversity_threshold
            sample.accepted = accepted

            if accepted:
                seen_responses.append(sample.response)

        return samples


# ---------------------------------------------------------------------------
# DataBuffer
# ---------------------------------------------------------------------------


class DataBuffer:
    """Rolling buffer of accepted generated samples."""

    def __init__(self, config: FlywheelConfig) -> None:
        self.config = config
        self._buffer: list[GeneratedSample] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, samples: list[GeneratedSample]) -> int:
        """Add only accepted samples, enforcing max_buffer_size (drop oldest).

        Returns the number of samples actually added.
        """
        accepted = [s for s in samples if s.accepted]
        n_added = 0
        for sample in accepted:
            self._buffer.append(sample)
            n_added += 1

        # Drop oldest to enforce max size
        if len(self._buffer) > self.config.max_buffer_size:
            overflow = len(self._buffer) - self.config.max_buffer_size
            self._buffer = self._buffer[overflow:]

        return n_added

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def sample(self, n: int) -> list[GeneratedSample]:
        """Return *n* randomly sampled items (with replacement if n > len)."""
        if not self._buffer:
            return []
        return random.choices(self._buffer, k=n)

    def export(self) -> list[tuple[str, str]]:
        """Return all accepted samples as (prompt, response) tuples."""
        return [(s.prompt, s.response) for s in self._buffer]

    def __len__(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict:
        """Return {"size", "mean_quality", "mean_diversity"}."""
        if not self._buffer:
            return {"size": 0, "mean_quality": 0.0, "mean_diversity": 0.0}
        mean_q = sum(s.quality_score for s in self._buffer) / len(self._buffer)
        mean_d = sum(s.diversity_score for s in self._buffer) / len(self._buffer)
        return {
            "size": len(self._buffer),
            "mean_quality": float(mean_q),
            "mean_diversity": float(mean_d),
        }


# ---------------------------------------------------------------------------
# DataFlywheelGenerator
# ---------------------------------------------------------------------------


class DataFlywheelGenerator:
    """Autonomous data generation and quality filtering loop.

    Each step the generator:
      1. Picks seed prompts (up to n_generate_per_step)
      2. Runs the model to generate a response for each
      3. Filters by quality / diversity
      4. Adds accepted samples to the rolling buffer
    """

    def __init__(
        self,
        model,
        config: FlywheelConfig,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        seed_prompts: list[str],
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.seed_prompts = seed_prompts

        self._quality_filter = QualityFilter(config)
        self._buffer = DataBuffer(config)
        self._step: int = 0

    # ------------------------------------------------------------------
    # Internal generation helpers
    # ------------------------------------------------------------------

    def _generate_response(self, prompt: str) -> str:
        """Generate a response for *prompt* via the model.

        Uses the model forward pass in a simple greedy / temperature-sampling
        loop. Falls back to an empty string on any error.
        """
        import torch

        try:
            input_ids = self.tokenizer_encode(prompt)
            if not input_ids:
                return ""

            device = next(self.model.parameters()).device
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Cap generation at a reasonable length
            max_new = min(64, self.model.config.max_seq_len - len(input_ids) - 1)
            if max_new <= 0:
                return ""

            generated: list[int] = []
            with torch.no_grad():
                for _ in range(max_new):
                    _, logits, _ = self.model(ids)
                    next_logits = logits[0, -1, :]  # (vocab,)

                    if self.config.temperature > 0:
                        next_logits = next_logits / self.config.temperature

                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                    generated.append(int(next_id))

                    ids = torch.cat(
                        [ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                        dim=1,
                    )

            return self.tokenizer_decode(generated)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(self) -> list[GeneratedSample]:
        """Generate one batch of samples from seed prompts.

        Picks up to n_generate_per_step prompts (cycling through seeds).
        """
        n = self.config.n_generate_per_step
        prompts = (
            [self.seed_prompts[i % len(self.seed_prompts)] for i in range(n)]
            if self.seed_prompts
            else []
        )

        samples: list[GeneratedSample] = []
        for prompt in prompts:
            response = self._generate_response(prompt)
            samples.append(
                GeneratedSample(
                    prompt=prompt,
                    response=response,
                    generation_step=self._step,
                )
            )
        return samples

    def run_step(self) -> dict:
        """Execute one full flywheel step.

        Returns:
            {"generated": int, "accepted": int, "buffer_size": int,
             "acceptance_rate": float}
        """
        existing_texts = [s.response for s in self._buffer._buffer]

        batch = self.generate_batch()
        filtered = self._quality_filter.filter(batch, existing_texts)
        n_accepted = self._buffer.add(filtered)

        n_generated = len(batch)
        acceptance_rate = n_accepted / n_generated if n_generated > 0 else 0.0

        self._step += 1

        return {
            "generated": n_generated,
            "accepted": n_accepted,
            "buffer_size": len(self._buffer),
            "acceptance_rate": float(acceptance_rate),
        }

    def get_training_data(self, n: int) -> list[tuple[str, str]]:
        """Sample *n* (prompt, response) pairs from the buffer."""
        samples = self._buffer.sample(n)
        return [(s.prompt, s.response) for s in samples]

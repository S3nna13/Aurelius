"""Aurelius — STILL: Self-Taught Iterative Learning Loop.

A model iteratively improves by generating candidate responses, verifying them
with a lightweight verifier (no reward model needed), filtering to keep high-
scoring examples, and fine-tuning on those examples (rejection sampling fine-
tuning, RFT-style).  The loop repeats for n_iterations.

References:
    - STILL / "Slow Thinking with LLMs" (RUCAIBox, 2024)
      https://github.com/RUCAIBox/Slow_Thinking_with_LLMs
    - STaR: Self-Taught Reasoner (Zelikman et al., 2022)
    - ReST / Rejection Sampling Fine-Tuning (Gulcehre et al., 2023)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class STILLConfig:
    """Hyper-parameters for the STILL training loop."""

    n_iterations: int = 3
    n_samples_per_prompt: int = 8  # how many responses to generate per prompt
    keep_top_k: int = 2  # keep k best per prompt
    min_score: float = 0.5  # minimum verifier score to keep
    temperature: float = 0.8
    max_new_tokens: int = 256


# ---------------------------------------------------------------------------
# Verifier protocol + built-in implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Verifier(Protocol):
    """Pluggable verification function."""

    def score(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
    ) -> float:
        """Return a score in [0, 1]."""
        ...


class ExactMatchVerifier:
    """Scores 1.0 if *response* contains the reference answer, 0.0 otherwise.

    The check is case-insensitive and strips surrounding whitespace from both
    the response and reference before testing containment.
    """

    def score(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
    ) -> float:
        if reference is None:
            return 0.0
        ref = reference.strip().lower()
        resp = response.strip().lower()
        return 1.0 if ref in resp else 0.0


class RubricVerifier:
    """Scores based on a list of required substrings / regex patterns.

    The score is the fraction of rubric items that appear (case-insensitive)
    in the response.  An empty rubric always returns 0.0.

    Args:
        rubric: List of required substrings or regex patterns.
        use_regex: If True, each item is treated as a regex; otherwise as a
                   plain substring.
    """

    def __init__(self, rubric: list[str], *, use_regex: bool = False) -> None:
        self.rubric = rubric
        self.use_regex = use_regex

    def score(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
    ) -> float:
        if not self.rubric:
            return 0.0

        resp_lower = response.lower()
        hits = 0
        for item in self.rubric:
            if self.use_regex:
                if re.search(item, resp_lower):
                    hits += 1
            else:
                if item.lower() in resp_lower:
                    hits += 1
        return hits / len(self.rubric)


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class STILLSample:
    """A single verified candidate held in the replay buffer."""

    prompt: str
    response: str
    score: float
    iteration: int


# ---------------------------------------------------------------------------
# STILLTrainer
# ---------------------------------------------------------------------------


class STILLTrainer:
    """Implements the STILL generate -> verify -> filter -> fine-tune loop.

    Args:
        model:     Language model with forward signature
                   ``(input_ids, labels=None) -> (loss_or_None, logits, kv)``.
                   The model is both the generator and the learner.
        optimizer: PyTorch optimizer already configured for *model*.
        verifier:  Any object implementing the :class:`Verifier` protocol.
        config:    :class:`STILLConfig` controlling generation and filtering.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        verifier: Verifier,
        config: STILLConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.verifier = verifier
        self.config = config

    # ------------------------------------------------------------------
    # Internal: token-level autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Generate up to ``config.max_new_tokens`` tokens autoregressively.

        Args:
            input_ids: ``(1, prompt_len)`` integer tensor.
            temperature: Sampling temperature; values > 0 enable stochastic
                sampling.

        Returns:
            ``(1, n_generated)`` tensor of newly generated token ids.
        """
        self.model.training  # access attribute without calling .eval()
        # temporarily disable gradient tracking at the call site via no_grad decorator
        cur_ids = input_ids
        generated: list[torch.Tensor] = []

        for _ in range(self.config.max_new_tokens):
            out = self.model(cur_ids)
            # Support both (loss, logits, kv) and plain logit tensors
            if isinstance(out, tuple):
                logits = out[1]
            else:
                logits = out

            next_logits = logits[:, -1, :]  # (1, vocab_size)

            if temperature > 0.0 and temperature != 1.0:
                next_logits = next_logits / temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated.append(next_token)
            cur_ids = next_token  # feed only the last token (stateless)

        if not generated:
            return input_ids.new_empty((1, 0))
        return torch.cat(generated, dim=1)  # (1, n_generated)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        prompts: list[str],
        input_ids_list: list[torch.Tensor],
        iteration: int = 0,
    ) -> list[list[STILLSample]]:
        """For each prompt sample ``n_samples_per_prompt`` completions.

        Args:
            prompts:       List of N prompt strings (for metadata / scoring).
            input_ids_list: Matching list of ``(1, prompt_len)`` tensors.
            iteration:     Current STILL iteration index (stored in samples).

        Returns:
            List of length N; each element is a list of
            ``n_samples_per_prompt`` :class:`STILLSample` objects.
        """
        all_candidates: list[list[STILLSample]] = []

        for prompt_str, input_ids in zip(prompts, input_ids_list):
            candidates: list[STILLSample] = []
            for _ in range(self.config.n_samples_per_prompt):
                gen_ids = self._generate_tokens(input_ids, self.config.temperature)  # (1, n_gen)
                # Decode to a simple space-separated token id string
                response_str = " ".join(str(t) for t in gen_ids[0].tolist())
                candidates.append(
                    STILLSample(
                        prompt=prompt_str,
                        response=response_str,
                        score=0.0,  # populated by filter_samples
                        iteration=iteration,
                    )
                )
            all_candidates.append(candidates)

        return all_candidates

    def filter_samples(
        self,
        candidates: list[list[STILLSample]],
        references: list[str] | None = None,
    ) -> list[STILLSample]:
        """Score and filter candidates, keeping top-k above min_score.

        For each prompt group:
        1. Score every candidate with the verifier.
        2. Discard candidates below ``config.min_score``.
        3. Keep at most ``config.keep_top_k`` with the highest scores.

        Args:
            candidates:  Output of :meth:`generate_candidates`.
            references:  Optional list of reference answers (one per prompt).

        Returns:
            Flat list of accepted :class:`STILLSample` objects with scores
            populated.
        """
        kept: list[STILLSample] = []

        for i, group in enumerate(candidates):
            ref = references[i] if references is not None and i < len(references) else None

            # Score every sample in the group
            for sample in group:
                sample.score = self.verifier.score(sample.prompt, sample.response, ref)

            # Filter by minimum score
            passing = [s for s in group if s.score >= self.config.min_score]

            # Keep top-k by score descending
            passing.sort(key=lambda s: s.score, reverse=True)
            kept.extend(passing[: self.config.keep_top_k])

        return kept

    def sft_step(self, samples: list[STILLSample]) -> dict:
        """One supervised fine-tuning gradient step on filtered samples.

        Each sample's ``response`` string (space-separated token ids) is used
        to build a target sequence; cross-entropy loss is computed only on
        response positions.

        Args:
            samples: Non-empty list of :class:`STILLSample` objects.

        Returns:
            Dict with at least ``'loss'`` (float) and ``'n_samples'`` (int).
        """
        if not samples:
            return {"loss": 0.0, "n_samples": 0}

        self.model.train()
        self.optimizer.zero_grad()

        losses: list[torch.Tensor] = []

        for sample in samples:
            try:
                token_ids = [int(t) for t in sample.response.split()]
            except ValueError:
                continue
            if not token_ids:
                continue

            # Build input: response tokens only
            input_ids = torch.tensor([token_ids], dtype=torch.long)  # (1, T)
            labels = input_ids.clone()

            out = self.model(input_ids, labels=labels)
            if isinstance(out, tuple):
                loss = out[0]
            else:
                # Model returned raw logits; compute loss manually
                logits = out
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                vocab_size = logits.shape[-1]
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                )

            if loss is not None and torch.is_tensor(loss):
                losses.append(loss)

        if not losses:
            return {"loss": 0.0, "n_samples": 0}

        avg_loss = torch.stack(losses).mean()
        avg_loss.backward()
        self.optimizer.step()
        return {"loss": float(avg_loss.item()), "n_samples": len(losses)}

    def run_iteration(
        self,
        prompts: list[str],
        input_ids_list: list[torch.Tensor],
        references: list[str] | None = None,
        iteration: int = 0,
    ) -> dict:
        """Full generate -> filter -> train iteration.

        Args:
            prompts:        List of prompt strings.
            input_ids_list: Matching list of ``(1, prompt_len)`` tensors.
            references:     Optional reference answers for verifier scoring.
            iteration:      Current iteration index (stored in samples).

        Returns:
            Stats dict with keys: ``'n_generated'``, ``'n_kept'``, ``'loss'``,
            ``'n_samples'``, ``'iteration'``.
        """
        candidates = self.generate_candidates(prompts, input_ids_list, iteration=iteration)
        n_generated = sum(len(g) for g in candidates)

        filtered = self.filter_samples(candidates, references=references)
        n_kept = len(filtered)

        if filtered:
            train_stats = self.sft_step(filtered)
        else:
            train_stats = {"loss": 0.0, "n_samples": 0}

        return {
            "n_generated": n_generated,
            "n_kept": n_kept,
            "iteration": iteration,
            **train_stats,
        }

    def run(
        self,
        prompts: list[str],
        input_ids_list: list[torch.Tensor],
        references: list[str] | None = None,
    ) -> list[dict]:
        """Run ``config.n_iterations`` STILL iterations.

        Args:
            prompts:        List of prompt strings.
            input_ids_list: Matching list of ``(1, prompt_len)`` tensors.
            references:     Optional reference answers.

        Returns:
            List of per-iteration stats dicts (length = ``n_iterations``).
        """
        history: list[dict] = []
        for i in range(self.config.n_iterations):
            stats = self.run_iteration(prompts, input_ids_list, references=references, iteration=i)
            logger.info(
                "STILL iter %d/%d — kept %d/%d samples, loss=%.4f",
                i + 1,
                self.config.n_iterations,
                stats["n_kept"],
                stats["n_generated"],
                stats["loss"],
            )
            history.append(stats)
        return history

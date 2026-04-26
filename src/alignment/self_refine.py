"""Aurelius — Self-Refine iterative refinement framework.

Implementation of Madaan et al. (2023) Self-Refine: Iterative Refinement with
Self-Feedback.  A model generates an initial output, critiques it, and refines
based on the critique -- all with the same model and no additional training.

For training purposes, Self-Refine generates (initial_output, critique,
refined_output) triplets that can be used as supervised training data
(e.g., for Chain-of-Hindsight fine-tuning).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SelfRefineConfig:
    """Configuration for the Self-Refine refinement loop."""

    n_refine_steps: int = 3
    temperature: float = 0.7
    stop_if_improved: bool = True
    max_new_tokens: int = 64
    min_improvement: float = 0.0


# ---------------------------------------------------------------------------
# Data class for a single refinement step
# ---------------------------------------------------------------------------


@dataclass
class SelfRefineStep:
    """Stores the artefacts produced during one self-refine iteration."""

    initial_ids: torch.Tensor  # token ids of the initial (pre-refinement) output
    critique_ids: torch.Tensor  # token ids of the critique
    refined_ids: torch.Tensor  # token ids of the refined output
    reward_before: float
    reward_after: float
    improvement: float  # reward_after - reward_before


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_refinement_gain(steps: list[SelfRefineStep]) -> dict:
    """Compute aggregate statistics over a list of refinement steps.

    Args:
        steps: List of SelfRefineStep produced by a refinement loop.

    Returns:
        dict with keys:
            mean_improvement -- average improvement across all steps.
            max_improvement  -- largest improvement observed.
            success_rate     -- fraction of steps where improvement > 0.
    """
    if not steps:
        return {"mean_improvement": 0.0, "max_improvement": 0.0, "success_rate": 0.0}

    improvements = [s.improvement for s in steps]
    mean_improvement = sum(improvements) / len(improvements)
    max_improvement = max(improvements)
    success_rate = sum(1 for v in improvements if v > 0) / len(improvements)

    return {
        "mean_improvement": mean_improvement,
        "max_improvement": max_improvement,
        "success_rate": success_rate,
    }


# ---------------------------------------------------------------------------
# SelfRefineTrainer
# ---------------------------------------------------------------------------


class SelfRefineTrainer:
    """Iterative self-refinement wrapper around a pure-PyTorch language model.

    The trainer implements the three-step Self-Refine loop:
        1. Generate an initial response.
        2. Critique that response using the same model.
        3. Produce a refined response conditioned on the critique.

    Optionally stops early when a reward improvement is observed.
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[torch.Tensor], float],
        n_refine_steps: int = 3,
        temperature: float = 0.7,
        stop_if_improved: bool = True,
    ) -> None:
        """
        Args:
            model:            A PyTorch Module that accepts (B, seq_len) input_ids
                              and returns a tuple (_, logits, _) or plain logits,
                              where logits has shape (B, seq_len, vocab_size).
            reward_fn:        Callable mapping a 1-D token-id tensor to a scalar
                              float reward.
            n_refine_steps:   Maximum number of critique-refine iterations.
            temperature:      Sampling temperature used during generation.
            stop_if_improved: If True, stop the loop as soon as reward improves.
        """
        self.model = model
        self.reward_fn = reward_fn
        self.n_refine_steps = n_refine_steps
        self.temperature = temperature
        self.stop_if_improved = stop_if_improved

    # ------------------------------------------------------------------
    # Low-level generation
    # ------------------------------------------------------------------

    def generate_with_ids(
        self,
        input_ids: torch.Tensor,
        max_new: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate new tokens auto-regressively given a context.

        Uses greedy decoding when temperature is zero, otherwise samples from
        the softmax distribution.

        Args:
            input_ids:   1-D LongTensor -- the prompt/context token ids.
            max_new:     Maximum number of new tokens to generate.
            temperature: Sampling temperature; 0.0 means greedy decoding.

        Returns:
            1-D LongTensor of generated token ids (not including input_ids).
        """
        self.model.eval()
        generated: list[int] = []
        current_ids = input_ids.unsqueeze(0)  # (1, seq_len)

        with torch.no_grad():
            for _ in range(max_new):
                output = self.model(current_ids)
                # Support (None, logits, None) or plain logits
                if isinstance(output, tuple):
                    logits = output[1]
                else:
                    logits = output
                # logits: (1, seq_len, vocab_size) -- take last position
                next_logits = logits[0, -1, :]  # (vocab_size,)

                if temperature <= 0.0:
                    next_token = int(next_logits.argmax().item())
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated.append(next_token)
                new_tok_t = torch.tensor([[next_token]], dtype=torch.long)
                current_ids = torch.cat([current_ids, new_tok_t], dim=1)

        return torch.tensor(generated, dtype=torch.long)

    # ------------------------------------------------------------------
    # Critique generation
    # ------------------------------------------------------------------

    def generate_critique(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        critique_prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a critique of an initial response.

        Concatenates [prompt | response | critique_instruction] and feeds it to
        the model to produce the critique tokens.

        Args:
            prompt_ids:          1-D LongTensor -- original prompt.
            response_ids:        1-D LongTensor -- initial model response.
            critique_prompt_ids: 1-D LongTensor -- tokens instructing the model
                                 to produce a critique (e.g. "What is wrong?").

        Returns:
            1-D LongTensor of critique token ids.
        """
        context = torch.cat([prompt_ids, response_ids, critique_prompt_ids], dim=0)
        return self.generate_with_ids(
            context,
            max_new=max(16, self.n_refine_steps * 8),
            temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # Refinement generation
    # ------------------------------------------------------------------

    def refine(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        critique_ids: torch.Tensor,
        refine_prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a refined response given the original response and critique.

        Concatenates [prompt | response | critique | refine_instruction] and
        generates the improved response.

        Args:
            prompt_ids:        1-D LongTensor -- original prompt.
            response_ids:      1-D LongTensor -- initial (or previous) response.
            critique_ids:      1-D LongTensor -- critique of the response.
            refine_prompt_ids: 1-D LongTensor -- tokens instructing the model to
                               produce an improved response.

        Returns:
            1-D LongTensor of refined response token ids.
        """
        context = torch.cat([prompt_ids, response_ids, critique_ids, refine_prompt_ids], dim=0)
        return self.generate_with_ids(
            context,
            max_new=max(16, self.n_refine_steps * 8),
            temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # Refinement loop
    # ------------------------------------------------------------------

    def run_refinement_loop(
        self,
        prompt_ids: torch.Tensor,
        initial_response_ids: torch.Tensor,
        critique_prompt_ids: torch.Tensor,
        refine_prompt_ids: torch.Tensor,
    ) -> list[SelfRefineStep]:
        """Run the full Self-Refine loop for up to n_refine_steps iterations.

        Each iteration:
            1. Scores the current response with reward_fn.
            2. Generates a critique.
            3. Generates a refined response.
            4. Scores the refined response.
            5. Records a SelfRefineStep.
            6. Optionally stops early if stop_if_improved and reward improved.

        Args:
            prompt_ids:            1-D LongTensor -- original task prompt.
            initial_response_ids:  1-D LongTensor -- first model response.
            critique_prompt_ids:   1-D LongTensor -- instruction to critique.
            refine_prompt_ids:     1-D LongTensor -- instruction to refine.

        Returns:
            List of SelfRefineStep, one per iteration executed (length <=
            n_refine_steps).
        """
        steps: list[SelfRefineStep] = []
        current_response_ids = initial_response_ids

        for _ in range(self.n_refine_steps):
            reward_before = float(self.reward_fn(current_response_ids))

            critique_ids = self.generate_critique(
                prompt_ids, current_response_ids, critique_prompt_ids
            )
            refined_ids = self.refine(
                prompt_ids, current_response_ids, critique_ids, refine_prompt_ids
            )

            reward_after = float(self.reward_fn(refined_ids))
            improvement = reward_after - reward_before

            step = SelfRefineStep(
                initial_ids=current_response_ids,
                critique_ids=critique_ids,
                refined_ids=refined_ids,
                reward_before=reward_before,
                reward_after=reward_after,
                improvement=improvement,
            )
            steps.append(step)

            # Advance to the refined response for the next iteration
            current_response_ids = refined_ids

            if self.stop_if_improved and improvement > 0:
                break

        return steps

    # ------------------------------------------------------------------
    # Training pair extraction
    # ------------------------------------------------------------------

    def create_training_pairs(
        self,
        steps: list[SelfRefineStep],
    ) -> list[dict]:
        """Extract (bad_response, critique, good_response) training pairs.

        Only steps where improvement > 0 are included, making these suitable
        for Chain-of-Hindsight supervised fine-tuning.

        Args:
            steps: List of SelfRefineStep from run_refinement_loop.

        Returns:
            List of dicts with keys:
                initial_ids  -- 1-D LongTensor (the initial / worse response).
                critique_ids -- 1-D LongTensor.
                refined_ids  -- 1-D LongTensor (the improved response).
                improvement  -- float scalar (reward_after - reward_before).
        """
        pairs: list[dict] = []
        for step in steps:
            if step.improvement > 0:
                pairs.append(
                    {
                        "initial_ids": step.initial_ids,
                        "critique_ids": step.critique_ids,
                        "refined_ids": step.refined_ids,
                        "improvement": step.improvement,
                    }
                )
        return pairs

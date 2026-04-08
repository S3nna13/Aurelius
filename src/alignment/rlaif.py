"""Aurelius — RLAIF (Reinforcement Learning from AI Feedback) with Constitutional AI-style revision loop.

The model critiques its own outputs against a set of constitutional principles and
revises them, without a human-in-the-loop. Revised outputs are then used as SFT targets,
enabling iterative self-improvement of the model's alignment.

Pipeline:
    Generate initial response
    → Critique against each principle (using the model itself)
    → Revise based on critique
    → Train on revised response as SFT target
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constitutional principles
# ---------------------------------------------------------------------------

PRINCIPLES = [
    "The response should be helpful, harmless, and honest.",
    "The response should not assist with illegal activities.",
    "The response should be respectful and avoid discrimination.",
    "The response should acknowledge uncertainty when appropriate.",
]


class ConstitutionalPrinciple:
    """A single constitutional principle with associated critique and revision prompts.

    Args:
        text: The principle statement.
        critique_prompt: Prompt used to ask the model to critique a response.
            Defaults to a generic critique instruction based on text.
        revision_prompt: Prompt used to ask the model to revise a response.
            Defaults to a generic revision instruction based on text.
    """

    def __init__(
        self,
        text: str,
        critique_prompt: str | None = None,
        revision_prompt: str | None = None,
    ) -> None:
        self.text = text
        self.critique_prompt = critique_prompt or (
            f"Identify any ways the response violates: {text}"
        )
        self.revision_prompt = revision_prompt or (
            f"Revise the response to better follow: {text}"
        )


# ---------------------------------------------------------------------------
# Helper: autoregressive token generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate up to max_new_tokens greedily/sampled from model.

    Works with any model whose forward() returns (loss, logits, present_kv).
    Uses the same interface as AureliusTransformer.generate but inline so we
    can keep it self-contained and avoid dependency on the full generate API.

    Args:
        model: AureliusTransformer (or compatible mock).
        input_ids: (1, seq_len) prompt tokens.
        max_new_tokens: Number of new tokens to produce.
        temperature: Sampling temperature; 1.0 = multinomial from softmax.

    Returns:
        (1, max_new_tokens) tensor of newly generated token ids.
    """
    generated: list[torch.Tensor] = []
    past_key_values = None
    cur_ids = input_ids

    for _ in range(max_new_tokens):
        _, logits, past_key_values = model(cur_ids, past_key_values=past_key_values)
        next_logits = logits[:, -1, :]  # (1, vocab)
        if temperature != 1.0:
            next_logits = next_logits / temperature
        probs = next_logits.softmax(dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated.append(next_token)
        cur_ids = next_token  # feed only the new token; cache holds context

    if not generated:
        # Return empty tensor with correct dtype/device
        return torch.zeros((1, 0), dtype=torch.long, device=input_ids.device)

    return torch.cat(generated, dim=1)  # (1, max_new_tokens)


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _perplexity_of_completion(
    model: nn.Module,
    prefix_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> float:
    """Compute average token-level cross-entropy (perplexity proxy) of completion.

    The model sees [prefix || completion] and we measure the loss only over
    the completion tokens.

    Args:
        model: Model with forward(input_ids, labels) → (loss, logits, kv).
        prefix_ids: (1, prefix_len) context tokens.
        completion_ids: (1, completion_len) tokens whose perplexity we measure.

    Returns:
        Mean cross-entropy loss (float) over completion tokens.
    """
    full_ids = torch.cat([prefix_ids, completion_ids], dim=1)  # (1, total)
    # Labels: mask prefix with -100, keep completion tokens
    labels = torch.full_like(full_ids, -100)
    prefix_len = prefix_ids.shape[1]
    labels[:, prefix_len:] = completion_ids

    loss, _, _ = model(full_ids, labels=labels)
    if loss is None:
        # Shouldn't happen since we passed labels, but guard
        return 0.0
    return loss.item()


# ---------------------------------------------------------------------------
# RLAIFCritique
# ---------------------------------------------------------------------------

class RLAIFCritique:
    """Generate AI critique of a response using the model itself.

    The model is prompted with [prompt] + [response] + [critique_instruction]
    and generates a textual critique. The critique tokens are returned directly
    so they can be fed into the reviser.

    Args:
        model: AureliusTransformer or compatible model.
        max_critique_tokens: Maximum number of critique tokens to generate.
    """

    def __init__(self, model: nn.Module, max_critique_tokens: int = 128) -> None:
        self.model = model
        self.max_critique_tokens = max_critique_tokens

    # ------------------------------------------------------------------
    # Internal: encode a plain string as token IDs using vocabulary hash
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_text(text: str, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Encode a UTF-8 string into token ids by hashing each character.

        This is a lightweight stand-in for a real tokenizer so that the module
        works without an external tokenizer dependency. Real usage would inject
        a tokenizer; for the self-contained pipeline we hash characters into
        the model's vocabulary.

        Args:
            text: String to encode.
            vocab_size: Size of the model vocabulary.
            device: Target device.

        Returns:
            (1, len(text)) LongTensor of token ids in [0, vocab_size).
        """
        ids = [ord(c) % vocab_size for c in text]
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def _vocab_size(self) -> int:
        """Infer vocab size from the model's embedding or lm_head."""
        # AureliusTransformer exposes config.vocab_size
        if hasattr(self.model, "config"):
            return self.model.config.vocab_size
        # Fallback: inspect lm_head weight shape
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head.weight.shape[0]
        # Last resort
        return 32000

    def _device(self) -> torch.device:
        """Get device of the model."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def critique(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        principle: ConstitutionalPrinciple,
    ) -> torch.Tensor:
        """Generate a textual critique of the response against the principle.

        Builds the input as [prompt_ids || response_ids || critique_instruction_ids]
        and autoregressively generates critique tokens.

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.
            response_ids: (1, response_len) response token ids.
            principle: The constitutional principle to critique against.

        Returns:
            critique_ids: (1, critique_len) generated critique token ids.
        """
        vocab = self._vocab_size()
        device = self._device()

        # Encode the critique instruction text into token ids
        instruction_ids = self._encode_text(
            principle.critique_prompt, vocab, device
        )

        # Concatenate: [prompt || response || critique_instruction]
        context_ids = torch.cat([prompt_ids, response_ids, instruction_ids], dim=1)

        # Generate critique tokens
        critique_ids = _generate_tokens(
            self.model, context_ids, max_new_tokens=self.max_critique_tokens
        )
        return critique_ids

    def score_response(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        principle: ConstitutionalPrinciple,
    ) -> float:
        """Score the response quality using model perplexity on a "Yes" completion.

        Constructs the query:
            "[prompt] [response] [principle text] Does this response follow the
             principle? Answer: Yes"

        Lower perplexity on "Yes" → model thinks the response follows the
        principle → higher score. We map perplexity to [0, 1] via:
            score = exp(-perplexity)

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.
            response_ids: (1, response_len) response token ids.
            principle: Principle to score against.

        Returns:
            Float in [0, 1]; higher is better.
        """
        vocab = self._vocab_size()
        device = self._device()

        # Build template prefix: prompt + response + scoring question
        scoring_question = (
            f"{principle.text} Does this response follow the principle? Answer:"
        )
        prefix_ids = torch.cat(
            [
                prompt_ids,
                response_ids,
                self._encode_text(scoring_question, vocab, device),
            ],
            dim=1,
        )

        # "Yes" completion (single token for simplicity)
        yes_ids = self._encode_text(" Yes", vocab, device)

        perplexity = _perplexity_of_completion(self.model, prefix_ids, yes_ids)
        # Map cross-entropy loss → [0, 1]: lower loss = higher score
        score = math.exp(-perplexity)
        # Clamp to [0, 1] to handle any numerical edge cases
        return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# RLAIFReviser
# ---------------------------------------------------------------------------

class RLAIFReviser:
    """Revise responses based on AI critique.

    The model is prompted with [prompt] + [original_response] + [critique] +
    [revision_instruction] and generates a revised response.

    Args:
        model: AureliusTransformer or compatible model.
        max_revision_tokens: Maximum number of tokens in the revised response.
    """

    def __init__(self, model: nn.Module, max_revision_tokens: int = 256) -> None:
        self.model = model
        self.max_revision_tokens = max_revision_tokens

    def _vocab_size(self) -> int:
        if hasattr(self.model, "config"):
            return self.model.config.vocab_size
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head.weight.shape[0]
        return 32000

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _encode_text(text: str, vocab_size: int, device: torch.device) -> torch.Tensor:
        ids = [ord(c) % vocab_size for c in text]
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def revise(
        self,
        prompt_ids: torch.Tensor,
        original_response_ids: torch.Tensor,
        critique_ids: torch.Tensor,
        principle: ConstitutionalPrinciple,
    ) -> torch.Tensor:
        """Generate a revised response incorporating the critique.

        Builds the input as:
            [prompt || original_response || critique || revision_instruction]
        and generates a new response autoregressively.

        Args:
            prompt_ids: (1, prompt_len) original prompt token ids.
            original_response_ids: (1, response_len) original (possibly flawed) response.
            critique_ids: (1, critique_len) generated critique token ids.
            principle: Principle used for revision guidance.

        Returns:
            revised_response_ids: (1, revised_len) new response token ids.
        """
        vocab = self._vocab_size()
        device = self._device()

        instruction_ids = self._encode_text(principle.revision_prompt, vocab, device)

        # [prompt || original_response || critique || revision_instruction]
        context_ids = torch.cat(
            [prompt_ids, original_response_ids, critique_ids, instruction_ids],
            dim=1,
        )

        revised_ids = _generate_tokens(
            self.model, context_ids, max_new_tokens=self.max_revision_tokens
        )
        return revised_ids


# ---------------------------------------------------------------------------
# RLAIFPipeline
# ---------------------------------------------------------------------------

class RLAIFPipeline:
    """Full RLAIF pipeline: generate → critique → revise → train.

    For each input prompt the pipeline:
        1. Generates an initial response using the model.
        2. For each constitutional principle: runs critique → revision.
        3. Trains on the final revised response as an SFT target.

    Args:
        model: AureliusTransformer or compatible model.
        optimizer: PyTorch optimizer for SFT updates.
        principles: List of ConstitutionalPrinciple objects. Defaults to the
            first two principles from PRINCIPLES.
        n_critique_rounds: Number of critique-revision rounds per principle.
            Currently one round per principle; parameter reserved for future use.
        max_response_tokens: Max tokens for the initial response generation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        principles: list[ConstitutionalPrinciple] | None = None,
        n_critique_rounds: int = 1,
        max_response_tokens: int = 128,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.principles = principles or [
            ConstitutionalPrinciple(p) for p in PRINCIPLES[:2]
        ]
        self.n_critique_rounds = n_critique_rounds
        self.max_response_tokens = max_response_tokens

        self.critiquer = RLAIFCritique(model)
        self.reviser = RLAIFReviser(model)

    def run_critique_revision_loop(
        self,
        prompt_ids: torch.Tensor,
    ) -> dict:
        """Run the critique-revision loop for all principles.

        Steps:
            1. Generate initial response from the model.
            2. For each principle: score initial response, critique it, revise it,
               score the revision.
            3. The revised response from each principle feeds into the next.

        Args:
            prompt_ids: (1, prompt_len) token ids for the input prompt.

        Returns:
            dict with keys:
                'initial_ids': (1, T) initial response token ids.
                'revised_ids': (1, T') final revised response token ids after
                    all principles have been applied.
                'scores': list of floats, one score per principle reflecting
                    the revised response quality.
        """
        # 1. Generate initial response
        initial_ids = _generate_tokens(
            self.model, prompt_ids, max_new_tokens=self.max_response_tokens
        )

        current_ids = initial_ids
        scores: list[float] = []

        # 2. Critique and revise for each principle
        for principle in self.principles:
            critique_ids = self.critiquer.critique(
                prompt_ids, current_ids, principle
            )
            revised_ids = self.reviser.revise(
                prompt_ids, current_ids, critique_ids, principle
            )
            score = self.critiquer.score_response(
                prompt_ids, revised_ids, principle
            )
            scores.append(score)
            current_ids = revised_ids  # chain revisions

        return {
            "initial_ids": initial_ids,
            "revised_ids": current_ids,
            "scores": scores,
        }

    def sft_step(
        self,
        prompt_ids: torch.Tensor,
        revised_ids: torch.Tensor,
    ) -> float:
        """Train the model on the revised response as an SFT target.

        The full sequence [prompt || revised_response] is fed to the model with
        labels equal to the revised tokens (prompt tokens are masked with -100).

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.
            revised_ids: (1, revised_len) revised response token ids.

        Returns:
            Scalar training loss as a Python float.
        """
        self.model.train()
        self.optimizer.zero_grad()

        full_ids = torch.cat([prompt_ids, revised_ids], dim=1)  # (1, total)

        # Mask prompt tokens in labels
        labels = torch.full_like(full_ids, -100)
        prompt_len = prompt_ids.shape[1]
        labels[:, prompt_len:] = revised_ids

        loss, _, _ = self.model(full_ids, labels=labels)

        if loss is None:
            return 0.0

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def run(self, prompt_ids: torch.Tensor) -> float:
        """Full RLAIF pipeline: critique-revision loop followed by SFT step.

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.

        Returns:
            Training loss from the SFT step as a Python float.
        """
        result = self.run_critique_revision_loop(prompt_ids)
        loss = self.sft_step(prompt_ids, result["revised_ids"])
        return loss


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_improvement_score(initial_score: float, revised_score: float) -> float:
    """Compute how much the revision improved over the initial response.

    Args:
        initial_score: Quality score of the initial response (in [0, 1]).
        revised_score: Quality score of the revised response (in [0, 1]).

    Returns:
        Delta score in [-1, 1]; positive means improvement.
    """
    return float(revised_score - initial_score)

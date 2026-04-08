"""Mixture-of-Agents (MoA) inference (Wang et al. 2024).

Multiple proposer models generate independent responses to the same prompt.
An aggregator model synthesises those proposals into a refined final response.
Multiple rounds of refinement are supported: each round the aggregator output
becomes the new context fed to the proposers.

Reference: "Mixture-of-Agents Enhances Large Language Model Capabilities"
           Wang et al., 2024 (arXiv:2406.04692).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# EOS token reused as separator between proposals in the aggregator context.
_SEP_TOKEN_ID: int = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoAConfig:
    """Configuration for Mixture-of-Agents inference.

    Attributes:
        n_proposers: Number of independent proposer models.
        n_rounds: Rounds of proposal → aggregation refinement.
        max_new_tokens: Maximum new tokens generated per model call.
        temperature: Sampling temperature for proposers (>0 enables sampling).
        aggregator_temperature: Temperature for the aggregator (0 = greedy).
    """
    n_proposers: int = 3
    n_rounds: int = 2
    max_new_tokens: int = 128
    temperature: float = 0.7
    aggregator_temperature: float = 0.0


# ---------------------------------------------------------------------------
# Internal generation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> torch.Tensor:
    """Auto-regressively generate tokens from *model* given a 1-D prompt.

    Args:
        model: Any model whose forward signature matches AureliusTransformer,
               i.e. ``forward(input_ids) -> (loss, logits, present_kv)``.
               ``input_ids`` passed to the model is always shaped ``(1, seq_len)``.
        input_ids: 1-D tensor of prompt token ids.
        max_new_tokens: Number of new tokens to generate.
        temperature: 0 (or very small) → greedy (argmax).
                     >0 → multinomial sampling.

    Returns:
        1-D tensor containing *only the newly generated tokens* (not the prompt).
    """
    # Work with a 1-D input; add batch dimension for model calls.
    prompt_len = input_ids.shape[0]
    current = input_ids.unsqueeze(0)  # (1, prompt_len)

    generated: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        _, logits, _ = model(current)  # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        if temperature <= 1e-8:
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1,)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)

        generated.append(next_token)
        current = torch.cat([current, next_token.unsqueeze(0)], dim=1)  # (1, seq+1)

    if not generated:
        return torch.tensor([], dtype=torch.long, device=input_ids.device)

    return torch.cat(generated, dim=0)  # (max_new_tokens,)


# ---------------------------------------------------------------------------
# Mixture-of-Agents
# ---------------------------------------------------------------------------

class MixtureOfAgents:
    """Mixture-of-Agents inference pipeline.

    Each round:
    1. Every proposer independently generates a response to the current context.
    2. The aggregator receives all proposals (separated by SEP tokens) and
       synthesises a refined response.

    After ``n_rounds`` the aggregator's last output is returned.

    Args:
        proposers: List of proposer model instances.
        aggregator: Aggregator model instance (may be the same object as a proposer).
        tokenizer_encode: Callable that converts a string to token ids.
                          In tests identity lambdas are fine (already have ids).
        tokenizer_decode: Callable that converts token ids to a string.
        cfg: :class:`MoAConfig` controlling the pipeline behaviour.
    """

    def __init__(
        self,
        proposers: list[nn.Module],
        aggregator: nn.Module,
        tokenizer_encode: Callable,
        tokenizer_decode: Callable,
        cfg: MoAConfig | None = None,
    ) -> None:
        self.proposers = proposers
        self.aggregator = aggregator
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.cfg = cfg if cfg is not None else MoAConfig()

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------

    def generate_proposals(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        """Generate one response per proposer, independently.

        Args:
            input_ids: 1-D tensor of prompt token ids.

        Returns:
            List of 1-D tensors, one per proposer, containing *only the
            response tokens* (not the prompt).
        """
        proposals: list[torch.Tensor] = []
        for proposer in self.proposers:
            response = _greedy_generate(
                proposer,
                input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
            )
            proposals.append(response)
        return proposals

    def build_aggregator_context(
        self,
        prompt_ids: torch.Tensor,
        proposals: list[torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate prompt and proposals into a single context tensor.

        Layout::

            [prompt] [proposal_1] [SEP] [proposal_2] [SEP] ... [proposal_N]

        The SEP token (id = 2, EOS token used as separator) is inserted
        *between* consecutive proposals but not after the last one.

        Args:
            prompt_ids: 1-D tensor of prompt token ids.
            proposals: List of 1-D response tensors from the proposers.

        Returns:
            1-D combined context tensor for the aggregator.
        """
        sep = torch.tensor([_SEP_TOKEN_ID], dtype=torch.long, device=prompt_ids.device)

        parts: list[torch.Tensor] = [prompt_ids]
        for idx, proposal in enumerate(proposals):
            parts.append(proposal)
            if idx < len(proposals) - 1:
                parts.append(sep)

        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full MoA pipeline with ``n_rounds`` of refinement.

        Round flow:
        1. Proposers generate from the current context.
        2. Aggregator context is built from the *original* prompt + round proposals.
        3. Aggregator generates a refined response.
        4. The aggregator response becomes the context for the next round's proposers.

        Args:
            input_ids: 1-D prompt tensor.

        Returns:
            1-D tensor of the aggregator's final response tokens.
        """
        original_prompt = input_ids
        current_context = input_ids  # updated each round

        last_agg_response = torch.tensor([], dtype=torch.long, device=input_ids.device)

        for _round in range(self.cfg.n_rounds):
            # Step 1: proposers generate from the current context.
            proposals = self.generate_proposals(current_context)

            # Step 2: build aggregator input (always anchored on original prompt).
            agg_input = self.build_aggregator_context(original_prompt, proposals)

            # Step 3: aggregator synthesises.
            last_agg_response = _greedy_generate(
                self.aggregator,
                agg_input,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.aggregator_temperature,
            )

            # Step 4: next round's context = original prompt + aggregator response.
            current_context = torch.cat([original_prompt, last_agg_response], dim=0)

        return last_agg_response

    @torch.no_grad()
    def generate_simple(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Single-round MoA: proposers generate, aggregator synthesises once.

        Args:
            input_ids: 1-D prompt tensor.
            max_new_tokens: Override ``cfg.max_new_tokens`` for this call.

        Returns:
            1-D tensor of the aggregator's response tokens.
        """
        effective_max = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens

        proposals = self.generate_proposals(input_ids)
        agg_input = self.build_aggregator_context(input_ids, proposals)

        return _greedy_generate(
            self.aggregator,
            agg_input,
            max_new_tokens=effective_max,
            temperature=self.cfg.aggregator_temperature,
        )

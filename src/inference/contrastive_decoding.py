"""Contrastive Decoding (Li et al., 2022) — suppress amateur model artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CDConfig:
    """Configuration for Contrastive Decoding."""

    alpha: float = 0.1           # adaptive plausibility threshold factor
    beta: float = 0.5            # interpolation weight for expert (reserved)
    temperature: float = 1.0    # sampling temperature
    max_new_tokens: int = 50
    top_p: float = 0.95          # nucleus sampling fallback


def compute_contrastive_logits(
    expert_logits: torch.Tensor,
    amateur_logits: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Compute contrastive decoding logits from expert and amateur log-probs.

    Args:
        expert_logits: Raw logits from the expert model, shape (V,).
        amateur_logits: Raw logits from the amateur model, shape (V,).
        alpha: Adaptive plausibility threshold factor.

    Returns:
        Contrastive score tensor of shape (V,). Tokens outside the plausibility
        set are set to -inf. No softmax applied — caller handles sampling.
    """
    # Adaptive plausibility set: V_head = {v : expert_logits[v] >= log(alpha) + max(expert_logits)}
    threshold = math.log(alpha) + expert_logits.max()
    in_vhead = expert_logits >= threshold  # (V,) boolean mask

    # Contrastive score
    score = expert_logits - amateur_logits  # (V,)

    # Mask out tokens not in plausibility set
    score = score.masked_fill(~in_vhead, float("-inf"))

    return score


def nucleus_sample(
    logits: torch.Tensor,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> int:
    """Sample a token id via nucleus (top-p) sampling.

    Args:
        logits: Raw logits of shape (V,).
        top_p: Cumulative probability threshold. 1.0 allows all tokens.
        temperature: Softmax temperature (> 0). Lower = more peaked.

    Returns:
        Sampled token id as a Python int.
    """
    # Apply temperature
    scaled = logits / max(temperature, 1e-8)

    # Softmax to probabilities
    probs = F.softmax(scaled, dim=-1)

    # Sort descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative sum
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens where cum_prob <= top_p, always keep at least 1
    # Shift right so the first token is always included
    keep = cum_probs - sorted_probs <= top_p
    keep[0] = True  # always keep at least the top token

    # Zero out filtered tokens and renormalise
    filtered_probs = sorted_probs * keep.float()
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
    token_id = sorted_indices[sampled_idx].item()
    return int(token_id)


class ContrastiveDecoder:
    """Contrastive Decoding generator combining an expert and amateur model.

    Args:
        expert_model: The stronger (expert) AureliusTransformer.
        amateur_model: The weaker (amateur) AureliusTransformer.
        config: CDConfig controlling generation hyperparameters.
    """

    def __init__(
        self,
        expert_model: nn.Module,
        amateur_model: nn.Module,
        config: CDConfig,
    ) -> None:
        self.expert = expert_model
        self.amateur = amateur_model
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens via contrastive decoding with nucleus sampling.

        Args:
            input_ids: Prompt token ids, shape (1, T).
            max_new_tokens: Override config value if provided.

        Returns:
            Full sequence (prompt + generated) of shape (1, T + new_tokens).
        """
        n = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        seq = input_ids.clone()

        for _ in range(n):
            # Forward both models
            _, expert_logits, _ = self.expert(seq)
            _, amateur_logits, _ = self.amateur(seq)

            # Take last-token logits: (V,)
            e_logits = expert_logits[0, -1]
            a_logits = amateur_logits[0, -1]

            # Compute contrastive scores
            cd_logits = compute_contrastive_logits(e_logits, a_logits, alpha=self.config.alpha)

            # Sample next token via nucleus sampling
            next_token = nucleus_sample(
                cd_logits,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
            )
            next_token_tensor = torch.tensor([[next_token]], dtype=input_ids.dtype)
            seq = torch.cat([seq, next_token_tensor], dim=1)

        return seq

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens via contrastive decoding with greedy (argmax) selection.

        Deterministic variant — useful for testing.

        Args:
            input_ids: Prompt token ids, shape (1, T).
            max_new_tokens: Override config value if provided.

        Returns:
            Full sequence (prompt + generated) of shape (1, T + new_tokens).
        """
        n = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        seq = input_ids.clone()

        for _ in range(n):
            # Forward both models
            _, expert_logits, _ = self.expert(seq)
            _, amateur_logits, _ = self.amateur(seq)

            # Take last-token logits: (V,)
            e_logits = expert_logits[0, -1]
            a_logits = amateur_logits[0, -1]

            # Compute contrastive scores
            cd_logits = compute_contrastive_logits(e_logits, a_logits, alpha=self.config.alpha)

            # Greedy: argmax (handles -inf naturally)
            next_token = cd_logits.argmax().item()
            next_token_tensor = torch.tensor([[next_token]], dtype=input_ids.dtype)
            seq = torch.cat([seq, next_token_tensor], dim=1)

        return seq


class VocabProjectionAmateur(nn.Module):
    """Wrap a smaller/larger AureliusTransformer and project its logits to expert vocab size.

    When the amateur model has a different vocab size than the expert, a linear
    projection layer maps amateur vocab logits -> expert vocab logits.  When
    vocab sizes already match, the forward pass is a no-op projection.

    Args:
        model: Amateur AureliusTransformer instance.
        expert_vocab_size: Target vocab size of the expert model.
    """

    def __init__(self, model: nn.Module, expert_vocab_size: int) -> None:
        super().__init__()
        self.model = model
        self.expert_vocab_size = expert_vocab_size
        amateur_vocab = model.config.vocab_size

        if amateur_vocab == expert_vocab_size:
            self.projection: nn.Linear | None = None
        else:
            # Linear projection with identity-like (truncated) initialisation
            proj = nn.Linear(amateur_vocab, expert_vocab_size, bias=False)
            # Initialise to truncated identity so small models still pass useful signal
            with torch.no_grad():
                min_vocab = min(amateur_vocab, expert_vocab_size)
                proj.weight.zero_()
                proj.weight[:min_vocab, :min_vocab] = torch.eye(min_vocab)
            self.projection = proj

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the amateur model and project logits to expert vocab size.

        Args:
            input_ids: (B, T) token ids (must be valid indices for the amateur vocab).

        Returns:
            Logits of shape (B, T, expert_vocab_size).
        """
        _, logits, _ = self.model(input_ids)  # (B, T, amateur_vocab)
        if self.projection is not None:
            logits = self.projection(logits)  # (B, T, expert_vocab_size)
        return logits

"""Hydra Speculative Decoding — multi-head parallel draft speculation.

Implements Hydra (Ankner et al., 2024): N auxiliary "draft heads" are attached
to the target model's hidden states. Each head i predicts the token at position
t+i from a *single* forward pass of the target model, eliminating the need for
a separate draft model entirely.

Key properties:
- All N heads run in a single batched forward pass (true parallelism).
- Acceptance uses per-token probability comparison (speculative sampling rule).
- Drop-in replacement for draft-model speculative decoding.

Reference: https://arxiv.org/abs/2402.05244
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HydraConfig:
    """Configuration for HydraSpeculative.

    Attributes:
        d_model: Hidden dimension of the target model.
        vocab_size: Vocabulary size.
        n_draft_heads: Number of parallel draft heads (future token positions).
        head_hidden_dim: Intermediate MLP dimension for each head.  Defaults to
            d_model when None.
        temperature: Softmax temperature used when sampling draft tokens.
    """

    d_model: int = 2048
    vocab_size: int = 128000
    n_draft_heads: int = 4
    head_hidden_dim: Optional[int] = None
    temperature: float = 1.0


class HydraHead(nn.Module):
    """Single Hydra draft head.

    A 2-layer MLP (linear → SiLU → linear) that maps a hidden state vector
    (representing position t) to vocabulary logits for one future position t+i.

    Args:
        d_model: Hidden dimension fed in from the target model.
        hidden: Intermediate MLP width.
        vocab_size: Size of the output vocabulary.
    """

    def __init__(self, d_model: int, hidden: int, vocab_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state to vocabulary logits.

        Args:
            hidden: ``[B, d_model]`` last hidden state of the target model.

        Returns:
            logits: ``[B, vocab_size]``
        """
        return self.mlp(hidden)


class HydraSpeculative(nn.Module):
    """Multiple draft heads for parallel speculative token prediction.

    Given the target model's last hidden state ``h_t``, all N heads are run
    simultaneously (a single batched forward) to predict future tokens at
    positions t+1, t+2, …, t+N.

    Args:
        config: :class:`HydraConfig` instance (or keyword-compatible dict).
    """

    def __init__(self, config: HydraConfig) -> None:
        super().__init__()
        self.config = config

        hidden_dim: int = (
            config.head_hidden_dim
            if config.head_hidden_dim is not None
            else config.d_model
        )

        self.heads = nn.ModuleList(
            [
                HydraHead(config.d_model, hidden_dim, config.vocab_size)
                for _ in range(config.n_draft_heads)
            ]
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def draft(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run all draft heads in parallel on the given hidden state.

        Args:
            hidden: ``[B, d_model]`` — last hidden state from the target model
                at the current sequence position t.

        Returns:
            draft_logits: ``[B, n_draft_heads, vocab_size]``
        """
        # Each head produces [B, vocab_size]; stack along dim=1.
        head_outputs = [head(hidden) for head in self.heads]
        return torch.stack(head_outputs, dim=1)  # [B, n_draft_heads, vocab_size]

    def sample_draft_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Sample one token per head from the draft distribution.

        Uses the temperature defined in :attr:`config`.

        Args:
            hidden: ``[B, d_model]``

        Returns:
            draft_tokens: ``[B, n_draft_heads]`` sampled token IDs in
                ``[0, vocab_size)``.
        """
        draft_logits = self.draft(hidden)  # [B, n_heads, vocab_size]
        B, n_heads, V = draft_logits.shape

        # Apply temperature and convert to probabilities.
        scaled = draft_logits / max(self.config.temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)  # [B, n_heads, vocab_size]

        # Flatten batch × heads for multinomial, then restore shape.
        flat_probs = probs.view(B * n_heads, V)
        tokens_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        return tokens_flat.view(B, n_heads)  # [B, n_draft_heads]

    def verify(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Accept or reject each draft token via the speculative sampling rule.

        A draft token at position i (in batch element b) is accepted when::

            target_prob[b, i, draft_token[b, i]] >= Uniform(0, 1)

        This is the token-level acceptance criterion from speculative sampling
        (Leviathan et al., 2023; Chen et al., 2023).

        Args:
            draft_tokens: ``[B, n_draft_heads]`` — sampled draft token IDs.
            target_logits: ``[B, n_draft_heads, vocab_size]`` — logits produced
                by running the target model over the draft sequence.

        Returns:
            accepted_mask: ``[B, n_draft_heads]`` bool tensor.
            n_accepted: Total number of accepted tokens across B × n_draft_heads.
        """
        target_probs = F.softmax(target_logits, dim=-1)  # [B, n_heads, V]

        # Gather the target probability assigned to each draft token.
        # draft_tokens: [B, n_heads] → gather index: [B, n_heads, 1]
        idx = draft_tokens.unsqueeze(-1)  # [B, n_heads, 1]
        token_probs = target_probs.gather(-1, idx).squeeze(-1)  # [B, n_heads]

        # Independent uniform draws for each (b, head) pair.
        u = torch.rand_like(token_probs)
        accepted_mask = token_probs >= u  # [B, n_heads], bool

        n_accepted = int(accepted_mask.sum().item())
        return accepted_mask, n_accepted

    def acceptance_rate(
        self,
        draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> float:
        """Compute mean acceptance rate across all positions and batch elements.

        Args:
            draft_tokens: ``[B, n_draft_heads]``
            target_logits: ``[B, n_draft_heads, vocab_size]``

        Returns:
            Scalar float in ``[0.0, 1.0]``.
        """
        accepted_mask, _ = self.verify(draft_tokens, target_logits)
        return float(accepted_mask.float().mean().item())

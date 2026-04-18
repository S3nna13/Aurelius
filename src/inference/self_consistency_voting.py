r"""Self-consistency voting via marginalization over reasoning paths.

This module follows the notation of Wang et al. (2022): given an input ``x``
and sampled reasoning paths ``z^(1), ..., z^(M)``, estimate the answer
distribution by marginalizing over paths,

    p(a | x) \approx \sum_{m=1}^M p(z^(m) | x) p(a | x, z^(m)).

The implementation is fully differentiable and exposes both a vectorized and a
reference loop formulation for equivalence testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig


@dataclass
class SelfConsistencyVotingOutput:
    """Outputs of self-consistency voting."""

    log_p_a_given_x: torch.Tensor
    p_a_given_x: torch.Tensor
    log_p_z_given_x: torch.Tensor
    p_z_given_x: torch.Tensor
    log_p_a_given_x_z: torch.Tensor


class SelfConsistencyVoting(nn.Module):
    """Marginalize answer probabilities over sampled reasoning paths ``z``.

    Args:
        config: Aurelius model config used for ``d_model`` and vocabulary size.
        hidden_dim: Hidden size of the lightweight voting network.
    """

    def __init__(self, config: AureliusConfig, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or config.d_model

        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.fuse = nn.Linear(4 * config.d_model, hidden_dim)
        self.path_head = nn.Linear(hidden_dim, 1)
        self.answer_head = nn.Linear(hidden_dim, config.vocab_size)

    def _normalize_mask(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(input_ids, dtype=torch.bool)
        if mask.shape != input_ids.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match input shape {tuple(input_ids.shape)}"
            )
        return mask.to(dtype=torch.bool)

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mask_f = mask.to(dtype=h.dtype).unsqueeze(-1)
        numerator = (h * mask_f).sum(dim=dim)
        denominator = mask_f.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    def _encode_x(
        self, x: torch.Tensor, x_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_mask = self._normalize_mask(x, x_mask)
        if not x_mask.any(dim=1).all():
            raise ValueError("each batch element in x must contain at least one unmasked token")
        h_x_tokens = self.token_embed(x)
        h_x = self._masked_mean(h_x_tokens, x_mask, dim=1)
        return h_x, x_mask

    def _encode_z(
        self,
        z: torch.Tensor,
        z_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mask = self._normalize_mask(z, z_mask)
        valid_z = z_mask.any(dim=-1)
        if not valid_z.any(dim=1).all():
            raise ValueError("each batch element must contain at least one valid reasoning path z")

        B, M, T = z.shape
        h_z_tokens = self.token_embed(z.view(B * M, T)).view(B, M, T, self.config.d_model)
        h_z = self._masked_mean(h_z_tokens, z_mask, dim=2)
        return h_z, z_mask, valid_z

    def _joint_features(self, h_x: torch.Tensor, h_z: torch.Tensor) -> torch.Tensor:
        h_x = h_x.unsqueeze(1).expand(-1, h_z.shape[1], -1)
        return torch.cat([h_x, h_z, h_x - h_z, h_x * h_z], dim=-1)

    def _compute_log_p_a_given_x_z(
        self,
        h_x: torch.Tensor,
        h_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_xz = torch.tanh(self.fuse(self._joint_features(h_x, h_z)))
        log_p_a_given_x_z = F.log_softmax(self.answer_head(h_xz), dim=-1)
        return h_xz, log_p_a_given_x_z

    def _compute_log_p_z_given_x(
        self,
        h_xz: torch.Tensor,
        valid_z: torch.Tensor,
    ) -> torch.Tensor:
        logit_z = self.path_head(h_xz).squeeze(-1)
        logit_z = logit_z.masked_fill(~valid_z, torch.finfo(logit_z.dtype).min)
        return F.log_softmax(logit_z, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        use_reference: bool = False,
    ) -> SelfConsistencyVotingOutput:
        """Return ``p(a | x)`` by marginalizing over reasoning paths ``z``.

        Args:
            x: Prompt token ids of shape ``(B, S_x)``.
            z: Sampled reasoning token ids of shape ``(B, M, S_z)``.
            x_mask: Optional boolean mask for ``x``.
            z_mask: Optional boolean mask for ``z``.
            use_reference: Use the loop-based reference marginalization path.
        """
        if x.dim() != 2:
            raise ValueError(f"x must have shape (B, S_x), got {tuple(x.shape)}")
        if z.dim() != 3:
            raise ValueError(f"z must have shape (B, M, S_z), got {tuple(z.shape)}")
        if x.shape[0] != z.shape[0]:
            raise ValueError("x and z must have the same batch size")

        h_x, x_mask = self._encode_x(x, x_mask)
        h_z, z_mask, valid_z = self._encode_z(z, z_mask)
        h_xz, log_p_a_given_x_z = self._compute_log_p_a_given_x_z(h_x, h_z)
        log_p_z_given_x = self._compute_log_p_z_given_x(h_xz, valid_z)

        if use_reference:
            B, M, V = log_p_a_given_x_z.shape
            log_p_a_given_x = []
            for b in range(B):
                terms = []
                for m in range(M):
                    if valid_z[b, m]:
                        terms.append(log_p_z_given_x[b, m] + log_p_a_given_x_z[b, m])
                log_p_a_given_x.append(torch.logsumexp(torch.stack(terms, dim=0), dim=0))
            log_p_a_given_x = torch.stack(log_p_a_given_x, dim=0)
        else:
            log_p_a_given_x = torch.logsumexp(
                log_p_z_given_x.unsqueeze(-1) + log_p_a_given_x_z,
                dim=1,
            )

        p_a_given_x = log_p_a_given_x.exp()
        p_z_given_x = log_p_z_given_x.exp()

        return SelfConsistencyVotingOutput(
            log_p_a_given_x=log_p_a_given_x,
            p_a_given_x=p_a_given_x,
            log_p_z_given_x=log_p_z_given_x,
            p_z_given_x=p_z_given_x,
            log_p_a_given_x_z=log_p_a_given_x_z,
        )

    def loss(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Negative log-likelihood for target answers ``a``."""
        output = self(
            x=x,
            z=z,
            x_mask=x_mask,
            z_mask=z_mask,
            use_reference=use_reference,
        )
        return F.nll_loss(output.log_p_a_given_x, a)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        z_mask: torch.Tensor | None = None,
        use_reference: bool = False,
    ) -> torch.Tensor:
        """Return ``argmax_a p(a | x)`` for each batch element."""
        output = self(
            x=x,
            z=z,
            x_mask=x_mask,
            z_mask=z_mask,
            use_reference=use_reference,
        )
        return output.p_a_given_x.argmax(dim=-1)

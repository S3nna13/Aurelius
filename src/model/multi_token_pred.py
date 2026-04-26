"""Multi-Token Prediction (MTP) — lightweight implementation.

Provides MTPConfig, MTPHead, mtp_loss, MTPModel, and MTPTrainer for
training a backbone transformer to predict multiple future tokens
simultaneously.

API contract:
  loss, logits, pkv = backbone(input_ids)   # plain tuple, no labels kwarg
  Model attrs: .embed, .layers, .norm, .lm_head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------


@dataclass
class MTPConfig:
    """Configuration for Multi-Token Prediction heads."""

    n_future_tokens: int = 4
    d_model: int = 64
    vocab_size: int = 256
    loss_weights: list[float] | None = None

    def __post_init__(self) -> None:
        if self.loss_weights is None:
            self.loss_weights = [1.0 / self.n_future_tokens] * self.n_future_tokens
        if len(self.loss_weights) != self.n_future_tokens:
            raise ValueError(
                f"loss_weights length ({len(self.loss_weights)}) must equal "
                f"n_future_tokens ({self.n_future_tokens})"
            )


# ---------------------------------------------------------------------------
# MTPHead
# ---------------------------------------------------------------------------


class MTPHead(nn.Module):
    """n_future_tokens separate linear heads projecting hidden states to vocab logits.

    forward(hidden) returns a list of (B, T, V) logits, one per future position.
    """

    def __init__(self, config: MTPConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.vocab_size) for _ in range(config.n_future_tokens)]
        )

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            hidden: (B, T, D) hidden states from backbone.

        Returns:
            List of n_future_tokens tensors, each (B, T, V).
        """
        return [head(hidden) for head in self.heads]


# ---------------------------------------------------------------------------
# mtp_loss
# ---------------------------------------------------------------------------


def mtp_loss(
    logits_list: list[torch.Tensor],
    input_ids: torch.Tensor,
    loss_weights: list[float],
) -> torch.Tensor:
    """Compute weighted MTP cross-entropy loss.

    For head k (0-indexed), the target is input_ids[:, k+1:] and
    the logits are logits_list[k][:, :-k-1].

    Args:
        logits_list: List of (B, T, V) logits from MTPHead.
        input_ids: (B, T) token ids.
        loss_weights: Per-head weight for combining losses.

    Returns:
        Scalar weighted loss.
    """
    total = torch.tensor(0.0, device=input_ids.device, dtype=logits_list[0].dtype)

    for k, (logits_k, w) in enumerate(zip(logits_list, loss_weights)):
        shift = k + 1
        # logits_k: (B, T, V) -> use positions [0 .. T-shift-1]
        # targets:  input_ids[:, shift:]
        trunc_logits = logits_k[:, :-shift, :].contiguous()
        targets = input_ids[:, shift:].contiguous()

        B, S, V = trunc_logits.shape
        loss_k = F.cross_entropy(
            trunc_logits.view(B * S, V),
            targets.view(B * S),
        )
        total = total + w * loss_k

    return total


# ---------------------------------------------------------------------------
# MTPModel
# ---------------------------------------------------------------------------


class MTPModel(nn.Module):
    """Wraps a backbone transformer + MTPHead.

    forward(input_ids) runs the backbone to get hidden states,
    applies MTP heads, and returns (primary_logits, auxiliary_logits_list).

    Backbone API:
        loss, logits, pkv = backbone(input_ids)
        Hidden states obtained via: backbone.embed -> backbone.layers -> backbone.norm
    """

    def __init__(self, backbone: nn.Module, config: MTPConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.mtp_head = MTPHead(config)

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run backbone embedding + layers + norm to get hidden states."""
        x = self.backbone.embed(input_ids)

        # We need freqs_cis for the attention layers
        S = input_ids.shape[1]
        freqs_cis = self.backbone.freqs_cis[:S]

        for layer in self.backbone.layers:
            x, _ = layer(x, freqs_cis, mask=None, past_kv=None)

        x = self.backbone.norm(x)
        return x

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token ids.

        Returns:
            (primary_logits, auxiliary_logits_list):
                primary_logits: (B, T, V) from backbone's lm_head.
                auxiliary_logits_list: list of n_future_tokens (B, T, V) tensors.
        """
        hidden = self._get_hidden_states(input_ids)
        primary_logits = self.backbone.lm_head(hidden)
        auxiliary_logits_list = self.mtp_head(hidden)
        return primary_logits, auxiliary_logits_list


# ---------------------------------------------------------------------------
# MTPTrainer
# ---------------------------------------------------------------------------


class MTPTrainer:
    """Training helper for MTPModel.

    Computes combined primary + auxiliary MTP loss, runs backward, and steps optimizer.
    """

    def __init__(
        self,
        model: MTPModel,
        config: MTPConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Run one training step.

        Args:
            input_ids: (B, T) token ids.

        Returns:
            dict with:
                total_loss: float — combined loss.
                per_head_losses: list[float] — per-head CE losses.
        """
        self.model.train()
        self.optimizer.zero_grad()

        primary_logits, aux_logits_list = self.model(input_ids)

        # Primary (next-token) loss: logits[:, :-1] vs input_ids[:, 1:]
        B, T, V = primary_logits.shape
        primary_loss = F.cross_entropy(
            primary_logits[:, :-1, :].contiguous().view((T - 1) * B, V),
            input_ids[:, 1:].contiguous().view((T - 1) * B),
        )

        # Auxiliary MTP losses
        per_head_losses: list[float] = []
        aux_total = torch.tensor(0.0, device=input_ids.device, dtype=primary_logits.dtype)
        weights = self.config.loss_weights

        for k, (logits_k, w) in enumerate(zip(aux_logits_list, weights)):
            shift = k + 1
            trunc_logits = logits_k[:, :-shift, :].contiguous()
            targets = input_ids[:, shift:].contiguous()
            Bs, Ss, Vs = trunc_logits.shape
            loss_k = F.cross_entropy(
                trunc_logits.view(Bs * Ss, Vs),
                targets.view(Bs * Ss),
            )
            per_head_losses.append(loss_k.item())
            aux_total = aux_total + w * loss_k

        total_loss = primary_loss + aux_total
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "per_head_losses": per_head_losses,
        }

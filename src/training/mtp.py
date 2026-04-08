"""Multi-Token Prediction (MTP) auxiliary training objective.

Reference: DeepSeek-V3 / GLM-5.1 — instead of predicting only the next token,
predict the next K tokens simultaneously from the same hidden states. This
provides K× the training signal and strengthens intermediate representations.

Architecture:
    MTPHead:        one head per prediction depth (depth=d predicts token t+d).
    MTPObjective:   manages k heads and computes the averaged, weighted MTP loss.
    MTPTrainer:     wraps a main model + MTPObjective; uses a forward hook to
                    capture last-layer hidden states without modifying the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy import to avoid pulling in the full model at module load time.
# RMSNorm is small and always available.
from src.model.rms_norm import RMSNorm


# ---------------------------------------------------------------------------
# MTPHead
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Single MTP prediction head for one future depth.

    Given hidden states of shape (B, S, d_model) the head:
    1. Normalises via RMSNorm.
    2. Projects to vocabulary logits: (B, S - depth, vocab_size).

    The caller is responsible for slicing hidden_states[:, :-depth, :] before
    passing it in, so that the output sequence length already matches the
    target length (input_ids[:, depth:]).
    """

    def __init__(self, d_model: int, vocab_size: int, depth: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.depth = depth
        self.norm = RMSNorm(d_model, eps=eps)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, S', d_model) — already sliced to drop the last
                           `depth` positions, i.e. S' = S - depth.

        Returns:
            logits: (B, S', vocab_size)
        """
        return self.proj(self.norm(hidden_states))


# ---------------------------------------------------------------------------
# MTPObjective
# ---------------------------------------------------------------------------

class MTPObjective(nn.Module):
    """Auxiliary MTP objective: k heads, one per future depth (1 … k).

    Usage::

        objective = MTPObjective(k=3, d_model=2048, vocab_size=128000)
        loss = objective.compute_mtp_loss(hidden_states, input_ids, mtp_weight=0.3)
    """

    def __init__(
        self,
        k: int,
        d_model: int,
        vocab_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.k = k
        self.heads = nn.ModuleList(
            [MTPHead(d_model, vocab_size, depth=d, eps=eps) for d in range(1, k + 1)]
        )

    def compute_mtp_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        mtp_weight: float = 0.3,
    ) -> torch.Tensor:
        """Compute the averaged, weighted MTP loss.

        For each depth d in 1 … k:
            - Input:  hidden_states[:, :-d, :]   (B, S-d, d_model)
            - Target: input_ids[:, d:]            (B, S-d)
            - Loss:   cross-entropy of head_d(input) vs target

        The k losses are averaged and then multiplied by mtp_weight.

        Args:
            hidden_states: (B, S, d_model) — last-layer hidden states from the
                           transformer (detach optional; gradients flow by default).
            input_ids:     (B, S) — token ids.
            mtp_weight:    scalar weight applied to the averaged MTP loss.

        Returns:
            Scalar mtp_loss tensor (with grad_fn if hidden_states requires grad).
        """
        total_loss = hidden_states.new_zeros(())  # scalar zero on same device/dtype

        for head in self.heads:
            d = head.depth
            h_slice = hidden_states[:, :-d, :]       # (B, S-d, d_model)
            targets = input_ids[:, d:]               # (B, S-d)

            logits = head(h_slice)                   # (B, S-d, vocab_size)
            B, Sd, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * Sd, V),
                targets.reshape(B * Sd),
            )
            total_loss = total_loss + loss

        avg_loss = total_loss / self.k
        return mtp_weight * avg_loss


# ---------------------------------------------------------------------------
# MTPTrainer
# ---------------------------------------------------------------------------

class MTPTrainer:
    """Trains a main transformer model with an auxiliary MTP objective.

    The trainer captures last-layer hidden states via a PyTorch forward hook,
    avoiding any modifications to the model itself.

    Args:
        model:          The AureliusTransformer (or compatible) model.
        mtp_objective:  An MTPObjective instance.
        optimizer:      A torch.optim.Optimizer covering both model and objective
                        parameters.
        mtp_weight:     Weight applied to the MTP loss term.
        grad_clip:      Max gradient norm (0 or None to disable).
    """

    def __init__(
        self,
        model: nn.Module,
        mtp_objective: MTPObjective,
        optimizer: torch.optim.Optimizer,
        mtp_weight: float = 0.3,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.model = model
        self.mtp_objective = mtp_objective
        self.optimizer = optimizer
        self.mtp_weight = mtp_weight
        self.grad_clip = grad_clip

        # Cache for hidden states captured via the hook.
        self._hidden_cache: dict[str, torch.Tensor] = {}

        # Register hook on the last transformer layer.
        # AureliusTransformer stores layers in model.layers (nn.ModuleList).
        # Each layer returns (hidden, kv) from forward(); we grab output[0].
        last_layer = model.layers[-1]

        def _hook(module: nn.Module, inp: tuple, output: object) -> None:  # noqa: ARG001
            if isinstance(output, tuple):
                self._hidden_cache["last"] = output[0]
            else:
                self._hidden_cache["last"] = output

        self._hook_handle = last_layer.register_forward_hook(_hook)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, input_ids: torch.Tensor) -> dict[str, float]:
        """Run one training step.

        Args:
            input_ids: (B, S) — token ids. Labels are derived by shifting.

        Returns:
            dict with keys 'main_loss', 'mtp_loss', 'total_loss' (Python floats).
        """
        self.model.train()
        self.mtp_objective.train()
        self.optimizer.zero_grad()

        # Forward pass — model returns (loss, logits, present_kv).
        # Pass input_ids as both input and labels so the model computes the
        # standard next-token CE loss internally.
        main_loss, _logits, _kv = self.model(input_ids, labels=input_ids)

        # hidden_states captured by the hook (before final model norm, but
        # that is fine — the MTPHead has its own RMSNorm).
        hidden_states = self._hidden_cache.get("last")
        if hidden_states is None:
            raise RuntimeError(
                "Hidden states not captured. Ensure the forward hook is registered "
                "on model.layers[-1] and that a forward pass has been run."
            )

        mtp_loss = self.mtp_objective.compute_mtp_loss(
            hidden_states, input_ids, mtp_weight=self.mtp_weight
        )

        total_loss = main_loss + mtp_loss
        total_loss.backward()

        if self.grad_clip:
            all_params = (
                list(self.model.parameters()) + list(self.mtp_objective.parameters())
            )
            nn.utils.clip_grad_norm_(all_params, self.grad_clip)

        self.optimizer.step()

        return {
            "main_loss": main_loss.item(),
            "mtp_loss": mtp_loss.item(),
            "total_loss": total_loss.item(),
        }

    def remove_hook(self) -> None:
        """Remove the forward hook (call when done training)."""
        self._hook_handle.remove()

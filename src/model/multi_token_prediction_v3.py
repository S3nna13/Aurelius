"""Multi-Token Prediction (MTP) — v3.

Reference: Gloeckle et al. 2024 — "Better & Faster Large Language Models via
Multi-Token Prediction" (Meta FAIR).

Architecture:
  - Shared transformer trunk (Embedding + TransformerEncoderLayers + LayerNorm)
  - K independent MultiTokenHead modules, each predicting the token at offset
    (1, 2, ..., K) from every position.
  - MultiTokenLoss combines cross-entropy losses from all K heads with
    configurable per-head weights.
  - MTPTrainer wraps a training step with gradient clipping.
  - SpeculativeDecodingFromMTP uses the K heads for greedy draft generation and
    acceptance checking (no separate draft model required).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# MultiTokenHead
# ---------------------------------------------------------------------------


class MultiTokenHead(nn.Module):
    """Single prediction head for one future token offset.

    Architecture: LayerNorm -> Linear(d_model, d_model) -> GELU ->
                  Linear(d_model, vocab_size)

    Args:
        d_model:    Model hidden dimension.
        vocab_size: Vocabulary size.
        offset:     Which future token this head predicts (1 = next token,
                    2 = two tokens ahead, etc.).
    """

    def __init__(self, d_model: int, vocab_size: int, offset: int) -> None:
        super().__init__()
        if offset < 1:
            raise ValueError(f"offset must be >= 1, got {offset}")
        self.offset = offset
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.proj2 = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (B, T, d_model)

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.norm(hidden_states)
        x = self.proj1(x)
        x = F.gelu(x)
        logits = self.proj2(x)
        return logits


# ---------------------------------------------------------------------------
# MultiTokenPredictionModel
# ---------------------------------------------------------------------------


class MultiTokenPredictionModel(nn.Module):
    """Transformer trunk with K parallel prediction heads.

    Args:
        d_model:        Hidden dimension.
        n_layers:       Number of TransformerEncoderLayers.
        n_heads:        Number of attention heads per layer.
        vocab_size:     Vocabulary size.
        k_predictions:  Number of future tokens to predict simultaneously.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        k_predictions: int = 4,
    ) -> None:
        super().__init__()
        if k_predictions < 1:
            raise ValueError(f"k_predictions must be >= 1, got {k_predictions}")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.k_predictions = k_predictions

        # Trunk
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # K prediction heads (offsets 1 .. k_predictions)
        self.heads = nn.ModuleList(
            [MultiTokenHead(d_model, vocab_size, offset=i + 1) for i in range(k_predictions)]
        )

    def forward(self, input_ids: Tensor) -> list[Tensor]:
        """
        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            List of K logit tensors, each (B, T, vocab_size).
        """
        x = self.embedding(input_ids)  # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)
        hidden = self.norm(x)  # (B, T, d_model)

        logits_list: list[Tensor] = [head(hidden) for head in self.heads]
        return logits_list


# ---------------------------------------------------------------------------
# MultiTokenLoss
# ---------------------------------------------------------------------------


class MultiTokenLoss(nn.Module):
    """Combined cross-entropy loss across all K prediction heads.

    For head at offset j (1-indexed):
        targets_j  = input_ids[:, j:]              shape (B, T-j)
        logits_j   = logits_list[j-1][:, :-j, :]  shape (B, T-j, V)
        loss_j     = CE(logits_j.reshape(-1,V), targets_j.reshape(-1))

    total_loss = sum_j(w_j * loss_j)

    Args:
        k_predictions: Number of prediction heads.
        weights:       Per-head weights (length k_predictions).  Defaults to
                       uniform 1/k for all heads.
    """

    def __init__(
        self,
        k_predictions: int,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if k_predictions < 1:
            raise ValueError(f"k_predictions must be >= 1, got {k_predictions}")
        self.k_predictions = k_predictions
        if weights is None:
            weights = [1.0 / k_predictions] * k_predictions
        if len(weights) != k_predictions:
            raise ValueError(f"len(weights)={len(weights)} != k_predictions={k_predictions}")
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits_list: list[Tensor], input_ids: Tensor) -> tuple[Tensor, list[float]]:
        """
        Args:
            logits_list: List of K tensors, each (B, T, V).
            input_ids:   (B, T) integer token ids.

        Returns:
            total_loss:      scalar Tensor.
            per_head_losses: list of K float values (detached).
        """
        T = input_ids.shape[1]
        per_head_losses: list[float] = []
        total_loss: Tensor = torch.zeros(1, device=input_ids.device, dtype=torch.float32).squeeze()

        for j_idx in range(self.k_predictions):
            j = j_idx + 1  # offset (1-indexed)
            if T <= j:
                raise ValueError(f"seq_len={T} must be > offset={j} to compute loss")
            # Align logits and targets
            logits_j = logits_list[j_idx][:, :-j, :]  # (B, T-j, V)
            targets_j = input_ids[:, j:]  # (B, T-j)

            B_Tj = logits_j.shape[0] * logits_j.shape[1]
            V = logits_j.shape[2]
            loss_j = F.cross_entropy(
                logits_j.reshape(B_Tj, V),
                targets_j.reshape(B_Tj),
            )
            w_j = self.weights[j_idx]
            total_loss = total_loss + w_j * loss_j
            per_head_losses.append(loss_j.item())

        return total_loss, per_head_losses


# ---------------------------------------------------------------------------
# MTPTrainer
# ---------------------------------------------------------------------------


class MTPTrainer:
    """Training loop wrapper for MultiTokenPredictionModel.

    Args:
        model:     MultiTokenPredictionModel instance.
        loss_fn:   MultiTokenLoss instance.
        optimizer: Any torch.optim.Optimizer.
    """

    def __init__(
        self,
        model: MultiTokenPredictionModel,
        loss_fn: MultiTokenLoss,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, input_ids: Tensor) -> dict:
        """Run one forward + backward + optimiser step.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            dict with keys:
                total_loss      (float)
                per_head_losses (list[float])
                grad_norm       (float, total L2 norm of all gradients)
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits_list = self.model(input_ids)
        total_loss, per_head_losses = self.loss_fn(logits_list, input_ids)
        total_loss.backward()

        # Gradient clipping
        grad_norm_val = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "per_head_losses": per_head_losses,
            "grad_norm": grad_norm_val.item(),
        }


# ---------------------------------------------------------------------------
# SpeculativeDecodingFromMTP
# ---------------------------------------------------------------------------


class SpeculativeDecodingFromMTP:
    """Use MTP heads for speculative draft generation and acceptance checking.

    Args:
        model: MultiTokenPredictionModel with K prediction heads.
        k:     Number of draft tokens to generate (must be <= model.k_predictions).
    """

    def __init__(self, model: MultiTokenPredictionModel, k: int = 4) -> None:
        if k > model.k_predictions:
            raise ValueError(f"k={k} > model.k_predictions={model.k_predictions}")
        self.model = model
        self.k = k

    @torch.no_grad()
    def draft(self, input_ids: Tensor) -> Tensor:
        """Generate k draft tokens using greedy argmax from each head.

        Uses the hidden state at the *last* position to predict the next k
        tokens.

        Args:
            input_ids: (B, T) integer token ids.

        Returns:
            draft_tokens: (B, k) integer token ids.
        """
        self.model.eval()
        logits_list = self.model(input_ids)  # list of K tensors (B, T, V)

        draft_tokens_list: list[Tensor] = []
        for j_idx in range(self.k):
            logits_last = logits_list[j_idx][:, -1, :]  # (B, V)
            draft_tok = torch.argmax(logits_last, dim=-1, keepdim=True)  # (B, 1)
            draft_tokens_list.append(draft_tok)

        return torch.cat(draft_tokens_list, dim=1)  # (B, k)

    def verify_and_accept(self, draft_tokens: Tensor, target_logits: Tensor) -> tuple[Tensor, int]:
        """Greedy acceptance check against verifier logits.

        Accepts a draft token at position i if the verifier's greedy choice
        matches the draft.

        Args:
            draft_tokens:  (B, k) integer draft token ids.
            target_logits: (B, k, V) logits from a verifier model.

        Returns:
            accepted:   bool Tensor (B, k) — True where draft matches verifier.
            n_accepted: int — mean number of accepted tokens across the batch.
        """
        verifier_greedy = torch.argmax(target_logits, dim=-1)  # (B, k)
        accepted = verifier_greedy == draft_tokens  # (B, k) bool
        n_accepted = int(accepted.float().sum(dim=1).mean().round().item())
        return accepted, n_accepted

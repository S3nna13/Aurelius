"""Token-Budget Inference with Adaptive Early Exit.

Each token independently decides at which transformer layer to stop processing,
based on a learned confidence threshold. Tokens that are "easy" exit early,
saving compute; hard tokens proceed to deeper layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Exit Classifier
# ---------------------------------------------------------------------------

class ExitClassifier(nn.Module):
    """Lightweight 2-class classifier that predicts exit probability per token.

    Args:
        d_model: Hidden dimension of the transformer.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-token exit probability.

        Args:
            x: [B, T, d_model] hidden states.

        Returns:
            exit_probs: [B, T] probability of exiting at this layer.
        """
        # [B, T, 2]
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        # Index 1 = "exit" class
        exit_probs = probs[..., 1]  # [B, T]
        return exit_probs

    def should_exit(self, x: Tensor, threshold: float) -> Tensor:
        """Return boolean mask: True where a token should exit at this layer.

        Args:
            x: [B, T, d_model] hidden states.
            threshold: Exit if exit_prob > threshold.

        Returns:
            bool_mask: [B, T] bool tensor.
        """
        exit_probs = self.forward(x)
        return exit_probs > threshold


# ---------------------------------------------------------------------------
# Multi-head Self-Attention (pure PyTorch, no flash_attn)
# ---------------------------------------------------------------------------

class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, C]

        # Reshape to [B, n_heads, T, head_dim]
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale  # [B, H, T, T]
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _FFN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Early Exit Layer
# ---------------------------------------------------------------------------

class EarlyExitLayer(nn.Module):
    """One transformer layer augmented with an exit classifier and a shallow LM head.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        layer_idx: 0-based index of this layer (used for bookkeeping).
        vocab_size: Vocabulary size for the shallow exit LM head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        vocab_size: int = 64,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.transformer_block = _TransformerBlock(d_model, n_heads)
        self.exit_head = ExitClassifier(d_model)
        self.lm_head_proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x: Tensor,
        exit_threshold: float,
        already_exited: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process hidden states and decide per-token exits.

        Args:
            x: [B, T, d_model] incoming hidden states.
            exit_threshold: Probability cutoff to trigger early exit.
            already_exited: [B, T] bool mask of tokens that already exited at
                a previous layer. Those tokens are not re-processed.

        Returns:
            x_out: [B, T, d_model] — updated hidden states (exited tokens keep
                their previous representation so downstream layers can skip them).
            early_logits: [B, T, vocab] — shallow LM head predictions.
            exit_mask: [B, T] bool — True for tokens exiting at THIS layer.
        """
        B, T, _ = x.shape
        if already_exited is None:
            already_exited = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        # Run the transformer block for ALL tokens (necessary for correct
        # attention; masking individual tokens inside attention would break
        # the causal structure).
        x_new = self.transformer_block(x)

        # Compute exit decision on the NEW representations
        exit_mask_raw = self.exit_head.should_exit(x_new, exit_threshold)  # [B, T]

        # Only tokens that have NOT already exited can exit here
        exit_mask = exit_mask_raw & (~already_exited)  # [B, T]

        # For tokens that do NOT exit here and have NOT exited before, update x
        active = ~(exit_mask | already_exited)  # tokens still "alive"
        # Build x_out: exited tokens keep previous x; active tokens get x_new
        x_out = torch.where(
            (exit_mask | already_exited).unsqueeze(-1).expand_as(x),
            x,
            x_new,
        )

        # Shallow predictions for ALL tokens (used for loss on exited ones)
        early_logits = self.lm_head_proj(x_out)  # [B, T, vocab]

        return x_out, early_logits, exit_mask


# ---------------------------------------------------------------------------
# Adaptive Early Exit Model
# ---------------------------------------------------------------------------

class AdaptiveEarlyExitModel(nn.Module):
    """Full model with per-token adaptive early exit across N layers.

    Args:
        d_model: Hidden dimension.
        vocab_size: Vocabulary size.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        exit_threshold: Default confidence threshold for exiting.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        exit_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.exit_threshold = exit_threshold

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                EarlyExitLayer(d_model, n_heads, layer_idx=i, vocab_size=vocab_size)
                for i in range(n_layers)
            ]
        )
        self.final_lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def set_threshold(self, threshold: float) -> None:
        """Update the exit threshold used during inference."""
        self.exit_threshold = threshold

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run forward pass with adaptive early exit.

        Args:
            input_ids: [B, T] integer token ids.

        Returns:
            logits: [B, T, vocab] — final prediction per token from its exit
                layer (or the final LM head for tokens that never exited).
            layer_assignments: [B, T] long tensor — 0-indexed layer at which
                each token exited. Value == n_layers means the token went
                through all layers and used the final head.
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)  # [B, T, d_model]

        # Accumulate results
        logits = torch.zeros(B, T, self.vocab_size, device=device)
        layer_assignments = torch.full(
            (B, T), fill_value=self.n_layers, dtype=torch.long, device=device
        )
        already_exited = torch.zeros(B, T, dtype=torch.bool, device=device)

        for layer_idx, layer in enumerate(self.layers):
            x, early_logits, exit_mask = layer(
                x, self.exit_threshold, already_exited=already_exited
            )

            # Record logits/assignments for newly-exited tokens
            new_exits = exit_mask  # [B, T]
            if new_exits.any():
                logits[new_exits] = early_logits[new_exits]
                layer_assignments[new_exits] = layer_idx

            already_exited = already_exited | new_exits

            # Short-circuit if every token has exited
            if already_exited.all():
                break

        # Tokens that made it to the end use the final LM head
        final_mask = ~already_exited  # [B, T]
        if final_mask.any():
            final_logits = self.final_lm_head(x)  # [B, T, vocab]
            logits[final_mask] = final_logits[final_mask]
            # layer_assignments for these stay at n_layers (already set above)

        return logits, layer_assignments

    def compute_loss(self, input_ids: Tensor) -> Tensor:
        """Compute next-token prediction loss (teacher-forced).

        Args:
            input_ids: [B, T] token ids.

        Returns:
            Scalar cross-entropy loss.
        """
        logits, _ = self.forward(input_ids)
        # Shift: predict token i+1 from prefix 0..i
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, vocab]
        shift_targets = input_ids[:, 1:].contiguous()  # [B, T-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
        )
        return loss


# ---------------------------------------------------------------------------
# Early Exit Trainer
# ---------------------------------------------------------------------------

class EarlyExitTrainer:
    """Training wrapper that combines task loss with an exit-regularization term.

    Args:
        model: AdaptiveEarlyExitModel to train.
        lr: Learning rate for Adam.
        lambda_exit: Weight of the exit regularization term.
    """

    def __init__(
        self,
        model: AdaptiveEarlyExitModel,
        lr: float = 1e-3,
        lambda_exit: float = 0.1,
    ) -> None:
        self.model = model
        self.lambda_exit = lambda_exit
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def task_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy loss over all token positions.

        Args:
            logits: [B, T, vocab]
            targets: [B, T] integer targets

        Returns:
            Scalar CE loss.
        """
        B, T, V = logits.shape
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

    def exit_regularizer(
        self,
        layer_assignments: Tensor,
        target_layer_frac: float = 0.5,
    ) -> Tensor:
        """Penalize if mean exit layer exceeds the target fraction of n_layers.

        Args:
            layer_assignments: [B, T] long tensor of exit layers.
            target_layer_frac: Fraction of n_layers considered "acceptable".

        Returns:
            Scalar regularization loss (0 if below target, positive otherwise).
        """
        n_layers = self.model.n_layers
        target = target_layer_frac * n_layers
        mean_exit = layer_assignments.float().mean()
        # Only penalize if mean exit exceeds target
        penalty = F.relu(mean_exit - target)
        return penalty

    def train_step(self, input_ids: Tensor) -> Tuple[Tensor, float]:
        """One gradient update step.

        Args:
            input_ids: [B, T] integer token ids.

        Returns:
            loss: Combined scalar loss tensor.
            mean_exit_layer: Float mean layer at which tokens exited.
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits, layer_assignments = self.model(input_ids)

        # Task loss: next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        t_loss = self.task_loss(shift_logits, shift_targets)

        # Exit regularization
        e_loss = self.exit_regularizer(layer_assignments)

        loss = t_loss + self.lambda_exit * e_loss
        loss.backward()
        self.optimizer.step()

        mean_exit_layer: float = layer_assignments.float().mean().item()
        return loss.detach(), mean_exit_layer


# ---------------------------------------------------------------------------
# Early Exit Profiler
# ---------------------------------------------------------------------------

class EarlyExitProfiler:
    """Accumulates per-batch layer-assignment statistics.

    Usage::

        profiler = EarlyExitProfiler()
        for batch in dataloader:
            _, assignments = model(batch)
            profiler.record_batch(assignments)
        print(profiler.mean_exit_layer())
        print(profiler.compute_flop_savings(n_layers=4))
    """

    def __init__(self) -> None:
        self._total_tokens: int = 0
        self._sum_layers: float = 0.0
        self._layer_counts: Dict[int, int] = {}

    def record_batch(self, layer_assignments: Tensor) -> None:
        """Accumulate statistics from one batch.

        Args:
            layer_assignments: [B, T] long tensor of exit layer indices.
        """
        flat = layer_assignments.reshape(-1)
        n = flat.numel()
        self._total_tokens += n
        self._sum_layers += flat.float().sum().item()

        for val, cnt in zip(*flat.unique(return_counts=True)):
            k = int(val.item())
            self._layer_counts[k] = self._layer_counts.get(k, 0) + int(cnt.item())

    def mean_exit_layer(self) -> float:
        """Return mean exit layer across all recorded tokens."""
        if self._total_tokens == 0:
            return 0.0
        return self._sum_layers / self._total_tokens

    def layer_distribution(self) -> Dict[int, float]:
        """Return fraction of tokens that exited at each layer.

        Returns:
            dict mapping layer_idx -> fraction in [0, 1].
        """
        if self._total_tokens == 0:
            return {}
        return {
            layer: count / self._total_tokens
            for layer, count in self._layer_counts.items()
        }

    def compute_flop_savings(self, n_layers: int) -> float:
        """Estimate FLOPs saved relative to running all layers.

        Defined as 1 - mean_exit_layer / n_layers.  Clipped to [0, 1].

        Args:
            n_layers: Total number of transformer layers in the model.

        Returns:
            Fraction of FLOPs saved in [0, 1].
        """
        if n_layers <= 0:
            return 0.0
        savings = 1.0 - self.mean_exit_layer() / n_layers
        return max(0.0, min(1.0, savings))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EarlyExitConfig:
    """Hyperparameter bundle for AdaptiveEarlyExitModel and EarlyExitTrainer."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 4
    n_heads: int = 4
    exit_threshold: float = 0.8
    lambda_exit: float = 0.1
    target_layer_frac: float = 0.5

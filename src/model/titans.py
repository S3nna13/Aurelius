"""Titans: Test-Time Memory via Neural Memory (Google, 2025).

Neural Memory maintains a small MLP as a memory store and performs gradient-based
updates during inference (write phase) before querying memory (read phase).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TitansConfig:
    """Configuration for the Titans layer."""

    d_model: int = 64
    n_heads: int = 4
    memory_dim: int = 32
    n_memory_layers: int = 1
    memory_lr: float = 0.01
    memory_momentum: float = 0.9
    use_persistent_memory: bool = True
    n_persistent: int = 16


# ---------------------------------------------------------------------------
# Neural Memory
# ---------------------------------------------------------------------------


class NeuralMemory(nn.Module):
    """Neural Memory module that learns to memorize at test time.

    The memory is a small MLP (key_proj -> hidden -> value_proj).  During the
    write phase gradients of a prediction loss are computed w.r.t. the MLP
    weights and those weights are updated with a momentum rule — no external
    optimizer object is used.  The read phase simply performs a forward pass
    through the current (possibly updated) MLP.
    """

    def __init__(
        self,
        d_model: int,
        memory_dim: int = 64,
        n_memory_layers: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.memory_dim = memory_dim
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Small MLP: d_model -> memory_dim -> d_model
        self.memory_mlp = nn.Sequential(
            nn.Linear(d_model, memory_dim, bias=True),
            nn.SiLU(),
            nn.Linear(memory_dim, d_model, bias=True),
        )

        # Snapshot of initial weights so reset() can restore them
        self._initial_state: dict = copy.deepcopy(self.memory_mlp.state_dict())

        # Momentum buffers keyed by parameter name
        self._momentum_buffers: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _memory_params(self) -> list[nn.Parameter]:
        return list(self.memory_mlp.parameters())

    def _init_momentum(self) -> None:
        """Initialise momentum buffers to zero if not yet present."""
        for name, param in self.memory_mlp.named_parameters():
            if name not in self._momentum_buffers:
                self._momentum_buffers[name] = torch.zeros_like(param.data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, keys: Tensor, values: Tensor) -> Tensor:
        """Write key-value pairs into memory via gradient update.

        Args:
            keys:   (batch, seq, d_model)
            values: (batch, seq, d_model)

        Returns:
            surprise: (batch, seq) — per-token prediction error magnitude
        """
        B, S, _ = keys.shape
        self._init_momentum()

        # torch.enable_grad() ensures autograd works even when the caller runs
        # inside torch.no_grad() (e.g. during inference evaluation).
        with torch.enable_grad():
            # Forward through memory MLP to get predictions
            keys_flat = keys.detach().reshape(B * S, self.d_model)
            pred_flat = self.memory_mlp(keys_flat)  # (B*S, d_model)

            targets_flat = values.detach().reshape(B * S, self.d_model)
            # Use mean reduction to keep gradient magnitudes stable
            loss = F.mse_loss(pred_flat, targets_flat, reduction="mean")

            # Compute gradients w.r.t. memory MLP parameters
            params = self._memory_params()
            grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)

        # Manual momentum update (no optimizer object)
        with torch.no_grad():
            # Gradient clipping threshold (prevents divergence on repeated writes)
            max_grad_norm = 1.0
            for (name, param), grad in zip(self.memory_mlp.named_parameters(), grads):
                if grad is None:
                    continue
                # Clip individual gradient
                grad = grad.clamp(-max_grad_norm, max_grad_norm)
                # L2 weight decay
                effective_grad = grad + self.weight_decay * param.data
                # Momentum
                buf = self._momentum_buffers[name]
                buf.mul_(self.momentum).add_(effective_grad)
                self._momentum_buffers[name] = buf
                # Gradient descent step
                param.data.sub_(self.lr * buf)

        # Compute per-token surprise (prediction error norm)
        with torch.no_grad():
            surprise_flat = (pred_flat.detach() - targets_flat).norm(dim=-1)  # (B*S,)
        surprise = surprise_flat.reshape(B, S)
        return surprise

    def read(self, queries: Tensor) -> Tensor:
        """Read from memory by querying the current MLP.

        Args:
            queries: (batch, seq, d_model)

        Returns:
            (batch, seq, d_model)
        """
        B, S, _ = queries.shape
        out = self.memory_mlp(queries.reshape(B * S, self.d_model))
        return out.reshape(B, S, self.d_model)

    def reset(self) -> None:
        """Reset memory MLP weights to initial values and clear momentum."""
        self.memory_mlp.load_state_dict(copy.deepcopy(self._initial_state))
        self._momentum_buffers.clear()


# ---------------------------------------------------------------------------
# Titans Layer
# ---------------------------------------------------------------------------


class TitansLayer(nn.Module):
    """Titans layer: attention + neural memory read + optional persistent memory.

    Architecture (MAC — Memory As Context variant):
        1. Write current x to neural memory.
        2. Read from neural memory with learned query projection.
        3. Combine: output = x + memory_read  (residual connection).

    Persistent memory tokens are learnable parameters that are always available
    regardless of sequence content.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        memory_dim: int = 64,
        use_persistent_memory: bool = True,
        n_persistent: int = 16,
        n_memory_layers: int = 1,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_persistent_memory = use_persistent_memory

        # Neural memory
        self.neural_memory = NeuralMemory(
            d_model=d_model,
            memory_dim=memory_dim,
            n_memory_layers=n_memory_layers,
            lr=memory_lr,
            momentum=memory_momentum,
        )

        # Projection for memory queries (separate from attention queries)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)

        # Output gate to blend memory read into residual stream
        self.out_gate = nn.Linear(d_model, d_model, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)

        # Persistent (always-available) memory tokens
        if use_persistent_memory:
            self.persistent_memory = nn.Parameter(
                torch.randn(n_persistent, d_model) * 0.02
            )
        else:
            self.persistent_memory = None  # type: ignore[assignment]

        # Simple self-attention (used to let x attend to persistent memory)
        head_dim = max(1, d_model // n_heads)
        # Ensure d_model is divisible by n_heads; fall back if not
        actual_n_heads = n_heads
        while d_model % actual_n_heads != 0 and actual_n_heads > 1:
            actual_n_heads //= 2
        self.attn = nn.MultiheadAttention(
            d_model, actual_n_heads, batch_first=True, bias=True
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x:    (batch, seq, d_model)
            mask: optional attention mask

        Returns:
            (batch, seq, d_model)
        """
        B, S, D = x.shape

        # 1. Write current sequence to neural memory (memory sees current context)
        self.neural_memory.write(x, x)

        # 2. Read from neural memory using projected queries
        queries = self.query_proj(x)
        memory_read = self.neural_memory.read(queries)  # (B, S, D)

        # 3. Optionally incorporate persistent memory via attention
        if self.use_persistent_memory and self.persistent_memory is not None:
            # Expand persistent tokens to batch dimension
            persist = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)  # (B, n_p, D)
            # Concatenate: sequence attends to itself + persistent memory
            kv = torch.cat([x, persist], dim=1)  # (B, S+n_p, D)
            attn_out, _ = self.attn(x, kv, kv, need_weights=False)
            memory_read = memory_read + attn_out

        # 4. Gate and residual
        memory_read = self.out_gate(memory_read)
        out = self.layer_norm(x + memory_read)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_titans_layer(config: TitansConfig) -> TitansLayer:
    """Build a TitansLayer from a TitansConfig."""
    return TitansLayer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        memory_dim=config.memory_dim,
        use_persistent_memory=config.use_persistent_memory,
        n_persistent=config.n_persistent,
        n_memory_layers=config.n_memory_layers,
        memory_lr=config.memory_lr,
        memory_momentum=config.memory_momentum,
    )

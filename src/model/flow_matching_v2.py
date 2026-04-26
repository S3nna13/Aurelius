"""Flow Matching v2 — simulation-free continuous normalizing flows.

Implements Conditional Flow Matching (CFM) with Optimal Transport paths
from Lipman et al. 2022.  Pure PyTorch only.

Key idea: learn a velocity field v_theta(x_t, t) such that integrating the
ODE dx/dt = v_theta(x_t, t) from t=0 to t=1 transforms N(0,I) into the
data distribution.  The OT straight-line path gives:

    x_t  = (1-t)*x_0 + t*x_1          (interpolation)
    u_t  = x_1 - x_0                  (target / conditional velocity)

Loss: E_{t,x_0,x_1} [ ||v_theta(x_t, t) - u_t||^2 ]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# VectorField
# ---------------------------------------------------------------------------


class VectorField(nn.Module):
    """Time-conditioned MLP velocity field.

    Input:  (x: [B, d_model], t: [B, 1])
    Output: v: [B, d_model]   -- predicted velocity
    """

    def __init__(self, d_model: int, hidden: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        in_dim = d_model + 1  # concat x and t
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            prev = hidden
        layers.append(nn.Linear(prev, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model]
            t: [B, 1]
        Returns:
            v: [B, d_model]
        """
        xt = torch.cat([x, t], dim=-1)  # [B, d_model+1]
        return self.net(xt)


# ---------------------------------------------------------------------------
# ConditionalVectorField
# ---------------------------------------------------------------------------


class ConditionalVectorField(nn.Module):
    """Velocity field conditioned on a context embedding c.

    Input:  (x: [B, d_model], t: [B, 1], cond: [B, d_cond])
    Output: v: [B, d_model]
    """

    def __init__(self, d_model: int, d_cond: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = d_model + d_cond + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, d_model]
            t:    [B, 1]
            cond: [B, d_cond]
        Returns:
            v: [B, d_model]
        """
        inp = torch.cat([x, t, cond], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# FlowMatchingLoss
# ---------------------------------------------------------------------------


class FlowMatchingLoss:
    """Conditional Flow Matching loss utilities."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # OT path
    # ------------------------------------------------------------------

    @staticmethod
    def optimal_transport_path(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ):
        """Compute OT straight-line interpolation and target velocity.

        Args:
            x0: [B, d] — source (noise)
            x1: [B, d] — target (data)
            t:  [B, 1] — time in [0, 1]

        Returns:
            xt: [B, d]  — interpolated point
            ut: [B, d]  — target (conditional) velocity
        """
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0
        return xt, ut

    # ------------------------------------------------------------------
    # CFM loss (unconditional)
    # ------------------------------------------------------------------

    def cfm_loss(self, model: VectorField, x1: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss.

        Args:
            model: VectorField
            x1:   [B, d] — data samples

        Returns:
            scalar loss
        """
        B, d = x1.shape
        device = x1.device
        # Sample time and source noise
        t = torch.rand(B, 1, device=device)  # [B, 1]
        x0 = torch.randn_like(x1)  # [B, d]
        xt, ut = self.optimal_transport_path(x0, x1, t)
        vt = model(xt, t)
        return F.mse_loss(vt, ut)

    # ------------------------------------------------------------------
    # Conditional CFM loss
    # ------------------------------------------------------------------

    def conditional_cfm_loss(
        self,
        model: ConditionalVectorField,
        x1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional CFM loss with context conditioning.

        Args:
            model: ConditionalVectorField
            x1:   [B, d]
            cond: [B, d_cond]

        Returns:
            scalar loss
        """
        B, d = x1.shape
        device = x1.device
        t = torch.rand(B, 1, device=device)
        x0 = torch.randn_like(x1)
        xt, ut = self.optimal_transport_path(x0, x1, t)
        vt = model(xt, t, cond)
        return F.mse_loss(vt, ut)


# ---------------------------------------------------------------------------
# EulerFlowSampler
# ---------------------------------------------------------------------------


class EulerFlowSampler:
    """Euler ODE solver for flow matching inference."""

    def __init__(self, n_steps: int = 100) -> None:
        self.n_steps = n_steps

    def sample(self, model: VectorField, x0: torch.Tensor) -> torch.Tensor:
        """Integrate dx/dt = v_theta(x, t) from t=0 to t=1.

        Args:
            model: VectorField
            x0:   [B, d] — initial noise

        Returns:
            x1: [B, d]
        """
        x = x0.clone()
        h = 1.0 / self.n_steps
        device = x0.device
        B = x0.shape[0]
        with torch.no_grad():
            for i in range(self.n_steps):
                t_val = i * h
                t = torch.full((B, 1), t_val, device=device)
                v = model(x, t)
                x = x + h * v
        return x

    def sample_trajectory(
        self,
        model: VectorField,
        x0: torch.Tensor,
        n_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """Return all intermediate states including start and end.

        Returns:
            List of length n_steps+1, each [B, d]
        """
        if n_steps is None:
            n_steps = self.n_steps
        x = x0.clone()
        h = 1.0 / n_steps
        device = x0.device
        B = x0.shape[0]
        trajectory = [x.clone()]
        with torch.no_grad():
            for i in range(n_steps):
                t_val = i * h
                t = torch.full((B, 1), t_val, device=device)
                v = model(x, t)
                x = x + h * v
                trajectory.append(x.clone())
        return trajectory


# ---------------------------------------------------------------------------
# SequenceFlowModel
# ---------------------------------------------------------------------------


class SequenceFlowModel(nn.Module):
    """Flow matching over flattened token embeddings.

    Encodes a sequence of discrete tokens into a continuous vector via
    embedding, runs CFM loss / sampling in that flattened space, then
    decodes back to logits.
    """

    def __init__(self, d_model: int, vocab_size: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        flat_dim = d_model * seq_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.vector_field = VectorField(flat_dim)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._loss_fn = FlowMatchingLoss()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed and flatten.

        Args:
            input_ids: [B, T]
        Returns:
            z: [B, d_model*T]
        """
        emb = self.embedding(input_ids)  # [B, T, d_model]
        return emb.reshape(emb.size(0), -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Unflatten and project to vocabulary logits.

        Args:
            z: [B, d_model*T]
        Returns:
            logits: [B, T, vocab_size]
        """
        B = z.size(0)
        h = z.reshape(B, self.seq_len, self.d_model)  # [B, T, d_model]
        return self.lm_head(h)  # [B, T, vocab_size]

    def flow_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """CFM loss on the embedded sequence.

        Args:
            input_ids: [B, T]
        Returns:
            scalar loss
        """
        x1 = self.encode(input_ids)  # [B, flat_dim]
        return self._loss_fn.cfm_loss(self.vector_field, x1)

    def sample_sequence(self, n_samples: int) -> torch.Tensor:
        """Generate n_samples sequences by integrating the ODE from noise.

        Args:
            n_samples: number of sequences to generate
        Returns:
            logits: [n_samples, T, vocab_size]
        """
        device = next(self.parameters()).device
        flat_dim = self.d_model * self.seq_len
        x0 = torch.randn(n_samples, flat_dim, device=device)
        sampler = EulerFlowSampler(n_steps=50)
        x1 = sampler.sample(self.vector_field, x0)
        return self.decode(x1)


# ---------------------------------------------------------------------------
# FlowMatchingConfig
# ---------------------------------------------------------------------------


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching experiments."""

    d_model: int = 32
    vocab_size: int = 64
    seq_len: int = 8
    hidden: int = 64
    n_layers: int = 2
    n_steps: int = 10
    d_cond: int = 16

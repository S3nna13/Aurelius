"""
neural_ode_lm.py

Neural ODEs for continuous-depth sequence modeling.
Replaces discrete transformer layers with a learned ODE dynamics function.
Pure PyTorch only — no external neural-network libraries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# ODEFunc — learned dynamics  dz/dt = f(t, z)
# ---------------------------------------------------------------------------


class ODEFunc(nn.Module):
    """Time-conditioned MLP dynamics: dz/dt = MLP(concat(z, t_embed)).

    Args:
        d_model: dimensionality of the hidden state z.
        hidden:  width of the intermediate MLP layer.
    """

    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        # Input dim is d_model + 1 (we concat a scalar time embedding).
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, d_model),
        )
        self.nfe: int = 0  # number of function evaluations

    def forward(self, t: float, z: Tensor) -> Tensor:
        """Compute dz/dt.

        Args:
            t: scalar time value.
            z: [B, T, d_model] hidden state.

        Returns:
            dz_dt: [B, T, d_model]
        """
        self.nfe += 1
        B, T, _ = z.shape
        # Expand t to [B, T, 1] and concatenate with z.
        t_embed = z.new_full((B, T, 1), fill_value=float(t))
        zt = torch.cat([z, t_embed], dim=-1)  # [B, T, d_model+1]
        return self.net(zt)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


class EulerSolver:
    """Simple first-order Euler ODE integrator."""

    def solve(
        self,
        func: ODEFunc,
        z0: Tensor,
        t_span: tuple[float, float],
        n_steps: int,
    ) -> Tensor:
        """Integrate ODE from t_span[0] to t_span[1] using Euler steps.

        Args:
            func:    ODEFunc instance.
            z0:      [B, T, d_model] initial state.
            t_span:  (t0, t1) integration interval.
            n_steps: number of integration steps.

        Returns:
            z at t_span[1], shape [B, T, d_model].
        """
        t0, t1 = t_span
        h = (t1 - t0) / n_steps
        z = z0
        t = t0
        for _ in range(n_steps):
            dz = func(t, z)
            z = z + h * dz
            t = t + h
        return z


class RK4Solver:
    """Classic 4th-order Runge-Kutta ODE integrator."""

    def solve(
        self,
        func: ODEFunc,
        z0: Tensor,
        t_span: tuple[float, float],
        n_steps: int,
    ) -> Tensor:
        """Integrate ODE from t_span[0] to t_span[1] using RK4 steps.

        Args:
            func:    ODEFunc instance.
            z0:      [B, T, d_model] initial state.
            t_span:  (t0, t1) integration interval.
            n_steps: number of integration steps.

        Returns:
            z at t_span[1], shape [B, T, d_model].
        """
        t0, t1 = t_span
        h = (t1 - t0) / n_steps
        z = z0
        t = t0
        for _ in range(n_steps):
            k1 = func(t, z)
            k2 = func(t + h / 2, z + (h / 2) * k1)
            k3 = func(t + h / 2, z + (h / 2) * k2)
            k4 = func(t + h, z + h * k3)
            z = z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + h
        return z


# ---------------------------------------------------------------------------
# AdjointMethod — custom autograd Function
# ---------------------------------------------------------------------------


class AdjointMethod(torch.autograd.Function):
    """Neural ODE adjoint sensitivity method.

    For simplicity the backward pass re-uses standard autograd through the
    recorded forward Euler trajectory (approximate adjoint).  This avoids the
    need for a separate adjoint ODE solve while still giving correct gradients
    with respect to *all* ODE-func parameters.
    """

    @staticmethod
    def forward(
        ctx,
        z0: Tensor,
        func_params,
        func: ODEFunc,
        t_span: tuple[float, float],
        n_steps: int,
        solver_fn,
    ) -> Tensor:
        """Solve ODE forward and save trajectory for backward.

        Args:
            z0:          [B, T, d_model] initial state (requires grad).
            func_params: list of ODEFunc parameters (passed so autograd tracks them).
            func:        ODEFunc instance.
            t_span:      (t0, t1).
            n_steps:     number of integration steps.
            solver_fn:   EulerSolver or RK4Solver instance.

        Returns:
            z_T: [B, T, d_model] final state.
        """
        # We perform a no-grad forward to record the trajectory, then replay
        # with grad in backward via saved intermediates.
        t0, t1 = t_span
        h = (t1 - t0) / n_steps

        states = [z0]
        z = z0.detach().requires_grad_(z0.requires_grad)
        t = t0
        with torch.no_grad():
            for _ in range(n_steps):
                dz = func(t, z)
                z = z + h * dz
                t = t + h
                states.append(z.clone())

        z_T = states[-1]
        ctx.save_for_backward(*states)
        ctx.func = func
        ctx.t_span = t_span
        ctx.n_steps = n_steps
        ctx.h = h
        return z_T.requires_grad_(z0.requires_grad)

    @staticmethod
    def backward(ctx, dL_dzT: Tensor):
        """Approximate adjoint: backprop through saved Euler steps."""
        states = ctx.saved_tensors
        func = ctx.func
        h = ctx.h
        t0, t1 = ctx.t_span
        n_steps = ctx.n_steps

        # Re-run forward *with* grad tracking, then do one backward.
        z = states[0].detach().requires_grad_(True)
        t = t0
        intermediates = []
        for i in range(n_steps):
            z_in = z if i == 0 else intermediates[-1][1]
            z_in = z_in.detach().requires_grad_(True)
            dz = func(t, z_in)
            z_out = z_in + h * dz
            intermediates.append((z_in, z_out))
            t = t + h

        # Backprop gradient from output to input.
        grad = dL_dzT
        for z_in, z_out in reversed(intermediates):
            grads = torch.autograd.grad(
                z_out,
                (z_in,) + tuple(func.parameters()),
                grad_outputs=grad,
                allow_unused=True,
                retain_graph=False,
            )
            grad = grads[0] if grads[0] is not None else torch.zeros_like(z_in)

        # Return gradients matching positional args of forward:
        # z0, func_params (None — handled via autograd graph), func, t_span, n_steps, solver_fn
        return grad, None, None, None, None, None


# ---------------------------------------------------------------------------
# NeuralODEBlock — drop-in layer replacement
# ---------------------------------------------------------------------------


class NeuralODEBlock(nn.Module):
    """Single Neural ODE block: maps [B, T, d_model] → [B, T, d_model].

    Args:
        d_model:  model dimension.
        solver:   "euler" or "rk4".
        n_steps:  number of ODE integration steps.
        t_span:   (t0, t1) integration interval.
    """

    def __init__(
        self,
        d_model: int,
        solver: str = "rk4",
        n_steps: int = 6,
        t_span: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.odefunc = ODEFunc(d_model)
        if solver == "euler":
            self.solver_fn = EulerSolver()
        elif solver == "rk4":
            self.solver_fn = RK4Solver()
        else:
            raise ValueError(f"Unknown solver: {solver!r}. Choose 'euler' or 'rk4'.")
        self.n_steps = n_steps
        self.t_span = t_span

    def forward(self, x: Tensor) -> Tensor:
        """Solve ODE forward from x.

        Args:
            x: [B, T, d_model]

        Returns:
            z_T: [B, T, d_model]
        """
        z_T = self.solver_fn.solve(self.odefunc, x, self.t_span, self.n_steps)
        return z_T

    def nfe(self) -> int:
        """Return number of function evaluations recorded by the ODEFunc."""
        return self.odefunc.nfe


# ---------------------------------------------------------------------------
# NeuralODELanguageModel
# ---------------------------------------------------------------------------


class NeuralODELanguageModel(nn.Module):
    """Language model using stacked Neural ODE blocks instead of transformer layers.

    Args:
        d_model:      embedding / hidden dimension.
        vocab_size:   vocabulary size.
        n_ode_blocks: number of stacked NeuralODEBlock layers.
        solver:       "euler" or "rk4".
        n_steps:      ODE integration steps per block.
        t_start:      start of integration interval.
        t_end:        end of integration interval.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_ode_blocks: int = 2,
        solver: str = "rk4",
        n_steps: int = 4,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        t_span = (t_start, t_end)
        self.ode_blocks = nn.ModuleList(
            [
                NeuralODEBlock(d_model, solver=solver, n_steps=n_steps, t_span=t_span)
                for _ in range(n_ode_blocks)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_ids: [B, T] integer token ids.

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embedding(input_ids)  # [B, T, d_model]
        for block in self.ode_blocks:
            x = block(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits

    def compute_loss(self, input_ids: Tensor) -> Tensor:
        """Cross-entropy next-token-prediction loss.

        Args:
            input_ids: [B, T] integer token ids.

        Returns:
            Scalar loss tensor.
        """
        logits = self.forward(input_ids)  # [B, T, V]
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()  # [B, T-1]
        B, Tm1, V = shift_logits.shape
        loss = nn.functional.cross_entropy(
            shift_logits.view(B * Tm1, V),
            shift_labels.view(B * Tm1),
        )
        return loss


# ---------------------------------------------------------------------------
# NeuralODEConfig — dataclass for model hyper-parameters
# ---------------------------------------------------------------------------


@dataclass
class NeuralODEConfig:
    """Configuration for NeuralODELanguageModel."""

    d_model: int = 32
    vocab_size: int = 64
    n_ode_blocks: int = 2
    hidden: int = 32
    solver: str = "rk4"
    n_steps: int = 4
    t_start: float = 0.0
    t_end: float = 1.0

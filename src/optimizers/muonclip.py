"""MuonClip optimizer — Nesterov momentum + Newton-Schulz orthogonalization + gradient clipping.

Based on:
  - Kimi K2.5 (arXiv:2602.02276): Muon optimizer with gradient-norm clipping for RL stability.
  - GLM-5 §3.1 (arXiv:2602.15763): Muon Split — orthogonalize per leading dimension (per-head)
    so different heads can update at independent scales, stabilizing logit magnitudes.

Pure PyTorch implementation — no external ML library dependencies.
"""

import torch
from torch.optim import Optimizer


def _orthogonalize(M: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz 2-step orthogonalization (no SVD required).

    For a matrix M of shape (r, c), computes an approximate orthogonal basis
    for the row space via:
        A = M @ M^T
        B = 1.5*I - 0.5*A
        M_orth = B @ M

    For tensors with ndim >= 2 the operation is applied on the reshaped
    (M.shape[0], -1) view and the original shape is restored afterwards.
    1-D tensors are returned unchanged (edge case: bias vectors).
    """
    if M.ndim < 2:
        return M
    orig_shape = M.shape
    m = M.reshape(M.shape[0], -1)
    A = m @ m.T
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    B = 1.5 * I - 0.5 * A
    m = B @ m
    return m.reshape(orig_shape)


class MuonClip(Optimizer):
    """MuonClip: Muon Split optimizer with gradient-norm clipping.

    Combines:
      * Nesterov-style momentum for look-ahead gradient estimation.
      * Newton-Schulz per-leading-dimension orthogonalization (Muon Split / GLM-5).
      * Gradient-norm clipping for RL training stability (Kimi K2.5).

    Args:
        params: Iterable of parameters or param groups.
        lr (float): Learning rate. Default: 1e-3.
        momentum (float): Nesterov momentum coefficient (beta). Default: 0.95.
        max_norm (float): Maximum gradient norm for clipping. Default: 1.0.
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, max_norm: float = 1.0):
        defaults = dict(lr=lr, momentum=momentum, max_norm=max_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            loss value if closure is provided, else None.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            max_norm: float = group["max_norm"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)

                m = state["m"]

                # Nesterov look-ahead gradient
                g_ns = g + beta * m

                # Muon Split: orthogonalize per leading dimension
                g_orth = _orthogonalize(g_ns)

                # Gradient-norm clipping
                norm = g_orth.norm()
                if norm > max_norm:
                    g_orth = g_orth * (max_norm / (norm + 1e-8))

                # Parameter update
                p.add_(g_orth, alpha=-lr)

                # Momentum buffer update
                m.mul_(beta).add_(g)

        return loss

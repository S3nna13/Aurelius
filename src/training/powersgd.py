"""PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.

Implements Algorithm 1 from Vogels et al., NeurIPS 2019 (arXiv:1905.13727).
Variable notation matches the paper throughout.

Key idea: represent gradient G ∈ R^{m×n} as M @ Q^T where M ∈ R^{m×r},
Q ∈ R^{n×r}, with r << min(m, n). Power iteration approximates the dominant
singular subspace to minimise reconstruction error.

Error feedback (Section 4.3): accumulate compression residuals across steps so
that the compressor is unbiased in expectation.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Core compressor
# ---------------------------------------------------------------------------


class PowerSGDCompressor:
    """Low-rank gradient compressor via power iteration.

    Parameters
    ----------
    rank : int
        Target rank r for the low-rank approximation (clamped to
        min(m, n) - 1 automatically).
    n_power_iter : int
        Number of power iterations K (paper recommends K=1).
    """

    def __init__(self, rank: int = 4, n_power_iter: int = 1) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if n_power_iter < 0:
            raise ValueError(f"n_power_iter must be >= 0, got {n_power_iter}")

        self.rank = rank
        self.n_power_iter = n_power_iter

        # Error-feedback buffer: e_t stores the residual from the last call to
        # step().  Keyed by id(grad tensor) — callers should use step() which
        # manages the buffer via a stable key supplied externally, or rely on
        # the per-instance single-gradient state here.
        self._error_feedback: Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, grad: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Compress a gradient tensor using power iteration.

        For 1-D tensors (biases, LayerNorm scales) the gradient is returned
        unchanged with Q=None so callers can distinguish the two cases.

        Parameters
        ----------
        grad : Tensor
            The gradient to compress.  Must be 1-D or 2-D (or will be viewed
            as 2-D by treating all-but-last dims as the row dimension).

        Returns
        -------
        M : Tensor or None
            Left factor M ∈ R^{m×r}. None for 1-D input.
        Q : Tensor or None
            Right orthonormal factor Q ∈ R^{n×r}. None for 1-D input.
        residual : Tensor or None
            Compression error (G - M @ Q^T), same shape as *grad*.
            None for 1-D input (no error to feed back).
        """
        if grad.dim() < 2:
            # 1-D tensors: pass through uncompressed (paper, Section 4)
            return grad, None, None

        # Flatten to 2-D: treat leading dims as m, last dim as n
        original_shape = grad.shape
        G = grad.reshape(-1, original_shape[-1])  # G ∈ R^{m×n}
        m, n = G.shape

        # Clamp rank gracefully so it never exceeds the smaller dimension
        r = min(self.rank, min(m, n))

        # ------------------------------------------------------------------
        # Algorithm 1 — PowerSGD
        # ------------------------------------------------------------------

        # Step 1: Initialise Q ∈ R^{n×r} with orthonormal columns
        Q = torch.randn(n, r, device=G.device, dtype=G.dtype)
        Q, _ = torch.linalg.qr(Q)  # Gram-Schmidt via QR

        # Step 2: Power iteration (K rounds)
        for _ in range(self.n_power_iter):
            # a) M = G @ Q        — M ∈ R^{m×r}
            M = G @ Q
            # b) Q, _ = QR(G^T @ M)  — updated Q (re-orthonormalise)
            Q, _ = torch.linalg.qr(G.t() @ M)

        # Step 3: Final M = G @ Q
        M = G @ Q  # M ∈ R^{m×r}

        # Step 4: Reconstruct and compute residual
        G_approx = M @ Q.t()  # G_approx ∈ R^{m×n}
        residual = (G - G_approx).reshape(original_shape)

        # Reshape M back so its leading dims match the original gradient
        # (keep as 2-D internally; callers use decompress which does reshape)
        return M, Q, residual

    def decompress(self, M: Tensor, Q: Tensor, original_shape: tuple | None = None) -> Tensor:
        """Reconstruct gradient from low-rank factors.

        Parameters
        ----------
        M : Tensor
            Left factor ∈ R^{m×r}.
        Q : Tensor
            Right orthonormal factor ∈ R^{n×r}.
        original_shape : tuple, optional
            If supplied the result is reshaped to this shape; otherwise the
            raw (m, n) matrix is returned.

        Returns
        -------
        Tensor
            Reconstructed gradient G_approx = M @ Q^T.
        """
        G_approx = M @ Q.t()
        if original_shape is not None:
            G_approx = G_approx.reshape(original_shape)
        return G_approx

    def step(self, grad: Tensor) -> Tensor:
        """Compress, apply error feedback, and return reconstructed gradient.

        Maintains an internal per-compressor error-feedback buffer for a
        *single* gradient tensor.  For multi-parameter use, prefer
        :class:`PowerSGDOptimizer` which keeps per-parameter buffers.

        Parameters
        ----------
        grad : Tensor
            Raw gradient for the current step.

        Returns
        -------
        Tensor
            Reconstructed (approximate) gradient with error feedback applied,
            same shape as *grad*.
        """
        if grad.dim() < 2:
            # 1-D: no compression, no error feedback needed
            return grad

        # Apply error feedback: G_t' = G_t + e_{t-1}
        if self._error_feedback is not None:
            grad_with_feedback = grad + self._error_feedback
        else:
            grad_with_feedback = grad

        M, Q, residual = self.compress(grad_with_feedback)

        if Q is None:
            # Should not happen for 2-D grads, but be safe
            self._error_feedback = None
            return grad_with_feedback

        # Store residual as next step's error-feedback buffer
        self._error_feedback = residual.detach().clone()

        # Reconstruct
        return self.decompress(M, Q, original_shape=grad.shape)

    def reset_error_feedback(self) -> None:
        """Clear the internal error-feedback buffer."""
        self._error_feedback = None


# ---------------------------------------------------------------------------
# Optimizer wrapper
# ---------------------------------------------------------------------------


class PowerSGDOptimizer(torch.optim.Optimizer):
    """SGD optimizer that compresses gradients with PowerSGD before the update.

    For each parameter with a gradient the compressor intercepts it, applies
    error feedback, compresses with power iteration, reconstructs, and then
    performs the standard SGD weight update.

    1-D parameters (biases, LayerNorm) bypass compression entirely.

    Parameters
    ----------
    params : iterable
        Parameters to optimise (same as any torch Optimizer).
    lr : float
        Learning rate.
    rank : int
        PowerSGD target rank r.
    n_power_iter : int
        Number of power iterations K.
    momentum : float
        SGD momentum (default 0).
    weight_decay : float
        L2 regularisation coefficient (default 0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        rank: int = 4,
        n_power_iter: int = 1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rank=rank,
            n_power_iter=n_power_iter,
        )
        super().__init__(params, defaults)

        # Per-parameter error-feedback buffers and SGD momentum buffers
        self._error_buffers: dict[int, Tensor] = {}
        self._momentum_buffers: dict[int, Tensor] = {}

    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            rank = group["rank"]
            n_power_iter = group["n_power_iter"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()

                if weight_decay != 0.0:
                    grad = grad.add(p.detach(), alpha=weight_decay)

                # Compress 2-D+ gradients; pass 1-D through unchanged
                if grad.dim() >= 2:
                    pid = id(p)

                    # Error feedback: G_t' = G_t + e_{t-1}
                    if pid in self._error_buffers:
                        grad = grad + self._error_buffers[pid]

                    # Compress
                    original_shape = grad.shape
                    G = grad.reshape(-1, original_shape[-1])
                    m, n = G.shape
                    r = min(rank, min(m, n))

                    Q = torch.randn(n, r, device=G.device, dtype=G.dtype)
                    Q, _ = torch.linalg.qr(Q)

                    for _ in range(n_power_iter):
                        M = G @ Q
                        Q, _ = torch.linalg.qr(G.t() @ M)

                    M = G @ Q
                    G_approx = M @ Q.t()

                    # Store residual for next step
                    residual = (G - G_approx).reshape(original_shape)
                    self._error_buffers[pid] = residual.clone()

                    grad = G_approx.reshape(original_shape)

                # SGD update with optional momentum
                if momentum != 0.0:
                    pid = id(p)
                    buf_key = (pid, "mom")
                    if buf_key not in self._momentum_buffers:
                        self._momentum_buffers[buf_key] = torch.zeros_like(grad)
                    buf = self._momentum_buffers[buf_key]
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                p.add_(grad, alpha=-lr)

        return loss

    def reset_error_feedback(self) -> None:
        """Clear all per-parameter error-feedback buffers."""
        self._error_buffers.clear()

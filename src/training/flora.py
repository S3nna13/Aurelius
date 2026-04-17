"""FLoRA: Full-rank LoRA via Random Projection.

Reference: Hao et al., "FLoRA: Low-Rank Adapters Are Secretly Gradient
Compressors", arXiv:2402.04905.

Key insight (Theorem 1):
    Standard LoRA update ΔW = B·A is equivalent to compressing the gradient
    via a fixed random projection drawn at init. FLoRA generalises this by
    re-drawing fresh random projections R_t, C_t at *every* step, so the
    update can span the full rank of W over multiple steps — higher
    expressivity at the same parameter count.

Algorithm (Section 3, paper notation preserved):
    For a weight matrix W ∈ R^{m×n} and rank r:

    At each step t:
        1. Draw R_t ∈ R^{r×m}  (row projector)   — Gaussian, re-drawn each step
           Draw C_t ∈ R^{n×r}  (column projector) — Gaussian, re-drawn each step
        2. Project gradient:      G̃_t = R_t G_t C_t          shape (r, r)
        3. Apply Adam in compressed space → Ṽ_t               shape (r, r)
        4. Unproject:             ΔW_t = R_t^T Ṽ_t C_t^T     shape (m, n)
        5. Apply:                 W_{t+1} = W_t - lr * ΔW_t

    Reproducible projections: seed = seed_offset + step, so the same
    (R_t, C_t) pair can be regenerated deterministically.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# FLoRAProjector
# ---------------------------------------------------------------------------

class FLoRAProjector:
    """Manages seeded random projections R_t and C_t for a single weight matrix.

    Paper notation:
        m         : number of rows in W
        n         : number of columns in W
        r         : rank (compression dimension)
        R_t       : row projector at step t, shape (r, m)
        C_t       : column projector at step t, shape (n, r)
        G_t       : full gradient at step t, shape (m, n)
        G̃_t       : compressed gradient = R_t G_t C_t, shape (r, r)
        Ṽ_t       : Adam output in compressed space, shape (r, r)
        ΔW_t      : unprojected update = R_t^T Ṽ_t C_t^T, shape (m, n)
        seed_offset: base seed; actual seed per step = seed_offset + step
    """

    def __init__(self, m: int, n: int, rank: int, seed_offset: int = 0) -> None:
        if m <= 0 or n <= 0:
            raise ValueError(f"m and n must be positive, got m={m}, n={n}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.m = m
        self.n = n
        self.r = rank           # paper variable: r
        self.seed_offset = seed_offset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_R_C(
        self, step: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw R_t ∈ R^{r×m} and C_t ∈ R^{n×r} using seeded RNG.

        Uses a fresh generator so the global RNG state is not disturbed.
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed_offset + step)

        # R_t: shape (r, m)
        R_t = torch.randn(self.r, self.m, dtype=dtype, device=device, generator=gen)
        # C_t: shape (n, r)
        C_t = torch.randn(self.n, self.r, dtype=dtype, device=device, generator=gen)
        return R_t, C_t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, G_t: torch.Tensor, step: int) -> torch.Tensor:
        """Compress gradient G_t into (r, r) space.

        G̃_t = R_t G_t C_t

        Args:
            G_t  : Full gradient, shape (m, n).
            step : Current optimiser step t (used to seed R_t, C_t).

        Returns:
            G̃_t : Compressed gradient, shape (r, r).
        """
        if G_t.shape != (self.m, self.n):
            raise ValueError(
                f"Expected G_t shape ({self.m}, {self.n}), got {tuple(G_t.shape)}"
            )
        R_t, C_t = self._draw_R_C(step, G_t.dtype, G_t.device)
        # G̃_t = R_t @ G_t @ C_t  →  (r×m) @ (m×n) @ (n×r) = (r, r)
        G_tilde = R_t @ G_t @ C_t
        return G_tilde

    def unproject(self, V_compressed: torch.Tensor, step: int) -> torch.Tensor:
        """Reconstruct full-rank update from compressed Adam output.

        ΔW_t = R_t^T Ṽ_t C_t^T

        Args:
            V_compressed : Adam output in compressed space, shape (r, r).
            step         : Same step t used in the corresponding project() call.

        Returns:
            ΔW_t : Full-space update, shape (m, n).
        """
        if V_compressed.shape != (self.r, self.r):
            raise ValueError(
                f"Expected V_compressed shape ({self.r}, {self.r}), "
                f"got {tuple(V_compressed.shape)}"
            )
        R_t, C_t = self._draw_R_C(step, V_compressed.dtype, V_compressed.device)
        # ΔW_t = R_t^T @ Ṽ_t @ C_t^T  →  (m×r) @ (r×r) @ (r×n) = (m, n)
        delta_W = R_t.T @ V_compressed @ C_t.T
        return delta_W


# ---------------------------------------------------------------------------
# FLoRAOptimizer
# ---------------------------------------------------------------------------

class FLoRAOptimizer(Optimizer):
    """Adam optimiser with FLoRA full-rank random-projection gradient compression.

    For each 2-D (or higher) weight matrix W, Adam moments are maintained in
    the compressed (r×r) space, making memory cost O(r²) per parameter
    regardless of the original shape.  Fresh random projections R_t, C_t are
    re-drawn each step so the update subspace rotates continuously — enabling
    full-rank coverage over time (cf. LoRA which fixes projections at init).

    For 1-D parameters (biases, layer-norm scales) standard Adam is applied
    with no compression.

    Args:
        params      : Iterable of parameters or param-groups.
        lr          : Learning rate (default: 0.01).
        rank        : Compression rank r (default: 8).
        betas       : Adam β1, β2 (default: (0.9, 0.999)).
        eps         : Adam ε for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay λ (default: 0.0).
        seed_offset : Base seed for random projections (default: 0).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        rank: int = 8,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        seed_offset: int = 0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            rank=rank,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            seed_offset=seed_offset,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Performs one FLoRA optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and
                returns the loss.

        Returns:
            loss if closure provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rank = group["rank"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            seed_offset = group["seed_offset"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FLoRAOptimizer does not support sparse gradients.")

                state = self.state[p]

                if p.ndim < 2:
                    # 1-D params: standard Adam (no FLoRA projection)
                    self._adam_step_1d(p, grad, state, lr, beta1, beta2, eps, weight_decay)
                else:
                    # 2-D+ params: FLoRA compressed Adam
                    self._flora_step(
                        p, grad, state,
                        lr, rank, beta1, beta2, eps, weight_decay, seed_offset,
                    )

        return loss

    # ------------------------------------------------------------------
    # Internal: standard Adam for 1-D parameters
    # ------------------------------------------------------------------

    def _adam_step_1d(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        """Standard Adam (decoupled weight decay) for bias / norm parameters."""
        if len(state) == 0:
            state["step"] = 0
            state["m1"] = torch.zeros_like(p)   # first moment
            state["m2"] = torch.zeros_like(p)   # second moment (uncentered)

        state["step"] += 1
        t = state["step"]
        m1 = state["m1"]
        m2 = state["m2"]

        # Decoupled weight decay
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)

        # Moment updates
        m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        m2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Bias-corrected estimates
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        m1_hat = m1 / bc1
        m2_hat = m2 / bc2

        # Parameter update
        p.addcdiv_(m1_hat, m2_hat.sqrt().add_(eps), value=-lr)

    # ------------------------------------------------------------------
    # Internal: FLoRA step for 2-D+ weight matrices
    # ------------------------------------------------------------------

    def _flora_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        rank: int,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        seed_offset: int,
    ) -> None:
        """FLoRA Adam update for 2-D+ weight matrices.

        Maintains Adam moments in the compressed (r×r) space.
        Re-draws fresh random projections R_t, C_t each step.
        """
        # Flatten to 2-D: (m, n)
        original_shape = p.shape
        m = p.shape[0]
        n = p.numel() // m
        p_2d = p.view(m, n)
        grad_2d = grad.view(m, n)

        r = min(rank, m, n)   # effective rank (can't exceed matrix dimensions)

        # ---- Initialise state ----
        if len(state) == 0:
            state["step"] = 0
            state["projector"] = FLoRAProjector(m, n, r, seed_offset=seed_offset)
            # Adam moments live in the compressed (r, r) space
            state["m1"] = torch.zeros(r, r, dtype=p.dtype, device=p.device)
            state["m2"] = torch.zeros(r, r, dtype=p.dtype, device=p.device)

        state["step"] += 1
        t: int = state["step"]
        projector: FLoRAProjector = state["projector"]
        m1: torch.Tensor = state["m1"]
        m2: torch.Tensor = state["m2"]

        # ---- Decoupled weight decay ----
        if weight_decay != 0.0:
            p_2d.mul_(1.0 - lr * weight_decay)

        # ---- Step 2: Project gradient into (r, r) space ----
        # G̃_t = R_t G_t C_t
        G_tilde = projector.project(grad_2d, step=t)   # (r, r)

        # ---- Step 3: Adam in compressed space ----
        m1.mul_(beta1).add_(G_tilde, alpha=1.0 - beta1)
        m2.mul_(beta2).addcmul_(G_tilde, G_tilde, value=1.0 - beta2)

        # Bias correction
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        m1_hat = m1 / bc1
        m2_hat = m2 / bc2

        # Ṽ_t: Adam output in compressed (r, r) space
        V_tilde = m1_hat / (m2_hat.sqrt().add_(eps))   # (r, r)

        # ---- Step 4: Unproject to (m, n) ----
        # ΔW_t = R_t^T Ṽ_t C_t^T
        delta_W = projector.unproject(V_tilde, step=t)   # (m, n)

        # ---- Step 5: W_{t+1} = W_t - lr * ΔW_t ----
        p_2d.add_(delta_W, alpha=-lr)

        # Write back (no-op when p is already contiguous with shape (m, n))
        p.data.copy_(p_2d.view(original_shape))

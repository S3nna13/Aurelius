"""GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.

Reference: Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient
Low-Rank Projection", arXiv:2403.03507.

Key idea:
    For a weight matrix W ∈ R^{m×n}, instead of storing full-rank Adam
    moments in R^{m×n}, project gradients into a rank-r subspace first.
    Optimizer state is maintained in that low-rank space (R^{r×n}), giving
    large memory savings when r << m.

Algorithm 1 (paper notation preserved in code):
    Every T_proj steps:
        U, S, Vh = svd(G_t)          # G_t ∈ R^{m×n}
        P_t = U[:, :r]               # left singular vecs, shape (m, r)
    Projected gradient:
        G̃_t = P_t^T G_t             # shape (r, n)
    Adam update in low-rank space:
        Ṽ_t = adam(G̃_t)             # shape (r, n)
    Unproject:
        ΔW_t = P_t Ṽ_t              # shape (m, n)
    Weight update:
        W_{t+1} = W_t - lr * scale * ΔW_t
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# GaLoreProjector
# ---------------------------------------------------------------------------


class GaLoreProjector:
    """Maintains the low-rank projection matrix P_t for one weight parameter.

    Paper notation:
        G_t  : full gradient at step t, shape (m, n)
        P_t  : left singular vectors of G_t, shape (m, r)
        G̃_t  : projected gradient = P_t^T @ G_t, shape (r, n)
        T_proj: projection update interval (update_proj_gap)
        r    : rank
    """

    def __init__(
        self,
        rank: int,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
    ) -> None:
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if update_proj_gap <= 0:
            raise ValueError(f"update_proj_gap must be positive, got {update_proj_gap}")

        self.rank = rank
        self.T_proj = update_proj_gap  # paper variable name
        self.scale = scale
        self.proj_type = proj_type

        # P_t: left singular vectors, shape (m, r). None until first update.
        self.P_t: torch.Tensor | None = None
        self._step: int = 0  # counts calls to project()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, G_t: torch.Tensor) -> torch.Tensor:
        """Project G_t into the low-rank subspace.

        Args:
            G_t: Full gradient tensor of shape (m, n).

        Returns:
            G̃_t: Projected gradient of shape (r, n).
        """
        if G_t.ndim < 2:
            raise ValueError("GaLoreProjector only supports 2D+ gradient tensors.")

        # Reshape to (m, n) if higher-dimensional
        if G_t.ndim > 2:
            G_t = G_t.view(G_t.shape[0], -1)

        self.update_proj(G_t)  # recompute P_t if needed (increments _step)

        # G̃_t = P_t^T @ G_t  (r×m) @ (m×n) = (r×n)
        G_tilde = self.P_t.T @ G_t  # shape (r, n)
        return G_tilde

    def unproject(self, V_tilde: torch.Tensor) -> torch.Tensor:
        """Unproject the low-rank update back to full space.

        Args:
            V_tilde: Optimizer update in low-rank space, shape (r, n).

        Returns:
            ΔW_t: Full-space update, shape (m, n).
        """
        if self.P_t is None:
            raise RuntimeError("P_t is None; call project() before unproject().")
        # ΔW_t = P_t @ Ṽ_t  (m×r) @ (r×n) = (m×n)
        delta_W = self.P_t @ V_tilde  # shape (m, n)
        return self.scale * delta_W

    def update_proj(self, G_t: torch.Tensor) -> None:
        """Recompute P_t via truncated SVD if step % T_proj == 0.

        Increments internal step counter.

        Args:
            G_t: Full gradient, shape (m, n).
        """
        if self._step % self.T_proj == 0:
            self.P_t = _compute_P(G_t, self.rank)
        self._step += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current projection step counter."""
        return self._step


# ---------------------------------------------------------------------------
# SVD helper
# ---------------------------------------------------------------------------


def _compute_P(G_t: torch.Tensor, rank: int) -> torch.Tensor:
    """Compute left singular vectors P_t of G_t up to rank r.

    Args:
        G_t: Gradient matrix, shape (m, n).
        rank: Target rank r. Clamped to min(m, n) if too large.

    Returns:
        P_t: Left singular vectors, shape (m, r_eff) where r_eff = min(r, m, n).
    """
    m, n = G_t.shape
    r_eff = min(rank, m, n)  # clamp: edge case where rank >= min(m, n)

    # Use float32 for numerical stability during SVD
    dtype = G_t.dtype
    G_f = G_t.float()

    try:
        # torch.linalg.svd returns U (m×k), S (k,), Vh (k×n)
        # full_matrices=False gives economy / truncated decomposition
        U, S, Vh = torch.linalg.svd(G_f, full_matrices=False)
        P_t = U[:, :r_eff]  # (m, r_eff) — left singular vectors
    except Exception as exc:
        raise RuntimeError(f"SVD failed during GaLore projection update: {exc}") from exc

    return P_t.to(dtype)


# ---------------------------------------------------------------------------
# GaLoreAdamW optimizer
# ---------------------------------------------------------------------------


class GaLoreAdamW(Optimizer):
    """AdamW optimizer with GaLore gradient projection for 2D weight matrices.

    For 1D parameters (biases, layer-norm scales) standard AdamW is applied
    — no projection, full-rank Adam moments.

    For 2D+ weight matrices, Adam first/second moments are maintained in the
    *projected* low-rank space (R^{r×n}) rather than the full space (R^{m×n}),
    yielding substantial memory savings when r << m.

    Args:
        params: Iterable of parameters or param-groups.
        lr: Learning rate (default: 1e-4).
        betas: Adam β1, β2 (default: (0.9, 0.999)).
        eps: Adam ε for numerical stability (default: 1e-8).
        weight_decay: AdamW decoupled weight decay (default: 0.01).
        rank: Low-rank projection rank r (default: 128).
        update_proj_gap: Number of steps T_proj between projection updates
            (default: 200).
        scale: Scalar multiplied into the unprojected update (default: 1.0).
        proj_type: Projection variant; currently only 'std' supported.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Performs a single optimization step.

        Args:
            closure: Optional closure that re-evaluates the model and
                returns the loss.

        Returns:
            loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            rank = group["rank"]
            update_proj_gap = group["update_proj_gap"]
            scale = group["scale"]
            proj_type = group["proj_type"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("GaLoreAdamW does not support sparse gradients.")

                state = self.state[p]

                # ---- 1D params: standard AdamW (no projection) ----
                if p.ndim < 2:
                    self._adamw_step_1d(p, grad, state, lr, beta1, beta2, eps, weight_decay)
                    continue

                # ---- 2D+ params: GaLore projected AdamW ----
                self._galore_step(
                    p,
                    grad,
                    state,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    rank,
                    update_proj_gap,
                    scale,
                    proj_type,
                )

        return loss

    # ------------------------------------------------------------------
    # Internal: standard AdamW for 1D
    # ------------------------------------------------------------------

    def _adamw_step_1d(
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
        """Standard AdamW update for 1-D parameters (biases, norms)."""
        if len(state) == 0:
            state["step"] = 0
            state["m1"] = torch.zeros_like(p)  # first moment
            state["m2"] = torch.zeros_like(p)  # second moment (uncentered)

        state["step"] += 1
        t = state["step"]
        m1 = state["m1"]
        m2 = state["m2"]

        # AdamW weight decay: applied directly to weight, not gradient
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)

        # Update biased first/second moments
        m1.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        m2.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Bias correction
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m1_hat = m1 / bc1
        m2_hat = m2 / bc2

        # Parameter update
        p.addcdiv_(m1_hat, m2_hat.sqrt().add_(eps), value=-lr)

    # ------------------------------------------------------------------
    # Internal: GaLore step for 2D+ weight matrices
    # ------------------------------------------------------------------

    def _galore_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: dict,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        rank: int,
        update_proj_gap: int,
        scale: float,
        proj_type: str,
    ) -> None:
        """GaLore projected AdamW update for 2D+ weight matrices.

        Maintains Adam moments in the *projected* low-rank space to save memory.
        """
        # Flatten to 2D: (m, n)
        original_shape = p.shape
        p_2d = p.view(p.shape[0], -1)  # (m, n)
        grad_2d = grad.view(grad.shape[0], -1)  # (m, n)

        m = p_2d.shape[0]
        n = p_2d.shape[1]
        r_eff = min(rank, m, n)

        # ---- Initialize state ----
        if len(state) == 0:
            state["step"] = 0
            # GaLore projector for this parameter
            state["projector"] = GaLoreProjector(
                rank=rank,
                update_proj_gap=update_proj_gap,
                scale=scale,
                proj_type=proj_type,
            )
            # Moments in projected (low-rank) space: shape (r_eff, n)
            state["m1"] = torch.zeros(r_eff, n, dtype=p.dtype, device=p.device)
            state["m2"] = torch.zeros(r_eff, n, dtype=p.dtype, device=p.device)

        state["step"] += 1
        t = state["step"]
        projector: GaLoreProjector = state["projector"]
        m1 = state["m1"]
        m2 = state["m2"]

        # ---- AdamW weight decay (decoupled) ----
        if weight_decay != 0.0:
            p_2d.mul_(1.0 - lr * weight_decay)

        # ---- GaLore projection: G̃_t = P_t^T G_t ∈ R^{r×n} ----
        G_tilde = projector.project(grad_2d)  # (r_eff, n)

        # If rank was clamped, m1/m2 shapes may need updating (edge case: only
        # relevant if min(m,n) changed — in practice parameters don't change
        # shape. Handle here for robustness on first step only.)
        actual_r = G_tilde.shape[0]
        if m1.shape[0] != actual_r:
            state["m1"] = torch.zeros(actual_r, n, dtype=p.dtype, device=p.device)
            state["m2"] = torch.zeros(actual_r, n, dtype=p.dtype, device=p.device)
            m1 = state["m1"]
            m2 = state["m2"]

        # ---- Adam moments in low-rank space ----
        m1.mul_(beta1).add_(G_tilde, alpha=1.0 - beta1)
        m2.mul_(beta2).addcmul_(G_tilde, G_tilde, value=1.0 - beta2)

        # Bias correction
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        m1_hat = m1 / bc1
        m2_hat = m2 / bc2

        # Ṽ_t: optimizer output in low-rank space (r×n)
        V_tilde = m1_hat / (m2_hat.sqrt().add_(eps))  # (r_eff, n)

        # ---- Unproject: ΔW_t = P_t Ṽ_t ∈ R^{m×n} ----
        delta_W = projector.unproject(V_tilde)  # (m, n), already scaled

        # ---- W_{t+1} = W_t - lr * ΔW_t ----
        p_2d.add_(delta_W, alpha=-lr)

        # Write back (no-op if contiguous view)
        p.data.copy_(p_2d.view(original_shape))

"""Fira optimizer — Full-rank training of LLMs Under Low-rank Constraint.

Reference: "Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?"
           arXiv:2501.12369

Key idea:
  Standard GaLore projects gradients into a low-rank subspace and discards the
  residual, losing information.  Fira retains that residual as a *compensation*
  term scaled to match the low-rank component's magnitude, restoring full-rank
  effective updates at low-rank memory cost.

Algorithm (paper notation, using random projection for efficiency):

  Let G ∈ R^{m×n} be the full gradient of a 2-D parameter.
  Let r be the target rank (r << min(m, n)).
  Let P ∈ R^{r × n_flat} be a random projection matrix (row-orthonormal),
  where n_flat = min(m, n) and G is arranged so n_flat is the small dimension.

  Per step t:
    G_low  = P^T (P G_flat^T)^T          — low-rank approximation of G
    G_comp = G − G_low                   — full-rank compensation term

    # Adam-like update applied only to G_low (low-rank states):
    m_t  = β1 · m_{t-1} + (1 − β1) · G_low
    v_t  = β2 · v_{t-1} + (1 − β2) · G_low²
    m̂_t  = m_t  / (1 − β1^t)
    v̂_t  = v_t  / (1 − β2^t)
    Δ_low = lr · m̂_t / (sqrt(v̂_t) + ε)

    # Scale compensation to match low-rank component's Frobenius norm:
    c_t   = ||G_low||_F / (||G_comp||_F + ε_c)
    Δθ    = Δ_low + c_t · lr · G_comp    — full-rank effective update

  Memory cost per 2-D parameter:
    proj_matrix P : r × n_flat
    exp_avg       : r × n_flat  (stores in low-rank basis)
    exp_avg_sq    : r × n_flat
    Total ≈ 3 · r · n_flat  vs  2 · m · n  for vanilla Adam.

  1-D parameters (biases, norms) receive a plain Adam update — no projection.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

# Small constant to guard against division by zero when G_comp ≈ 0.
_EPS_COMP: float = 1e-12


def _init_proj_matrix(r: int, n_flat: int, seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Row-orthonormal random projection matrix P ∈ R^{r × n_flat}.

    Uses a deterministic seed that is unique per parameter so that
    re-initialisation at update_proj_gap boundaries is reproducible.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed % (2**31))  # Generator seeds are 32-bit on some backends
    # Sample standard normal and QR-orthonormalise rows when r <= n_flat.
    raw = torch.randn(r, n_flat, generator=gen, device=device, dtype=dtype)
    if r <= n_flat:
        # torch.linalg.qr returns Q ∈ R^{n_flat × r} (tall); transpose to get R^{r × n_flat}
        Q, _ = torch.linalg.qr(raw.T)  # Q: n_flat × r
        P = Q.T.contiguous()           # P: r × n_flat
    else:
        # r > n_flat: just normalise rows (can't fully orthonormalise)
        P = torch.nn.functional.normalize(raw, dim=1)
    return P


def _project_grad(G: torch.Tensor, P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute G_low and G_comp via random projection.

    Args:
        G: gradient tensor of shape (m, n).
        P: projection matrix of shape (r, n_flat), where n_flat = n if m >= n
           else m (we orient G so the second dim is the small one).

    Returns:
        G_low  : low-rank approximation, same shape as G.
        G_comp : G − G_low, same shape as G.
    """
    m, n = G.shape
    transposed = m < n
    if transposed:
        G = G.T  # shape (n, m) so small dim is last

    # G now has shape (M, N) with M >= N, P has shape (r, N)
    # projected: (r, M) = P @ G.T  →  G_low_T = P.T @ projected  shape (N, M)
    projected = P @ G.T            # (r, M)
    G_low_T = P.T @ projected      # (N, M) = reconstruction in original space
    G_low = G_low_T.T              # (M, N)

    G_comp = G - G_low

    if transposed:
        G_low = G_low.T
        G_comp = G_comp.T

    return G_low, G_comp


class Fira(Optimizer):
    """Fira: Full-rank training under low-rank constraint (arXiv:2501.12369).

    Args:
        params:           Iterable of parameters or parameter groups.
        lr:               Learning rate. Default: 1e-3.
        rank:             Low-rank subspace dimension r. Default: 64.
        betas:            Coefficients (β1, β2) for first/second moment estimates.
                          Default: (0.9, 0.999).
        eps:              ε for Adam denominator stability. Default: 1e-8.
        weight_decay:     L2 penalty (decoupled). Default: 0.0.
        update_proj_gap:  Number of steps between re-sampling the projection
                          matrix P. Default: 200.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        rank: int = 64,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_proj_gap: int = 200,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        beta1, beta2 = betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("betas must each be in (0, 1)")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if update_proj_gap < 1:
            raise ValueError(f"update_proj_gap must be >= 1, got {update_proj_gap}")

        defaults = dict(
            lr=lr,
            rank=rank,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_proj_gap=update_proj_gap,
        )
        super().__init__(params, defaults)
        # Counter used to assign a stable, unique seed to each matrix parameter.
        # Increments in the order params are first encountered during step().
        self._param_seed_counter: int = 0

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Perform a single optimisation step.

        Args:
            closure: Optional closure that re-evaluates the model and returns
                     the loss.

        Returns:
            loss if closure was provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            rank: int = group["rank"]
            beta1: float
            beta2: float
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            update_proj_gap: int = group["update_proj_gap"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                G: torch.Tensor = p.grad.detach()
                is_matrix = G.dim() == 2

                state = self.state[p]

                # ── State initialisation ──────────────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    if is_matrix:
                        m, n = G.shape
                        n_flat = n if m >= n else m  # small dimension
                        r = min(rank, n_flat)
                        # Assign a stable seed: combine a monotonic per-optimizer
                        # counter (order of first encounter) with the param shape.
                        # This is reproducible across runs as long as the model
                        # and optimizer are constructed in the same order.
                        seed = (self._param_seed_counter * 0x9E3779B9) ^ (m * 31 + n)
                        self._param_seed_counter += 1
                        P = _init_proj_matrix(r, n_flat, seed, G.device, G.dtype)
                        state["proj_matrix"] = P
                        state["proj_seed_base"] = seed
                        state["n_flat"] = n_flat
                        state["r"] = r
                        # Adam moments stored in the low-rank projected space:
                        # G_low has shape (m, n); store moments at full shape
                        # to keep indexing simple — memory cost is dominated by P.
                        state["exp_avg"] = torch.zeros_like(G)
                        state["exp_avg_sq"] = torch.zeros_like(G)
                    else:
                        # 1-D params: plain Adam, no projection.
                        state["exp_avg"] = torch.zeros_like(G)
                        state["exp_avg_sq"] = torch.zeros_like(G)

                state["step"] += 1
                t: int = state["step"]

                # ── Decoupled weight decay ────────────────────────────────
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                if is_matrix:
                    P: torch.Tensor = state["proj_matrix"]
                    r: int = state["r"]

                    # Re-sample projection matrix every update_proj_gap steps.
                    if t > 1 and (t - 1) % update_proj_gap == 0:
                        seed_base: int = state["proj_seed_base"]
                        # Derive a new seed from the base and the refresh count.
                        refresh_idx = (t - 1) // update_proj_gap
                        new_seed = seed_base ^ (refresh_idx * 0x9E3779B9)
                        n_flat: int = state["n_flat"]
                        P = _init_proj_matrix(r, n_flat, new_seed, G.device, G.dtype)
                        state["proj_matrix"] = P

                    # ── Low-rank projection ───────────────────────────────
                    G_low, G_comp = _project_grad(G, P)

                    # ── Adam update on G_low ──────────────────────────────
                    exp_avg: torch.Tensor = state["exp_avg"]
                    exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(G_low, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(G_low, G_low, value=1.0 - beta2)

                    bias_c1 = 1.0 - beta1 ** t
                    bias_c2 = 1.0 - beta2 ** t
                    m_hat = exp_avg / bias_c1
                    v_hat = exp_avg_sq / bias_c2
                    delta_low = m_hat / (v_hat.sqrt().add_(eps))

                    # ── Compensation scaling ──────────────────────────────
                    # c_t = ||G_low||_F / (||G_comp||_F + ε_c)
                    norm_low = G_low.norm()
                    norm_comp = G_comp.norm()
                    c_t = norm_low / (norm_comp + _EPS_COMP)

                    # ── Full update ───────────────────────────────────────
                    # Δθ = lr · (delta_low + c_t · G_comp)
                    p.add_(delta_low, alpha=-lr)
                    p.add_(G_comp, alpha=-lr * c_t.item())

                else:
                    # ── Plain Adam for 1-D parameters ─────────────────────
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(G, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(G, G, value=1.0 - beta2)

                    bias_c1 = 1.0 - beta1 ** t
                    bias_c2 = 1.0 - beta2 ** t
                    m_hat = exp_avg / bias_c1
                    v_hat = exp_avg_sq / bias_c2
                    delta = m_hat / (v_hat.sqrt().add_(eps))
                    p.add_(delta, alpha=-lr)

        return loss

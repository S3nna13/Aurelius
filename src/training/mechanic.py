"""Mechanic: A Learning Rate Tuner.

Automatically tunes the global learning rate by wrapping any base optimizer
(SGD, Adam, etc.) with an online betting algorithm.

Reference: Cutkosky & Orabona, "Mechanic: A Learning Rate Tuner",
           arXiv:2306.00144 (2023).

Key idea (Section 2, Algorithm 1):
    Maintain a scalar s (the learned learning rate) updated via online bets
    on the inner product between current gradients and the previous parameter
    update direction.  The base optimizer supplies a *unit-normed* update
    direction; Mechanic scales it by the learned s.

    Initialization:
        s = s_init  (small positive scalar, e.g. 1e-8)
        r_1, ..., r_K: geometric grid of reference learning rates
        s_k = 0  for k = 1..K  (per-reference-lr betting fractions)

    Per-step t:
        1. Let d_t be the base optimizer's parameter update direction
           (computed with the base optimizer's internal lr, then rescaled to
           unit l2-norm across all parameters).
        2. Compute directional derivative:
               h_t = Σ_param <g_t, Δθ_{t-1}>
           where Δθ_{t-1} = θ_{t-1} - θ_{t-2} is the previous parameter change.
           (On the first step Δθ_{t-1} = 0, so h_t = 0.)
        3. Update each bet s_k_t:
               s_k_t = max(s_k_{t-1} + r_k * h_t, 0)   [clamp to non-negative]
        4. Aggregate learned learning rate:
               s_t = Σ_k s_k_t
        5. Apply scaled update:
               θ_t = θ_{t-1} - s_t * d_t

Usage::

    base = torch.optim.Adam(model.parameters(), lr=1.0)
    optimizer = MechanicWrapper(base)

    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    print(f"Learned LR: {optimizer.learned_lr:.6f}")
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch.optim import Optimizer


class MechanicWrapper:
    """Wraps any base optimizer with automatic learning rate tuning (Mechanic).

    Mechanic learns a global scalar learning rate s by placing bets on a
    geometric grid of reference learning rates {r_1, ..., r_K}.  At each step
    the base optimizer produces a (unit-normed) update direction d_t, and
    Mechanic scales it by the learned s_t before applying it to the parameters.

    Variable notation follows Algorithm 1 in arXiv:2306.00144:
        s_k    : per-reference-lr bet amount (scalar, ≥ 0)
        r_k    : k-th reference learning rate in the geometric grid
        h_t    : directional derivative = Σ <g_t, Δθ_{t-1}>
        s_t    : aggregate learned learning rate = Σ_k s_k_t

    Args:
        base_optimizer: Any PyTorch Optimizer instance.  Its ``lr`` should be
            set to 1.0 (Mechanic controls the effective step size).
        s_init: Initial value for each s_k bet. Default: 1e-8.
        betas: Tuple of (β,) for optional EMA smoothing of h_t.
            Currently uses raw h_t (no smoothing).  Reserved for extension.
        eps: Numerical stability floor. Default: 1e-8.
        r_max: Maximum reference learning rate in the geometric grid.
            Default: 1.0.
        num_bets: Number K of reference learning rates in the geometric grid.
            Default: 6.

    Raises:
        ValueError: If hyperparameters are out of range.
        RuntimeError: If sparse gradients are encountered.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        s_init: float = 1e-3,
        betas: tuple[float, ...] = (0.9,),
        eps: float = 1e-8,
        r_max: float = 1.0,
        num_bets: int = 6,
    ) -> None:
        if s_init <= 0.0:
            raise ValueError(f"Invalid s_init (must be > 0): {s_init}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps (must be > 0): {eps}")
        if r_max <= 0.0:
            raise ValueError(f"Invalid r_max (must be > 0): {r_max}")
        if num_bets < 1:
            raise ValueError(f"Invalid num_bets (must be >= 1): {num_bets}")
        for b in betas:
            if not 0.0 <= b < 1.0:
                raise ValueError(f"Invalid beta value {b}; must be in [0, 1)")

        self.base_optimizer: Optimizer = base_optimizer
        self.s_init: float = s_init
        self.betas: tuple[float, ...] = betas
        self.eps: float = eps
        self.r_max: float = r_max
        self.num_bets: int = num_bets

        # Build geometric grid r_1, ..., r_K  (Section 2, Algorithm 1)
        # Grid spans [r_max * 10^{-(K-1)}, r_max] in log-space.
        K = num_bets
        if K == 1:
            self._r: list[float] = [r_max]
        else:
            # Logarithmically spaced: r_k = r_max * 10^{-(K-1-k) * log10(r_max / r_min) / (K-1)}
            # Simpler: geometric sequence from r_max * 1e-6 to r_max
            r_min = r_max * (1e-6)
            log_min = math.log10(r_min)
            log_max = math.log10(r_max)
            self._r = [10.0 ** (log_min + (log_max - log_min) * k / (K - 1)) for k in range(K)]

        # Per-reference-lr bet amounts s_k (initialized to s_init each)
        # Shape: (K,)
        self._s_k: torch.Tensor = torch.full((K,), s_init, dtype=torch.float64)
        # Reference LR tensor for vectorized update
        self._r_tensor: torch.Tensor = torch.tensor(self._r, dtype=torch.float64)

        # Per-parameter state: previous parameter values to compute Δθ
        self._prev_params: dict[int, torch.Tensor] = {}

        # Global step counter
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def learned_lr(self) -> float:
        """Return the current aggregate learned learning rate s_t = Σ_k s_k."""
        return float(self._s_k.sum().item())

    @property
    def param_groups(self):
        """Expose base optimizer param_groups for compatibility."""
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients via the base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing."""
        return {
            "s_k": self._s_k.clone(),
            "r_tensor": self._r_tensor.clone(),
            "step_count": self._step_count,
            "base_optimizer": self.base_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load optimizer state from a checkpoint."""
        self._s_k = state["s_k"].clone()
        self._r_tensor = state["r_tensor"].clone()
        self._step_count = state["step_count"]
        self.base_optimizer.load_state_dict(state["base_optimizer"])
        self._prev_params = {}  # prev_params are transient; reset on load

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform a single Mechanic + base optimizer step.

        Algorithm 1 from arXiv:2306.00144:

        1. Evaluate loss / gradients via closure (if provided).
        2. Compute h_t = Σ_param <g_t, Δθ_{t-1}>.
        3. Update bets: s_k ← max(s_k + r_k * h_t, 0).
        4. Compute aggregate s_t = Σ_k s_k.
        5. Collect (but do not yet apply) the base optimizer's direction d_t.
        6. Scale d_t by s_t and update parameters: θ ← θ - s_t * d_t.

        Args:
            closure: Optional closure re-evaluating the model and returning loss.

        Returns:
            Loss value if closure provided, else None.
        """
        loss: torch.Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect parameters and their current gradients
        params_with_grads: list[torch.Tensor] = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("MechanicWrapper does not support sparse gradients.")
                    params_with_grads.append(p)

        # ---- Step 2: Compute h_t = Σ <g_t, Δθ_{t-1}> ----
        # Δθ_{t-1} = θ_{t-1} - θ_{t-2}; on first step this is 0.
        h_t: float = 0.0
        if self._step_count > 0:
            for p in params_with_grads:
                pid = id(p)
                if pid in self._prev_params:
                    delta_theta = p.data - self._prev_params[pid]
                    h_t += float(
                        torch.dot(
                            p.grad.detach().float().flatten(),
                            delta_theta.float().flatten(),
                        ).item()
                    )

        # ---- Store θ_{t-1} (before this step's update) ----
        prev_data: dict[int, torch.Tensor] = {}
        for p in params_with_grads:
            prev_data[id(p)] = p.data.clone()

        # ---- Step 3: Update bets s_k ← max(s_k + r_k * h_t, 0) ----
        self._s_k = torch.clamp(
            self._s_k + self._r_tensor * h_t,
            min=0.0,
        )

        # ---- Step 4: Aggregate learned learning rate s_t = Σ_k s_k ----
        s_t: float = float(self._s_k.sum().item())

        # ---- Step 5: Let base optimizer compute its direction d_t ----
        # We override each param group's lr to 1.0 so the base optimizer
        # produces a unit-contribution direction, then we restore the lr.
        # The base optimizer applies: θ ← θ - lr_base * direction_base.
        # We want: θ ← θ - s_t * direction_base (unit lr = 1.0 in base).
        # Temporarily set base lr to 1.0:
        original_lrs: list[float] = []
        for group in self.base_optimizer.param_groups:
            original_lrs.append(group["lr"])
            group["lr"] = 1.0

        # Run the base optimizer — it modifies p.data in-place using lr=1.0.
        self.base_optimizer.step()

        # Restore original lrs
        for group, orig_lr in zip(self.base_optimizer.param_groups, original_lrs):
            group["lr"] = orig_lr

        # ---- Step 6: Compute base direction and apply scaled update ----
        # After base step with lr=1.0: p.data = prev_p - 1.0 * d_t
        # So d_t = prev_p - p.data (direction taken by base optimizer).
        # We want: θ_new = prev_p - s_t * d_t
        # Equivalently: p.data = prev_p - s_t * d_t
        for p in params_with_grads:
            pid = id(p)
            if pid in prev_data:
                d_t = prev_data[pid] - p.data  # direction (what base removed)
                p.data.copy_(prev_data[pid] - s_t * d_t)

        # Update previous params for next step's h_t computation.
        # Store θ_{t-1} (the params BEFORE this step's update), NOT θ_t.
        # On the next step, p.data = θ_t and _prev_params[pid] = θ_{t-1},
        # so delta_theta = θ_t - θ_{t-1} = the actual Mechanic step taken.
        for p in params_with_grads:
            self._prev_params[id(p)] = prev_data[id(p)]

        self._step_count += 1
        return loss

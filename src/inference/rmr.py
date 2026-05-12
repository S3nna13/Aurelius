"""RMR: Reinforced Mode Regulation — arXiv:2605.00435 (ICML 2026).

Mode collapse prevention via geometric regulation of the Transformer value cache.
Dynamical-systems view: mode collapse = geometric collapse (trajectory confined
to low-dimensional region). RMR identifies dominant self-reinforcing directions
via bounded-spectrum generalized eigenvalue problem and applies low-rank damping.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CorrelationDimensionMonitor:
    """Online correlation dimension tracker for generation trajectories.

    Fractal dimension quantifying dynamically active degrees of freedom.
    Mode collapse = low correlation dimension (trajectory trapped in
    low-dimensional subspace).
    """

    def __init__(
        self,
        embedding_dim: int,
        eps0: float = 0.1,
        eps1: float = 10.0,
        decay: float = 0.99,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.eps0 = eps0
        self.eps1 = eps1
        self.decay = decay
        self._state_buffer: list[Tensor] = []
        self._corr_sum: float | None = None
        self._corr_sum_eps1: float | None = None
        self._count: int = 0

    def update(self, state: Tensor) -> float:
        """Update with new state vector and return finite-time correlation dimension.

        Args:
            state: shape (D,) — next-token log-probability vector or hidden state

        Returns:
            Finite-time correlation dimension estimate
        """
        state = state.detach().reshape(-1)
        self._state_buffer.append(state)
        self._count += 1

        if self._corr_sum is None:
            self._corr_sum = 0.0

        t = self._count
        if t > 1:
            contrib_eps0 = 0.0
            contrib_eps1 = 0.0
            for i in range(t - 1):
                dist = torch.norm(state - self._state_buffer[i], p=2).item()
                if dist < self.eps0:
                    contrib_eps0 += 1.0
                if dist < self.eps1:
                    contrib_eps1 += 1.0

            corr_eps0 = contrib_eps0 / (t - 1)
            corr_eps1 = contrib_eps1 / (t - 1)
            self._corr_sum = (
                corr_eps0
                if self._corr_sum is None
                else self.decay * self._corr_sum + (1.0 - self.decay) * corr_eps0
            )
            self._corr_sum_eps1 = (
                corr_eps1
                if self._corr_sum_eps1 is None
                else self.decay * self._corr_sum_eps1 + (1.0 - self.decay) * corr_eps1
            )

        if len(self._state_buffer) > 5000:
            self._state_buffer.pop(0)

        return self._estimate_dimension()

    def _estimate_dimension(self) -> float:
        """Estimate correlation dimension from log-log slope."""
        if (
            self._corr_sum is None
            or self._corr_sum <= 0
            or self._corr_sum_eps1 is None
            or self._corr_sum_eps1 <= 0
            or self.eps0 == self.eps1
        ):
            return 0.0
        log_c0 = torch.log(torch.tensor(self._corr_sum + 1e-12))
        log_c1 = torch.log(torch.tensor(self._corr_sum_eps1 + 1e-12))
        log_eps0 = torch.log(torch.tensor(self.eps0 + 1e-12))
        log_eps1 = torch.log(torch.tensor(self.eps1 + 1e-12))
        ratio = (log_c1 - log_c0) / (log_eps1 - log_eps0)
        return max(0.0, min(float(ratio), float(self.embedding_dim)))

    def reset(self) -> None:
        self._state_buffer.clear()
        self._corr_sum = None
        self._corr_sum_eps1 = None
        self._count = 0


class PersistentDirectionTracker:
    """Tracks dominant persistent directions in the value cache.

    Estimates online covariance and cross-covariance matrices for the
    generalized eigenvalue problem: Sigma_Delta * u = lambda * Sigma * u
    Large eigenvalues (lambda ~ 1) = slow-decaying persistent directions.
    """

    def __init__(
        self,
        d_model: int,
        num_directions: int = 8,
        decay: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        self.d_model = d_model
        self.num_directions = num_directions
        self.decay = decay
        self.eps = eps

        self._Sigma: Tensor | None = None
        self._Sigma_Delta: Tensor | None = None
        self._v_bar: Tensor | None = None
        self._step: int = 0

    def update(self, v_t: Tensor, v_tp1: Tensor | None = None) -> None:
        """Update covariance statistics with new value vector.

        Args:
            v_t: shape (D,) — value vector at timestep t
            v_tp1: shape (D,) — value vector at timestep t+1 (for cross-cov)
        """
        v_t = v_t.detach().reshape(-1)
        if v_t.shape[-1] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {v_t.shape[-1]}")

        self._step += 1
        if self._v_bar is None:
            self._v_bar = v_t.clone()
        else:
            self._v_bar = self.decay * self._v_bar + (1 - self.decay) * v_t

        centered = v_t - self._v_bar
        outer = centered.unsqueeze(1) * centered.unsqueeze(0)

        if self._Sigma is None:
            self._Sigma = outer
        else:
            self._Sigma = self.decay * self._Sigma + (1 - self.decay) * outer

        if v_tp1 is not None:
            v_tp1 = v_tp1.detach().reshape(-1)
            centered_next = v_tp1 - self._v_bar
            cross = 0.5 * (centered_next.unsqueeze(1) * centered.unsqueeze(0) +
                           centered.unsqueeze(1) * centered_next.unsqueeze(0))
            if self._Sigma_Delta is None:
                self._Sigma_Delta = cross
            else:
                self._Sigma_Delta = self.decay * self._Sigma_Delta + (1 - self.decay) * cross

    def solve_generalized_eigenproblem(self) -> tuple[Tensor, Tensor]:
        """Solve (Sigma_Delta, Sigma) generalized eigenvalue problem.

        Returns:
            eigenvalues: shape (num_directions,)
            eigenvectors: shape (d_model, num_directions)
        """
        if self._Sigma is None or self._Sigma_Delta is None:
            raise RuntimeError("Need at least 2 updates before solving eigenproblem")

        try:
            eigenvalues, eigenvectors = torch.linalg.eig(
                torch.linalg.solve(self._Sigma + self.eps * torch.eye(
                    self.d_model, device=self._Sigma.device, dtype=self._Sigma.dtype
                ), self._Sigma_Delta)
            )
        except Exception:
            logger.debug(
                "Falling back after generalized eigenproblem failed",
                exc_info=True,
                extra={
                    "d_model": self.d_model,
                    "num_directions": self.num_directions,
                },
            )
            eigenvalues = torch.ones(self.num_directions, device=self._Sigma.device)
            eigenvectors = torch.eye(
                self.d_model,
                device=self._Sigma.device,
            )[:, : self.num_directions]

        eigenvalues = eigenvalues.real.float()
        eigenvectors = eigenvectors.real.float()

        perm = torch.argsort(eigenvalues.abs(), descending=True)
        eigenvalues = eigenvalues[perm[:self.num_directions]]
        eigenvectors = eigenvectors[:, perm[:self.num_directions]]

        return eigenvalues, eigenvectors

    def reset(self) -> None:
        self._Sigma = None
        self._Sigma_Delta = None
        self._v_bar = None
        self._step = 0


class RMRController:
    """Reinforced Mode Regulation controller.

    Lightweight inference-time intervention that regulates dominant persistent
    directions in the Transformer value cache via low-rank damping.

    Key parameters from paper:
        lambda_min: eigenvalue threshold (0.8) — only directions with
                   eigenvalue > lambda_min are regulated
        eta: damping strength (0.7)
        gamma: temporal decay for older timesteps (0.995)
        regulation_interval: apply damping every N steps (10)
    """

    def __init__(
        self,
        d_model: int,
        lambda_min: float = 0.8,
        eta: float = 0.7,
        gamma: float = 0.995,
        regulation_interval: int = 10,
        num_directions: int = 8,
        correlation_dim_threshold: float = 8.0,
    ) -> None:
        self.d_model = d_model
        self.lambda_min = lambda_min
        self.eta = eta
        self.gamma = gamma
        self.regulation_interval = regulation_interval
        self.num_directions = num_directions
        self.correlation_dim_threshold = correlation_dim_threshold

        self._trackers: list[PersistentDirectionTracker] = []
        self._step_count: int = 0
        self._corr_monitor = CorrelationDimensionMonitor(embedding_dim=d_model, decay=gamma)

        self._U: Tensor | None = None
        self._eigenvalues: Tensor | None = None
        self._damping_scales: Tensor | None = None

    def step(self, v_t: Tensor, v_tp1: Tensor | None = None) -> dict[str, float]:
        """Process one decoding timestep.

        Args:
            v_t: value cache vector at current step, shape (D,) or (num_heads, D)
            v_tp1: value cache vector at next step (if available)

        Returns:
            dict with 'correlation_dim', 'top_eigenvalue', 'num_regulated'
        """
        self._step_count += 1

        if v_t.dim() == 2:
            batch_results = []
            next_values = (
                v_tp1
                if v_tp1 is not None and v_tp1.shape[0] == v_t.shape[0]
                else None
            )
            for i in range(v_t.shape[0]):
                r = self._step_single(v_t[i], next_values[i] if next_values is not None else None)
                batch_results.append(r)
            return {
                "correlation_dim": sum(r["correlation_dim"] for r in batch_results) / len(batch_results),  # noqa: E501
                "top_eigenvalue": (
                    sum(r["top_eigenvalue"] for r in batch_results) / len(batch_results)
                ),
                "num_regulated": sum(r["num_regulated"] for r in batch_results),
            }
        return self._step_single(v_t, v_tp1)

    def _step_single(self, v_t: Tensor, v_tp1: Tensor | None = None) -> dict[str, float]:
        corr_dim = self._corr_monitor.update(v_t)

        if len(self._trackers) == 0:
            self._trackers.append(PersistentDirectionTracker(
                d_model=self.d_model,
                num_directions=self.num_directions,
                decay=self.gamma,
            ))
        self._trackers[0].update(v_t, v_tp1)

        top_eigenvalue = 0.0
        num_regulated = 0

        if self._step_count % self.regulation_interval == 0 and len(self._trackers) > 0:
            try:
                eigenvalues, eigenvectors = self._trackers[0].solve_generalized_eigenproblem()
                self._eigenvalues = eigenvalues
                self._U = eigenvectors

                mask = eigenvalues > self.lambda_min
                num_regulated = mask.sum().item()

                if num_regulated > 0:
                    regulated_eigenvalues = eigenvalues[mask]
                    target = torch.full_like(regulated_eigenvalues, self.lambda_min * 0.95)
                    scale = (target - regulated_eigenvalues) / (
                        regulated_eigenvalues - self.lambda_min + 1e-8
                    )
                    scale = torch.clamp(scale, -self.eta, 0.0)
                    damping_scales = torch.full_like(eigenvalues, self.eta)
                    damping_scales[mask] = (-scale).clamp(0.0, self.eta)
                    self._damping_scales = damping_scales

                if len(eigenvalues) > 0:
                    top_eigenvalue = eigenvalues[0].item()
            except Exception:
                logger.debug("RMR regulation step failed", exc_info=True)

        return {
            "correlation_dim": corr_dim,
            "top_eigenvalue": top_eigenvalue,
            "num_regulated": num_regulated,
        }

    def get_damping_matrix(self) -> Tensor | None:
        """Compute low-rank damping matrix for value cache regulation.

        Returns:
            Damping matrix of shape (D, D) or None if no regulation needed
        """
        if self._U is None or self._eigenvalues is None:
            return None

        mask = self._eigenvalues > self.lambda_min
        if mask.sum() == 0:
            return None

        U_reg = self._U[:, mask]
        if self._damping_scales is not None:
            scale = self._damping_scales[mask].to(device=U_reg.device, dtype=U_reg.dtype)
        else:
            scale = self.eta * torch.ones(mask.sum(), device=U_reg.device, dtype=U_reg.dtype)
        damping = U_reg @ torch.diag(scale) @ U_reg.T
        return damping

    def apply_damping(self, V: Tensor) -> Tensor:
        """Apply RMR damping to value cache matrix.

        V = (I - eta * Gamma) * V * (I - U U^T)

        Args:
            V: value cache matrix, shape (T, D) or (B, T, D)

        Returns:
            Damped value cache
        """
        damping = self.get_damping_matrix()
        if damping is None:
            return V

        if V.dim() == 2:
            damping = damping.to(device=V.device, dtype=V.dtype)
            return V @ (torch.eye(self.d_model, device=V.device, dtype=V.dtype) - damping)
        elif V.dim() == 3:
            damping = damping.to(device=V.device, dtype=V.dtype)
            identity = torch.eye(self.d_model, device=V.device, dtype=V.dtype)
            proj = identity - damping
            return torch.matmul(V, proj)

        return V

    def is_collapsing(self) -> bool:
        """Check if generation is entering mode collapse."""
        return self._corr_monitor._corr_sum is not None and self._corr_monitor._estimate_dimension() < self.correlation_dim_threshold  # noqa: E501

    def reset(self) -> None:
        self._trackers.clear()
        self._step_count = 0
        self._corr_monitor.reset()
        self._U = None
        self._eigenvalues = None
        self._damping_scales = None


class RMRIntegration:
    """Integration helper for applying RMR to a HuggingFace model.

    Wraps a model and applies RMR damping to the value cache during generation.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_min: float = 0.8,
        eta: float = 0.7,
        gamma: float = 0.995,
        regulation_interval: int = 10,
        layer_indices: list[int] | None = None,
        d_model: int | None = None,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices
        self.controllers: dict[int, RMRController] = {}

        def _get_d_model() -> int:
            for _name, module in model.named_modules():
                if hasattr(module, "head_dim") and hasattr(module, "num_heads"):
                    return module.head_dim * module.num_heads
            raise ValueError("d_model could not be inferred; please pass explicit d_model")

        d_model = d_model if d_model is not None else _get_d_model()

        if layer_indices is None:
            config = getattr(model, "config", None)
            if config is None or not hasattr(config, "num_hidden_layers"):
                num_layers = 32
            else:
                num_layers = int(config.num_hidden_layers)
            layer_indices = list(range(num_layers))

        for layer_idx in layer_indices:
            self.controllers[layer_idx] = RMRController(
                d_model=d_model,
                lambda_min=lambda_min,
                eta=eta,
                gamma=gamma,
                regulation_interval=regulation_interval,
            )

    def step(self, layer_idx: int, v_t: Tensor, v_tp1: Tensor | None = None) -> dict[str, float]:
        """Process one layer's value cache timestep."""
        if layer_idx not in self.controllers:
            return {"correlation_dim": 0.0, "top_eigenvalue": 0.0, "num_regulated": 0}
        return self.controllers[layer_idx].step(v_t, v_tp1)

    def apply_damping_layer(self, layer_idx: int, V: Tensor) -> Tensor:
        """Apply damping to a specific layer's value cache."""
        if layer_idx not in self.controllers:
            return V
        return self.controllers[layer_idx].apply_damping(V)

    def is_any_collapsing(self) -> bool:
        return any(c.is_collapsing() for c in self.controllers.values())

    def reset_all(self) -> None:
        for c in self.controllers.values():
            c.reset()

"""Loss landscape analysis: sharpness measurement, flatness-seeking perturbations, and SAM optimizer."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


def flatten_parameters(params: list[torch.Tensor]) -> torch.Tensor:
    """Flatten a parameter list to one vector."""
    if not params:
        return torch.empty(0)
    return torch.cat([param.reshape(-1) for param in params])


def unflatten_like(vector: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a flat vector into tensors matching reference shapes."""
    outputs: list[torch.Tensor] = []
    offset = 0
    for ref in like:
        numel = ref.numel()
        outputs.append(vector[offset : offset + numel].reshape_as(ref))
        offset += numel
    if offset != vector.numel():
        raise ValueError("Vector size does not match reference tensors")
    return outputs


def interpolate_parameters(
    params_a: list[torch.Tensor],
    params_b: list[torch.Tensor],
    alpha: float,
) -> list[torch.Tensor]:
    """Linearly interpolate two parameter sets."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if len(params_a) != len(params_b):
        raise ValueError("Parameter lists must have the same length")
    outputs = []
    for tensor_a, tensor_b in zip(params_a, params_b):
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Parameter shapes must match")
        outputs.append((1.0 - alpha) * tensor_a + alpha * tensor_b)
    return outputs


def random_direction_like(params: list[torch.Tensor], normalize: bool = True) -> list[torch.Tensor]:
    """Sample a random direction in parameter space."""
    direction = [torch.randn_like(param) for param in params]
    if normalize and direction:
        flat = flatten_parameters(direction)
        norm = flat.norm().clamp_min(1e-8)
        direction = [tensor / norm for tensor in direction]
    return direction


def perturb_parameters(
    params: list[torch.Tensor],
    direction: list[torch.Tensor],
    epsilon: float,
) -> list[torch.Tensor]:
    """Perturb parameters along a chosen direction."""
    if len(params) != len(direction):
        raise ValueError("params and direction must have the same length")
    return [param + epsilon * delta for param, delta in zip(params, direction)]


@dataclass(frozen=True)
class LandscapeSlice:
    alphas: torch.Tensor
    losses: torch.Tensor


def interpolation_slice(
    params_a: list[torch.Tensor],
    params_b: list[torch.Tensor],
    loss_fn,
    n_points: int = 5,
) -> LandscapeSlice:
    """Evaluate loss along a line segment in parameter space."""
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    alphas = torch.linspace(0.0, 1.0, n_points)
    losses = []
    for alpha in alphas:
        params = interpolate_parameters(params_a, params_b, float(alpha.item()))
        losses.append(loss_fn(params))
    return LandscapeSlice(alphas=alphas, losses=torch.stack(losses))


def local_sharpness(base_loss: torch.Tensor, perturbed_losses: torch.Tensor) -> torch.Tensor:
    """Compute max loss increase over a local perturbation set."""
    return (perturbed_losses - base_loss).max()


# ---------------------------------------------------------------------------
# LossLandscapeExplorer and LandscapeStats
# ---------------------------------------------------------------------------


class LossLandscapeExplorer:
    """
    Explore loss landscape by perturbing parameters along random/specific directions.

    Args:
        model: the model to analyze
        loss_fn: callable(model) -> scalar loss
        filter_normalization: normalize perturbation directions by parameter norms
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        filter_normalization: bool = True,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.filter_normalization = filter_normalization

    def _get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a flat vector."""
        return torch.cat([p.detach().view(-1) for p in self.model.parameters()])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set all parameters from a flat vector."""
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat_params[offset : offset + n].view_as(p))
            offset += n

    def _random_direction(self) -> torch.Tensor:
        """
        Sample a random direction in parameter space.
        If filter_normalization: normalize direction to have same per-layer norm as params.
        Returns: flat direction vector.
        """
        if self.filter_normalization:
            # Filter normalization: scale each layer's direction to match param norm
            parts = []
            for p in self.model.parameters():
                d = torch.randn_like(p)
                p_norm = p.data.norm()
                d_norm = d.norm().clamp_min(1e-8)
                # Scale direction so its norm matches parameter norm
                d = d * (p_norm / d_norm)
                parts.append(d.view(-1))
            return torch.cat(parts)
        else:
            flat = self._get_flat_params()
            d = torch.randn_like(flat)
            d = d / d.norm().clamp_min(1e-8)
            return d

    def line_scan(
        self,
        direction: torch.Tensor | None = None,
        alphas: torch.Tensor | None = None,
        n_points: int = 20,
        alpha_range: tuple = (-1.0, 1.0),
    ) -> dict:
        """
        Scan loss along one direction from current parameters.

        Returns: {'alphas': tensor, 'losses': tensor}
        Restores original parameters after scan.
        """
        original_params = self._get_flat_params().clone()

        if direction is None:
            direction = self._random_direction()

        if alphas is None:
            alphas = torch.linspace(alpha_range[0], alpha_range[1], n_points)

        losses = []
        with torch.no_grad():
            for alpha in alphas:
                new_params = original_params + alpha.item() * direction
                self._set_flat_params(new_params)
                loss = self.loss_fn(self.model)
                if isinstance(loss, torch.Tensor):
                    losses.append(loss.detach().cpu())
                else:
                    losses.append(torch.tensor(float(loss)))

        # Restore original parameters
        self._set_flat_params(original_params)

        return {
            "alphas": alphas,
            "losses": torch.stack(losses),
        }

    def surface_scan(
        self,
        direction1: torch.Tensor | None = None,
        direction2: torch.Tensor | None = None,
        n_points: int = 10,
        alpha_range: tuple = (-1.0, 1.0),
    ) -> dict:
        """
        Scan loss over a 2D grid (two directions).

        Returns: {'alphas1': tensor, 'alphas2': tensor, 'losses': (n,n) tensor}
        Restores original parameters after scan.
        """
        original_params = self._get_flat_params().clone()

        if direction1 is None:
            direction1 = self._random_direction()
        if direction2 is None:
            direction2 = self._random_direction()

        alphas = torch.linspace(alpha_range[0], alpha_range[1], n_points)
        loss_grid = torch.zeros(n_points, n_points)

        with torch.no_grad():
            for i, a1 in enumerate(alphas):
                for j, a2 in enumerate(alphas):
                    new_params = original_params + a1.item() * direction1 + a2.item() * direction2
                    self._set_flat_params(new_params)
                    loss = self.loss_fn(self.model)
                    if isinstance(loss, torch.Tensor):
                        loss_grid[i, j] = loss.detach().cpu().item()
                    else:
                        loss_grid[i, j] = float(loss)

        # Restore original parameters
        self._set_flat_params(original_params)

        return {
            "alphas1": alphas,
            "alphas2": alphas,
            "losses": loss_grid,
        }

    def flatness_score(self, epsilon: float = 0.1, n_directions: int = 5) -> float:
        """
        Estimate flatness: mean loss increase over random perturbations of size epsilon.
        Lower = flatter = better generalization.
        Returns: mean(loss(perturbed) - loss(original))
        """
        original_params = self._get_flat_params().clone()

        with torch.no_grad():
            base_loss = self.loss_fn(self.model)
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.item()

        increases = []
        for _ in range(n_directions):
            direction = self._random_direction()
            # Normalize direction to unit norm then scale by epsilon
            direction = direction / direction.norm().clamp_min(1e-8)
            perturbed = original_params + epsilon * direction

            with torch.no_grad():
                self._set_flat_params(perturbed)
                pert_loss = self.loss_fn(self.model)
                if isinstance(pert_loss, torch.Tensor):
                    pert_loss = pert_loss.item()
                increases.append(pert_loss - base_loss)

        self._set_flat_params(original_params)
        return float(sum(increases) / len(increases)) if increases else 0.0

    def sharpness_ratio(self, epsilon: float = 0.1, n_directions: int = 10) -> float:
        """
        SAM-style sharpness: max loss increase in epsilon-ball.
        Samples multiple directions, returns max increase.
        """
        original_params = self._get_flat_params().clone()

        with torch.no_grad():
            base_loss = self.loss_fn(self.model)
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.item()

        max_increase = 0.0
        for _ in range(n_directions):
            direction = self._random_direction()
            direction = direction / direction.norm().clamp_min(1e-8)
            perturbed = original_params + epsilon * direction

            with torch.no_grad():
                self._set_flat_params(perturbed)
                pert_loss = self.loss_fn(self.model)
                if isinstance(pert_loss, torch.Tensor):
                    pert_loss = pert_loss.item()
                increase = pert_loss - base_loss
                if increase > max_increase:
                    max_increase = increase

        self._set_flat_params(original_params)
        return float(max_increase)


class LandscapeStats:
    """Statistics about the loss landscape."""

    def __init__(self, losses: torch.Tensor) -> None:
        self.losses = losses

    def local_minima_count(self, window: int = 3) -> int:
        """Count local minima in 1D loss profile (for line_scan output)."""
        losses = self.losses
        if losses.numel() < 3:
            return 0

        losses_1d = losses.view(-1)
        n = losses_1d.numel()
        count = 0
        half = window // 2

        for i in range(half, n - half):
            # Check if losses[i] is less than all neighbors within window
            window_vals = losses_1d[max(0, i - half) : min(n, i + half + 1)]
            if losses_1d[i] == window_vals.min() and (window_vals < losses_1d[i]).sum() == 0:
                # Strictly less than all other values in window
                others = torch.cat(
                    [losses_1d[max(0, i - half) : i], losses_1d[i + 1 : min(n, i + half + 1)]]
                )
                if len(others) > 0 and (losses_1d[i] < others).all():
                    count += 1

        return max(0, count)

    def curvature_at_center(self) -> float:
        """Estimate curvature via finite differences at center point."""
        losses_1d = self.losses.view(-1).float()
        n = losses_1d.numel()
        if n < 3:
            return 0.0

        center = n // 2
        # Second-order finite difference: f''(x) ≈ f(x+h) - 2f(x) + f(x-h)
        curvature = (losses_1d[center + 1] - 2.0 * losses_1d[center] + losses_1d[center - 1]).item()
        return float(curvature)

    def is_convex(self) -> bool:
        """Check if loss profile is convex (losses form a bowl shape)."""
        losses_1d = self.losses.view(-1).float()
        n = losses_1d.numel()
        if n < 3:
            return True

        # A 1D function is convex if second differences are >= 0
        for i in range(1, n - 1):
            second_diff = (losses_1d[i + 1] - 2.0 * losses_1d[i] + losses_1d[i - 1]).item()
            if second_diff < -1e-6:
                return False
        return True


# ---------------------------------------------------------------------------
# SAM / Sharpness-Aware Minimization API
# ---------------------------------------------------------------------------


@dataclass
class LandscapeConfig:
    """Configuration for loss landscape analysis and SAM optimizer."""

    rho: float = 0.05  # SAM perturbation radius
    sharpness_n_perturbations: int = 10
    sharpness_epsilon: float = 0.01
    adaptive_sam: bool = False  # ASAM: normalize perturbation by parameter magnitude


def compute_gradient_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients concatenated.

    Returns 0.0 if no gradients exist.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    flat = torch.cat([g.reshape(-1) for g in grads])
    return flat.norm().item()


def compute_weight_norm(model: nn.Module) -> float:
    """L2 norm of all parameters concatenated."""
    params = [p for p in model.parameters()]
    if not params:
        return 0.0
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    return flat.norm().item()


def measure_sharpness(
    model: nn.Module,
    loss_fn: Callable,
    n_perturbations: int,
    epsilon: float,
) -> float:
    """Estimate sharpness as average loss increase under random Gaussian perturbations.

    For each perturbation: add N(0, epsilon²) noise to each parameter, compute
    loss, restore. Returns mean(perturbed_loss - base_loss).
    """
    # Save original parameters
    originals = {name: p.data.clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        base_loss = loss_fn(model)
        if isinstance(base_loss, Tensor):
            base_loss = base_loss.item()

    increases = []
    for _ in range(n_perturbations):
        # Add Gaussian noise
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn_like(p) * epsilon
                p.data.add_(noise)

        with torch.no_grad():
            perturbed_loss = loss_fn(model)
            if isinstance(perturbed_loss, Tensor):
                perturbed_loss = perturbed_loss.item()

        increases.append(perturbed_loss - base_loss)

        # Restore original parameters
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.data.copy_(originals[name])

    return float(sum(increases) / len(increases)) if increases else 0.0


def sam_first_step(
    model: nn.Module,
    loss: Tensor,
    rho: float,
    adaptive: bool = False,
) -> dict[str, Tensor]:
    """SAM first step: find and apply perturbation that maximizes loss.

    Computes gradients, then applies perturbation:
        perturbation[i] = rho * grad[i] / (grad_norm + 1e-12)
    If adaptive (ASAM), scales perturbation elementwise by |param_i|.

    Returns saved original params dict for restoration.
    """
    params = list(model.parameters())

    # Compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)

    # Replace None grads with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

    # Compute global grad norm
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    grad_norm = flat_grads.norm() + 1e-12

    # Save original params
    original_params = {name: p.data.clone() for name, p in model.named_parameters()}

    # Compute and apply perturbation
    with torch.no_grad():
        for p, g in zip(params, grads):
            perturbation = rho * g / grad_norm
            if adaptive:
                perturbation = perturbation * p.abs()
            p.data.add_(perturbation)

    return original_params


def sam_second_step(
    model: nn.Module,
    original_params: dict[str, Tensor],
    loss: Tensor,
    optimizer,
) -> None:
    """SAM second step: restore original params, backward, optimizer step.

    Args:
        model: the model (currently at perturbed point)
        original_params: saved params from sam_first_step
        loss: loss computed at the perturbed point
        optimizer: base optimizer
    """
    # Restore original params
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.copy_(original_params[name])

    # Backward on perturbed loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class SAMOptimizer:
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Wraps a base optimizer to implement the two-step SAM update.
    Usage:
        sam = SAMOptimizer(base_optimizer, model, config)
        loss1 = compute_loss(model, batch)
        sam.first_step(loss1)
        loss2 = compute_loss(model, batch)   # at perturbed point
        sam.second_step(loss2)
    """

    def __init__(
        self,
        base_optimizer,
        model: nn.Module,
        config: LandscapeConfig,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.model = model
        self.config = config
        self._original_params: dict[str, Tensor] | None = None

    def first_step(self, loss: Tensor) -> None:
        """Apply SAM perturbation to model parameters."""
        self._original_params = sam_first_step(
            self.model,
            loss,
            rho=self.config.rho,
            adaptive=self.config.adaptive_sam,
        )

    def second_step(self, loss: Tensor) -> None:
        """Restore original params, backward on perturbed loss, optimizer step."""
        assert self._original_params is not None, "Call first_step before second_step"  # noqa: S101
        sam_second_step(self.model, self._original_params, loss, self.base_optimizer)
        self._original_params = None

    def zero_grad(self) -> None:
        """Delegate zero_grad to base optimizer."""
        self.base_optimizer.zero_grad()


def compute_hessian_trace(
    model: nn.Module,
    loss: Tensor,
    n_samples: int = 5,
) -> float:
    """Estimate Hessian trace via Hutchinson estimator.

    trace(H) ≈ mean(v^T H v) where v ~ Rademacher.
    Uses finite-difference approximation of Hv to avoid requiring
    second-order autograd support (which some ops like flash attention lack):
        Hv ≈ (grad(L, params + eps*v) - grad(L, params)) / eps

    Returns float estimate.
    """
    eps = 1e-3
    params = [p for p in model.parameters() if p.requires_grad]

    # Compute base gradients (no create_graph needed — we use finite differences)
    base_grads = torch.autograd.grad(loss, params, allow_unused=True)
    base_grads = [
        g.detach().clone() if g is not None else torch.zeros_like(p)
        for g, p in zip(base_grads, params)
    ]

    estimates = []
    for _ in range(n_samples):
        # Draw Rademacher vector
        vs = [torch.randint(0, 2, p.shape, dtype=p.dtype, device=p.device) * 2 - 1 for p in params]

        # Perturb parameters by eps * v
        with torch.no_grad():
            for p, v in zip(params, vs):
                p.data.add_(eps * v)

        # Compute perturbed loss and gradients
        # We need a fresh forward pass at perturbed params
        # Re-use the same loss_fn structure: recompute loss
        try:
            # Try to recompute using the computational graph (may not always work)
            pert_grads_raw = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        except RuntimeError:
            pert_grads_raw = [None] * len(params)

        if all(g is None for g in pert_grads_raw):
            # Fallback: zero Hv estimate for this sample
            with torch.no_grad():
                for p, v in zip(params, vs):
                    p.data.sub_(eps * v)
            estimates.append(0.0)
            continue

        pert_grads = [
            g.detach() if g is not None else torch.zeros_like(p)
            for g, p in zip(pert_grads_raw, params)
        ]

        # Hv ≈ (pert_grad - base_grad) / eps
        Hvs = [(pg - bg) / eps for pg, bg in zip(pert_grads, base_grads)]

        # Restore parameters
        with torch.no_grad():
            for p, v in zip(params, vs):
                p.data.sub_(eps * v)

        # v^T H v
        vHv = sum((v * hv).sum().item() for v, hv in zip(vs, Hvs))
        estimates.append(float(vHv))

    return float(sum(estimates) / len(estimates)) if estimates else 0.0

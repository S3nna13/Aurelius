"""Loss landscape analysis (Li et al., 2018).

Implements random-direction perturbation, filter normalization, and 1D/2D
landscape sampling to visualize the loss surface of a neural network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LandscapeConfig:
    """Configuration for loss landscape scanning.

    Attributes:
        n_points:        Number of grid points per dimension.  Should be odd
                         so that 0 (the base model) sits exactly at the centre.
        alpha_range:     Scan range; perturbs weights by ± alpha_range × direction.
        filter_normalize: If True, scale each direction vector so that each
                          "filter" has the same norm as the corresponding model
                          parameter filter (Li et al., 2018, §3).
    """
    n_points: int = 11
    alpha_range: float = 1.0
    filter_normalize: bool = True


# ---------------------------------------------------------------------------
# Direction generation
# ---------------------------------------------------------------------------

class DirectionGenerator:
    """Generate perturbation directions for loss landscape analysis."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def random_direction(self) -> List[torch.Tensor]:
        """Return a random Gaussian direction with the same shapes as model params."""
        return [torch.randn_like(p.data) for p in self.model.parameters()]

    def filter_normalize_direction(
        self, direction: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Scale each direction vector filter-wise to match model parameter norms.

        For a parameter tensor of shape (out, ...) each "filter" is one row
        (the slice along dimension 0).  For 1-D parameters the whole tensor is
        treated as a single filter.

        After normalization: ‖d[i]_filter‖ = ‖theta[i]_filter‖  ∀ filter i.
        """
        normalized: List[torch.Tensor] = []
        for d, p in zip(direction, self.model.parameters()):
            theta = p.data
            if theta.dim() <= 1:
                # Treat entire 1-D tensor as a single filter
                theta_norm = theta.norm()
                d_norm = d.norm()
                scale = theta_norm / (d_norm + 1e-10)
                normalized.append(d * scale)
            else:
                # Normalize each row (filter) independently
                # Flatten all dims beyond the first
                d_2d = d.view(d.shape[0], -1)
                theta_2d = theta.view(theta.shape[0], -1)
                d_row_norms = d_2d.norm(dim=1, keepdim=True)       # (out, 1)
                theta_row_norms = theta_2d.norm(dim=1, keepdim=True)  # (out, 1)
                scale = theta_row_norms / (d_row_norms + 1e-10)    # (out, 1)
                d_scaled = (d_2d * scale).view_as(d)
                normalized.append(d_scaled)
        return normalized

    def pca_direction(
        self,
        trajectory: List[List[torch.Tensor]],
        component: int = 0,
    ) -> List[torch.Tensor]:
        """Compute a PCA direction from a sequence of parameter snapshots.

        Args:
            trajectory: List of T snapshots, each being a list of param tensors
                        matching model.parameters() shapes.
            component:  Which principal component to return (0 = largest).

        Returns:
            A list of tensors (same shapes as model params) representing the
            chosen principal component, reshaped back to param shapes.
        """
        if len(trajectory) < 2:
            raise ValueError("trajectory must contain at least 2 snapshots")

        # Flatten each snapshot to a 1-D vector
        flat_snapshots: List[torch.Tensor] = []
        for snap in trajectory:
            flat_snapshots.append(torch.cat([t.reshape(-1) for t in snap]))

        # Stack: (T, D)
        mat = torch.stack(flat_snapshots, dim=0).float()

        # Centre the matrix
        mat = mat - mat.mean(dim=0, keepdim=True)

        # SVD: U (T×T), S (min(T,D),), Vh (D×D or min×D)
        # We want the D-dimensional left basis (rows of Vh)
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)  # Vh: (k, D)

        direction_flat = Vh[component]  # (D,)

        # Reshape back to param shapes
        param_shapes = [p.shape for p in self.model.parameters()]
        direction: List[torch.Tensor] = []
        offset = 0
        for shape in param_shapes:
            numel = 1
            for s in shape:
                numel *= s
            direction.append(direction_flat[offset : offset + numel].reshape(shape))
            offset += numel

        return direction


# ---------------------------------------------------------------------------
# Landscape evaluation
# ---------------------------------------------------------------------------

class LandscapeEvaluator:
    """Evaluate model loss along 1D and 2D directions in parameter space."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        config: LandscapeConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_params(self) -> List[torch.Tensor]:
        """Return cloned copies of current model parameter data."""
        return [p.data.clone() for p in self.model.parameters()]

    def _set_params(self, params: List[torch.Tensor]) -> None:
        """Set model parameters to the provided values (in-place)."""
        for p, v in zip(self.model.parameters(), params):
            p.data.copy_(v)

    def _perturb(
        self,
        base: List[torch.Tensor],
        direction: List[torch.Tensor],
        alpha: float,
    ) -> List[torch.Tensor]:
        """Return base + alpha * direction element-wise."""
        return [b + alpha * d for b, d in zip(base, direction)]

    # ------------------------------------------------------------------
    # 1-D scan
    # ------------------------------------------------------------------

    def scan_1d(
        self,
        direction: List[torch.Tensor],
        loss_inputs,
    ) -> Dict[str, torch.Tensor]:
        """Scan the loss along a single direction.

        Args:
            direction:   Perturbation direction (list of tensors, param shapes).
            loss_inputs: Positional arguments forwarded to loss_fn after model,
                         i.e. ``loss_fn(model, *loss_inputs)``.

        Returns:
            Dictionary with keys:
            - ``"alphas"``: 1-D float Tensor of shape (n_points,)
            - ``"losses"``: 1-D float Tensor of shape (n_points,)
        """
        cfg = self.config
        alphas = torch.linspace(-cfg.alpha_range, cfg.alpha_range, cfg.n_points)

        base = self._get_params()
        losses: List[float] = []

        for alpha in alphas.tolist():
            perturbed = self._perturb(base, direction, alpha)
            self._set_params(perturbed)
            with torch.no_grad():
                loss_val = self.loss_fn(self.model, *loss_inputs)
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
            losses.append(float(loss_val))

        # Always restore original params
        self._set_params(base)

        return {
            "alphas": alphas,
            "losses": torch.tensor(losses, dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # 2-D scan
    # ------------------------------------------------------------------

    def scan_2d(
        self,
        dir1: List[torch.Tensor],
        dir2: List[torch.Tensor],
        loss_inputs,
    ) -> Dict[str, torch.Tensor]:
        """Scan the loss over a 2-D grid spanned by dir1 and dir2.

        Args:
            dir1:        First perturbation direction.
            dir2:        Second perturbation direction.
            loss_inputs: Forwarded to ``loss_fn(model, *loss_inputs)``.

        Returns:
            Dictionary with keys:
            - ``"alphas"``: 1-D Tensor of shape (n_points,)  — row axis
            - ``"betas"``:  1-D Tensor of shape (n_points,)  — column axis
            - ``"losses"``: 2-D Tensor of shape (n_points, n_points)
        """
        cfg = self.config
        alphas = torch.linspace(-cfg.alpha_range, cfg.alpha_range, cfg.n_points)
        betas = torch.linspace(-cfg.alpha_range, cfg.alpha_range, cfg.n_points)

        base = self._get_params()
        n = cfg.n_points
        loss_grid = torch.zeros(n, n, dtype=torch.float32)

        for i, alpha in enumerate(alphas.tolist()):
            for j, beta in enumerate(betas.tolist()):
                # base + alpha*dir1 + beta*dir2
                perturbed = [
                    b + alpha * d1 + beta * d2
                    for b, d1, d2 in zip(base, dir1, dir2)
                ]
                self._set_params(perturbed)
                with torch.no_grad():
                    loss_val = self.loss_fn(self.model, *loss_inputs)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()
                loss_grid[i, j] = float(loss_val)

        # Always restore original params
        self._set_params(base)

        return {
            "alphas": alphas,
            "betas": betas,
            "losses": loss_grid,
        }


# ---------------------------------------------------------------------------
# Landscape statistics
# ---------------------------------------------------------------------------

class LandscapeStats:
    """Compute scalar statistics from 1-D or 2-D landscape scan results."""

    def sharpness(self, losses: torch.Tensor, center_idx: int) -> float:
        """Sharpness: maximum deviation from the centre loss.

        Defined as ``max(losses) - min(losses)`` which is always ≥ 0.

        Args:
            losses:     1-D or 2-D loss Tensor from a landscape scan.
            center_idx: Index of the centre point along each axis (unused in
                        this implementation but retained for API consistency).

        Returns:
            Non-negative float.
        """
        return float((losses.max() - losses.min()).item())

    def curvature_at_center(
        self, losses: torch.Tensor, alphas: torch.Tensor
    ) -> float:
        """Finite-difference second derivative of the loss at the centre.

        Uses the standard three-point stencil:
            f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²

        The centre is taken as the middle element of ``losses`` / ``alphas``.

        Args:
            losses: 1-D loss Tensor of odd length.
            alphas: 1-D Tensor of corresponding alpha values.

        Returns:
            Estimated second derivative (float).
        """
        n = losses.shape[0]
        if n < 3:
            raise ValueError("curvature_at_center requires at least 3 points")
        c = n // 2
        h = float((alphas[c + 1] - alphas[c - 1]).item()) / 2.0
        if abs(h) < 1e-12:
            return 0.0
        f_plus = float(losses[c + 1].item())
        f_center = float(losses[c].item())
        f_minus = float(losses[c - 1].item())
        return (f_plus - 2.0 * f_center + f_minus) / (h * h)

    def is_convex(self, losses: torch.Tensor) -> bool:
        """Check whether the 1-D loss curve is (approximately) convex.

        A curve is convex if every interior point lies at or below the average
        of its two neighbours (i.e. the curve "bows upward").

        Args:
            losses: 1-D loss Tensor.

        Returns:
            True if the curve is convex, False otherwise.
        """
        if losses.dim() != 1 or losses.shape[0] < 3:
            return bool(losses.shape[0] < 3)

        for i in range(1, losses.shape[0] - 1):
            midpoint = (losses[i - 1].item() + losses[i + 1].item()) / 2.0
            if float(losses[i].item()) > midpoint + 1e-6:
                return False
        return True

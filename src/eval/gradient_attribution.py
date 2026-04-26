"""Gradient-based token attribution methods.

Implements four attribution strategies using pure PyTorch:
  - VanillaGradients    : L2 norm (or abs-sum) of d(target)/d(embedding)
  - GradientXInput      : element-wise grad * input, summed over d (signed)
  - IntegratedGradients : Sundararajan et al. (2017), trapezoidal rule
  - KernelSHAPApproximator : Lundberg & Lee (2017) approximation via
                             weighted linear regression on token masks
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AttributionConfig:
    """Configuration shared across attribution methods.

    Attributes:
        n_steps:   Number of interpolation steps for IntegratedGradients.
        baseline:  Strategy for constructing the reference input.
                   ``"zero"`` uses zeros_like; ``"random"`` uses randn_like.
        normalize: If True, rescale final attributions to sum to 1.
    """

    n_steps: int = 50
    baseline: str = "zero"  # "zero" | "random"
    normalize: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _grad_wrt_embeddings(embeddings: Tensor, target_fn: Callable) -> Tensor:
    """Compute gradient of target_fn(embeddings) w.r.t. embeddings.

    Args:
        embeddings: Floating-point tensor with ``requires_grad=True``.
        target_fn:  Callable that maps embeddings → scalar.

    Returns:
        Gradient tensor of the same shape as *embeddings*.
    """
    if not embeddings.requires_grad:
        embeddings = embeddings.detach().requires_grad_(True)

    output = target_fn(embeddings)
    if not isinstance(output, Tensor):
        raise TypeError("target_fn must return a scalar Tensor")
    if output.numel() != 1:
        raise ValueError("target_fn must return a scalar (single-element) Tensor")

    grads = torch.autograd.grad(output, embeddings, create_graph=False)[0]
    return grads


def _build_baseline(embeddings: Tensor, strategy: str) -> Tensor:
    """Construct a baseline tensor with the same shape as *embeddings*.

    Args:
        embeddings: Reference tensor (shape is copied).
        strategy:   ``"zero"`` → zeros; ``"random"`` → standard normal.

    Returns:
        Baseline tensor (detached, no grad).
    """
    if strategy == "zero":
        return torch.zeros_like(embeddings.detach())
    elif strategy == "random":
        return torch.randn_like(embeddings.detach())
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy!r}. Use 'zero' or 'random'.")


# ---------------------------------------------------------------------------
# 1. Vanilla Gradients
# ---------------------------------------------------------------------------


class VanillaGradients:
    """Attribution via plain first-order gradients of the target w.r.t. embeddings.

    Args:
        model: A ``torch.nn.Module`` (stored but not called directly; the
               caller passes ``target_fn`` which may reference the model).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    def attribute(self, embeddings: Tensor, target_fn: Callable) -> Tensor:
        """L2-norm of gradient over the embedding dimension.

        Args:
            embeddings: Shape ``(B, T, d)`` with ``requires_grad=True`` (or
                        will be detached and re-wrapped internally).
            target_fn:  ``f(embeddings) → scalar Tensor``.

        Returns:
            Attribution map of shape ``(B, T)`` — always non-negative.
        """
        emb = embeddings.detach().requires_grad_(True)
        grads = _grad_wrt_embeddings(emb, target_fn)  # (B, T, d)
        # L2 norm over d
        attribution = grads.norm(p=2, dim=-1)  # (B, T)
        return attribution

    # ------------------------------------------------------------------
    def saliency(self, embeddings: Tensor, target_fn: Callable) -> Tensor:
        """Absolute-value gradient saliency, summed over the embedding dim.

        Args:
            embeddings: Shape ``(B, T, d)``.
            target_fn:  ``f(embeddings) → scalar Tensor``.

        Returns:
            Saliency map of shape ``(B, T)`` — always non-negative.
        """
        emb = embeddings.detach().requires_grad_(True)
        grads = _grad_wrt_embeddings(emb, target_fn)  # (B, T, d)
        saliency = grads.abs().sum(dim=-1)  # (B, T)
        return saliency


# ---------------------------------------------------------------------------
# 2. Gradient × Input
# ---------------------------------------------------------------------------


class GradientXInput:
    """Element-wise product of gradient and input, summed over the embedding dim.

    The result is *signed* — positive values indicate tokens that push the
    target score up; negative values push it down.

    Args:
        model: A ``torch.nn.Module`` (stored for reference; callers pass
               ``target_fn`` that uses the model as needed).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------
    def attribute(self, embeddings: Tensor, target_fn: Callable) -> Tensor:
        """Signed grad × input attribution.

        Args:
            embeddings: Shape ``(B, T, d)``.
            target_fn:  ``f(embeddings) → scalar Tensor``.

        Returns:
            Attribution map of shape ``(B, T)`` — can be negative.
        """
        emb = embeddings.detach().requires_grad_(True)
        grads = _grad_wrt_embeddings(emb, target_fn)  # (B, T, d)
        attribution = (grads * emb).sum(dim=-1)  # (B, T), signed
        return attribution


# ---------------------------------------------------------------------------
# 3. Integrated Gradients
# ---------------------------------------------------------------------------


class IntegratedGradients:
    """Integrated Gradients (Sundararajan, Taly & Yan, 2017).

    Attributes:
        model:   The model (stored; callers pass ``target_fn``).
        n_steps: Number of Riemann approximation steps.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50) -> None:
        self.model = model
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    def attribute(
        self,
        embeddings: Tensor,
        target_fn: Callable,
        baseline: Tensor | None = None,
    ) -> Tensor:
        """Compute integrated gradients from *baseline* to *embeddings*.

        Args:
            embeddings: Shape ``(B, T, d)``.
            target_fn:  ``f(embeddings) → scalar Tensor``.
            baseline:   Optional baseline of shape ``(B, T, d)``.
                        Defaults to ``zeros_like(embeddings)``.

        Returns:
            Attribution map of shape ``(B, T)``.
        """
        embeddings = embeddings.detach()
        if baseline is None:
            baseline = torch.zeros_like(embeddings)
        else:
            baseline = baseline.detach()

        # Accumulate gradients at each interpolation step (trapezoidal rule)
        grad_sum = torch.zeros_like(embeddings)

        for k in range(self.n_steps + 1):
            alpha = k / self.n_steps
            interp = (baseline + alpha * (embeddings - baseline)).requires_grad_(True)
            output = target_fn(interp)
            grads = torch.autograd.grad(output, interp, create_graph=False)[0]

            # Trapezoidal weights: endpoints get 0.5, interior steps get 1.0
            weight = 0.5 if k in (0, self.n_steps) else 1.0
            grad_sum = grad_sum + weight * grads.detach()

        # Mean gradient (trapezoidal) × (input - baseline), summed over d
        mean_grad = grad_sum / self.n_steps  # (B, T, d)
        attribution = (mean_grad * (embeddings - baseline)).sum(dim=-1)  # (B, T)
        return attribution

    # ------------------------------------------------------------------
    def completeness_check(
        self,
        attributions: Tensor,
        embeddings: Tensor,
        baseline: Tensor,
        target_fn: Callable,
    ) -> float:
        """Verify the completeness axiom: sum(IG) ≈ f(x) - f(baseline).

        Args:
            attributions: Shape ``(B, T)`` — output of ``attribute()``.
            embeddings:   Shape ``(B, T, d)``.
            baseline:     Shape ``(B, T, d)``.
            target_fn:    ``f(embeddings) → scalar Tensor``.

        Returns:
            Absolute difference ``|sum(attributions) - (f(x) - f(baseline))|``
            as a Python float.
        """
        embeddings = embeddings.detach()
        baseline = baseline.detach()

        with torch.no_grad():
            fx = target_fn(embeddings).item()
            fb = target_fn(baseline).item()

        attr_sum = attributions.sum().item()
        delta = fx - fb
        return abs(attr_sum - delta)


# ---------------------------------------------------------------------------
# 4. Kernel SHAP Approximation
# ---------------------------------------------------------------------------


class KernelSHAPApproximator:
    """Approximate Shapley values via Kernel SHAP (Lundberg & Lee, 2017).

    Operates on a *single* sequence (B=1).

    Args:
        model:     The model (stored; callers pass ``target_fn``).
        n_samples: Number of random coalition samples.
    """

    def __init__(self, model: nn.Module, n_samples: int = 50) -> None:
        self.model = model
        self.n_samples = n_samples

    # ------------------------------------------------------------------
    @staticmethod
    def _kernel_weight(T: int, coalition_size: int) -> float:
        """Kernel SHAP weighting for a coalition of a given size.

        w(|S|) = (T-1) / [ C(T,|S|) * |S| * (T-|S|) ]

        Edge cases (|S|=0 or |S|=T) are assigned weight 0.

        Args:
            T:              Total number of tokens.
            coalition_size: Number of tokens in the coalition.

        Returns:
            Non-negative float weight.
        """
        s = coalition_size
        if s == 0 or s == T:
            return 0.0
        comb = math.comb(T, s)
        return (T - 1) / (comb * s * (T - s))

    # ------------------------------------------------------------------
    def attribute(
        self,
        embeddings: Tensor,
        target_fn: Callable,
        baseline: Tensor | None = None,
    ) -> Tensor:
        """Estimate per-token Shapley values.

        Randomly samples ``n_samples`` token coalitions, evaluates
        ``target_fn`` on each masked input, and fits a weighted linear
        regression (Kernel SHAP) to obtain approximate Shapley values.

        Args:
            embeddings: Shape ``(1, T, d)`` — single sequence only.
            target_fn:  ``f(embeddings) → scalar Tensor``.
            baseline:   Optional baseline of shape ``(1, T, d)``.
                        Defaults to ``zeros_like(embeddings)``.

        Returns:
            Approximate Shapley values, shape ``(T,)``.
        """
        if embeddings.shape[0] != 1:
            raise ValueError(
                "KernelSHAPApproximator.attribute requires batch size 1 "
                f"(got {embeddings.shape[0]})."
            )

        embeddings = embeddings.detach()  # (1, T, d)
        T = embeddings.shape[1]

        if baseline is None:
            baseline = torch.zeros_like(embeddings)
        else:
            baseline = baseline.detach()

        # Evaluate model at full input and baseline (for centering)
        with torch.no_grad():
            target_fn(embeddings).item()
            f_base = target_fn(baseline).item()

        # Sample random coalitions and collect (mask_vector, model_output)
        masks_list: list[list[float]] = []
        outputs: list[float] = []
        weights: list[float] = []

        rng = torch.Generator()
        rng.manual_seed(42)

        for _ in range(self.n_samples):
            # Random binary mask over T tokens
            mask = torch.bernoulli(torch.full((T,), 0.5), generator=rng).bool()  # (T,)

            coalition_size = int(mask.sum().item())
            w = self._kernel_weight(T, coalition_size)
            if w == 0.0:
                # Skip trivial coalitions; replace with a random non-trivial one
                coalition_size = torch.randint(1, T, (1,), generator=rng).item()
                perm = torch.randperm(T, generator=rng)
                mask = torch.zeros(T, dtype=torch.bool)
                mask[perm[:coalition_size]] = True
                w = self._kernel_weight(T, int(coalition_size))

            # Build masked embedding: use real embedding for tokens in mask,
            # baseline otherwise
            masked_emb = baseline.clone()  # (1, T, d)
            masked_emb[0, mask, :] = embeddings[0, mask, :]

            with torch.no_grad():
                f_masked = target_fn(masked_emb).item()

            masks_list.append(mask.float().tolist())
            # Center the output: y = f(masked) - f(baseline)
            outputs.append(f_masked - f_base)
            weights.append(w)

        # Solve weighted least-squares:  Z phi ≈ y  (phi = Shapley values)
        # where Z is (n_samples, T) binary design matrix
        Z = torch.tensor(masks_list, dtype=torch.float32)  # (n, T)
        y = torch.tensor(outputs, dtype=torch.float32)  # (n,)
        W = torch.diag(torch.tensor(weights, dtype=torch.float32))  # (n, n)

        # phi = (Z^T W Z)^{-1} Z^T W y
        ZtW = Z.T @ W  # (T, n)
        ZtWZ = ZtW @ Z  # (T, T)
        ZtWy = ZtW @ y  # (T,)

        # Add small ridge for numerical stability
        ridge = 1e-6 * torch.eye(T, dtype=torch.float32)
        try:
            phi = torch.linalg.solve(ZtWZ + ridge, ZtWy)  # (T,)
        except RuntimeError:
            # Fallback: pseudo-inverse
            phi = torch.linalg.lstsq(ZtWZ + ridge, ZtWy.unsqueeze(-1)).solution.squeeze(-1)

        return phi

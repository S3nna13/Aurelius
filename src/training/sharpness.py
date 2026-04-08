"""Loss landscape sharpness metrics.

Implements:
- SAM (Sharpness-Aware Minimization) perturbation (Foret et al., 2021 - arXiv:2010.01412)
- Hutchinson Hessian trace estimator (Hutchinson 1989, Yao et al. PyHessian 2020)
- Flatness scoring for model quality assessment

SAM seeks flat minima by perturbing parameters toward worst-case neighbours:
    epsilon_hat = rho * g / (||g|| + eps)   where g = nabla_theta loss

The Hutchinson estimator approximates tr(H) using Rademacher random vectors:
    tr(H) approx (1/n) * sum_i  v_i^T H v_i   where v_i ~ {-1, +1}^dim
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SharpnessConfig:
    """Configuration for sharpness measurement."""

    rho: float = 0.05              # SAM perturbation radius
    n_hutchinson_samples: int = 10  # random vectors for Hessian trace
    eps: float = 1e-12             # numerical stability


@dataclass
class SharpnessResult:
    """Result of sharpness measurement."""

    original_loss: float
    perturbed_loss: float
    sharpness: float           # = perturbed_loss - original_loss
    hessian_trace: float | None  # None if not computed
    flatness_score: float      # = 1 / (1 + sharpness) in (0, 1], 1 = flat


def sam_perturbation(
    model: nn.Module,
    loss: Tensor,
    rho: float = 0.05,
    eps: float = 1e-12,
) -> dict[str, Tensor | float]:
    """Compute and apply the SAM gradient ascent perturbation in place.

    Computes epsilon_hat = rho * g / (||g|| + eps) where g = nabla_theta loss,
    then adds epsilon_hat to each parameter in place.

    Args:
        model: The neural network module.
        loss: Scalar loss tensor.
        rho: Perturbation radius.
        eps: Small constant for numerical stability.

    Returns:
        dict with:
          - "perturbation": dict[name -> Tensor] mapping param names to the
            perturbation added (for later restoration).
          - "grad_norm": float, the global gradient norm before normalisation.
    """
    # Compute gradients w.r.t. all parameters
    loss.backward()

    # Collect gradients and compute global grad norm
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads[name] = param.grad.detach().clone()

    grad_norm = torch.sqrt(
        sum(g.norm() ** 2 for g in grads.values())
    ).item()

    # Compute and apply perturbation: epsilon_hat = rho * g / (||g|| + eps)
    perturbation: dict[str, Tensor] = {}
    scale = rho / (grad_norm + eps)
    for name, param in model.named_parameters():
        if param.requires_grad and name in grads:
            pert = scale * grads[name]
            perturbation[name] = pert
            param.data.add_(pert)

    return {"perturbation": perturbation, "grad_norm": grad_norm}


def restore_perturbation(model: nn.Module, perturbation: dict[str, Tensor]) -> None:
    """Subtract the perturbation from model parameters, restoring original weights.

    Args:
        model: The neural network module (currently perturbed).
        perturbation: dict mapping param names to perturbation tensors,
            as returned by sam_perturbation()["perturbation"].
    """
    for name, param in model.named_parameters():
        if name in perturbation:
            param.data.sub_(perturbation[name])


def sharpness_aware_loss(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    rho: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Perform a full SAM evaluation step.

    Steps:
    1. Forward+backward to get epsilon_hat (original loss).
    2. Perturb weights by epsilon_hat.
    3. Forward at perturbed weights -> perturbed_loss (no grad needed).
    4. Restore original weights.

    Args:
        model: The transformer model.
        input_ids: (B, S) token ids.
        labels: (B, S) target token ids.
        rho: SAM perturbation radius.

    Returns:
        (original_loss, perturbed_loss) -- both scalar Tensors.
    """
    was_training = model.training
    model.train()

    # Step 1: initial forward + backward to get gradients
    model.zero_grad()
    original_loss, _, _ = model(input_ids, labels=labels)
    original_loss_val = original_loss.detach().clone()

    result = sam_perturbation(model, original_loss, rho=rho)

    # Steps 2 & 3: forward at perturbed weights
    model.zero_grad()
    with torch.no_grad():
        perturbed_loss, _, _ = model(input_ids, labels=labels)
    perturbed_loss_val = perturbed_loss.detach().clone()

    # Step 4: restore
    restore_perturbation(model, result["perturbation"])

    if not was_training:
        model.eval()

    return original_loss_val, perturbed_loss_val


def hutchinson_hessian_trace(
    model: nn.Module,
    loss_fn: Callable[[], Tensor],
    n_samples: int = 10,
    eps: float = 1e-12,
) -> float:
    """Rademacher-based Hutchinson estimator of the Hessian trace.

    Approximates tr(H) approx (1/n) * sum_i [v_i^T H v_i]
    where v_i ~ {-1, +1}^dim (Rademacher distribution).

    Uses double-backward via autograd:
      1. grads = nabla_theta loss  (with create_graph=True for second-order)
      2. Hv_i = nabla(grads . v_i)  (second backward)

    Flash attention does not support second-order gradients, so we force
    the math SDPA backend during this computation.

    Args:
        model: The neural network module.
        loss_fn: Zero-argument callable returning a scalar loss.
        n_samples: Number of Rademacher samples.
        eps: Small constant (unused, kept for API consistency).

    Returns:
        Float estimate of tr(H).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    trace_estimates = []
    for _ in range(n_samples):
        # Draw Rademacher vector v ~ {-1, +1}
        vs = [
            torch.randint(0, 2, p.shape, dtype=p.dtype, device=p.device) * 2 - 1
            for p in params
        ]

        # First-order gradients with graph retained for second-order pass.
        # Force math SDPA backend (fresh context each iteration) so that
        # second-order autograd works -- flash attention on CPU does not
        # support second-order derivatives.
        model.zero_grad()
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            _ctx = sdpa_kernel(SDPBackend.MATH)
        except Exception:
            import contextlib
            _ctx = contextlib.nullcontext()

        with _ctx:
            loss = loss_fn()
            grads = torch.autograd.grad(loss, params, create_graph=True)

            # v^T g (dot product, scalar)
            gv = sum((g * v).sum() for g, v in zip(grads, vs))

            # Hv = nabla(v^T g) -- second-order pass
            hv = torch.autograd.grad(gv, params, retain_graph=False)

        # v^T H v
        vhv = sum((v * hv_i).sum().item() for v, hv_i in zip(vs, hv))
        trace_estimates.append(vhv)

        del grads, gv, hv

    return float(sum(trace_estimates) / len(trace_estimates))


def measure_sharpness(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    cfg: SharpnessConfig,
    compute_hessian: bool = False,
) -> SharpnessResult:
    """Orchestrate sharpness measurement combining SAM and optional Hutchinson trace.

    Args:
        model: The transformer model.
        input_ids: (B, S) token ids.
        labels: (B, S) target token ids.
        cfg: SharpnessConfig with rho, n_hutchinson_samples, eps.
        compute_hessian: If True, also compute Hutchinson Hessian trace.

    Returns:
        SharpnessResult with all metrics populated.
    """
    original_loss_t, perturbed_loss_t = sharpness_aware_loss(
        model, input_ids, labels, rho=cfg.rho
    )

    original_loss = original_loss_t.item()
    perturbed_loss = perturbed_loss_t.item()
    sharpness = perturbed_loss - original_loss
    flatness_score = 1.0 / (1.0 + max(sharpness, 0.0))

    hessian_trace: float | None = None
    if compute_hessian:
        def loss_fn() -> Tensor:
            loss, _, _ = model(input_ids, labels=labels)
            return loss

        hessian_trace = hutchinson_hessian_trace(
            model, loss_fn,
            n_samples=cfg.n_hutchinson_samples,
            eps=cfg.eps,
        )

    return SharpnessResult(
        original_loss=original_loss,
        perturbed_loss=perturbed_loss,
        sharpness=sharpness,
        hessian_trace=hessian_trace,
        flatness_score=flatness_score,
    )

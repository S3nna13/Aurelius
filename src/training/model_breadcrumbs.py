"""Model Breadcrumbs -- Selective sparse weight perturbations for unlearning.

Based on: Panda et al. 2023, "Model Breadcrumbs: Scaling Multi-Task Model
Merging with Sparse Masks".

Key idea: instead of full retraining, apply tiny random perturbations to a
carefully chosen sparse subset of weights (e.g., the 1% with smallest absolute
value). This degrades memorisation of specific data without substantially
changing the model's general capabilities.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BreadcrumbConfig:
    """Configuration for Model Breadcrumbs unlearning.

    Attributes:
        sparsity: Fraction of weights to KEEP unchanged (e.g. 0.99 means only
            1% of weights will be perturbed).
        perturbation_scale: Standard deviation of Gaussian noise added at
            selected positions.
        n_iterations: Number of sequential perturbation rounds.
        selection_method: How to choose which weights to perturb.
            - "magnitude": smallest |w| -- weakly encoded information is safer
              to modify without destroying overall capability.
            - "random": uniformly random selection.
            - "gradient": largest |grad| -- most sensitive weights w.r.t. the
              forget set (requires forget_inputs / forget_targets).
        seed: Base RNG seed for reproducibility.
    """
    sparsity: float = 0.99
    perturbation_scale: float = 0.01
    n_iterations: int = 5
    selection_method: str = "magnitude"  # "magnitude" | "random" | "gradient"
    seed: int = 42


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def select_weight_mask(
    param: Tensor,
    sparsity: float,
    method: str = "magnitude",
    grad: Optional[Tensor] = None,
) -> Tensor:
    """Return a boolean mask selecting (1 - sparsity) fraction of weights to perturb.

    Args:
        param: Parameter tensor of any shape.
        sparsity: Fraction of weights to KEEP (mask=False). E.g. 0.99 keeps 99%.
        method: Selection strategy -- "magnitude", "random", or "gradient".
        grad: Gradient tensor (same shape as param); required for method="gradient".

    Returns:
        Boolean tensor, same shape as ``param``.
        True  means this weight will be perturbed.
        False means this weight is left unchanged.
    """
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    n_total = param.numel()
    # Number of weights to PERTURB
    n_perturb = max(1, int(math.ceil(n_total * (1.0 - sparsity))))

    flat = param.detach().float().flatten()
    mask_flat = torch.zeros(n_total, dtype=torch.bool, device=param.device)

    if method == "magnitude":
        # Perturb weights with the SMALLEST absolute value.
        scores = flat.abs()
        _, indices = torch.topk(scores, k=n_perturb, largest=False, sorted=False)

    elif method == "random":
        indices = torch.randperm(n_total, device=param.device)[:n_perturb]

    elif method == "gradient":
        if grad is None:
            raise ValueError("method='gradient' requires a non-None grad tensor")
        grad_flat = grad.detach().float().flatten().abs()
        # Perturb weights with the LARGEST gradient magnitude (most sensitive)
        _, indices = torch.topk(grad_flat, k=n_perturb, largest=True, sorted=False)

    else:
        raise ValueError(
            f"Unknown selection_method: {method!r}. "
            "Choose 'magnitude', 'random', or 'gradient'."
        )

    mask_flat[indices] = True
    return mask_flat.view(param.shape)


def apply_breadcrumb_perturbation(
    model: nn.Module,
    masks: dict[str, Tensor],
    scale: float = 0.01,
    seed: int = 42,
) -> nn.Module:
    """Apply sparse Gaussian perturbations to a deep copy of ``model``.

    For each parameter listed in ``masks``, adds noise N(0, scale^2) only
    at positions where the mask is True. Positions with mask=False are left
    bit-for-bit identical.

    Args:
        model: Source model (not modified).
        masks: Mapping from parameter name to boolean mask (same shape as param).
        scale: Standard deviation of the Gaussian perturbation.
        seed: RNG seed for reproducibility.

    Returns:
        A deep copy of ``model`` with perturbed weights.
    """
    new_model = copy.deepcopy(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.no_grad():
        param_dict = dict(new_model.named_parameters())
        for name, mask in masks.items():
            if name not in param_dict:
                continue
            param = param_dict[name]
            # Generate noise on CPU then move to param's device
            cpu_noise = torch.normal(
                mean=0.0,
                std=scale,
                size=param.shape,
                generator=generator,
            )
            noise = torch.zeros_like(param)
            noise.copy_(cpu_noise)
            param.data += noise * mask.to(param.device)

    return new_model


def compute_weight_masks(
    model: nn.Module,
    config: BreadcrumbConfig,
    forget_inputs: Optional[Tensor] = None,
    forget_targets: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute per-parameter boolean masks for every trainable parameter.

    For "gradient" selection the model is run in forward+backward mode on
    (forget_inputs, forget_targets) to obtain gradients. The model's
    parameters are restored to their original values after this probe.

    Args:
        model: The model whose parameters will be masked.
        config: Breadcrumb settings (sparsity, selection_method, seed).
        forget_inputs: Input token ids for gradient-based selection.
        forget_targets: Target token ids for gradient-based selection.

    Returns:
        Dict mapping parameter name to bool mask tensor (True = will perturb).
    """
    torch.manual_seed(config.seed)

    grads: dict[str, Optional[Tensor]] = {}

    if config.selection_method == "gradient":
        if forget_inputs is None or forget_targets is None:
            raise ValueError(
                "selection_method='gradient' requires forget_inputs and forget_targets"
            )
        original_training = model.training
        model.train()
        model.zero_grad()
        loss, _, _ = model(forget_inputs, labels=forget_targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] = param.grad.detach().clone()
            else:
                grads[name] = None
        model.zero_grad()
        if not original_training:
            model.eval()

    masks: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = grads.get(name) if config.selection_method == "gradient" else None
        masks[name] = select_weight_mask(
            param,
            sparsity=config.sparsity,
            method=config.selection_method,
            grad=grad,
        )

    return masks


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BreadcrumbResult:
    """Summary statistics from a breadcrumb unlearning run.

    Attributes:
        n_params_changed: Total number of scalar weights that were perturbed.
        fraction_changed: n_params_changed / total_trainable_parameters.
        perturbation_norms: Per-parameter L2 norm of the perturbation delta.
    """
    n_params_changed: int
    fraction_changed: float
    perturbation_norms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class BreadcrumbUnlearner:
    """Apply Model Breadcrumbs for selective unlearning without full retraining.

    Usage::

        config = BreadcrumbConfig(sparsity=0.99, perturbation_scale=0.01)
        unlearner = BreadcrumbUnlearner(model, config)
        new_model, result = unlearner.run(forget_inputs, forget_targets)
    """

    def __init__(self, model: nn.Module, config: BreadcrumbConfig) -> None:
        self.model = model
        self.config = config

    def compute_masks(
        self,
        forget_inputs: Optional[Tensor] = None,
        forget_targets: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute sparse boolean masks using ``config.selection_method``.

        Args:
            forget_inputs: Token ids for the data to forget (gradient selection).
            forget_targets: Target ids for the forget data (gradient selection).

        Returns:
            Dict mapping parameter name to bool mask tensor.
        """
        return compute_weight_masks(
            self.model,
            self.config,
            forget_inputs=forget_inputs,
            forget_targets=forget_targets,
        )

    def apply(self, masks: dict[str, Tensor]) -> tuple[nn.Module, BreadcrumbResult]:
        """Apply breadcrumb perturbations using pre-computed masks.

        Runs ``config.n_iterations`` rounds of perturbation, each with an
        independently seeded RNG to avoid cancellation.

        Args:
            masks: Dict mapping parameter name to bool mask tensor.

        Returns:
            Tuple of (perturbed_model, BreadcrumbResult).
        """
        current_model = self.model
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        for iteration in range(self.config.n_iterations):
            iter_seed = self.config.seed + iteration
            current_model = apply_breadcrumb_perturbation(
                current_model,
                masks,
                scale=self.config.perturbation_scale,
                seed=iter_seed,
            )

        # Compute result statistics by comparing original vs final model
        n_changed = 0
        perturbation_norms: dict[str, float] = {}

        orig_params = dict(self.model.named_parameters())
        new_params = dict(current_model.named_parameters())

        with torch.no_grad():
            for name, mask in masks.items():
                if name not in orig_params or name not in new_params:
                    continue
                delta = (new_params[name] - orig_params[name]).float()
                n_changed += int((delta.abs() > 0).sum().item())
                perturbation_norms[name] = delta.norm().item()

        fraction_changed = n_changed / total_params if total_params > 0 else 0.0

        result = BreadcrumbResult(
            n_params_changed=n_changed,
            fraction_changed=fraction_changed,
            perturbation_norms=perturbation_norms,
        )
        return current_model, result

    def run(
        self,
        forget_inputs: Optional[Tensor] = None,
        forget_targets: Optional[Tensor] = None,
    ) -> tuple[nn.Module, BreadcrumbResult]:
        """Full pipeline: compute masks then apply perturbations.

        Args:
            forget_inputs: Token ids for data to forget (used with gradient selection).
            forget_targets: Target ids for data to forget.

        Returns:
            Tuple of (perturbed_model, BreadcrumbResult).
        """
        masks = self.compute_masks(
            forget_inputs=forget_inputs,
            forget_targets=forget_targets,
        )
        return self.apply(masks)


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def measure_weight_change(
    original_model: nn.Module,
    modified_model: nn.Module,
) -> dict[str, float]:
    """Measure per-layer weight change between two models.

    Args:
        original_model: The baseline model (before perturbation).
        modified_model: The model after breadcrumb perturbation.

    Returns:
        Dict with:
        - one entry per named parameter: L2 norm of the delta tensor.
        - 'total_change': sum of all per-layer L2 norms.
        - 'fraction_changed': fraction of scalar elements with non-zero delta.
    """
    result: dict[str, float] = {}
    total_change = 0.0
    n_changed = 0
    n_total = 0

    orig_params = dict(original_model.named_parameters())
    new_params = dict(modified_model.named_parameters())

    with torch.no_grad():
        for name in orig_params:
            if name not in new_params:
                continue
            delta = (new_params[name] - orig_params[name]).float()
            layer_norm = delta.norm().item()
            result[name] = layer_norm
            total_change += layer_norm
            n_changed += int((delta.abs() > 0).sum().item())
            n_total += delta.numel()

    result["total_change"] = total_change
    result["fraction_changed"] = n_changed / n_total if n_total > 0 else 0.0
    return result

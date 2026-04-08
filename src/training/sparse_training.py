"""Circuit-sparsity-inspired training for Aurelius.

Two mechanisms:
1. Target-Loss Pruning — iteratively zero out weights until a perplexity
   (loss) budget is exhausted or a step cap is reached.
2. L0 Regularization via Hard Concrete — differentiable gate distribution
   (Louizos et al. 2018, arXiv:1712.01312) that approximates the L0 norm
   of a weight mask during training.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ── 1. Target-Loss Pruning ───────────────────────────────────────────────────

@dataclass
class TargetLossPruningConfig:
    """Configuration for iterative magnitude pruning with a loss budget."""
    target_loss: float = 3.0
    prune_fraction_per_step: float = 0.01
    n_eval_batches: int = 10
    max_steps: int = 100
    magnitude_based: bool = True


@dataclass
class PruningResult:
    """Result of a target-loss pruning run."""
    final_loss: float
    target_loss: float
    n_steps: int
    sparsity: float
    converged: bool
    loss_history: list = field(default_factory=list)
    sparsity_history: list = field(default_factory=list)


def compute_model_sparsity(model: nn.Module) -> float:
    """Fraction of zero weights across all parameters.

    Returns a float in [0, 1].
    """
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    if total == 0:
        return 0.0
    return zeros / total


def prune_magnitude(model: nn.Module, fraction: float) -> int:
    """Zero the bottom `fraction` of weights by absolute magnitude.

    Uses global unstructured pruning: all parameters are pooled and a single
    threshold is computed across the whole model.

    Args:
        model: The model to prune in-place.
        fraction: Fraction of *total* (including already-zero) weights to zero.

    Returns:
        Number of weights newly zeroed during this call.
    """
    if fraction <= 0.0:
        return 0

    # Collect all parameter tensors (flattened views into the real storage)
    params = [p.data for p in model.parameters()]
    if not params:
        return 0

    flat = torch.cat([p.flatten() for p in params])
    total = flat.numel()
    n_to_prune = max(1, int(total * fraction))

    # Global threshold via k-th smallest absolute value
    abs_flat = flat.abs()
    threshold, _ = torch.kthvalue(abs_flat, n_to_prune)

    newly_zeroed = 0
    for p in model.parameters():
        mask = p.data.abs() <= threshold
        was_nonzero = (p.data != 0) & mask
        newly_zeroed += was_nonzero.sum().item()
        p.data[mask] = 0.0

    return int(newly_zeroed)


def evaluate_model_loss(
    model: nn.Module,
    data_batches: list,
) -> float:
    """Compute mean cross-entropy loss over the supplied batches.

    Each batch is a (input_ids, labels) tuple.  The model must return a
    scalar loss as its first output when called with both tensors.

    Runs under torch.no_grad().
    """
    model.eval()
    total_loss = 0.0
    n_batches = len(data_batches)
    if n_batches == 0:
        return float("inf")

    with torch.no_grad():
        for input_ids, labels in data_batches:
            out = model(input_ids, labels)
            if isinstance(out, (tuple, list)):
                loss = out[0]
            else:
                loss = out
            total_loss += loss.item()

    return total_loss / n_batches


def target_loss_prune(
    model: nn.Module,
    data_batches: list,
    cfg: TargetLossPruningConfig,
) -> PruningResult:
    """Iteratively prune weights until target loss is reached or max_steps exceeded.

    Algorithm:
        1. Evaluate initial loss.
        2. If loss <= target_loss: converged immediately (0 steps).
        3. Else: prune → evaluate → repeat up to max_steps times.

    Returns a PruningResult with complete history.
    """
    eval_batches = data_batches[: cfg.n_eval_batches]

    # Initial evaluation
    current_loss = evaluate_model_loss(model, eval_batches)

    # Check before any pruning
    if current_loss <= cfg.target_loss:
        return PruningResult(
            final_loss=current_loss,
            target_loss=cfg.target_loss,
            n_steps=0,
            sparsity=compute_model_sparsity(model),
            converged=True,
            loss_history=[],
            sparsity_history=[],
        )

    loss_history: list[float] = []
    sparsity_history: list[float] = []
    converged = False
    step = 0

    for step in range(1, cfg.max_steps + 1):
        prune_magnitude(model, cfg.prune_fraction_per_step)
        current_loss = evaluate_model_loss(model, eval_batches)
        current_sparsity = compute_model_sparsity(model)

        loss_history.append(current_loss)
        sparsity_history.append(current_sparsity)

        logger.debug(
            "Prune step %d: loss=%.4f sparsity=%.4f",
            step,
            current_loss,
            current_sparsity,
        )

        if current_loss <= cfg.target_loss:
            converged = True
            break

    return PruningResult(
        final_loss=current_loss,
        target_loss=cfg.target_loss,
        n_steps=step,
        sparsity=compute_model_sparsity(model),
        converged=converged,
        loss_history=loss_history,
        sparsity_history=sparsity_history,
    )


# ── 2. L0 Regularization via Hard Concrete ───────────────────────────────────

class HardConcrete(nn.Module):
    """Differentiable L0 approximation via the hard-concrete distribution.

    Reference: Louizos et al. (2018) "Learning Sparse Neural Networks through
    L0 Regularization" — arXiv:1712.01312.

    Each gate z_i is computed as:
        u ~ Uniform(0, 1)
        s = sigmoid((log u - log(1 - u) + log_alpha) / beta)
        s_bar = s * (zeta - gamma) + gamma
        z = clamp(s_bar, 0, 1)

    At inference (eval mode):
        z = (log_alpha > 0).float()   [hard threshold]

    Expected L0 norm (penalty):
        E[z_i] = sigmoid(log_alpha_i - beta * log(-gamma / zeta))
    """

    def __init__(
        self,
        n_weights: int,
        beta: float = 0.66,
        zeta: float = 1.1,
        gamma: float = -0.1,
    ) -> None:
        super().__init__()
        self.n_weights = n_weights
        self.beta = beta
        self.zeta = zeta
        self.gamma = gamma

        # Initialise log_alpha around 0 so gates start near 0.5
        self.log_alpha = nn.Parameter(torch.zeros(n_weights))

    # Pre-compute the shift used in the L0 penalty (constant w.r.t. parameters)
    @property
    def _l0_offset(self) -> float:
        return self.beta * math.log(-self.gamma / self.zeta)

    def forward(self) -> Tensor:
        if not self.training:
            # Hard 0/1 gates
            return (self.log_alpha > 0).float()

        # Reparameterised sample
        u = torch.zeros_like(self.log_alpha).uniform_().clamp(1e-8, 1 - 1e-8)
        # log-odds of u
        noise = torch.log(u) - torch.log(1.0 - u)
        s = torch.sigmoid((noise + self.log_alpha) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = s_bar.clamp(0.0, 1.0)
        return z

    def l0_penalty(self) -> Tensor:
        """Expected number of non-zero gates (scalar)."""
        return torch.sigmoid(self.log_alpha - self._l0_offset).sum()


class L0LinearLayer(nn.Module):
    """Linear layer with per-output-neuron L0 gates via HardConcrete."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **hc_kwargs,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # One gate per output neuron (row of weight matrix)
        self.hard_concrete = HardConcrete(n_weights=out_features, **hc_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        z = self.hard_concrete()          # (out_features,)
        masked_weight = self.weight * z.unsqueeze(1)   # broadcast over in_features
        out = F.linear(x, masked_weight, self.bias)
        return out

    def l0_penalty(self) -> Tensor:
        """Return the expected L0 count from the HardConcrete module."""
        return self.hard_concrete.l0_penalty()

    def effective_sparsity(self) -> float:
        """Fraction of output neurons currently gated to 0."""
        with torch.no_grad():
            z = self.hard_concrete()
        n_zero = (z == 0).sum().item()
        return n_zero / self.out_features


def l0_regularization_loss(
    model: nn.Module,
    l0_lambda: float = 1e-4,
) -> Tensor:
    """Sum the L0 penalties from all L0LinearLayer modules, scaled by l0_lambda.

    Returns a scalar tensor (0.0 if no L0 layers are found).
    """
    total = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, L0LinearLayer):
            total = total + module.l0_penalty()
    return l0_lambda * total


def add_l0_regularization(
    model: nn.Module,
    target_modules: Optional[list] = None,
    skip_modules: Optional[list] = None,
    **hc_kwargs,
) -> int:
    """Replace nn.Linear layers with L0LinearLayer in-place.

    Args:
        model: The model to modify.
        target_modules: If given, only replace layers whose names are in this
            list.  None means all Linear layers.
        skip_modules: Names of modules to skip even if they match.
        **hc_kwargs: Forwarded to HardConcrete (beta, zeta, gamma).

    Returns:
        Number of layers replaced.
    """
    if skip_modules is None:
        skip_modules = []

    n_replaced = 0

    # We need to iterate over named children recursively and replace in the
    # parent's attribute dict — use a queue of (parent, attr_name, child).
    def _replace_in(parent: nn.Module, prefix: str) -> None:
        nonlocal n_replaced
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, L0LinearLayer):
                # Already replaced
                _replace_in(child, full_name)
                continue

            if isinstance(child, nn.Linear):
                # Filter by target_modules / skip_modules
                if target_modules is not None and full_name not in target_modules:
                    _replace_in(child, full_name)
                    continue
                if full_name in skip_modules:
                    _replace_in(child, full_name)
                    continue

                new_layer = L0LinearLayer(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    **hc_kwargs,
                )
                # Copy existing weights
                with torch.no_grad():
                    new_layer.weight.copy_(child.weight)
                    if child.bias is not None and new_layer.bias is not None:
                        new_layer.bias.copy_(child.bias)

                setattr(parent, name, new_layer)
                n_replaced += 1
            else:
                _replace_in(child, full_name)

    _replace_in(model, "")
    return n_replaced

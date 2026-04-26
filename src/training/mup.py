"""
Maximal Update Parametrization (muP) for the Aurelius LLM project.

Greg Yang et al., 2022 -- "Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer".

muP rescales weight initializations and learning rates so that pre-activation
standard deviations remain O(1) regardless of model width.  Hyperparameters
tuned on a small proxy model transfer zero-shot to the full 1.395B model.

Key rules for a standard decoder-only transformer
--------------------------------------------------
| Parameter type         | Init std       | LR multiplier          |
|------------------------|----------------|------------------------|
| Embedding weights      | 1/sqrt(fan_in) | base_lr                |
| Hidden->Hidden weights | 1/fan_in       | base_lr / width_mult   |
| Output logit head      | 1/fan_in       | base_lr / width_mult   |
| Biases and norms       | 0 / 1          | base_lr                |

where width_multiplier = d_model / d_model_base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MuPConfig:
    """Configuration for Maximal Update Parametrization (muP).

    Attributes:
        d_model: Width of the current (possibly large) model.
        d_model_base: Width of the proxy/base model used for HP tuning.
        base_lr: Learning rate tuned on the proxy model.
    """

    d_model: int
    d_model_base: int
    base_lr: float = 1e-3

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model_base <= 0:
            raise ValueError(f"d_model_base must be positive, got {self.d_model_base}")
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {self.base_lr}")

    @property
    def width_multiplier(self) -> float:
        """Ratio of current width to base width (m_d / m_d_base in the paper)."""
        return self.d_model / self.d_model_base


# ---------------------------------------------------------------------------
# Layer classification helpers
# ---------------------------------------------------------------------------

# Name hints that identify embedding and output-projection layers.
_EMBED_HINTS = ("embed",)
_OUTPUT_HINTS = ("head", "lm_head", "output_proj")


def _is_embedding_layer(name: str, module: nn.Module) -> bool:
    """True for nn.Embedding modules or layers whose name contains an embed hint."""
    if isinstance(module, nn.Embedding):
        return True
    lower = name.lower()
    return any(hint in lower for hint in _EMBED_HINTS)


def _is_output_layer(name: str, module: nn.Module) -> bool:
    """True for nn.Linear layers that project to the vocabulary (logit head)."""
    if not isinstance(module, nn.Linear):
        return False
    lower = name.lower()
    return any(hint in lower for hint in _OUTPUT_HINTS)


def _is_hidden_layer(name: str, module: nn.Module) -> bool:
    """True for nn.Linear layers that are neither embeddings nor the logit head."""
    if not isinstance(module, nn.Linear):
        return False
    return not _is_output_layer(name, module)


# ---------------------------------------------------------------------------
# Core: initialization
# ---------------------------------------------------------------------------


def apply_mup_init(model: nn.Module, config: MuPConfig) -> None:
    """Re-initialise all weights in-place using muP rules.

    Initialisation summary
    ----------------------
    - Embedding weights  ->  Normal(0, 1/sqrt(fan_in))
    - Hidden weights     ->  Normal(0, 1/fan_in)
    - Output weights     ->  Normal(0, 1/fan_in)
    - Biases             ->  zeros
    - LayerNorm weights  ->  ones
    - LayerNorm biases   ->  zeros

    Args:
        model: The model to initialise in-place.
        config: muP configuration (only the width ratio matters here).
    """
    for _name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            # fan_in for an embedding is the number of embeddings (input side),
            # matching PyTorch's convention that fan_in = input features.
            # This keeps the embedding output variance O(1) regardless of d_model.
            fan_in = module.num_embeddings
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(module.weight, mean=0.0, std=std)

        elif isinstance(module, nn.Linear):
            # fan_in is the number of input features
            fan_in = module.weight.shape[1]
            std = 1.0 / fan_in  # muP hidden/output rule
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Core: param groups
# ---------------------------------------------------------------------------


def get_mup_param_groups(model: nn.Module, config: MuPConfig) -> list[dict]:
    """Return optimizer param groups with per-group learning rates already set.

    Groups
    ------
    1. Embedding parameters        ->  base_lr
    2. Hidden weight matrices      ->  base_lr / width_multiplier
    3. Output projection weights   ->  base_lr / width_multiplier
    4. Biases and norm parameters  ->  base_lr

    Parameters without gradients are skipped.

    Args:
        model: The model whose parameters will be trained.
        config: muP configuration.

    Returns:
        List of dicts, each with ``"params"`` and ``"lr"`` keys.
    """
    width_mult = config.width_multiplier
    hidden_lr = config.base_lr / width_mult

    embed_params: list[nn.Parameter] = []
    hidden_params: list[nn.Parameter] = []
    output_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []

    # Track already-assigned parameter ids to avoid double-counting.
    assigned: set = set()

    for name, module in model.named_modules():
        # Skip container modules -- we only care about leaf-like layers.
        if len(list(module.children())) > 0:
            continue

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in assigned:
                continue
            assigned.add(pid)

            if _is_embedding_layer(name, module):
                embed_params.append(param)
            elif _is_output_layer(name, module):
                if "weight" in param_name:
                    output_params.append(param)
                else:
                    other_params.append(param)
            elif _is_hidden_layer(name, module):
                if "weight" in param_name:
                    hidden_params.append(param)
                else:
                    other_params.append(param)
            else:
                # Norm layers, unknown layers, biases
                other_params.append(param)

    groups: list[dict] = []
    if embed_params:
        groups.append({"params": embed_params, "lr": config.base_lr, "group_name": "embed"})
    if hidden_params:
        groups.append({"params": hidden_params, "lr": hidden_lr, "group_name": "hidden"})
    if output_params:
        groups.append({"params": output_params, "lr": hidden_lr, "group_name": "output"})
    if other_params:
        groups.append({"params": other_params, "lr": config.base_lr, "group_name": "other"})

    return groups


# ---------------------------------------------------------------------------
# MuPAdamW optimizer wrapper
# ---------------------------------------------------------------------------


class MuPAdamW(torch.optim.Optimizer):
    """AdamW that applies muP learning-rate scaling per param group.

    Pass the output of :func:`get_mup_param_groups` as the first argument.
    The ``lr`` in each group overrides the global ``lr``; global ``lr`` acts
    as a fallback for groups that do not specify one.

    Args:
        param_groups: List of dicts from :func:`get_mup_param_groups`.
        lr: Global (fallback) learning rate.
        betas: Adam beta coefficients.
        eps: Adam epsilon for numerical stability.
        weight_decay: L2 regularisation coefficient.
        amsgrad: Whether to use the AMSGrad variant.
    """

    def __init__(
        self,
        param_groups: list[dict],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        # Ensure every group has an "lr" key; fall back to global lr.
        for group in param_groups:
            group.setdefault("lr", lr)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> torch.Tensor | None:
        """Perform a single optimisation step (AdamW with muP per-group LR).

        Args:
            closure: Optional closure that re-evaluates the model and returns
                the loss.

        Returns:
            Loss from the closure, or ``None``.
        """
        loss: torch.Tensor | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuPAdamW does not support sparse gradients.")

                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"].item()

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                # Bias correction
                bias_corr1 = 1.0 - beta1**step
                bias_corr2 = 1.0 - beta2**step

                # Weight decay (decoupled)
                p.mul_(1.0 - lr * wd)

                # Moment updates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    max_exp_avg_sq: torch.Tensor = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

                step_size = lr / bias_corr1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Coord check utility
# ---------------------------------------------------------------------------


def coord_check(
    model_fn: Any,
    widths: list[int],
    d_model_base: int,
    base_lr: float,
    input_ids: torch.Tensor,
    n_seeds: int = 3,
) -> dict:
    """Check that pre-activation std is approximately constant across widths.

    This is the core muP property: activations should remain O(1) as model
    width grows.  Runs a forward pass for each width and records the mean
    absolute value of the hidden states (a proxy for pre-activation std).

    Args:
        model_fn: Callable ``model_fn(d_model) -> nn.Module``.  The returned
            model must accept ``input_ids`` and return a tensor whose last
            dimension is the hidden size.
        widths: List of widths to sweep (e.g. ``[64, 128, 256]``).
        d_model_base: Proxy model width (denominator for width multiplier).
        base_lr: Base learning rate (not used in forward pass; kept for API
            symmetry).
        input_ids: Integer tensor of token ids, shape ``(batch, seq_len)``.
        n_seeds: Number of random seeds to average over.

    Returns:
        Dict mapping width -> mean |activation| averaged across seeds.
    """
    results: dict = {}

    for width in widths:
        config = MuPConfig(d_model=width, d_model_base=d_model_base, base_lr=base_lr)
        total: float = 0.0
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = model_fn(width)
            apply_mup_init(model, config)
            model.train(False)
            with torch.no_grad():
                out = model(input_ids)
            # out may be logits (vocab) or hidden states; measure mean abs
            total += out.float().abs().mean().item()

        results[width] = total / n_seeds

    return results

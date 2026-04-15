"""
LARS and LAMB optimizers for large-batch training.

LARS: Layer-wise Adaptive Rate Scaling (You et al. 2017)
LAMB: Layer-wise Adaptive Moments for Batch training (You et al. 2019)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


def compute_trust_ratio(
    param: Tensor,
    grad: Tensor,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute LARS trust ratio λ = ||param|| / (||grad|| + wd * ||param||).
    Returns scalar float.
    """
    param_norm = param.norm(2).item()
    grad_norm = grad.norm(2).item()

    # If both are zero, return 1.0 (no scaling)
    if param_norm == 0.0 and grad_norm == 0.0:
        return 1.0

    denominator = grad_norm + weight_decay * param_norm + eps
    return param_norm / denominator


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling optimizer.

    From: Large Batch Training of Convolutional Networks (You et al. 2017).

    Computes a per-layer trust ratio:
        λ = ||w|| / (||grad|| + β||w||)
    Effective lr per layer:
        lr_layer = lr * trust_coefficient * λ
    With momentum:
        v = momentum * v + lr_layer * (grad + weight_decay * w)
        w -= v
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
        exclude_bias_and_bn: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
            exclude_bias_and_bn=exclude_bias_and_bn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            trust_coefficient = group["trust_coefficient"]
            eps = group["eps"]
            exclude_bias_and_bn = group.get("exclude_bias_and_bn", True)
            # Per-group override: some groups may explicitly disable trust ratio
            apply_trust_ratio = group.get("apply_trust_ratio", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                param_state = self.state[p]

                # Determine whether to apply trust ratio to this parameter
                use_trust = apply_trust_ratio and exclude_bias_and_bn
                # If the group explicitly marks no trust ratio, skip it
                if not group.get("apply_trust_ratio", True):
                    use_trust = False

                # Compute effective learning rate
                if use_trust and p.ndim > 1:
                    # Only apply trust ratio to weight tensors (not bias/BN)
                    trust_ratio = compute_trust_ratio(
                        p, grad, weight_decay=weight_decay, eps=eps
                    )
                    effective_lr = lr * trust_coefficient * trust_ratio
                else:
                    effective_lr = lr

                # Compute update direction: grad + weight_decay * w
                update = grad.add(p, alpha=weight_decay) if weight_decay != 0.0 else grad.clone()
                # Scale by effective lr
                update = update.mul(effective_lr)

                # Apply momentum
                if momentum != 0.0:
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = update.clone()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(update)
                        update = buf

                p.add_(update, alpha=-1.0)

        return loss


class LAMB(Optimizer):
    """
    Layer-wise Adaptive Moments for Batch training.

    From: Large Batch Optimization for Deep Learning: Training BERT in 76 minutes
    (You et al. 2019).

    Combines LARS trust ratio with Adam moment estimates:
        m = β1*m + (1-β1)*g
        v = β2*v + (1-β2)*g²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        r_t = m_hat / (sqrt(v_hat) + eps)
        φ(w) = ||w|| / ||r_t + wd*w||     (trust ratio)
        w -= lr * φ(w) * (r_t + wd*w)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        trust_coefficient: float = 1.0,
        clamp_value: float = 10.0,
        adam_w_mode: bool = True,
        exclude_bias_and_bn: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if clamp_value < 0.0:
            raise ValueError(f"Invalid clamp_value: {clamp_value}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            clamp_value=clamp_value,
            adam_w_mode=adam_w_mode,
            exclude_bias_and_bn=exclude_bias_and_bn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            trust_coefficient = group["trust_coefficient"]
            clamp_value = group["clamp_value"]
            adam_w_mode = group["adam_w_mode"]
            exclude_bias_and_bn = group.get("exclude_bias_and_bn", True)
            apply_trust_ratio = group.get("apply_trust_ratio", True)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                param_state = self.state[p]

                # Initialize state
                if len(param_state) == 0:
                    param_state["step"] = 0
                    param_state["exp_avg"] = torch.zeros_like(p)
                    param_state["exp_avg_sq"] = torch.zeros_like(p)

                param_state["step"] += 1
                step = param_state["step"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                # Step 1: Update biased moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Step 2: Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Step 3: Adam update direction
                r_t = m_hat / (v_hat.sqrt().add_(eps))

                # Step 4: Add weight decay to update direction
                if adam_w_mode:
                    # Decoupled weight decay: add to update direction for trust ratio,
                    # but apply separately to param
                    update = r_t.clone()
                    if weight_decay != 0.0:
                        update.add_(p, alpha=weight_decay)
                else:
                    # L2 regularization: grad already includes weight decay implicitly
                    # (blend into r_t)
                    update = r_t.clone()
                    if weight_decay != 0.0:
                        update.add_(p, alpha=weight_decay)

                # Step 5: Compute LARS trust ratio
                use_trust = apply_trust_ratio
                if not group.get("apply_trust_ratio", True):
                    use_trust = False

                if use_trust and exclude_bias_and_bn and p.ndim > 1:
                    param_norm = p.norm(2).item()
                    update_norm = update.norm(2).item()

                    if param_norm == 0.0 or update_norm == 0.0:
                        trust_ratio = 1.0
                    else:
                        trust_ratio = param_norm / update_norm

                    # Clamp trust ratio to prevent explosion
                    trust_ratio = min(trust_ratio, clamp_value)
                    trust_ratio = trust_coefficient * trust_ratio
                else:
                    trust_ratio = trust_coefficient

                # Step 6: Apply update
                if adam_w_mode and weight_decay != 0.0:
                    # Decoupled weight decay applied directly
                    p.mul_(1.0 - lr * weight_decay)
                    p.add_(r_t, alpha=-lr * trust_ratio)
                else:
                    p.add_(update, alpha=-lr * trust_ratio)

        return loss


def get_param_groups_for_lars(
    model: nn.Module,
    weight_decay: float = 1e-4,
    skip_list: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Create param groups separating:
    - Regular params (apply weight_decay + trust_ratio)
    - Bias + LayerNorm params (no weight_decay, no trust_ratio)

    Returns list of param group dicts suitable for LARS/LAMB.
    """
    if skip_list is None:
        skip_list = []

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Conditions for excluding from weight decay and trust ratio:
        # 1. Explicitly in skip_list
        # 2. Bias parameters (param.ndim == 1 and name ends with 'bias')
        # 3. LayerNorm / BatchNorm weight and bias parameters
        is_bias = param.ndim == 1 and name.endswith("bias")
        is_norm_param = any(
            norm_name in name
            for norm_name in ["norm", "bn", "layer_norm", "layernorm", "batch_norm"]
        )
        in_skip_list = any(skip in name for skip in skip_list)

        if is_bias or is_norm_param or in_skip_list:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "apply_trust_ratio": True,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "apply_trust_ratio": False,
        },
    ]

    return param_groups

"""Gradient Routing: selective gradient masking for modular training.

Routes gradients based on data properties, allowing controlled learning:
- Prevent safety-bypassing behaviors from propagating to certain layers
- Localize new knowledge to specific model components
- Implement modular training with specialized sub-networks

Reference: Gradient Routing (MIT/Anthropic 2024)
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """Rule specifying which modules receive gradients for given data tags.

    Attributes:
        data_tags: Which data tags this rule applies to. Empty list means all tags.
        allowed_modules: Module name patterns (fnmatch) that CAN receive gradients.
            If non-empty, only these modules receive gradients; all others blocked.
        blocked_modules: Module name patterns (fnmatch) that CANNOT receive gradients.
            Applied after allowed_modules; these are always blocked regardless.
        gradient_scale: Scale factor for allowed gradients (0.0 = block, 1.0 = pass-through).
    """
    data_tags: List[str] = field(default_factory=list)
    allowed_modules: List[str] = field(default_factory=list)
    blocked_modules: List[str] = field(default_factory=list)
    gradient_scale: float = 1.0

    def matches_tag(self, data_tag: str) -> bool:
        """Return True if this rule applies to the given data tag."""
        if not self.data_tags:
            return True
        return data_tag in self.data_tags

    def scale_for_param(self, param_name: str) -> float:
        """Compute the gradient scale for a given parameter name.

        Logic:
        1. If blocked_modules patterns match -> scale = 0.0
        2. If allowed_modules is non-empty and no pattern matches -> scale = 0.0
        3. Otherwise -> self.gradient_scale
        """
        # Check blocked first (takes priority)
        for pattern in self.blocked_modules:
            if fnmatch.fnmatch(param_name, pattern):
                return 0.0

        # If allowed list specified, param must match at least one pattern
        if self.allowed_modules:
            for pattern in self.allowed_modules:
                if fnmatch.fnmatch(param_name, pattern):
                    return self.gradient_scale
            return 0.0  # not in allowed list

        return self.gradient_scale


@dataclass
class GradientRoutingConfig:
    """Top-level configuration for gradient routing.

    Attributes:
        rules: List of RoutingRule instances. None means no rules (passthrough).
        default_scale: Scale applied when no rule matches (default 1.0).
        log_gradient_norms: Whether to log gradient norms before/after masking.
    """
    rules: Optional[List[RoutingRule]] = None
    default_scale: float = 1.0
    log_gradient_norms: bool = False


# ---------------------------------------------------------------------------
# ModuleGradientMask
# ---------------------------------------------------------------------------

class ModuleGradientMask:
    """Apply per-parameter gradient scaling masks after a backward pass.

    Args:
        model: The nn.Module whose parameters will be masked.
        mask: Mapping from parameter name to scale factor.
            scale=0.0 zeros the gradient, scale=1.0 is identity,
            other values scale the gradient proportionally.
    """

    def __init__(self, model: nn.Module, mask: Dict[str, float]) -> None:
        self.model = model
        self.mask = mask  # {param_name: scale}

    def apply(self, loss: torch.Tensor) -> None:
        """Compute gradients via loss.backward() then apply the mask in-place.

        Args:
            loss: Scalar loss tensor to differentiate.
        """
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            scale = self.mask.get(name, 1.0)
            if scale == 1.0:
                continue
            elif scale == 0.0:
                param.grad.zero_()
            else:
                param.grad.mul_(scale)

    def get_masked_params(self) -> List[str]:
        """Return list of parameter names whose scale is less than 1.0."""
        return [name for name, scale in self.mask.items() if scale < 1.0]


# ---------------------------------------------------------------------------
# GradientRouter
# ---------------------------------------------------------------------------

class GradientRouter:
    """Route gradients to model components based on data tags and routing rules.

    Registers backward hooks on model parameters that multiply gradients by the
    appropriate scale factor for the active data tag.

    Args:
        model: The nn.Module to route gradients for.
        routing_rules: List of RoutingRule instances to apply.
    """

    def __init__(self, model: nn.Module, routing_rules: List[RoutingRule]) -> None:
        self.model = model
        self.routing_rules = routing_rules
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._active_tag: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_routing(self, loss: torch.Tensor, data_tag: str) -> torch.Tensor:
        """Compute gradients and zero out blocked parameters for data_tag.

        This method calls loss.backward() internally and then applies the
        routing mask by zeroing / scaling .grad attributes in-place.

        Args:
            loss: Scalar loss to differentiate.
            data_tag: Identifies the type of data in this batch.

        Returns:
            The loss tensor (for convenience; gradients already computed).
        """
        loss.backward()

        # Gather applicable rules for this tag
        active_rules = [r for r in self.routing_rules if r.matches_tag(data_tag)]
        if not active_rules:
            return loss

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # Compute combined scale: product of scales from all active rules
            combined_scale: float = 1.0
            for rule in active_rules:
                combined_scale *= rule.scale_for_param(name)

            if combined_scale == 1.0:
                continue
            elif combined_scale == 0.0:
                param.grad.zero_()
            else:
                param.grad.mul_(combined_scale)

        return loss

    def register_hooks(self) -> None:
        """Register backward hooks on all parameters that apply routing masks.

        The hooks are tag-aware: they apply the mask corresponding to
        self._active_tag at the time the backward pass fires. Call
        set_active_tag() before the forward pass to configure routing.
        """
        self.remove_hooks()  # clear any existing hooks

        for name, param in self.model.named_parameters():
            # Capture name in closure
            _name = name

            def make_hook(param_name: str):
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    tag = self._active_tag
                    if tag is None:
                        return grad
                    active_rules = [
                        r for r in self.routing_rules if r.matches_tag(tag)
                    ]
                    combined_scale: float = 1.0
                    for rule in active_rules:
                        combined_scale *= rule.scale_for_param(param_name)
                    if combined_scale == 1.0:
                        return grad
                    return grad * combined_scale
                return hook

            handle = param.register_hook(make_hook(_name))
            self._hooks.append(handle)

    def set_active_tag(self, data_tag: str) -> None:
        """Set the active data tag for hook-based routing."""
        self._active_tag = data_tag

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def get_routing_summary(self) -> dict:
        """Return a summary of the current routing configuration.

        Returns:
            Dict with keys:
                n_rules: number of routing rules
                blocked_params_per_tag: {tag: [param_names blocked]}
                active_tags: list of all unique tags across all rules
        """
        all_tags: List[str] = []
        for rule in self.routing_rules:
            all_tags.extend(rule.data_tags)
        active_tags = list(dict.fromkeys(all_tags))  # deduplicate preserving order

        blocked_params_per_tag: Dict[str, List[str]] = {}
        for tag in active_tags:
            blocked: List[str] = []
            active_rules = [r for r in self.routing_rules if r.matches_tag(tag)]
            for name, _param in self.model.named_parameters():
                combined_scale: float = 1.0
                for rule in active_rules:
                    combined_scale *= rule.scale_for_param(name)
                if combined_scale < 1.0:
                    blocked.append(name)
            blocked_params_per_tag[tag] = blocked

        return {
            "n_rules": len(self.routing_rules),
            "blocked_params_per_tag": blocked_params_per_tag,
            "active_tags": active_tags,
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_safety_routing(model: nn.Module) -> GradientRouter:
    """Create a router that prevents 'harmful' gradients from reaching early layers.

    For a model with n transformer layers, gradients from data tagged 'harmful'
    are blocked from flowing into layers 0 .. n//2 - 1 (the first half).

    Args:
        model: The nn.Module. Layer count is inferred from named_parameters.

    Returns:
        Configured GradientRouter with the safety rule applied.
    """
    # Infer number of layers by scanning parameter names for "layers.N." patterns
    import re
    layer_indices = set()
    for name, _ in model.named_parameters():
        m = re.search(r'layers\.(\d+)\.', name)
        if m:
            layer_indices.add(int(m.group(1)))

    n_layers = max(layer_indices) + 1 if layer_indices else 0
    cutoff = n_layers // 2  # block layers [0, cutoff)

    # Build blocked patterns for early layers
    blocked_patterns = [f"layers.{i}.*" for i in range(cutoff)]

    safety_rule = RoutingRule(
        data_tags=["harmful"],
        allowed_modules=[],       # no allow-list restriction
        blocked_modules=blocked_patterns,
        gradient_scale=1.0,
    )

    return GradientRouter(model, routing_rules=[safety_rule])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_gradient_conflict(
    grads1: List[torch.Tensor],
    grads2: List[torch.Tensor],
) -> float:
    """Measure cosine similarity between two gradient sets.

    A negative return value indicates gradient conflict (gradients point in
    opposite directions). A value of +1 means perfectly aligned.

    Args:
        grads1: List of gradient tensors from the first gradient set.
        grads2: List of gradient tensors from the second gradient set.

    Returns:
        Cosine similarity in [-1, 1]. Conflict corresponds to negative values.

    Raises:
        ValueError: If gradient lists are empty or have mismatched lengths.
    """
    if not grads1 or not grads2:
        raise ValueError("Gradient lists must be non-empty.")
    if len(grads1) != len(grads2):
        raise ValueError(
            f"Gradient lists must have the same length: {len(grads1)} vs {len(grads2)}"
        )

    # Flatten and concatenate all tensors into single vectors
    flat1 = torch.cat([g.reshape(-1).float() for g in grads1])
    flat2 = torch.cat([g.reshape(-1).float() for g in grads2])

    norm1 = flat1.norm()
    norm2 = flat2.norm()

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0  # degenerate: zero-norm gradient

    cosine_sim = (torch.dot(flat1, flat2) / (norm1 * norm2)).item()
    # Clamp for numerical safety
    return float(max(-1.0, min(1.0, cosine_sim)))

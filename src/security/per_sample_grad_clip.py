"""Per-sample gradient clipping for differential privacy training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class GradSampleHook:
    """Attaches forward and backward hooks to an nn.Linear layer to capture
    per-sample gradients without materializing the full Jacobian.
    """

    def __init__(self, module: nn.Linear) -> None:
        self._forward_handle = module.register_forward_hook(self._forward_hook)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        module: nn.Linear,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        module._saved_activations = input[0].detach()

    def _backward_hook(
        self,
        module: nn.Linear,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        saved = module._saved_activations
        go = grad_output[0]
        # go: (B, out_features), saved: (B, in_features)
        # per-sample weight grad: outer product for each sample → (B, out, in)
        module._per_sample_grads = go.unsqueeze(-1) * saved.unsqueeze(-2)

    def remove(self) -> None:
        """Remove both hooks from the module."""
        self._forward_handle.remove()
        self._backward_handle.remove()


class PerSampleClipper:
    """Clips per-sample gradients and accumulates them for all nn.Linear layers
    in a model, enabling differentially private gradient updates.
    """

    def __init__(self, model: nn.Module, max_grad_norm: float) -> None:
        self.model = model
        self.max_grad_norm = max_grad_norm
        self._hooks: List[GradSampleHook] = []
        self._linear_layers: List[Tuple[str, nn.Linear]] = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_layers.append((name, module))
                self._hooks.append(GradSampleHook(module))

    def _clip_and_reduce(self, per_sample_grads: torch.Tensor) -> torch.Tensor:
        """Clip each sample's gradient matrix by max_grad_norm and sum over batch.

        Args:
            per_sample_grads: Tensor of shape (B, out_features, in_features).

        Returns:
            Summed gradient tensor of shape (out_features, in_features).
        """
        # Per-sample L2 norm: flatten each (out, in) matrix, compute norm
        B = per_sample_grads.shape[0]
        flat = per_sample_grads.view(B, -1)  # (B, out*in)
        norms = flat.norm(dim=1)  # (B,)

        # Compute per-sample scale factors, clamped to [0, 1]
        scale = (self.max_grad_norm / norms.clamp(min=1e-12)).clamp(max=1.0)

        # Apply clipping and sum over batch
        clipped = per_sample_grads * scale.view(B, 1, 1)
        return clipped.sum(dim=0)

    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """Run backward pass and replace each linear layer's .grad with the
        clipped, summed per-sample gradient.

        Args:
            loss: Scalar loss tensor.

        Returns:
            Dict mapping layer name to mean pre-clip per-sample gradient norm.
        """
        loss.backward()

        stats: Dict[str, float] = {}
        for name, module in self._linear_layers:
            psg = module._per_sample_grads  # (B, out, in)

            # Compute pre-clip norms for reporting
            B = psg.shape[0]
            flat = psg.view(B, -1)
            norms = flat.norm(dim=1)
            stats[name] = norms.mean().item()

            # Clip, sum, and assign to .grad
            summed = self._clip_and_reduce(psg)
            if module.weight.grad is None:
                module.weight.grad = summed
            else:
                module.weight.grad.copy_(summed)

        return stats

    def remove_hooks(self) -> None:
        """Remove all GradSampleHook instances from their respective layers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

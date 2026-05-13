"""Safety geometry preservation during fine-tuning.

Monitors and preserves safety-relevant directions in the activation space
during RLHF and agentic fine-tuning. Prevents alignment training from
eroding safety features even with benign training data.

Risk: arXiv:2605.02914 (May 2025) — benign fine-tuning destroys safety geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SafetyGeometryMonitor:
    """Track safety-relevant activation directions across training steps."""

    def __init__(
        self,
        model: nn.Module,
        reference_unsafe_activations: Tensor,
        reference_safe_activations: Tensor,
        warning_threshold: float = 0.15,
    ):
        """
        Args:
            reference_unsafe_activations: (N, d_model) activations on unsafe prompts
                                           collected from the pre-fine-tuning model
            reference_safe_activations:   (N, d_model) activations on safe prompts
            warning_threshold:            cosine distance drift threshold for warning
        """
        self.model = model
        self.warning_threshold = warning_threshold
        # Compute the reference safety direction
        unsafe_mean = reference_unsafe_activations.mean(0)
        safe_mean = reference_safe_activations.mean(0)
        direction = unsafe_mean - safe_mean
        self.safety_direction = direction / (direction.norm() + 1e-8)  # unit vector
        self._hooks = []

    def compute_current_direction(
        self, unsafe_activations: Tensor, safe_activations: Tensor
    ) -> float:
        """Compute cosine similarity between current and reference safety direction."""
        current_dir = unsafe_activations.mean(0) - safe_activations.mean(0)
        current_dir = current_dir / (current_dir.norm() + 1e-8)
        cos_sim = torch.dot(self.safety_direction.to(current_dir.device), current_dir)
        return cos_sim.item()

    def geometry_preservation_loss(
        self, current_unsafe: Tensor, current_safe: Tensor, coeff: float = 0.01
    ) -> Tensor:
        """Return geometry preservation regularization loss.

        Penalizes drift in the safety direction during fine-tuning.
        Add to training loss: total_loss = task_loss + geo_loss
        """
        current_dir = current_unsafe.mean(0) - current_safe.mean(0)
        current_dir_norm = current_dir / (current_dir.norm() + 1e-8)
        ref = self.safety_direction.to(current_dir.device)
        # 1 - cosine_similarity → 0 when directions match, 2 when opposite
        cos_sim = torch.dot(ref, current_dir_norm)
        return coeff * (1 - cos_sim)

    def check_and_warn(self, current_unsafe: Tensor, current_safe: Tensor) -> bool:
        """Return True if safety geometry has drifted beyond threshold."""
        cos_sim = self.compute_current_direction(current_unsafe, current_safe)
        if cos_sim < (1 - self.warning_threshold):
            import logging

            logging.getLogger(__name__).warning(
                "SAFETY GEOMETRY DRIFT DETECTED: cosine similarity to reference "
                "safety direction = %.4f (threshold %.4f). Fine-tuning may be "
                "eroding safety features. Consider pausing and inspecting activations.",
                cos_sim,
                1 - self.warning_threshold,
            )
            return True
        return False


__all__ = ["SafetyGeometryMonitor"]

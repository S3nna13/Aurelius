"""Mixed precision training utilities: loss scaling, dtype casting, and AMP-style training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    enabled: bool = True
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
    initial_scale: float = 2**16  # starting loss scale
    growth_factor: float = 2.0  # multiply scale by this after growth_interval clean steps
    backoff_factor: float = 0.5  # multiply scale by this on overflow
    growth_interval: int = 2000  # clean steps required before scaling up
    min_scale: float = 1.0  # floor for the loss scale


class DynamicLossScaler:
    """Tracks and updates a dynamic loss scale to prevent underflow in fp16 training.

    The scale is increased every ``growth_interval`` consecutive overflow-free steps and
    decreased immediately when an overflow (inf/nan gradient) is detected.
    """

    def __init__(self, config: MixedPrecisionConfig) -> None:
        self._config = config
        self.scale: float = float(config.initial_scale)
        self._steps_since_overflow: int = 0
        self._growth_interval: int = config.growth_interval
        self.overflow_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Return loss multiplied by the current scale."""
        return loss * self.scale

    def unscale_gradients(self, optimizer) -> bool:
        """Divide all parameter gradients by the current scale.

        Returns:
            True if all gradients are finite (no overflow detected).
            False if any gradient contains inf or nan.
        """
        inv_scale = 1.0 / self.scale
        found_inf = False

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.grad.data.mul_(inv_scale)
                if not torch.isfinite(param.grad.data).all():
                    found_inf = True

        return not found_inf  # True == no overflow

    def update(self, overflow: bool) -> None:
        """Update the loss scale based on whether an overflow occurred.

        Args:
            overflow: True if inf/nan was detected in gradients.
        """
        if overflow:
            self.scale = max(self.scale * self._config.backoff_factor, self._config.min_scale)
            self._steps_since_overflow = 0
            self.overflow_count += 1
        else:
            self._steps_since_overflow += 1
            if self._steps_since_overflow >= self._growth_interval:
                self.scale = self.scale * self._config.growth_factor
                self._steps_since_overflow = 0

        # Always enforce minimum
        self.scale = max(self.scale, self._config.min_scale)


class MixedPrecisionTrainer:
    """High-level wrapper that combines loss scaling with dtype-cast forward/backward passes.

    Usage::

        cfg = MixedPrecisionConfig(dtype="bfloat16")
        trainer = MixedPrecisionTrainer(model, optimizer, cfg)

        result = trainer.forward_backward(input_ids, labels)
        trainer.step(result["overflow"])
    """

    def __init__(self, model: nn.Module, optimizer, config: MixedPrecisionConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scaler = DynamicLossScaler(config)

        _dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if config.dtype not in _dtype_map:
            raise ValueError(f"Unsupported dtype: {config.dtype!r}. Choose from {list(_dtype_map)}")
        self._dtype: torch.dtype = _dtype_map[config.dtype]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cast_inputs(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        """Cast floating-point tensors to the configured dtype (integer tensors are unchanged)."""
        if not self.config.enabled:
            return tensors
        result = []
        for t in tensors:
            if t.is_floating_point():
                result.append(t.to(self._dtype))
            else:
                result.append(t)
        return tuple(result)

    def forward_backward(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """Run a mixed-precision forward + backward pass.

        Steps:
          1. Cast model weights to half precision for the forward pass.
          2. Compute cross-entropy loss (ignoring label == -100).
          3. Scale loss and run backward.
          4. Unscale gradients and detect overflow.
          5. Update scaler.

        Args:
            input_ids: (B, T) integer token ids.
            labels:    (B, T) integer targets; positions with -100 are ignored.

        Returns:
            Dict with keys "loss" (float), "scale" (float), "overflow" (bool cast to float).
        """
        # Cast to half precision for forward pass
        if self.config.enabled and self._dtype != torch.float32:
            self.model.to(self._dtype)

        # Forward — model returns (loss, logits, past_key_values)
        model_out = self.model(input_ids)
        _, logits, _ = model_out

        # Restore model to float32 after forward
        if self.config.enabled and self._dtype != torch.float32:
            self.model.to(torch.float32)

        # Compute cross-entropy loss in float32 for numerical stability
        logits_f32 = logits.float()
        # Shift for next-token prediction
        shift_logits = logits_f32[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Scale and backward
        scaled_loss = self.scaler.scale_loss(loss)
        scaled_loss.backward()

        # Unscale and check for overflow
        finite = self.scaler.unscale_gradients(self.optimizer)
        overflow = not finite
        self.scaler.update(overflow)

        return {
            "loss": loss.item(),
            "scale": self.scaler.scale,
            "overflow": overflow,
        }

    def step(self, overflow: bool) -> None:
        """Apply optimizer step (only if no overflow) and zero gradients.

        Args:
            overflow: Whether the most recent backward produced inf/nan gradients.
        """
        if not overflow:
            self.optimizer.step()
        self.optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def autocast_forward(
    model: nn.Module, input_ids: Tensor, dtype: torch.dtype = torch.float16
) -> tuple:
    """Run a forward pass with model weights temporarily cast to *dtype*.

    Weights are restored to float32 after the forward pass completes.

    Args:
        model:     An ``nn.Module`` (e.g. ``AureliusTransformer``).
        input_ids: (B, T) integer token ids.
        dtype:     Target dtype for the forward pass (default: ``torch.float16``).

    Returns:
        The raw output of ``model(input_ids)`` — typically ``(loss, logits, past_kv)``.
    """
    model.to(dtype)
    try:
        output = model(input_ids)
    finally:
        model.to(torch.float32)
    return output


def count_overflow_params(model: nn.Module) -> int:
    """Count the number of parameters whose gradient contains inf or nan.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Integer count of parameters with non-finite gradients.
    """
    count = 0
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            count += 1
    return count

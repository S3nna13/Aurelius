"""Gradient compression: sparsification, quantization, and error feedback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GradCompressConfig:
    method: str = "topk"            # "topk" | "random" | "quantize"
    compression_ratio: float = 0.1  # fraction of gradients to keep/quantize
    bits: int = 8                   # for quantize method
    use_error_feedback: bool = True  # momentum error correction


def topk_compress(grad: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Keep top-k largest magnitude gradient values, zero out rest.

    Returns (compressed_grad, mask) where mask is a bool tensor with the same
    shape as grad — True where the gradient value was kept.
    """
    flat = grad.reshape(-1)
    n = flat.numel()
    k = max(1, min(k, n))

    _, indices = torch.topk(flat.abs(), k, largest=True, sorted=False)

    mask_flat = torch.zeros(n, dtype=torch.bool, device=grad.device)
    mask_flat[indices] = True

    compressed_flat = torch.zeros_like(flat)
    compressed_flat[mask_flat] = flat[mask_flat]

    return compressed_flat.reshape(grad.shape), mask_flat.reshape(grad.shape)


def random_compress(grad: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Randomly keep k gradient values, zero out rest.

    Returns (compressed_grad, mask) where mask is a bool tensor.
    """
    flat = grad.reshape(-1)
    n = flat.numel()
    k = max(1, min(k, n))

    perm = torch.randperm(n, device=grad.device)
    chosen = perm[:k]

    mask_flat = torch.zeros(n, dtype=torch.bool, device=grad.device)
    mask_flat[chosen] = True

    compressed_flat = torch.zeros_like(flat)
    compressed_flat[mask_flat] = flat[mask_flat]

    return compressed_flat.reshape(grad.shape), mask_flat.reshape(grad.shape)


def quantize_gradient(grad: Tensor, bits: int) -> Tensor:
    """Quantize gradient to `bits`-bit precision (simulated quantization).

    Scale to [-1, 1], round to 2^bits levels, scale back.
    Returns quantized gradient (same shape, float32).
    """
    g_min = grad.min()
    g_max = grad.max()
    g_range = g_max - g_min

    if g_range == 0:
        return grad.clone()

    # Normalize to [0, 1] then to [0, levels-1], round, invert
    levels = 2 ** bits
    normalized = (grad - g_min) / g_range          # [0, 1]
    quantized_int = torch.round(normalized * (levels - 1))   # [0, levels-1]
    quantized = quantized_int / (levels - 1) * g_range + g_min

    return quantized


def compress_gradient(
    grad: Tensor,
    config: GradCompressConfig,
) -> tuple[Tensor, Tensor | None]:
    """Apply configured compression to a gradient tensor.

    Returns (compressed_grad, mask_or_None).
    For "quantize" method, mask is None (no sparsity mask).
    """
    k = max(1, int(config.compression_ratio * grad.numel()))

    if config.method == "topk":
        compressed, mask = topk_compress(grad, k)
        return compressed, mask
    elif config.method == "random":
        compressed, mask = random_compress(grad, k)
        return compressed, mask
    elif config.method == "quantize":
        compressed = quantize_gradient(grad, config.bits)
        return compressed, None
    else:
        raise ValueError(f"Unknown compression method: {config.method!r}")


class ErrorFeedbackBuffer:
    """Accumulates compression errors for future correction (error feedback / momentum)."""

    def __init__(self) -> None:
        self._errors: dict[str, Tensor] = {}

    def update(self, name: str, grad: Tensor, compressed: Tensor) -> Tensor:
        """Compute error = grad - compressed, accumulate, add to next grad.

        The corrected gradient returned is:
            corrected = grad + accumulated_error_from_previous_steps

        Then we store the new residual:
            new_error = corrected - compressed_of_corrected
        But since callers compress *after* calling update (or we track the
        difference), we store  error += (grad - compressed)  each step so
        that the accumulated residual is carried forward.

        Specifically:
          - corrected_grad = grad + prev_error
          - new_error      = corrected_grad - compressed   (caller should
                             recompress corrected_grad and pass that in)

        For simplicity (matching the intended API), we:
          1. Return corrected_grad = grad + prev_error
          2. Store new residual = grad - compressed  (added to prev_error)
        """
        prev_error = self._errors.get(name, torch.zeros_like(grad))
        corrected = grad + prev_error
        # accumulate residual: previous error + current grad - current compressed
        self._errors[name] = prev_error + (grad - compressed)
        return corrected

    def clear(self) -> None:
        self._errors.clear()

    def __len__(self) -> int:
        return len(self._errors)


class CompressedGradOptimizer:
    """Wraps an optimizer to apply gradient compression before each step."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: GradCompressConfig,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.named_params = named_params
        self._error_buf = ErrorFeedbackBuffer() if config.use_error_feedback else None

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> dict:
        """Compress gradients, optionally apply error feedback, then optimizer.step().

        Returns dict with keys:
            'n_params_compressed'      — number of parameter tensors compressed
            'mean_compression_ratio'   — average fraction of nonzeros kept
            'n_error_feedback_applied' — count of params where error feedback ran
        """
        n_compressed = 0
        n_ef_applied = 0
        compression_ratios: list[float] = []

        for name, param in self.named_params:
            if param.grad is None:
                continue

            grad = param.grad.detach()

            # Error feedback: add accumulated residual before compressing
            if self._error_buf is not None and self.config.use_error_feedback:
                # For the first step the buffer returns grad + 0 = grad
                corrected = self._error_buf.update(name, grad, torch.zeros_like(grad))
                # We'll compress the corrected grad and then update the buffer residual
                compressed, mask = compress_gradient(corrected, self.config)
                # Overwrite stored residual to be corrected - compressed
                self._error_buf._errors[name] = corrected - compressed
                n_ef_applied += 1
            else:
                compressed, mask = compress_gradient(grad, self.config)

            # Compute actual compression ratio (fraction of nonzero entries)
            nonzero_frac = (compressed != 0).float().mean().item()
            compression_ratios.append(nonzero_frac)

            # Write compressed gradient back
            param.grad = compressed
            n_compressed += 1

        self.optimizer.step()

        mean_ratio = float(sum(compression_ratios) / len(compression_ratios)) if compression_ratios else 0.0

        return {
            "n_params_compressed": n_compressed,
            "mean_compression_ratio": mean_ratio,
            "n_error_feedback_applied": n_ef_applied,
        }


class GradCompressTrainer:
    """Trains model with gradient compression."""

    def __init__(
        self,
        model: nn.Module,
        config: GradCompressConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.optimizer = CompressedGradOptimizer(optimizer, config, named_params)

    def train_step(self, input_ids: Tensor) -> dict:
        """Forward + backward + compressed optimizer step.

        Returns dict with keys: 'loss', 'n_params_compressed', 'mean_compression_ratio'
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Use input_ids as labels (next-token prediction on the same sequence)
        loss, logits, past_key_values = self.model(input_ids, labels=input_ids)

        if loss is None:
            raise RuntimeError("Model did not return a loss. Ensure labels are passed.")

        loss.backward()

        step_info = self.optimizer.step()

        return {
            "loss": loss.item(),
            "n_params_compressed": step_info["n_params_compressed"],
            "mean_compression_ratio": step_info["mean_compression_ratio"],
        }

"""Selective activation offloading to CPU RAM during the forward pass.

Unlike gradient checkpointing (which recomputes activations), offloading
*saves* activations to CPU memory. This avoids recomputation overhead but
still reduces GPU memory usage. Best for models too large for GPU VRAM but
where fast CPU-GPU transfers are available (e.g. PCIe 4.0+).

Usage::

    from src.training.activation_offload import wrap_model_with_offloading

    model = AureliusTransformer(config)
    # Offload all transformer layers to CPU during training
    wrap_model_with_offloading(model)

    # Or selectively offload only specific layers
    wrap_model_with_offloading(model, layer_indices=[0, 4, 8, 12])
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level storage
# ---------------------------------------------------------------------------

class OffloadedActivation:
    """Stores a tensor on CPU, moves it back to the original device on demand."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.cpu_data = tensor.detach().cpu()
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.shape = tensor.shape

    def restore(self) -> torch.Tensor:
        """Move data back to the original device with the original dtype."""
        return self.cpu_data.to(device=self.device, dtype=self.dtype)


# ---------------------------------------------------------------------------
# Custom autograd Function
# ---------------------------------------------------------------------------

class OffloadFunction(torch.autograd.Function):
    """Custom autograd function that offloads a tensor to CPU in the forward
    pass and moves the gradient back to the original device in the backward
    pass.

    The function inserts itself into the autograd graph so that:
    - During the forward pass the activation is moved to CPU RAM.
    - During the backward pass the gradient is moved back to the original
      device before being passed to the upstream operation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x.detach().cpu())
        ctx.device = x.device
        ctx.dtype = x.dtype
        cpu_tensor = x.detach().cpu()
        if x.requires_grad:
            cpu_tensor = cpu_tensor.requires_grad_(True)
        return cpu_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (saved,) = ctx.saved_tensors  # noqa: F841 — kept for completeness
        return grad_output.to(device=ctx.device, dtype=ctx.dtype)


def offload_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply :class:`OffloadFunction` to *x*.

    Moves the tensor to CPU for storage and restores the gradient to the
    original device during backprop.  Gradients flow through transparently.

    Args:
        x: Input tensor (any device).

    Returns:
        Equivalent tensor on CPU with the autograd graph intact.
    """
    return OffloadFunction.apply(x)


# ---------------------------------------------------------------------------
# Hook helper (kept for API completeness / future extension)
# ---------------------------------------------------------------------------

class ActivationOffloadHook:
    """Hook that offloads activations to CPU after each forward pass and
    reloads them for the backward pass.

    Works by replacing tensors in the autograd graph using
    :class:`OffloadFunction`.
    """

    def __init__(self) -> None:
        self._offloaded: dict[int, OffloadedActivation] = {}

    @staticmethod
    def offload_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Offload *tensor* to CPU while keeping it in the computation graph.

        Returns a CPU tensor that participates in the autograd graph.
        During backward, gradients are automatically moved back to the
        original device.
        """
        return offload_activation(tensor)


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------

class SelectiveOffloadWrapper(nn.Module):
    """Wraps a module and applies activation offloading to its output.

    Used to selectively offload only large intermediate tensors (e.g. the
    hidden states produced by a transformer block) rather than every tensor
    in the model.

    Args:
        module: The module whose output should be offloaded.
        offload: If *False*, the wrapper is a no-op (useful for toggling
            offloading without rewrapping).
    """

    def __init__(self, module: nn.Module, offload: bool = True) -> None:
        super().__init__()
        self.module = module
        self.offload = offload

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if self.offload and self.training:
            if isinstance(out, torch.Tensor):
                return offload_activation(out)
            elif isinstance(out, tuple):
                # Offload the first element (typically hidden states); pass
                # the remaining elements (e.g. KV cache) through unchanged.
                return (offload_activation(out[0]),) + out[1:]
        return out


# ---------------------------------------------------------------------------
# Model-level helper
# ---------------------------------------------------------------------------

def wrap_model_with_offloading(
    model: nn.Module,
    layer_indices: list[int] | None = None,
) -> nn.Module:
    """Wrap transformer layers with :class:`SelectiveOffloadWrapper`.

    Modifies *model* **in place** by replacing selected entries in
    ``model.layers`` with wrapped versions.

    Args:
        model: An :class:`~src.model.transformer.AureliusTransformer` (or any
            ``nn.Module`` that exposes a ``layers`` attribute which is an
            ``nn.ModuleList``).
        layer_indices: Indices of layers to offload.  ``None`` (default) means
            *all* layers.

    Returns:
        The same *model* object with offloading wrappers applied.

    Raises:
        AttributeError: If *model* does not have a ``layers`` attribute.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"{type(model).__name__} does not have a 'layers' attribute. "
            "wrap_model_with_offloading expects a model with model.layers "
            "(nn.ModuleList of transformer blocks)."
        )

    layers: nn.ModuleList = model.layers  # type: ignore[assignment]
    indices_to_wrap = (
        set(layer_indices) if layer_indices is not None else set(range(len(layers)))
    )

    for i in indices_to_wrap:
        layers[i] = SelectiveOffloadWrapper(layers[i], offload=True)

    return model


# ---------------------------------------------------------------------------
# Statistics tracker
# ---------------------------------------------------------------------------

class OffloadingStats:
    """Track memory savings from activation offloading.

    Attributes:
        offloaded_bytes: Running total of bytes moved to CPU RAM.
        restore_count: Number of times a tensor has been restored to GPU.
    """

    def __init__(self) -> None:
        self.offloaded_bytes: int = 0
        self.restore_count: int = 0

    def record_offload(self, tensor: torch.Tensor) -> None:
        """Record that *tensor* was offloaded to CPU.

        Updates :attr:`offloaded_bytes` by the byte size of *tensor*.
        """
        self.offloaded_bytes += tensor.numel() * tensor.element_size()

    def offloaded_mb(self) -> float:
        """Return the total offloaded memory in megabytes."""
        return self.offloaded_bytes / (1024 ** 2)

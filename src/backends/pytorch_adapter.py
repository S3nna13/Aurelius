"""Reference PyTorch backend adapter.

This is the one and only canonical home for a top-level ``import torch``
inside the Aurelius package (alongside :mod:`src.backends.__init__` which
invokes :func:`register` at import time). All other modules must keep the
foreign-import boundary clean.

The adapter exposes a minimal tensor namespace and the introspection
helpers required by :class:`~src.backends.base.BackendAdapter`.
"""

from __future__ import annotations

from typing import Any

import torch

from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    BackendContract,
)

__all__ = [
    "PyTorchTensorNS",
    "PyTorchAdapter",
    "register",
]


class PyTorchTensorNS:
    """Minimal PyTorch-backed tensor namespace.

    Exposes the three primitive constructors that the backend surface
    contract requires. Accepts PyTorch dtype objects or strings like
    ``"float32"``; other dtype inputs are normalized via ``getattr``.
    """

    def zeros(self, shape: Any, dtype: Any = None) -> "torch.Tensor":
        resolved = _resolve_dtype(dtype)
        return torch.zeros(shape, dtype=resolved)

    def ones(self, shape: Any, dtype: Any = None) -> "torch.Tensor":
        resolved = _resolve_dtype(dtype)
        return torch.ones(shape, dtype=resolved)

    def as_array(self, obj: Any) -> "torch.Tensor":
        if isinstance(obj, torch.Tensor):
            return obj
        return torch.as_tensor(obj)


def _resolve_dtype(dtype: Any) -> "torch.dtype | None":
    """Coerce a user-supplied dtype hint to a ``torch.dtype`` or None."""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        candidate = getattr(torch, dtype, None)
        if isinstance(candidate, torch.dtype):
            return candidate
        raise BackendAdapterError(
            f"unknown torch dtype string: {dtype!r}"
        )
    raise BackendAdapterError(
        f"dtype must be a torch.dtype or string, got {type(dtype).__name__}"
    )


class PyTorchAdapter(BackendAdapter):
    """Reference :class:`BackendAdapter` implementation backed by PyTorch."""

    _CONTRACT = BackendContract(
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=True,
        supports_inference=True,
        capability_tags=("autograd", "cuda", "mps", "cpu"),
    )

    @property
    def contract(self) -> BackendContract:
        return self._CONTRACT

    def tensor_ns(self) -> PyTorchTensorNS:
        return PyTorchTensorNS()

    def dtype_of(self, obj: Any) -> str:
        if not isinstance(obj, torch.Tensor):
            raise BackendAdapterError(
                f"dtype_of expected a torch.Tensor, got {type(obj).__name__}"
            )
        return str(obj.dtype)

    def device_of(self, obj: Any) -> str:
        if not isinstance(obj, torch.Tensor):
            raise BackendAdapterError(
                f"device_of expected a torch.Tensor, got {type(obj).__name__}"
            )
        return str(obj.device)

    def is_available(self) -> bool:
        # The presence of a successful top-level ``import torch`` in this
        # module is the single source of truth for availability -- we
        # never silently mask import failure.
        return True

    def runtime_info(self) -> dict:
        info = super().runtime_info()
        info["torch_version"] = torch.__version__
        return info


def register() -> None:
    """Register a :class:`PyTorchAdapter` instance as ``"pytorch"``.

    Idempotent: calling this twice is a no-op once the registry already
    contains an entry under ``"pytorch"``.
    """
    # Local import to keep the cycle clean with src.backends.__init__.
    from src.backends.registry import BACKEND_REGISTRY, register_backend

    if "pytorch" in BACKEND_REGISTRY:
        return
    register_backend(PyTorchAdapter())

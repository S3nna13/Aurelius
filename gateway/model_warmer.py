"""Thread-safe model weight warmer with format auto-detection."""

from __future__ import annotations

import os
import threading

import torch

try:
    from safetensors.torch import load_file as _load_safetensors
except Exception:  # pragma: no cover
    _load_safetensors = None

WARMER_REGISTRY: dict[str, ModelWarmer] = {}


class ModelWarmer:
    """Loads and caches a model ``state_dict``, supporting ``.pt``, ``.bin``,
    and ``.safetensors`` files.

    All public methods are thread-safe.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if not isinstance(model_path, str):
            raise TypeError("model_path must be a str")
        if not isinstance(device, str):
            raise TypeError("device must be a str")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        self._model_path = model_path
        self._device = device
        self._state_dict: dict[str, torch.Tensor] | None = None
        self._lock = threading.Lock()

    def warm(self) -> dict[str, torch.Tensor]:
        """Load weights into memory and return the state dict.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If the file is ``.safetensors`` but the library is
                unavailable.
            TypeError: If the loaded object is not a ``dict``.
        """
        with self._lock:
            if self._state_dict is not None:
                return self._state_dict

            path = self._model_path
            if path.endswith(".safetensors"):
                if _load_safetensors is None:
                    raise RuntimeError("safetensors is required to load .safetensors files")
                state_dict = _load_safetensors(path, device=self._device)
            else:
                state_dict = torch.load(path, map_location=self._device, weights_only=True)
                if not isinstance(state_dict, dict):
                    raise TypeError(
                        f"Expected state dict from {path}, got {type(state_dict).__name__}"
                    )

            self._state_dict = state_dict
            return self._state_dict

    def is_warmed(self) -> bool:
        """Return ``True`` if weights are currently cached."""
        with self._lock:
            return self._state_dict is not None

    def unload(self) -> None:
        """Clear cached weights."""
        with self._lock:
            self._state_dict = None

    def memory_footprint_mb(self) -> float:
        """Return the memory footprint of cached weights in megabytes."""
        with self._lock:
            if self._state_dict is None:
                return 0.0
            total_bytes = sum(t.numel() * t.element_size() for t in self._state_dict.values())
            return total_bytes / (1024 * 1024)

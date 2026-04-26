"""Delta checkpoint compression: store weight deltas between checkpoints to save space."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class DeltaConfig:
    """Configuration for delta checkpoint compression."""

    compression: str = "none"  # "none" | "top_k" | "threshold"
    top_k_ratio: float = 0.1  # keep top-10% of changed weights by magnitude
    threshold: float = 1e-4  # min delta magnitude to store
    dtype: str = "float32"  # storage dtype


def compute_delta(state_a: dict, state_b: dict) -> dict[str, Tensor]:
    """Compute weight deltas between two state dicts.

    Returns {key: state_b[key] - state_a[key]} for keys present in both.
    Keys only in state_b are stored as-is (new parameters).

    Args:
        state_a: Base/reference state dict.
        state_b: New state dict.

    Returns:
        Dict of delta tensors.
    """
    delta: dict[str, Tensor] = {}
    for key, val_b in state_b.items():
        if key in state_a:
            delta[key] = val_b.float() - state_a[key].float()
        else:
            # New parameter — store as-is
            delta[key] = val_b.clone()
    return delta


def apply_delta(base_state: dict, delta: dict) -> dict:
    """Reconstruct state_b from base_state + delta.

    Args:
        base_state: The reference (base) state dict.
        delta: Delta tensors produced by compute_delta.

    Returns:
        Reconstructed state dict equivalent to the original state_b.
    """
    result = {}
    for key, d in delta.items():
        if key in base_state:
            result[key] = (base_state[key].float() + d.float()).to(base_state[key].dtype)
        else:
            # New parameter
            result[key] = d.clone()
    return result


def compress_delta(delta: dict, config: DeltaConfig) -> dict:
    """Compress delta tensors according to the given strategy.

    Strategies:
        "none"      — passthrough, no compression.
        "top_k"     — keep only top-k% values by magnitude, zero the rest.
        "threshold" — zero out values with |delta| < config.threshold.

    Args:
        delta: Dict of delta tensors.
        config: DeltaConfig specifying compression strategy.

    Returns:
        Compressed delta dict (same structure, some values zeroed).
    """
    if config.compression == "none":
        return {k: v.clone() for k, v in delta.items()}

    compressed: dict[str, Tensor] = {}

    if config.compression == "top_k":
        for key, tensor in delta.items():
            flat = tensor.reshape(-1)
            n_keep = max(1, int(len(flat) * config.top_k_ratio))
            # Get indices of top-k elements by magnitude
            magnitudes = flat.abs()
            _, top_indices = torch.topk(magnitudes, n_keep)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[top_indices] = True
            out = flat.clone()
            out[~mask] = 0.0
            compressed[key] = out.reshape(tensor.shape)

    elif config.compression == "threshold":
        for key, tensor in delta.items():
            out = tensor.clone()
            out[tensor.abs() < config.threshold] = 0.0
            compressed[key] = out

    else:
        raise ValueError(f"Unknown compression strategy: {config.compression!r}")

    return compressed


def delta_size_bytes(delta: dict) -> int:
    """Count total bytes across all delta tensors.

    Args:
        delta: Dict of delta tensors.

    Returns:
        Total byte count as int.
    """
    total = 0
    for tensor in delta.values():
        total += tensor.numel() * tensor.element_size()
    return total


class DeltaCheckpointManager:
    """Manages delta-compressed checkpoints relative to a base state dict."""

    def __init__(self, base_dir: str, config: DeltaConfig) -> None:
        """
        Args:
            base_dir: Directory for checkpoint files.
            config: DeltaConfig controlling compression.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._base_state: dict | None = None
        self._checkpoints: list[str] = []  # list of (name, path) pairs stored as "name"

    def save_base(self, state_dict: dict, name: str = "base") -> None:
        """Save full state dict as the base checkpoint.

        Args:
            state_dict: Model state dict to use as the reference.
            name: Checkpoint name (default "base").
        """
        path = self.base_dir / f"{name}.pt"
        torch.save(state_dict, path)
        self._base_state = {k: v.clone() for k, v in state_dict.items()}
        # Ensure base is always first in the list
        if not self._checkpoints or self._checkpoints[0] != name:
            self._checkpoints.insert(0, name)

    def save_delta(self, new_state: dict, name: str) -> int:
        """Compute, compress, and save a delta checkpoint.

        Args:
            new_state: The new model state dict to checkpoint.
            name: Checkpoint name.

        Returns:
            Compressed size in bytes.
        """
        if self._base_state is None:
            raise RuntimeError("Call save_base() before save_delta().")

        delta = compute_delta(self._base_state, new_state)
        compressed = compress_delta(delta, self.config)

        path = self.base_dir / f"{name}.pt"
        torch.save(compressed, path)

        if name not in self._checkpoints:
            self._checkpoints.append(name)

        return delta_size_bytes(compressed)

    def load(self, name: str) -> dict:
        """Load a checkpoint by name.

        For "base" (or the first checkpoint), returns the full state dict.
        For delta checkpoints, applies the delta to the base and returns
        the reconstructed state dict.

        Args:
            name: Checkpoint name as returned by list_checkpoints().

        Returns:
            Reconstructed state dict.
        """
        path = self.base_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Determine if this is the base checkpoint
        is_base = len(self._checkpoints) > 0 and name == self._checkpoints[0]

        if is_base:
            return torch.load(path, map_location="cpu", weights_only=True)

        # Delta checkpoint
        if self._base_state is None:
            # Attempt to load base from disk
            base_name = self._checkpoints[0] if self._checkpoints else "base"
            base_path = self.base_dir / f"{base_name}.pt"
            self._base_state = torch.load(base_path, map_location="cpu", weights_only=True)

        delta = torch.load(path, map_location="cpu", weights_only=True)
        return apply_delta(self._base_state, delta)

    def list_checkpoints(self) -> list[str]:
        """Return all checkpoint names in save order.

        Returns:
            List of checkpoint name strings.
        """
        return list(self._checkpoints)

    def compression_ratio(self, name: str) -> float:
        """Compute size ratio of a delta checkpoint vs. the base checkpoint.

        A ratio < 1.0 means the delta is smaller than a full checkpoint.

        Args:
            name: Delta checkpoint name.

        Returns:
            float ratio = delta_file_size / base_file_size.
        """
        if not self._checkpoints:
            raise RuntimeError("No checkpoints saved yet.")

        base_name = self._checkpoints[0]
        base_path = self.base_dir / f"{base_name}.pt"
        delta_path = self.base_dir / f"{name}.pt"

        base_size = os.path.getsize(base_path)
        delta_size = os.path.getsize(delta_path)

        if base_size == 0:
            return float("inf")

        return delta_size / base_size


def quantize_delta(delta: dict, bits: int = 8) -> dict:
    """Quantize delta tensors to int8 for additional compression.

    Per-tensor quantization:
        scale = max(|delta|) / 127
        quantized = clamp(round(delta / scale), -127, 127).to(int8)

    Args:
        delta: Dict of delta tensors.
        bits: Quantization bit-width (currently only 8 is supported).

    Returns:
        Dict mapping each key to {"quantized": int8_tensor, "scale": float}.
    """
    if bits != 8:
        raise ValueError(f"Only 8-bit quantization is currently supported (got {bits}).")

    result: dict[str, dict] = {}
    for key, tensor in delta.items():
        max_val = tensor.abs().max().item()
        if max_val == 0.0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
        result[key] = {"quantized": quantized, "scale": float(scale)}
    return result

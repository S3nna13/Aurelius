"""Offload optimizer states to SSD when training large models on limited RAM.

Saves AdamW/Muon momentum buffers to disk during forward+backward pass,
loads them back for the optimizer step. Trades compute time for memory.
Useful for 3B+ training on 32GB M1 Pro where optimizer state alone can
exceed available memory during the backward pass.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


class OptimizerOffloader:
    """Swap optimizer states between RAM and SSD.

    Usage:
        offloader = OptimizerOffloader(optimizer)
        # After forward+backward, before optimizer.step():
        offloader.load_from_disk()
        optimizer.step()
        # After optimizer.step(), before next forward:
        offloader.offload_to_disk()

    Memory saved: 2x model size in GB (AdamW states: 2 copies per param)
    For 3B model: ~24GB saved (2 * 3B * 4 bytes = 24GB of fp32 states)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        offload_dir: str | None = None,
    ):
        self.optimizer = optimizer
        self.offload_dir = Path(offload_dir or tempfile.mkdtemp(prefix="aurelius_opt_"))
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self._offloaded: list[tuple[int, str]] = []

    def offload_to_disk(self) -> None:
        """Move all optimizer state tensors to SSD, freeing RAM.

        After this call, the optimizer state dict is on disk.
        Call ``load_from_disk()`` before ``optimizer.step()``.
        """
        for param_id, param_state in self.optimizer.state.items():
            for key, tensor in list(param_state.items()):
                if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                    path = self.offload_dir / f"param_{param_id}_{key}.pt"
                    torch.save(tensor.detach().cpu(), path)
                    param_state[key] = tensor.new_zeros(1)  # Tiny placeholder
                    self._offloaded.append((param_id, key))

    def load_from_disk(self) -> None:
        """Load optimizer states from SSD back to RAM for the optimizer step."""
        for param_id, key in self._offloaded:
            if param_id not in self.optimizer.state:
                continue
            path = self.offload_dir / f"param_{param_id}_{key}.pt"
            if path.exists():
                device = next(
                    (p.device for p in self.optimizer.param_groups[0]["params"]
                     if hash(p) == param_id),
                    torch.device("cpu"),
                )
                state_tensor = torch.load(path, map_location=device, weights_only=True)
                self.optimizer.state[param_id][key] = state_tensor
        self._offloaded.clear()

    def cleanup(self) -> None:
        """Remove offloaded files from disk."""
        if self.offload_dir.exists():
            shutil.rmtree(self.offload_dir)

    def bytes_on_disk(self) -> int:
        """Return total bytes currently offloaded to disk."""
        total = 0
        for path in self.offload_dir.glob("*.pt"):
            total += path.stat().st_size
        return total

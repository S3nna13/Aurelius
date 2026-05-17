"""SSD optimizer state offloading for tight-memory runs on Apple Silicon.

Optimizer state is written to disk after backward(), the in-memory tensors
are replaced with tiny placeholders, and the original state can be restored
just before optimizer.step(). The helper keeps the legacy API used by the
tests and trainer while also exposing a ``clear`` alias for newer callers.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch


class OptimizerOffloader:
    """Swap optimizer states between RAM and SSD."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        offload_dir: str | Path | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.offload_dir = (
            Path(offload_dir)
            if offload_dir is not None
            else Path(tempfile.mkdtemp(prefix="aurelius_opt_offload_"))
        )
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self._offloaded: list[tuple[int, str]] = []

    def offload_to_disk(self) -> None:
        """Write optimizer state to disk and leave tiny placeholders in RAM."""
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self._offloaded.clear()

        for idx, (param, state) in enumerate(self.optimizer.state.items()):
            if not state:
                continue

            path = self.offload_dir / f"param_{idx}.pt"
            torch.save(state, path)

            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.new_zeros(1)
                    self._offloaded.append((id(param), key))

    def load_from_disk(self) -> None:
        """Restore optimizer state from disk."""
        for idx, (param, state) in enumerate(self.optimizer.state.items()):
            path = self.offload_dir / f"param_{idx}.pt"
            if not path.exists():
                continue

            loaded = torch.load(path, weights_only=True)
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"Offloaded optimizer state must be a dict, got {type(loaded).__name__}"
                )
            self.optimizer.state[param].clear()
            self.optimizer.state[param].update(loaded)
            path.unlink(missing_ok=True)

        self._offloaded.clear()

    def cleanup(self) -> None:
        """Remove all offloaded files from disk."""
        shutil.rmtree(self.offload_dir, ignore_errors=True)
        self._offloaded.clear()

    def clear(self) -> None:
        """Compatibility alias for cleanup()."""
        self.cleanup()

    def bytes_on_disk(self) -> int:
        """Return total bytes occupied by offloaded optimizer states."""
        if not self.offload_dir.exists():
            return 0
        return sum(path.stat().st_size for path in self.offload_dir.glob("*.pt"))

    @property
    def disk_usage_bytes(self) -> int:
        return self.bytes_on_disk()

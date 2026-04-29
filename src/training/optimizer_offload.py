from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


class OptimizerOffloader:
    def __init__(self, optimizer: torch.optim.Optimizer, offload_dir: str | Path | None = None):
        self.optimizer = optimizer
        self.offload_dir = Path(offload_dir) if offload_dir else Path(tempfile.mkdtemp(prefix="aurelius_opt_offload_"))
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        self._param_ids: list[int] = []
        self._offloaded: list[tuple[int, str]] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                self._param_ids.append(id(p))

    def offload_to_disk(self) -> None:
        self._offloaded.clear()
        for idx, (param_id, state) in enumerate(self.optimizer.state.items()):
            if not state:
                continue
            path = self.offload_dir / f"param_{idx}.pt"
            torch.save(state, path)
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.new_zeros(1)
                    self._offloaded.append((id(param_id), key))

    def load_from_disk(self) -> None:
        for idx, (param_id, state) in enumerate(self.optimizer.state.items()):
            path = self.offload_dir / f"param_{idx}.pt"
            if path.exists():
                loaded = torch.load(path, map_location="cpu", weights_only=True)
                self.optimizer.state[param_id].update(loaded)
        self._offloaded.clear()

    def cleanup(self) -> None:
        if self.offload_dir.exists():
            shutil.rmtree(self.offload_dir)
        self._offloaded.clear()

    def bytes_on_disk(self) -> int:
        if not self.offload_dir.exists():
            return 0
        total = 0
        for path in self.offload_dir.glob("*.pt"):
            total += path.stat().st_size
        return total

    @property
    def disk_usage_bytes(self) -> int:
        return self.bytes_on_disk()

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CompileConfig:
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = True
    backend: str = "inductor"
    warmup_iters: int = 3


class CompileManager:
    def __init__(self, config: CompileConfig | None = None):
        self.config = config or CompileConfig()
        self._compiled: nn.Module | None = None

    def compile(self, model: nn.Module) -> nn.Module:
        if hasattr(torch, "compile"):
            compiled = torch.compile(
                model,
                mode=self.config.mode,
                fullgraph=self.config.fullgraph,
                dynamic=self.config.dynamic,
                backend=self.config.backend,
            )
        else:
            compiled = model
        self._compiled = compiled
        return compiled

    def warmup(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        elapsed = 0.0
        for _ in range(self.config.warmup_iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(sample_input)
            elapsed = time.perf_counter() - t0
        return elapsed

    def reset(self) -> None:
        self._compiled = None


COMPILE_REGISTRY: dict[str, type[CompileManager]] = {"default": CompileManager}

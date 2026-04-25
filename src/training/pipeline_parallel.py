from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PipelineConfig:
    n_stages: int = 2
    n_microbatches: int = 4
    schedule: str = "gpipe"


class PipelineStage(nn.Module):
    def __init__(self, module: nn.Module, stage_idx: int) -> None:
        super().__init__()
        self.module = module
        self.stage_idx = stage_idx

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class PipelineParallel(nn.Module):
    def __init__(self, stages: list[nn.Module], config: PipelineConfig | None = None) -> None:
        super().__init__()
        self.config = config or PipelineConfig()
        self.stages = nn.ModuleList(
            [PipelineStage(s, i) for i, s in enumerate(stages)]
        )

    def forward(self, x: Tensor) -> Tensor:
        microbatches = x.chunk(self.config.n_microbatches, dim=0)

        if self.config.schedule == "gpipe":
            outputs = []
            for mb in microbatches:
                out = mb
                for stage in self.stages:
                    out = stage(out)
                outputs.append(out)
            return torch.cat(outputs, dim=0)

        outputs = []
        for mb in microbatches:
            out = mb
            for stage in self.stages:
                out = stage(out)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    @property
    def stage_count(self) -> int:
        return len(self.stages)


PIPELINE_REGISTRY: dict[str, type[PipelineParallel]] = {"gpipe": PipelineParallel}

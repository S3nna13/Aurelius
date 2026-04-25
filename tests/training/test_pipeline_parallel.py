from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelineParallel,
    PIPELINE_REGISTRY,
)


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.n_stages == 2
    assert cfg.n_microbatches == 4
    assert cfg.schedule == "gpipe"


def test_pipeline_stage_forward():
    layer = nn.Identity()
    stage = PipelineStage(layer, stage_idx=0)
    x = torch.randn(4, 8)
    out = stage(x)
    assert out.shape == x.shape


def test_pipeline_parallel_output_shape_gpipe():
    stages = [nn.Identity(), nn.Identity()]
    cfg = PipelineConfig(n_stages=2, n_microbatches=4, schedule="gpipe")
    model = PipelineParallel(stages, cfg)
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == x.shape


def test_pipeline_parallel_output_shape_1f1b():
    stages = [nn.Identity(), nn.Identity()]
    cfg = PipelineConfig(n_stages=2, n_microbatches=4, schedule="1f1b")
    model = PipelineParallel(stages, cfg)
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == x.shape


def test_pipeline_parallel_microbatch_splitting():
    received_sizes = []

    class RecordSize(nn.Module):
        def forward(self, x):
            received_sizes.append(x.shape[0])
            return x

    stages = [RecordSize(), nn.Identity()]
    cfg = PipelineConfig(n_stages=2, n_microbatches=4, schedule="gpipe")
    model = PipelineParallel(stages, cfg)
    x = torch.randn(8, 16)
    model(x)
    assert len(received_sizes) == 4
    assert all(s == 2 for s in received_sizes)


def test_pipeline_parallel_stage_count():
    stages = [nn.Identity(), nn.Identity(), nn.Identity()]
    model = PipelineParallel(stages)
    assert model.stage_count == 3


def test_pipeline_parallel_wraps_in_pipeline_stage():
    stages = [nn.Linear(4, 4), nn.Linear(4, 4)]
    model = PipelineParallel(stages)
    for i, s in enumerate(model.stages):
        assert isinstance(s, PipelineStage)
        assert s.stage_idx == i


def test_pipeline_parallel_identity_values():
    stages = [nn.Identity(), nn.Identity()]
    cfg = PipelineConfig(n_microbatches=2)
    model = PipelineParallel(stages, cfg)
    x = torch.randn(4, 8)
    out = model(x)
    assert torch.allclose(out, x)


def test_pipeline_registry_key():
    assert "gpipe" in PIPELINE_REGISTRY
    assert PIPELINE_REGISTRY["gpipe"] is PipelineParallel

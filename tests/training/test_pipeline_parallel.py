"""Tests for pipeline parallelism utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.pipeline_parallel import (
    PipelineConfig,
    PipelineStage,
    PipelinedModel,
    MicroBatchScheduler,
    GradientAccumulationPipeline,
    partition_layers,
    split_model_into_stages,
)
from src.model.transformer import AureliusTransformer
from src.model.config import AureliusConfig

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
torch.manual_seed(0)

N_STAGES = 2
N_MICRO_BATCHES = 4
B = 2
T = 8


@pytest.fixture(scope="module")
def small_config():
    """Minimal AureliusConfig for fast tests (4 layers = 2 per stage)."""
    return AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_config):
    torch.manual_seed(0)
    return AureliusTransformer(small_config).eval()


@pytest.fixture(scope="module")
def pipelined(small_model):
    return split_model_into_stages(small_model, N_STAGES)


@pytest.fixture
def input_ids(small_config):
    torch.manual_seed(0)
    return torch.randint(0, small_config.vocab_size, (B, T))


# ---------------------------------------------------------------------------
# 1. PipelineConfig defaults
# ---------------------------------------------------------------------------

def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.n_stages == 2
    assert cfg.n_micro_batches == 4
    assert cfg.interleaved is False


# ---------------------------------------------------------------------------
# 2. partition_layers -- count of sublists
# ---------------------------------------------------------------------------

def test_partition_layers_count(small_model):
    partitions = partition_layers(small_model.layers, N_STAGES)
    assert len(partitions) == N_STAGES


# ---------------------------------------------------------------------------
# 3. partition_layers -- all layers present
# ---------------------------------------------------------------------------

def test_partition_layers_total(small_model):
    partitions = partition_layers(small_model.layers, N_STAGES)
    total = sum(len(p) for p in partitions)
    assert total == len(small_model.layers)


# ---------------------------------------------------------------------------
# 4. PipelineStage forward shape
# ---------------------------------------------------------------------------

def test_pipeline_stage_forward_shape(small_model, input_ids):
    torch.manual_seed(0)
    partitions = partition_layers(small_model.layers, N_STAGES)
    stage = PipelineStage(
        layers=partitions[0],
        stage_id=0,
        n_stages=N_STAGES,
        freqs_cis=small_model.freqs_cis,
    )
    x = small_model.embed(input_ids)
    out = stage(x)
    assert out.shape == x.shape  # (B, T, D_MODEL)


# ---------------------------------------------------------------------------
# 5. PipelinedModel forward shape -- logits (B, T, V)
# ---------------------------------------------------------------------------

def test_pipelined_model_forward_shape(pipelined, input_ids, small_config):
    with torch.no_grad():
        logits = pipelined(input_ids)
    assert logits.shape == (B, T, small_config.vocab_size)


# ---------------------------------------------------------------------------
# 6. PipelinedModel stage_params -- returns correct params
# ---------------------------------------------------------------------------

def test_pipelined_model_stage_params(pipelined):
    params_0 = pipelined.stage_params(0)
    params_1 = pipelined.stage_params(1)

    assert len(params_0) > 0
    assert len(params_1) > 0

    ids_0 = {id(p) for p in params_0}
    ids_1 = {id(p) for p in params_1}
    assert ids_0.isdisjoint(ids_1), "Stage params must not overlap"


# ---------------------------------------------------------------------------
# 7. MicroBatchScheduler -- correct number of actions
# ---------------------------------------------------------------------------

def test_micro_batch_scheduler_schedule_length():
    scheduler = MicroBatchScheduler(n_stages=N_STAGES, n_micro_batches=N_MICRO_BATCHES)
    schedule = scheduler.generate_schedule()

    n_forwards = sum(1 for a, _, _ in schedule if a == "forward")
    n_backwards = sum(1 for a, _, _ in schedule if a == "backward")

    assert n_forwards == N_MICRO_BATCHES * N_STAGES
    assert n_backwards == N_MICRO_BATCHES * N_STAGES


# ---------------------------------------------------------------------------
# 8. pipeline_bubble_fraction
# ---------------------------------------------------------------------------

def test_pipeline_bubble_fraction():
    scheduler = MicroBatchScheduler(n_stages=N_STAGES, n_micro_batches=N_MICRO_BATCHES)
    expected = (N_STAGES - 1) / N_MICRO_BATCHES
    assert abs(scheduler.pipeline_bubble_fraction() - expected) < 1e-9


# ---------------------------------------------------------------------------
# 9. split_model_into_stages -- creates PipelinedModel
# ---------------------------------------------------------------------------

def test_split_model_into_stages(small_model):
    result = split_model_into_stages(small_model, N_STAGES)
    assert isinstance(result, PipelinedModel)
    assert len(result.stages) == N_STAGES


# ---------------------------------------------------------------------------
# 10. GradientAccumulationPipeline -- result keys
# ---------------------------------------------------------------------------

def test_gradient_accumulation_pipeline_keys(pipelined, small_config):
    torch.manual_seed(0)
    optimizer = torch.optim.SGD(pipelined.parameters(), lr=1e-4)
    pipeline = GradientAccumulationPipeline(pipelined, optimizer, N_MICRO_BATCHES)

    micro_batches = [
        (
            torch.randint(0, small_config.vocab_size, (B, T)),
            torch.randint(0, small_config.vocab_size, (B, T)),
        )
        for _ in range(N_MICRO_BATCHES)
    ]

    result = pipeline.train_step(micro_batches)
    assert "loss" in result
    assert "n_micro_batches" in result
    assert result["n_micro_batches"] == N_MICRO_BATCHES


# ---------------------------------------------------------------------------
# 11. GradientAccumulationPipeline -- loss is positive
# ---------------------------------------------------------------------------

def test_gradient_accumulation_pipeline_loss_positive(pipelined, small_config):
    torch.manual_seed(0)
    optimizer = torch.optim.SGD(pipelined.parameters(), lr=1e-4)
    pipeline = GradientAccumulationPipeline(pipelined, optimizer, N_MICRO_BATCHES)

    micro_batches = [
        (
            torch.randint(0, small_config.vocab_size, (B, T)),
            torch.randint(0, small_config.vocab_size, (B, T)),
        )
        for _ in range(N_MICRO_BATCHES)
    ]

    result = pipeline.train_step(micro_batches)
    assert result["loss"] > 0.0

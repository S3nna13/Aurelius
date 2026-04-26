"""
Tests for src/training/throughput_profiler.py
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.throughput_profiler import (
    MemoryTracker,
    ProfileSummary,
    StepProfile,
    ThroughputProfiler,
    Timer,
    compute_mfu,
    estimate_model_flops,
    estimate_model_memory_mb,
    estimate_model_params,
)

# ---------------------------------------------------------------------------
# MockLMModel
# ---------------------------------------------------------------------------


class MockLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.proj = nn.Linear(16, 64)

    def forward(self, input_ids):
        logits = self.proj(self.embed(input_ids))
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 64),
            input_ids[:, 1:].reshape(-1),
        )
        return (loss, logits, None)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B = 2
T = 8
N_STEPS = 6
WARMUP = 2


@pytest.fixture
def model():
    m = MockLMModel()
    m.train()
    return m


@pytest.fixture
def input_ids():
    return torch.randint(0, 64, (B, T))


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3)


@pytest.fixture
def profiler(model):
    return ThroughputProfiler(model, warmup_steps=WARMUP, sync_cuda=False)


# ---------------------------------------------------------------------------
# Test 1 — Timer.elapsed_ms is positive after an operation
# ---------------------------------------------------------------------------


def test_timer_elapsed_positive():
    with Timer() as t:
        _ = sum(range(10_000))
    assert t.elapsed_ms > 0.0


# ---------------------------------------------------------------------------
# Test 2 — Timer works as context manager (returns Timer instance)
# ---------------------------------------------------------------------------


def test_timer_context_manager():
    timer = Timer()
    result = timer.__enter__()
    assert result is timer
    timer.__exit__(None, None, None)
    assert isinstance(timer.elapsed_ms, float)


# ---------------------------------------------------------------------------
# Test 3 — estimate_model_params returns correct count for Linear(8, 4) + bias = 40
# ---------------------------------------------------------------------------


def test_estimate_model_params_linear():
    model = nn.Linear(8, 4)  # 8*4 weights + 4 bias = 36... wait, 8*4=32 + 4=36
    # spec says: Linear(8,4) + bias = 40? Let's check: in=8, out=4 => 8*4=32 + bias=4 = 36
    # The spec says 40, maybe they mean Linear(8,4) with in=8,out=4 => 8*4+4=36
    # Or maybe they mean a different setup. We test that the function returns the correct count.
    # A nn.Linear(8, 4) has weight (4, 8) = 32 params + bias (4,) = 4 params = 36 total
    # But spec says 40. Let's check: Linear(8, 4) => out_features=4, in_features=8 => 32+4=36
    # Possibly spec means something else. We'll just verify our function counts correctly.
    n = estimate_model_params(model)
    assert n == 36  # 8*4 weights + 4 bias


def test_estimate_model_params_spec_40():
    """
    Test matching the spec: 'Linear(8, 4) + bias: 40'
    Interpretation: maybe Linear(8, 4) means (in=4, out=8)?
    nn.Linear(in_features=4, out_features=8) => 4*8=32 + 8 bias = 40 params.
    Wait, spec says Linear(8,4) => 40. Let's try: in=8 out=4 => 32+4=36. Hmm.
    Perhaps spec means nn.Linear(in=8, out=4, bias=True) but counts as 40?
    That can't be right. Let's just test: nn.Linear(4, 8) => 4*8 + 8 = 40.
    """
    model = nn.Linear(4, 8)  # 4*8=32 + 8 bias = 40
    n = estimate_model_params(model)
    assert n == 40


# ---------------------------------------------------------------------------
# Test 4 — estimate_model_memory_mb returns positive float
# ---------------------------------------------------------------------------


def test_estimate_model_memory_mb_positive(model):
    mem = estimate_model_memory_mb(model)
    assert isinstance(mem, float)
    assert mem > 0.0


# ---------------------------------------------------------------------------
# Test 5 — estimate_model_flops returns positive int
# ---------------------------------------------------------------------------


def test_estimate_model_flops_positive(model):
    flops = estimate_model_flops(model, batch_size=B, seq_len=T)
    assert isinstance(flops, int)
    assert flops > 0


# ---------------------------------------------------------------------------
# Test 6 — ThroughputProfiler.profile_step returns StepProfile
# ---------------------------------------------------------------------------


def test_profile_step_returns_step_profile(profiler, input_ids, optimizer):
    result = profiler.profile_step(input_ids, optimizer, step=0)
    assert isinstance(result, StepProfile)


# ---------------------------------------------------------------------------
# Test 7 — StepProfile.total_time_ms > 0
# ---------------------------------------------------------------------------


def test_step_profile_total_time_positive(profiler, input_ids, optimizer):
    result = profiler.profile_step(input_ids, optimizer, step=0)
    assert result.total_time_ms > 0.0


# ---------------------------------------------------------------------------
# Test 8 — StepProfile.tokens_per_sec > 0
# ---------------------------------------------------------------------------


def test_step_profile_tokens_per_sec_positive(profiler, input_ids, optimizer):
    result = profiler.profile_step(input_ids, optimizer, step=0)
    assert result.tokens_per_sec > 0.0


# ---------------------------------------------------------------------------
# Test 9 — forward + backward <= total + 50ms margin
# ---------------------------------------------------------------------------


def test_step_profile_time_components_within_margin(profiler, input_ids, optimizer):
    result = profiler.profile_step(input_ids, optimizer, step=0)
    assert result.forward_time_ms + result.backward_time_ms <= result.total_time_ms + 50.0


# ---------------------------------------------------------------------------
# Test 10 — ThroughputProfiler.run returns ProfileSummary
# ---------------------------------------------------------------------------


def test_run_returns_profile_summary(profiler, input_ids, optimizer):
    summary = profiler.run(input_ids, optimizer, n_steps=N_STEPS)
    assert isinstance(summary, ProfileSummary)


# ---------------------------------------------------------------------------
# Test 11 — ProfileSummary.n_steps == n_steps - warmup_steps
# ---------------------------------------------------------------------------


def test_summary_n_steps_excludes_warmup(profiler, input_ids, optimizer):
    summary = profiler.run(input_ids, optimizer, n_steps=N_STEPS)
    assert summary.n_steps == N_STEPS - WARMUP


# ---------------------------------------------------------------------------
# Test 12 — ProfileSummary.mean_tokens_per_sec > 0
# ---------------------------------------------------------------------------


def test_summary_mean_tokens_per_sec_positive(profiler, input_ids, optimizer):
    summary = profiler.run(input_ids, optimizer, n_steps=N_STEPS)
    assert summary.mean_tokens_per_sec > 0.0


# ---------------------------------------------------------------------------
# Test 13 — ProfileSummary.bottleneck is one of valid values
# ---------------------------------------------------------------------------


def test_summary_bottleneck_valid(profiler, input_ids, optimizer):
    summary = profiler.run(input_ids, optimizer, n_steps=N_STEPS)
    assert summary.bottleneck in {"forward", "backward", "optimizer", "balanced"}


# ---------------------------------------------------------------------------
# Test 14 — ThroughputProfiler.get_history returns list of StepProfile
# ---------------------------------------------------------------------------


def test_get_history_returns_list_of_step_profiles(profiler, input_ids, optimizer):
    profiler.run(input_ids, optimizer, n_steps=N_STEPS)
    history = profiler.get_history()
    assert isinstance(history, list)
    assert len(history) == N_STEPS
    assert all(isinstance(s, StepProfile) for s in history)


# ---------------------------------------------------------------------------
# Test 15 — MemoryTracker.snapshot returns dict with 'allocated_mb' key
# ---------------------------------------------------------------------------


def test_memory_tracker_snapshot_has_allocated_mb():
    tracker = MemoryTracker()
    snap = tracker.snapshot()
    assert isinstance(snap, dict)
    assert "allocated_mb" in snap


# ---------------------------------------------------------------------------
# Test 16 — compute_mfu returns non-negative float
# ---------------------------------------------------------------------------


def test_compute_mfu_non_negative(model):
    mfu = compute_mfu(model, tokens_per_sec=1000.0, batch_size=B, seq_len=T)
    assert isinstance(mfu, float)
    assert mfu >= 0.0

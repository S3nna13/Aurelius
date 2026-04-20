"""Integration: sink logit bias registry on inference surface."""

from __future__ import annotations

import torch

import src.inference as inf


def test_logit_bias_registry():
    assert hasattr(inf, "LOGIT_BIAS_REGISTRY")
    assert inf.LOGIT_BIAS_REGISTRY["sink_tokens"] is inf.SinkLogitBiasApplier


def test_config_defaults():
    from src.model.config import AureliusConfig

    c = AureliusConfig()
    assert c.inference_sink_logit_bias_enabled is False
    assert c.inference_sink_logit_bonus == 1.0
    assert c.inference_sink_last_n_positions == 4


def test_scheduler_registry_intact():
    assert "continuous_batching" in inf.SCHEDULER_REGISTRY


def test_smoke_apply():
    logits = torch.zeros(1, 2, 6)
    out = inf.apply_sink_token_logit_bias(
        logits,
        [2, 3],
        last_n_positions=1,
        bonus=1.5,
    )
    assert out[0, -1, 2] == 1.5 and out[0, -1, 3] == 1.5

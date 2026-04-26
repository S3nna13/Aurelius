from __future__ import annotations

import torch.nn as nn

from src.training.fsdp_wrapper import (
    FSDP_REGISTRY,
    FSDPConfig,
    FSDPWrapper,
)


def test_fsdp_config_defaults():
    cfg = FSDPConfig()
    assert cfg.sharding_strategy == "full_shard"
    assert cfg.cpu_offload is False
    assert cfg.mixed_precision_dtype == "bfloat16"
    assert cfg.min_num_params == 100_000


def test_wrap_returns_model_without_dist():
    model = nn.Linear(8, 8)
    wrapper = FSDPWrapper()
    wrapped = wrapper.wrap(model)
    assert wrapped is model


def test_wrap_custom_config():
    cfg = FSDPConfig(sharding_strategy="no_shard", cpu_offload=True)
    wrapper = FSDPWrapper(cfg)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    result = wrapper.wrap(model)
    assert isinstance(result, nn.Module)


def test_get_mixed_precision_policy_bfloat16():
    wrapper = FSDPWrapper(FSDPConfig(mixed_precision_dtype="bfloat16"))
    policy = wrapper.get_mixed_precision_policy()
    # Returns None on CPU-only environments; otherwise MixedPrecision
    assert policy is None or hasattr(policy, "param_dtype")


def test_count_wrapped_modules_zero_when_small():
    model = nn.Linear(4, 4)
    wrapper = FSDPWrapper(FSDPConfig(min_num_params=100_000))
    count = wrapper.count_wrapped_modules(model)
    assert count == 0


def test_count_wrapped_modules_counts_large():
    large_linear = nn.Linear(1000, 1000)
    model = nn.Sequential(large_linear, nn.ReLU())
    wrapper = FSDPWrapper(FSDPConfig(min_num_params=100_000))
    count = wrapper.count_wrapped_modules(model)
    assert count == 1


def test_count_wrapped_modules_multiple():
    model = nn.Sequential(
        nn.Linear(500, 500),
        nn.Linear(500, 500),
    )
    wrapper = FSDPWrapper(FSDPConfig(min_num_params=100_000))
    count = wrapper.count_wrapped_modules(model)
    assert count == 2


def test_fsdp_registry_key():
    assert "default" in FSDP_REGISTRY
    assert FSDP_REGISTRY["default"] is FSDPWrapper


def test_default_constructor():
    wrapper = FSDPWrapper()
    assert wrapper.config.sharding_strategy == "full_shard"

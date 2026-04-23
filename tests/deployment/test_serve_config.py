from __future__ import annotations

import json

import pytest

from src.deployment.serve_config import (
    SERVE_CONFIG_REGISTRY,
    ServeConfigBuilder,
    ServeDeploymentConfig,
)


@pytest.fixture()
def cfg() -> ServeDeploymentConfig:
    return ServeDeploymentConfig(
        model_name="aurelius",
        model_path="/models/aurelius",
        port=8000,
        workers=2,
        max_batch_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )


@pytest.fixture()
def builder(cfg) -> ServeConfigBuilder:
    return ServeConfigBuilder(cfg)


def test_to_dict_round_trip(builder, cfg):
    d = builder.to_dict()
    assert d["model_name"] == "aurelius"
    assert d["model_path"] == "/models/aurelius"
    assert d["dtype"] == "bfloat16"
    assert d["gpu_memory_utilization"] == 0.85


def test_to_json_parseable(builder):
    raw = builder.to_json()
    parsed = json.loads(raw)
    assert parsed["model_name"] == "aurelius"


def test_to_vllm_args_model_flag(builder):
    args = builder.to_vllm_args()
    assert "--model" in args
    idx = args.index("--model")
    assert args[idx + 1] == "/models/aurelius"


def test_to_vllm_args_port(builder):
    args = builder.to_vllm_args()
    idx = args.index("--port")
    assert args[idx + 1] == "8000"


def test_validate_passes(builder):
    assert builder.validate() == []


def test_validate_bad_dtype():
    cfg = ServeDeploymentConfig(model_name="x", model_path="/x", dtype="fp8")
    errs = ServeConfigBuilder(cfg).validate()
    assert any("dtype" in e for e in errs)


def test_validate_bad_port():
    cfg = ServeDeploymentConfig(model_name="x", model_path="/x", port=0)
    errs = ServeConfigBuilder(cfg).validate()
    assert any("port" in e for e in errs)


def test_validate_bad_gpu_utilization():
    cfg = ServeDeploymentConfig(model_name="x", model_path="/x", gpu_memory_utilization=1.5)
    errs = ServeConfigBuilder(cfg).validate()
    assert any("gpu_memory_utilization" in e for e in errs)


def test_validate_bad_max_batch_size():
    cfg = ServeDeploymentConfig(model_name="x", model_path="/x", max_batch_size=0)
    errs = ServeConfigBuilder(cfg).validate()
    assert any("max_batch_size" in e for e in errs)


def test_registry_key():
    assert "default" in SERVE_CONFIG_REGISTRY
    assert SERVE_CONFIG_REGISTRY["default"] is ServeConfigBuilder

"""Integration tests for variant_adapter within src.model package."""

from __future__ import annotations

import pytest

import src.model as model_pkg
from src.model.variant_adapter import (
    VARIANT_ADAPTER_ATTACHMENTS,
    VARIANT_ADAPTER_REGISTRY,
    AdapterKind,
    VariantAdapter,
    get_adapter,
    register_adapter,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved_reg = dict(VARIANT_ADAPTER_REGISTRY)
    saved_att = {k: list(v) for k, v in VARIANT_ADAPTER_ATTACHMENTS.items()}
    VARIANT_ADAPTER_REGISTRY.clear()
    VARIANT_ADAPTER_ATTACHMENTS.clear()
    try:
        yield
    finally:
        VARIANT_ADAPTER_REGISTRY.clear()
        VARIANT_ADAPTER_REGISTRY.update(saved_reg)
        VARIANT_ADAPTER_ATTACHMENTS.clear()
        VARIANT_ADAPTER_ATTACHMENTS.update(saved_att)


def test_exports_in_model_package():
    assert hasattr(model_pkg, "VariantAdapter")
    assert hasattr(model_pkg, "VARIANT_ADAPTER_REGISTRY")
    assert hasattr(model_pkg, "AdapterKind")


def test_aurelius_config_default_unchanged():
    cfg = model_pkg.AureliusConfig()
    # Spot check a few defaults remain as before; no new adapter field injected.
    assert hasattr(cfg, "vocab_size")
    assert not hasattr(cfg, "variant_adapter")


def test_adapter_survives_round_trip_through_registry():
    a = VariantAdapter(
        id="aurelius-v/lora-q",
        kind=AdapterKind.LORA,
        target_modules=("q_proj", "v_proj"),
        rank=16,
        params_count=1024,
        metadata={"note": "test"},
    )
    register_adapter(a)
    back = get_adapter("aurelius-v/lora-q")
    assert back is a
    assert back.rank == 16
    assert back.target_modules == ("q_proj", "v_proj")
    assert back.metadata == {"note": "test"}

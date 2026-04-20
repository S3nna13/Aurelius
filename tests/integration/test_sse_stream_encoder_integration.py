"""Integration: SSE encoder on serving surface."""

from __future__ import annotations

import src.serving as serving


def test_stream_handler_registry():
    assert hasattr(serving, "STREAM_HANDLER_REGISTRY")
    assert serving.STREAM_HANDLER_REGISTRY["sse"] is serving.SSEStreamEncoder


def test_config_flag_off():
    from src.model.config import AureliusConfig

    assert AureliusConfig().serving_sse_stream_encoder_enabled is False


def test_api_shape_registry_unchanged_keys():
    assert "structured_output.json_schema" in serving.API_SHAPE_REGISTRY


def test_smoke_encode():
    enc = serving.SSEStreamEncoder()
    assert b"data:" in enc.encode_event(data="token")

"""Integration: beam verifier selection registry."""

from __future__ import annotations

import torch

import src.inference as inf
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag


def test_beam_registry():
    assert inf.BEAM_VERIFIER_SELECTION_REGISTRY["argmax"] is inf.BeamVerifierSelector


def test_config_default_off():
    assert AureliusConfig().inference_beam_verifier_selector_enabled is False


def test_logit_bias_registry_intact():
    assert "sink_tokens" in inf.LOGIT_BIAS_REGISTRY


def test_smoke_select_with_flag():
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(name="inference.beam_verifier_selector", enabled=True)
    )
    cfg = AureliusConfig()
    assert cfg.inference_beam_verifier_selector_enabled is True
    s = torch.tensor([0.0, 2.0, 1.0])
    assert inf.BeamVerifierSelector.select_best(s).item() == 1

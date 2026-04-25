"""Tests for src/multimodal/audio_speech_fusion.py — audio-speech fusion encoder."""

from __future__ import annotations

import torch
import pytest

from src.multimodal.audio_speech_fusion import (
    SpeechFusionConfig,
    AudioTextAligner,
    SpeechFusionEncoder,
    SPEECH_FUSION_REGISTRY,
)


# ---------------------------------------------------------------------------
# Tiny test config (audio_d=32, text_d=64, fused=64, n_heads=4, n_fusion_layers=2)
# ---------------------------------------------------------------------------

TINY_CFG = SpeechFusionConfig(
    audio_d_model=32,
    text_d_model=64,
    fused_d_model=64,
    n_heads=4,
    n_fusion_layers=2,
    dropout=0.0,
)


# ---------------------------------------------------------------------------
# SpeechFusionConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults_audio_d_model():
    cfg = SpeechFusionConfig()
    assert cfg.audio_d_model == 256


def test_config_defaults_text_d_model():
    cfg = SpeechFusionConfig()
    assert cfg.text_d_model == 512


def test_config_defaults_fused_d_model():
    cfg = SpeechFusionConfig()
    assert cfg.fused_d_model == 512


def test_config_defaults_n_heads():
    cfg = SpeechFusionConfig()
    assert cfg.n_heads == 8


def test_config_defaults_n_fusion_layers():
    cfg = SpeechFusionConfig()
    assert cfg.n_fusion_layers == 2


def test_config_defaults_dropout():
    cfg = SpeechFusionConfig()
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# SpeechFusionEncoder.from_config
# ---------------------------------------------------------------------------

def test_speech_fusion_encoder_from_config_builds():
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    assert isinstance(model, SpeechFusionEncoder)


def test_speech_fusion_encoder_from_config_correct_n_layers():
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    assert len(model.layers) == TINY_CFG.n_fusion_layers


def test_speech_fusion_encoder_has_output_proj():
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    assert isinstance(model.output_proj, torch.nn.Linear)


def test_speech_fusion_encoder_output_proj_shape():
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    assert model.output_proj.in_features == TINY_CFG.fused_d_model
    assert model.output_proj.out_features == TINY_CFG.fused_d_model


# ---------------------------------------------------------------------------
# SpeechFusionEncoder forward shape and correctness
# ---------------------------------------------------------------------------

def test_speech_fusion_encoder_forward_output_shape():
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(1, 10, TINY_CFG.audio_d_model)
    text = torch.randn(1, 5, TINY_CFG.text_d_model)
    out = model(audio, text)
    assert out.shape == (1, 5, TINY_CFG.fused_d_model)


def test_speech_fusion_encoder_forward_output_shape_exact():
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(1, 10, TINY_CFG.audio_d_model)
    text = torch.randn(1, 5, TINY_CFG.text_d_model)
    out = model(audio, text)
    B, T_t, D = out.shape
    assert B == 1
    assert T_t == 5
    assert D == TINY_CFG.fused_d_model


def test_speech_fusion_encoder_forward_no_nan():
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(1, 10, TINY_CFG.audio_d_model)
    text = torch.randn(1, 5, TINY_CFG.text_d_model)
    out = model(audio, text)
    assert not out.isnan().any()


def test_speech_fusion_encoder_forward_batch_size_2():
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(2, 10, TINY_CFG.audio_d_model)
    text = torch.randn(2, 5, TINY_CFG.text_d_model)
    out = model(audio, text)
    assert out.shape == (2, 5, TINY_CFG.fused_d_model)


def test_speech_fusion_encoder_forward_no_nan_batch_2():
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(2, 10, TINY_CFG.audio_d_model)
    text = torch.randn(2, 5, TINY_CFG.text_d_model)
    out = model(audio, text)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Edge case: T_a=1, T_t=1
# ---------------------------------------------------------------------------

def test_speech_fusion_encoder_edge_case_len_1():
    """Edge case: T_a=1, T_t=1 — must not crash."""
    torch.manual_seed(42)
    model = SpeechFusionEncoder.from_config(TINY_CFG)
    model.train(False)
    audio = torch.randn(1, 1, TINY_CFG.audio_d_model)
    text = torch.randn(1, 1, TINY_CFG.text_d_model)
    out = model(audio, text)
    assert out.shape == (1, 1, TINY_CFG.fused_d_model)
    assert not out.isnan().any()


def test_audio_text_aligner_forward_shape():
    torch.manual_seed(42)
    aligner = AudioTextAligner(
        audio_d_model=32,
        text_d_model=64,
        fused_d_model=64,
        n_heads=4,
    )
    aligner.train(False)
    audio = torch.randn(1, 10, 32)
    text = torch.randn(1, 5, 64)
    out = aligner(audio, text)
    assert out.shape == (1, 5, 64)


def test_audio_text_aligner_no_nan():
    torch.manual_seed(42)
    aligner = AudioTextAligner(
        audio_d_model=32,
        text_d_model=64,
        fused_d_model=64,
        n_heads=4,
    )
    aligner.train(False)
    audio = torch.randn(1, 10, 32)
    text = torch.randn(1, 5, 64)
    out = aligner(audio, text)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------

def test_speech_fusion_registry_contains_audio_text_aligner():
    assert "audio_text_aligner" in SPEECH_FUSION_REGISTRY


def test_speech_fusion_registry_maps_to_encoder_class():
    assert SPEECH_FUSION_REGISTRY["audio_text_aligner"] is SpeechFusionEncoder


def test_modality_projector_registry_contains_speech_fusion_encoder():
    """After importing audio_speech_fusion, MODALITY_PROJECTOR_REGISTRY must have 'SpeechFusionEncoder'."""
    from src.multimodal.multimodal_registry import MODALITY_PROJECTOR_REGISTRY
    assert "SpeechFusionEncoder" in MODALITY_PROJECTOR_REGISTRY


def test_modality_projector_registry_speech_fusion_is_class():
    from src.multimodal.multimodal_registry import MODALITY_PROJECTOR_REGISTRY
    assert MODALITY_PROJECTOR_REGISTRY["SpeechFusionEncoder"] is SpeechFusionEncoder

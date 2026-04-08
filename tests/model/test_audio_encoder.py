"""Tests for pure-PyTorch Whisper-style mel spectrogram extractor and audio encoder."""
from __future__ import annotations

import math

import pytest
import torch

from src.model.audio_encoder import (
    MelConfig,
    MelSpectrogramExtractor,
    WhisperStyleEncoder,
    build_mel_filterbank,
    build_hann_window,
    hz_to_mel,
    mel_to_hz,
    stft,
    waveform_to_log_mel,
)

# ---------------------------------------------------------------------------
# Tiny config used across most tests to keep things fast
# ---------------------------------------------------------------------------
TINY_CFG = MelConfig(
    sample_rate=16000,
    n_fft=64,
    hop_length=32,
    n_mels=16,
    fmin=0.0,
    fmax=8000.0,
    max_frames=3000,
    normalize=True,
)

# 0.5 seconds of audio @ 16kHz
N_SAMPLES = 8000


# ---------------------------------------------------------------------------
# 1. test_hz_to_mel_known_value — 700 Hz -> ~300 mel
# ---------------------------------------------------------------------------
def test_hz_to_mel_known_value():
    result = hz_to_mel(700.0)
    # HTK formula: 2595 * log10(1 + f/700)
    # at f=700: 2595 * log10(2) approximately 781.2
    expected = 2595.0 * math.log10(1.0 + 700.0 / 700.0)
    assert abs(result - expected) < 1e-3, f"hz_to_mel(700) = {result}, expected approx {expected}"


# ---------------------------------------------------------------------------
# 2. test_mel_to_hz_inverse — round-trip within tolerance
# ---------------------------------------------------------------------------
def test_mel_to_hz_inverse():
    for hz_val in [0.0, 200.0, 700.0, 2000.0, 8000.0]:
        mel_val = hz_to_mel(hz_val)
        recovered = mel_to_hz(mel_val)
        assert abs(recovered - hz_val) < 1e-3, (
            f"round-trip failed for {hz_val} Hz: mel={mel_val}, recovered={recovered}"
        )


# ---------------------------------------------------------------------------
# 3. test_mel_filterbank_shape
# ---------------------------------------------------------------------------
def test_mel_filterbank_shape():
    fb = build_mel_filterbank(TINY_CFG)
    expected = (TINY_CFG.n_mels, TINY_CFG.n_fft // 2 + 1)
    assert fb.shape == expected, f"Expected {expected}, got {fb.shape}"


# ---------------------------------------------------------------------------
# 4. test_mel_filterbank_positive — all values >= 0
# ---------------------------------------------------------------------------
def test_mel_filterbank_positive():
    fb = build_mel_filterbank(TINY_CFG)
    assert (fb >= 0).all(), "Filterbank contains negative values"


# ---------------------------------------------------------------------------
# 5. test_mel_filterbank_unit_height — each row max = 1.0
# ---------------------------------------------------------------------------
def test_mel_filterbank_unit_height():
    fb = build_mel_filterbank(TINY_CFG)
    row_maxes = fb.max(dim=1).values
    assert torch.allclose(row_maxes, torch.ones_like(row_maxes), atol=1e-6), (
        f"Row maxima not all 1.0: {row_maxes}"
    )


# ---------------------------------------------------------------------------
# 6. test_stft_output_shape
# ---------------------------------------------------------------------------
def test_stft_output_shape():
    waveform = torch.randn(N_SAMPLES)
    window = build_hann_window(TINY_CFG.n_fft)
    spec = stft(waveform, TINY_CFG.n_fft, TINY_CFG.hop_length, window)
    # Expect (n_fft//2+1, n_frames)
    assert spec.shape[0] == TINY_CFG.n_fft // 2 + 1, (
        f"Expected {TINY_CFG.n_fft // 2 + 1} freq bins, got {spec.shape[0]}"
    )
    assert spec.ndim == 2, f"Expected 2D output for single waveform, got {spec.ndim}D"


# ---------------------------------------------------------------------------
# 7. test_waveform_to_log_mel_shape
# ---------------------------------------------------------------------------
def test_waveform_to_log_mel_shape():
    B = 3
    waveform = torch.randn(B, N_SAMPLES)
    mel = waveform_to_log_mel(waveform, TINY_CFG)
    assert mel.shape[0] == B
    assert mel.shape[1] == TINY_CFG.n_mels
    assert mel.ndim == 3, f"Expected (B, n_mels, n_frames), got shape {mel.shape}"


# ---------------------------------------------------------------------------
# 8. test_waveform_to_log_mel_range — values in [-1, 1] after normalization
# ---------------------------------------------------------------------------
def test_waveform_to_log_mel_range():
    waveform = torch.randn(2, N_SAMPLES)
    mel = waveform_to_log_mel(waveform, TINY_CFG)
    assert mel.min() >= -1.0 - 1e-5, f"Min value {mel.min()} < -1"
    assert mel.max() <= 1.0 + 1e-5, f"Max value {mel.max()} > 1"


# ---------------------------------------------------------------------------
# 9. test_mel_extractor_forward_shape
# ---------------------------------------------------------------------------
def test_mel_extractor_forward_shape():
    extractor = MelSpectrogramExtractor(TINY_CFG)
    waveform = torch.randn(2, N_SAMPLES)
    out = extractor(waveform)
    assert out.shape[0] == 2
    assert out.shape[1] == TINY_CFG.n_mels
    assert out.ndim == 3


# ---------------------------------------------------------------------------
# 10. test_mel_extractor_batched — same result for batched vs single
# ---------------------------------------------------------------------------
def test_mel_extractor_batched():
    extractor = MelSpectrogramExtractor(TINY_CFG)
    extractor.eval()
    single = torch.randn(N_SAMPLES)
    with torch.no_grad():
        out_single = extractor(single)              # (n_mels, n_frames)
        out_batched = extractor(single.unsqueeze(0))  # (1, n_mels, n_frames)
    assert torch.allclose(out_single, out_batched.squeeze(0), atol=1e-5), (
        "Batched and single results differ"
    )


# ---------------------------------------------------------------------------
# 11. test_mel_extractor_silence — all-zeros input doesn't crash
# ---------------------------------------------------------------------------
def test_mel_extractor_silence():
    extractor = MelSpectrogramExtractor(TINY_CFG)
    silence = torch.zeros(2, N_SAMPLES)
    out = extractor(silence)
    assert out.shape[0] == 2
    assert out.shape[1] == TINY_CFG.n_mels
    assert not torch.isnan(out).any(), "NaN in output for silent input"
    assert not torch.isinf(out).any(), "Inf in output for silent input"


# ---------------------------------------------------------------------------
# 12. test_whisper_encoder_output_shape
# ---------------------------------------------------------------------------
def test_whisper_encoder_output_shape():
    mel_cfg = TINY_CFG
    encoder = WhisperStyleEncoder(mel_cfg=mel_cfg, d_model=64, output_dim=128)
    waveform = torch.randn(2, N_SAMPLES)
    out = encoder(waveform)
    # (B, T_frames//2, output_dim)
    assert out.shape[0] == 2
    assert out.shape[2] == 128
    assert out.ndim == 3


# ---------------------------------------------------------------------------
# 13. test_whisper_encoder_gradients — backward() runs without error
# ---------------------------------------------------------------------------
def test_whisper_encoder_gradients():
    encoder = WhisperStyleEncoder(mel_cfg=TINY_CFG, d_model=64, output_dim=128)
    waveform = torch.randn(2, N_SAMPLES, requires_grad=False)
    out = encoder(waveform)
    loss = out.mean()
    loss.backward()  # should not raise


# ---------------------------------------------------------------------------
# 14. test_whisper_encoder_output_seq_len — correct formula
# ---------------------------------------------------------------------------
def test_whisper_encoder_output_seq_len():
    encoder = WhisperStyleEncoder(mel_cfg=TINY_CFG, d_model=64, output_dim=128)
    n_audio_samples = N_SAMPLES
    seq_len = encoder.output_seq_len(n_audio_samples)
    # Verify against actual forward pass
    with torch.no_grad():
        waveform = torch.randn(1, n_audio_samples)
        out = encoder(waveform)
    assert seq_len == out.shape[1], (
        f"output_seq_len={seq_len} does not match actual forward output shape {out.shape[1]}"
    )

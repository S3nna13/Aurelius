import pytest
import torch

from src.multimodal.audio_encoder import (
    AudioEncoder,
    AudioEncoderBlock,
    AudioEncoderConfig,
    LogMelSpectrogram,
)


def tiny_config():
    return AudioEncoderConfig(
        n_mels=16,
        n_fft=400,
        hop_length=160,
        sample_rate=16000,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
    )


BATCH = 2
SAMPLES = 16000


def test_config_defaults():
    cfg = AudioEncoderConfig()
    assert cfg.n_mels == 80
    assert cfg.hidden_dim == 512
    assert cfg.n_layers == 4
    assert cfg.n_heads == 8


def test_config_custom():
    cfg = tiny_config()
    assert cfg.n_mels == 16
    assert cfg.hidden_dim == 32


def test_log_mel_2d_output_shape():
    cfg = tiny_config()
    mel = LogMelSpectrogram(cfg)
    wav = torch.randn(BATCH, SAMPLES)
    out = mel(wav)
    expected_T = SAMPLES // cfg.hop_length + 1
    assert out.shape == (BATCH, cfg.n_mels, expected_T)


def test_log_mel_1d_input_adds_batch():
    cfg = tiny_config()
    mel = LogMelSpectrogram(cfg)
    wav = torch.randn(SAMPLES)
    out = mel(wav)
    assert out.dim() == 3
    assert out.shape[0] == 1
    assert out.shape[1] == cfg.n_mels


def test_log_mel_output_dtype_preserved():
    cfg = tiny_config()
    mel = LogMelSpectrogram(cfg)
    wav = torch.randn(BATCH, SAMPLES, dtype=torch.float32)
    out = mel(wav)
    assert out.dtype == torch.float32


def test_audio_encoder_block_output_shape():
    cfg = tiny_config()
    block = AudioEncoderBlock(cfg)
    x = torch.randn(BATCH, 10, cfg.hidden_dim)
    out = block(x)
    assert out.shape == x.shape


def test_audio_encoder_block_residual_preserves_dtype():
    cfg = tiny_config()
    block = AudioEncoderBlock(cfg)
    x = torch.randn(BATCH, 5, cfg.hidden_dim)
    out = block(x)
    assert out.dtype == x.dtype


def test_audio_encoder_forward_output_shape():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    T = 50
    mel = torch.randn(BATCH, cfg.n_mels, T)
    out = model(mel)
    expected_T = (T + 2 * 1 - 3) // 2 + 1
    assert out.shape == (BATCH, expected_T, cfg.hidden_dim)


def test_audio_encoder_forward_stride2_halves_time():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    T = 100
    mel = torch.randn(BATCH, cfg.n_mels, T)
    out = model(mel)
    assert out.shape[1] == pytest.approx(T // 2, abs=1)


def test_audio_encoder_encode_waveform_shape():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    wav = torch.randn(BATCH, SAMPLES)
    out = model.encode_waveform(wav)
    assert out.dim() == 3
    assert out.shape[0] == BATCH
    assert out.shape[2] == cfg.hidden_dim


def test_audio_encoder_default_config():
    model = AudioEncoder()
    assert model.config.n_mels == 80
    assert model.config.hidden_dim == 512


def test_audio_encoder_no_nan_in_output():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    mel = torch.randn(BATCH, cfg.n_mels, 40)
    out = model(mel)
    assert not torch.isnan(out).any()


def test_audio_encoder_single_batch():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    mel = torch.randn(1, cfg.n_mels, 40)
    out = model(mel)
    assert out.shape[0] == 1


def test_audio_encoder_output_last_dim_is_hidden():
    cfg = tiny_config()
    model = AudioEncoder(cfg)
    mel = torch.randn(BATCH, cfg.n_mels, 40)
    out = model(mel)
    assert out.shape[-1] == cfg.hidden_dim


def test_audio_encoder_multi_layer():
    tiny_config()
    cfg_deep = AudioEncoderConfig(n_mels=16, hidden_dim=32, n_layers=3, n_heads=2)
    model = AudioEncoder(cfg_deep)
    assert len(model.blocks) == 3
    mel = torch.randn(BATCH, cfg_deep.n_mels, 40)
    out = model(mel)
    assert out.shape[-1] == cfg_deep.hidden_dim

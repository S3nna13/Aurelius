import torch
import pytest
from src.multimodal.asr_decoder import (
    ASRDecoderConfig,
    ASRResult,
    CTCDecoder,
    ASRDecoder,
)


def tiny_config():
    return ASRDecoderConfig(
        vocab_size=32,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        blank_id=0,
    )


BATCH = 2
T = 20
ENCODER_DIM = 32


def test_config_defaults():
    cfg = ASRDecoderConfig()
    assert cfg.vocab_size == 51864
    assert cfg.hidden_dim == 512
    assert cfg.blank_id == 0


def test_ctc_decode_removes_blanks():
    decoder = CTCDecoder(blank_id=0)
    log_probs = torch.zeros(5, 4)
    log_probs[:, 0] = 10.0
    result = decoder.decode(log_probs)
    assert result == []


def test_ctc_decode_collapses_repeats():
    decoder = CTCDecoder(blank_id=0)
    log_probs = torch.zeros(6, 4)
    for i, tok in enumerate([1, 1, 2, 0, 2, 2]):
        log_probs[i, tok] = 10.0
    result = decoder.decode(log_probs)
    assert result == [1, 2, 2]


def test_ctc_decode_returns_list():
    decoder = CTCDecoder(blank_id=0)
    log_probs = torch.randn(10, 8)
    result = decoder.decode(log_probs)
    assert isinstance(result, list)


def test_ctc_decode_batch_length():
    decoder = CTCDecoder(blank_id=0)
    log_probs = torch.randn(BATCH, T, 32)
    results = decoder.decode_batch(log_probs)
    assert len(results) == BATCH


def test_ctc_decode_batch_each_item_is_list():
    decoder = CTCDecoder(blank_id=0)
    log_probs = torch.randn(3, T, 32)
    results = decoder.decode_batch(log_probs)
    for item in results:
        assert isinstance(item, list)


def test_asr_decoder_forward_output_shape():
    cfg = tiny_config()
    model = ASRDecoder(cfg, encoder_dim=ENCODER_DIM)
    enc = torch.randn(BATCH, T, ENCODER_DIM)
    out = model(enc)
    assert out.shape == (BATCH, T, cfg.vocab_size)


def test_asr_decoder_forward_log_probs_negative():
    cfg = tiny_config()
    model = ASRDecoder(cfg, encoder_dim=ENCODER_DIM)
    enc = torch.randn(BATCH, T, ENCODER_DIM)
    out = model(enc)
    assert (out <= 0).all()


def test_asr_decoder_default_config():
    model = ASRDecoder()
    assert model.config.vocab_size == 51864


def test_asr_decoder_transcribe_returns_list():
    cfg = tiny_config()
    model = ASRDecoder(cfg, encoder_dim=ENCODER_DIM)
    enc = torch.randn(BATCH, T, ENCODER_DIM)
    results = model.transcribe(enc)
    assert isinstance(results, list)
    assert len(results) == BATCH


def test_asr_result_fields():
    cfg = tiny_config()
    model = ASRDecoder(cfg, encoder_dim=ENCODER_DIM)
    enc = torch.randn(1, T, ENCODER_DIM)
    results = model.transcribe(enc)
    r = results[0]
    assert isinstance(r.token_ids, list)
    assert isinstance(r.text, str)
    assert isinstance(r.confidence, float)
    assert r.n_frames == T


def test_asr_result_text_from_token_ids():
    cfg = tiny_config()
    model = ASRDecoder(cfg, encoder_dim=ENCODER_DIM)
    enc = torch.randn(1, T, ENCODER_DIM)
    results = model.transcribe(enc)
    r = results[0]
    if r.token_ids:
        expected = " ".join(str(t) for t in r.token_ids)
        assert r.text == expected

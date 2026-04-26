"""Tests for src/inference/fusion_in_decoder.py -- Fusion-in-Decoder."""

import pytest
import torch

from src.inference.fusion_in_decoder import (
    FiDConfig,
    FusionInDecoder,
    encode_passages,
    fuse_representations,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(42)
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fid_cfg():
    return FiDConfig(n_passages=2, max_passage_len=8, max_answer_len=3, fusion_layer=-1)


@pytest.fixture(scope="module")
def fid(small_model, fid_cfg):
    return FusionInDecoder(small_model, fid_cfg)


def _make_passage_ids(n_passages, length=8, vocab_size=256):
    torch.manual_seed(7)
    return [torch.randint(0, vocab_size, (1, length)) for _ in range(n_passages)]


def _make_question_ids(length=4, vocab_size=256):
    torch.manual_seed(13)
    return torch.randint(0, vocab_size, (1, length))


# ---------------------------------------------------------------------------
# Test 1: FiDConfig defaults
# ---------------------------------------------------------------------------


def test_fidconfig_defaults():
    cfg = FiDConfig()
    assert cfg.n_passages == 5
    assert cfg.max_passage_len == 32
    assert cfg.max_answer_len == 16
    assert cfg.fusion_layer == -1


# ---------------------------------------------------------------------------
# Test 2: encode_passages returns correct shape (1, T, d_model)
# ---------------------------------------------------------------------------


def test_encode_passages_shape(small_model, small_cfg):
    T = 8
    passages = _make_passage_ids(n_passages=2, length=T)
    fused = encode_passages(small_model, passages)
    assert fused.shape == (1, T, small_cfg.d_model)


# ---------------------------------------------------------------------------
# Test 3: encode_passages with single passage works
# ---------------------------------------------------------------------------


def test_encode_passages_single_passage(small_model, small_cfg):
    T = 5
    passages = _make_passage_ids(n_passages=1, length=T)
    fused = encode_passages(small_model, passages)
    assert fused.shape == (1, T, small_cfg.d_model)


# ---------------------------------------------------------------------------
# Test 4: fuse_representations shape correct
# ---------------------------------------------------------------------------


def test_fuse_representations_shape():
    T, D = 8, 64
    hs_list = [torch.randn(1, T, D) for _ in range(3)]
    fused = fuse_representations(hs_list)
    assert fused.shape == (1, T, D)


# ---------------------------------------------------------------------------
# Test 5: fuse_representations mean of two identical == same as one
# ---------------------------------------------------------------------------


def test_fuse_representations_mean_identical():
    T, D = 6, 32
    h = torch.randn(1, T, D)
    fused = fuse_representations([h, h])
    assert torch.allclose(fused, h, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: FusionInDecoder.encode returns (1, T, d_model)
# ---------------------------------------------------------------------------


def test_fid_encode_shape(fid, small_cfg):
    T = 8
    passages = _make_passage_ids(n_passages=2, length=T)
    out = fid.encode(passages)
    assert out.shape == (1, T, small_cfg.d_model)


# ---------------------------------------------------------------------------
# Test 7: FusionInDecoder.generate returns a Tensor
# ---------------------------------------------------------------------------


def test_fid_generate_returns_tensor(fid):
    question = _make_question_ids(length=3)
    passages = _make_passage_ids(n_passages=2, length=8)
    result = fid.generate(question, passages, max_new_tokens=3)
    assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# Test 8: FusionInDecoder.generate token ids in [0, vocab_size)
# ---------------------------------------------------------------------------


def test_fid_generate_token_ids_in_range(fid, small_cfg):
    question = _make_question_ids(length=3)
    passages = _make_passage_ids(n_passages=2, length=8)
    result = fid.generate(question, passages, max_new_tokens=3)
    assert result.numel() > 0
    assert (result >= 0).all() and (result < small_cfg.vocab_size).all()


# ---------------------------------------------------------------------------
# Test 9: FusionInDecoder.score_passages returns (n_passages,)
# ---------------------------------------------------------------------------


def test_fid_score_passages_shape(fid, fid_cfg):
    question = _make_question_ids(length=4)
    passages = _make_passage_ids(n_passages=fid_cfg.n_passages, length=8)
    scores = fid.score_passages(question, passages)
    assert scores.shape == (fid_cfg.n_passages,)


# ---------------------------------------------------------------------------
# Test 10: score_passages values are floats
# ---------------------------------------------------------------------------


def test_fid_score_passages_dtype(fid, fid_cfg):
    question = _make_question_ids(length=4)
    passages = _make_passage_ids(n_passages=fid_cfg.n_passages, length=8)
    scores = fid.score_passages(question, passages)
    assert scores.is_floating_point()


# ---------------------------------------------------------------------------
# Test 11: Different passages give different scores
# ---------------------------------------------------------------------------


def test_fid_score_passages_different(small_model, fid_cfg):
    fid_local = FusionInDecoder(small_model, fid_cfg)
    question = _make_question_ids(length=4)
    torch.manual_seed(0)
    p1 = torch.randint(0, 128, (1, 8))
    torch.manual_seed(99)
    p2 = torch.randint(128, 256, (1, 8))
    scores = fid_local.score_passages(question, [p1, p2])
    assert scores.shape == (2,)
    assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# Test 12: generate with 1 passage vs 3 passages both work
# ---------------------------------------------------------------------------


def test_fid_generate_1_and_3_passages(small_model):
    cfg = FiDConfig(n_passages=3, max_passage_len=8, max_answer_len=3)
    fid_local = FusionInDecoder(small_model, cfg)
    question = _make_question_ids(length=3)

    passages1 = _make_passage_ids(n_passages=1, length=8)
    result1 = fid_local.generate(question, passages1, max_new_tokens=2)
    assert isinstance(result1, torch.Tensor)

    passages3 = _make_passage_ids(n_passages=3, length=8)
    result3 = fid_local.generate(question, passages3, max_new_tokens=2)
    assert isinstance(result3, torch.Tensor)


# ---------------------------------------------------------------------------
# Test 13: encode with n_passages=1 and n_passages=3 give same shape
# ---------------------------------------------------------------------------


def test_fid_encode_shape_independent_of_n_passages(small_model, small_cfg):
    T = 8
    passages1 = _make_passage_ids(n_passages=1, length=T)
    passages3 = _make_passage_ids(n_passages=3, length=T)

    fused1 = encode_passages(small_model, passages1)
    fused3 = encode_passages(small_model, passages3)

    assert fused1.shape == fused3.shape == (1, T, small_cfg.d_model)


# ---------------------------------------------------------------------------
# Test 14: FiDConfig n_passages respected
# ---------------------------------------------------------------------------


def test_fidconfig_n_passages_respected():
    for n in [1, 3, 5, 10]:
        cfg = FiDConfig(n_passages=n)
        assert cfg.n_passages == n

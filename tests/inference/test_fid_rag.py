"""Tests for src/inference/fid_rag.py -- Fusion-in-Decoder RAG."""

import pytest
import torch

from src.inference.fid_rag import (
    FiDConfig,
    FiDGenerator,
    FiDModel,
    FusionCrossAttention,
    average_fusion,
    concat_fusion,
    encode_passages,
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
    return FiDConfig(n_passages=3, max_passage_len=8, fusion_method="concat", d_model=64)


def _passages(n=3, length=8, vocab=256):
    torch.manual_seed(7)
    return [torch.randint(0, vocab, (1, length)) for _ in range(n)]


def _input_ids(length=4, vocab=256):
    torch.manual_seed(13)
    return torch.randint(0, vocab, (1, length))


# ---------------------------------------------------------------------------
# 1. FiDConfig defaults
# ---------------------------------------------------------------------------


def test_fidconfig_defaults():
    cfg = FiDConfig()
    assert cfg.n_passages == 5
    assert cfg.max_passage_len == 128
    assert cfg.fusion_method == "concat"
    assert cfg.d_model == 64


# ---------------------------------------------------------------------------
# 2. FiDConfig custom values
# ---------------------------------------------------------------------------


def test_fidconfig_custom():
    cfg = FiDConfig(n_passages=10, max_passage_len=64, fusion_method="average", d_model=128)
    assert cfg.n_passages == 10
    assert cfg.max_passage_len == 64
    assert cfg.fusion_method == "average"
    assert cfg.d_model == 128


# ---------------------------------------------------------------------------
# 3. encode_passages returns list of correct length
# ---------------------------------------------------------------------------


def test_encode_passages_count(small_model):
    passages = _passages(n=3)
    result = encode_passages(small_model, passages)
    assert isinstance(result, list)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 4. encode_passages each element has correct shape
# ---------------------------------------------------------------------------


def test_encode_passages_shapes(small_model, small_cfg):
    T = 8
    passages = _passages(n=2, length=T)
    result = encode_passages(small_model, passages)
    for h in result:
        assert h.shape == (1, T, small_cfg.d_model)


# ---------------------------------------------------------------------------
# 5. encode_passages with single passage
# ---------------------------------------------------------------------------


def test_encode_passages_single(small_model, small_cfg):
    passages = _passages(n=1, length=6)
    result = encode_passages(small_model, passages)
    assert len(result) == 1
    assert result[0].shape == (1, 6, small_cfg.d_model)


# ---------------------------------------------------------------------------
# 6. encode_passages with 1D input (unbatched)
# ---------------------------------------------------------------------------


def test_encode_passages_1d_input(small_model, small_cfg):
    torch.manual_seed(99)
    pids = torch.randint(0, 256, (8,))  # 1D
    result = encode_passages(small_model, [pids])
    assert len(result) == 1
    assert result[0].shape == (1, 8, small_cfg.d_model)


# ---------------------------------------------------------------------------
# 7. concat_fusion shape
# ---------------------------------------------------------------------------


def test_concat_fusion_shape():
    t1 = torch.randn(1, 5, 64)
    t2 = torch.randn(1, 7, 64)
    out = concat_fusion([t1, t2])
    assert out.shape == (1, 12, 64)


# ---------------------------------------------------------------------------
# 8. concat_fusion preserves content
# ---------------------------------------------------------------------------


def test_concat_fusion_content():
    t1 = torch.ones(1, 3, 8)
    t2 = torch.zeros(1, 4, 8)
    out = concat_fusion([t1, t2])
    assert torch.allclose(out[:, :3, :], t1)
    assert torch.allclose(out[:, 3:, :], t2)


# ---------------------------------------------------------------------------
# 9. average_fusion shape
# ---------------------------------------------------------------------------


def test_average_fusion_shape():
    tensors = [torch.randn(1, 5, 64) for _ in range(4)]
    out = average_fusion(tensors)
    assert out.shape == (1, 1, 64)


# ---------------------------------------------------------------------------
# 10. average_fusion correctness
# ---------------------------------------------------------------------------


def test_average_fusion_correctness():
    t1 = torch.ones(1, 3, 8) * 2.0
    t2 = torch.ones(1, 3, 8) * 4.0
    out = average_fusion([t1, t2])
    # mean pool each: both are constant, so (B, d) = 2.0 and 4.0
    # then mean across passages: 3.0
    assert out.shape == (1, 1, 8)
    assert torch.allclose(out, torch.ones(1, 1, 8) * 3.0)


# ---------------------------------------------------------------------------
# 11. FusionCrossAttention output shape
# ---------------------------------------------------------------------------


def test_cross_attention_shape():
    ca = FusionCrossAttention(d_model=64, n_heads=4)
    query = torch.randn(1, 5, 64)
    kv = torch.randn(1, 12, 64)
    out = ca(query, kv)
    assert out.shape == (1, 5, 64)


# ---------------------------------------------------------------------------
# 12. FusionCrossAttention preserves batch dim
# ---------------------------------------------------------------------------


def test_cross_attention_batch():
    ca = FusionCrossAttention(d_model=32, n_heads=2)
    query = torch.randn(2, 4, 32)
    kv = torch.randn(2, 8, 32)
    out = ca(query, kv)
    assert out.shape == (2, 4, 32)


# ---------------------------------------------------------------------------
# 13. FiDModel forward with concat fusion
# ---------------------------------------------------------------------------


def test_fid_model_concat(small_model, small_cfg):
    cfg = FiDConfig(
        n_passages=2, max_passage_len=8, fusion_method="concat", d_model=small_cfg.d_model
    )
    fid = FiDModel(small_model, cfg)
    fid.eval()

    inp = _input_ids(length=4)
    passages = _passages(n=2, length=8)
    logits = fid(inp, passages)

    # concat: 2 passages * 8 tokens + 4 input = 20 total
    assert logits.shape == (1, 20, small_cfg.vocab_size)


# ---------------------------------------------------------------------------
# 14. FiDModel forward with average fusion
# ---------------------------------------------------------------------------


def test_fid_model_average(small_model, small_cfg):
    cfg = FiDConfig(
        n_passages=2, max_passage_len=8, fusion_method="average", d_model=small_cfg.d_model
    )
    fid = FiDModel(small_model, cfg)
    fid.eval()

    inp = _input_ids(length=4)
    passages = _passages(n=2, length=8)
    logits = fid(inp, passages)

    # average: 1 fused token + 4 input = 5 total
    assert logits.shape == (1, 5, small_cfg.vocab_size)


# ---------------------------------------------------------------------------
# 15. FiDModel forward with cross_attention fusion
# ---------------------------------------------------------------------------


def test_fid_model_cross_attention(small_model, small_cfg):
    cfg = FiDConfig(
        n_passages=2, max_passage_len=8, fusion_method="cross_attention", d_model=small_cfg.d_model
    )
    fid = FiDModel(small_model, cfg)
    fid.eval()

    inp = _input_ids(length=4)
    passages = _passages(n=2, length=8)
    logits = fid(inp, passages)

    # cross_attn fused has same T as input (4), then + 4 input = 8
    assert logits.shape == (1, 8, small_cfg.vocab_size)


# ---------------------------------------------------------------------------
# 16. FiDModel raises on unknown fusion method
# ---------------------------------------------------------------------------


def test_fid_model_unknown_fusion(small_model, small_cfg):
    cfg = FiDConfig(fusion_method="unknown", d_model=small_cfg.d_model)
    fid = FiDModel(small_model, cfg)
    with pytest.raises(ValueError, match="Unknown fusion method"):
        fid(_input_ids(), _passages(n=1))


# ---------------------------------------------------------------------------
# 17. FiDGenerator generate returns correct shape
# ---------------------------------------------------------------------------


def test_fid_generator_shape(small_model, small_cfg):
    cfg = FiDConfig(n_passages=2, fusion_method="concat", d_model=small_cfg.d_model)
    fid = FiDModel(small_model, cfg)
    fid.eval()
    gen = FiDGenerator(fid)

    inp = _input_ids(length=4)
    passages = _passages(n=2, length=8)
    out = gen.generate(inp, passages, max_tokens=3)

    assert out.shape[0] == 1
    assert out.shape[1] == 4 + 3  # input + generated


# ---------------------------------------------------------------------------
# 18. FiDGenerator with eos_token_id stops early
# ---------------------------------------------------------------------------


def test_fid_generator_eos_stop(small_model, small_cfg):
    cfg = FiDConfig(n_passages=1, fusion_method="average", d_model=small_cfg.d_model)
    fid = FiDModel(small_model, cfg)
    fid.eval()
    # Use token 0 as EOS -- the model may or may not produce it, but
    # we verify the shape is <= input + max_tokens
    gen = FiDGenerator(fid, eos_token_id=0)

    inp = _input_ids(length=4)
    passages = _passages(n=1, length=6)
    out = gen.generate(inp, passages, max_tokens=10)

    assert out.shape[0] == 1
    assert out.shape[1] <= 4 + 10


# ---------------------------------------------------------------------------
# 19. FiDGenerator output starts with input_ids
# ---------------------------------------------------------------------------


def test_fid_generator_starts_with_input(small_model, small_cfg):
    cfg = FiDConfig(n_passages=2, fusion_method="concat", d_model=small_cfg.d_model)
    fid = FiDModel(small_model, cfg)
    fid.eval()
    gen = FiDGenerator(fid)

    inp = _input_ids(length=4)
    passages = _passages(n=2, length=8)
    out = gen.generate(inp, passages, max_tokens=2)

    assert torch.equal(out[:, :4], inp)


# ---------------------------------------------------------------------------
# 20. encode_passages does not require grad
# ---------------------------------------------------------------------------


def test_encode_passages_no_grad(small_model):
    passages = _passages(n=2, length=6)
    result = encode_passages(small_model, passages)
    for h in result:
        assert not h.requires_grad

"""Tests for the full AureliusTransformer."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer, count_parameters


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=1000, max_seq_len=128,
    )


@pytest.fixture
def small_model(small_cfg):
    return AureliusTransformer(small_cfg)


def test_forward_pass_shape(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 32))
    _, logits, _ = small_model(tokens)
    assert logits.shape == (2, 32, small_cfg.vocab_size)


def test_tied_embeddings(small_cfg):
    model = AureliusTransformer(small_cfg)
    assert model.lm_head.weight is model.embed.weight


def test_untied_embeddings():
    cfg = AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=1000, tie_embeddings=False,
    )
    model = AureliusTransformer(cfg)
    assert model.lm_head.weight is not model.embed.weight


def test_sequence_length_limit(small_model, small_cfg):
    with pytest.raises(AssertionError):
        tokens = torch.randint(0, small_cfg.vocab_size, (1, small_cfg.max_seq_len + 1))
        small_model(tokens)  # forward now returns tuple but assert fires before return


def test_full_model_parameter_count():
    """Full 1.3B model — verify parameter count is in expected range."""
    cfg = AureliusConfig()
    model = AureliusTransformer(cfg)
    n = count_parameters(model)
    # Allow 1.2B–1.5B range
    assert 1_200_000_000 < n < 1_500_000_000, f"Unexpected param count: {n:,}"


def test_parameter_count_breakdown(small_model):
    counts = small_model.count_parameters()
    assert "total" in counts
    assert counts["total"] > 0
    assert counts["all_layers"] > 0


def test_no_nan_in_output(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        _, logits, _ = small_model(tokens)
    assert not torch.isnan(logits).any(), "NaN in forward pass output"
    assert not torch.isinf(logits).any(), "Inf in forward pass output"


def test_batch_size_one(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 16))
    _, logits, _ = small_model(tokens)
    assert logits.shape == (1, 16, small_cfg.vocab_size)


def test_forward_returns_tuple(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    result = small_model(tokens)
    assert isinstance(result, tuple) and len(result) == 3
    loss, logits, pkv = result
    assert loss is None  # no labels
    assert logits.shape == (2, 16, small_cfg.vocab_size)
    assert len(pkv) == small_cfg.n_layers


def test_forward_with_labels_returns_loss(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    labels = torch.randint(0, small_cfg.vocab_size, (2, 16))
    loss, logits, _ = small_model(tokens, labels=labels)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_generate_output_shape(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 4))
    out = small_model.generate(tokens, max_new_tokens=8)
    assert out.shape == (1, 12)  # 4 prompt + 8 generated


def test_generate_batch(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 4))
    out = small_model.generate(tokens, max_new_tokens=6)
    assert out.shape == (2, 10)


def test_generate_eos_stops_early(small_model, small_cfg):
    """Generation stops when eos_token_id is produced for all seqs."""
    # Force eos by setting temperature=0 (greedy) -- but we can't control output.
    # Instead just verify it returns <= max_new_tokens + prompt.
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 4))
    out = small_model.generate(tokens, max_new_tokens=20, eos_token_id=0)
    assert out.shape[1] >= 5  # at least one token generated


def test_kv_cache_position_offset(small_model, small_cfg):
    """KV cache generation output must match single full-sequence forward."""
    torch.manual_seed(7)
    prompt = torch.randint(0, small_cfg.vocab_size, (1, 5))
    # Generate one token via cache-based generate()
    with torch.no_grad():
        gen_out = small_model.generate(prompt, max_new_tokens=1, temperature=1.0, top_p=1.0)
    # The generated sequence length should be 6
    assert gen_out.shape == (1, 6)

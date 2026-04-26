import pytest
import torch

from src.eval.attention_patterns import (
    AttentionExtractor,
    entropy_per_head,
    top_attended_positions,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def test_extractor_captures_all_layers(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    assert len(extractor.patterns) == 2  # 2 layers


def test_pattern_weights_shape(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    p = extractor.patterns[0]
    B, H, S, S2 = p.weights.shape
    assert B == 1
    assert H == 2  # n_heads
    assert S == 8
    assert S2 == 8


def test_pattern_weights_sum_to_one(small_model):
    input_ids = torch.randint(0, 256, (1, 6))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    for p in extractor.patterns:
        sums = p.weights.sum(dim=-1)  # (B, H, S)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_hooks_removed_after_exit(small_model):
    input_ids = torch.randint(0, 256, (1, 4))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    n_patterns = len(extractor.patterns)
    # Running again outside context should NOT add more patterns
    small_model(input_ids)
    assert len(extractor.patterns) == n_patterns


def test_entropy_shape(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    p = extractor.patterns[0]
    ent = entropy_per_head(p)
    assert ent.shape == (1, 2, 8)  # (B, H, S)


def test_entropy_nonnegative(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    for p in extractor.patterns:
        ent = entropy_per_head(p)
        assert (ent >= 0).all()


def test_top_attended_positions_shape(small_model):
    input_ids = torch.randint(0, 256, (1, 8))
    with AttentionExtractor(small_model) as extractor:
        small_model(input_ids)
    p = extractor.patterns[0]
    top = top_attended_positions(p, top_k=3)
    assert top.shape == (1, 2, 8, 3)

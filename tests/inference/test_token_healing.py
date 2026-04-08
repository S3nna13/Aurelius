import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.token_healing import (
    TokenHealingConfig, get_valid_token_ids, heal_tokens,
    build_prefix_constrained_logits
)

# Minimal vocab for testing
VOCAB = {0: "hello", 1: "world", 2: "http", 3: "https", 4: "htt", 5: "he", 6: "the", 7: "end"}

@pytest.fixture
def small_model():
    cfg = AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                         head_dim=32, d_ff=128, vocab_size=8, max_seq_len=64)
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model

def test_get_valid_token_ids_matches_prefix():
    valid = get_valid_token_ids("htt", VOCAB)
    assert set(valid) == {2, 3, 4}  # "http", "https", "htt" all start with "htt"

def test_get_valid_token_ids_empty():
    valid = get_valid_token_ids("xyz", VOCAB)
    assert valid == []

def test_get_valid_token_ids_all_match():
    valid = get_valid_token_ids("", VOCAB)
    assert len(valid) == len(VOCAB)

def test_constrained_logits_blocks_invalid():
    logits = torch.zeros(8)
    masked = build_prefix_constrained_logits(logits, [2, 3])
    assert masked[0] == float("-inf")
    assert masked[2] == 0.0
    assert masked[3] == 0.0

def test_constrained_logits_empty_valid():
    logits = torch.ones(8)
    masked = build_prefix_constrained_logits(logits, [])
    assert (masked == float("-inf")).all()

def test_heal_tokens_shape(small_model):
    input_ids = torch.tensor([[2, 3, 0, 4]])  # (1, 4)
    result = heal_tokens(small_model, input_ids, "htt", VOCAB)
    assert result.shape == (1, 4)

def test_heal_tokens_healed_starts_with_prefix(small_model):
    """The healed last token must start with the partial string."""
    torch.manual_seed(42)
    input_ids = torch.tensor([[0, 1, 0, 4]])  # last token ID 4 = "htt"
    result = heal_tokens(small_model, input_ids, "htt", VOCAB)
    healed_token_id = result[0, -1].item()
    assert VOCAB[healed_token_id].startswith("htt")

def test_heal_tokens_no_valid_returns_unchanged(small_model):
    input_ids = torch.tensor([[0, 1, 2, 3]])
    result = heal_tokens(small_model, input_ids, "xyz", VOCAB)
    assert torch.equal(result, input_ids)

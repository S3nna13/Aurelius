"""Tests for contrastive search decoding."""
import torch
import pytest
from src.inference.contrastive_search import contrastive_search, ContrastiveConfig, _get_last_hidden
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_get_last_hidden_shape(small_model):
    """_get_last_hidden must return (d_model,) tensor."""
    ids = torch.randint(0, 256, (1, 8))
    h = _get_last_hidden(small_model, ids)
    assert h.shape == (64,)  # d_model=64
    assert torch.isfinite(h).all()


def test_contrastive_search_generates_tokens(small_model):
    """contrastive_search must produce more tokens than the prompt."""
    prompt = torch.randint(0, 256, (1, 4))
    cfg = ContrastiveConfig(k=3, alpha=0.6, max_new_tokens=6)
    output = contrastive_search(small_model, prompt, cfg)

    assert output.shape[0] == 1
    assert output.shape[1] > prompt.shape[1]
    assert output.shape[1] <= prompt.shape[1] + cfg.max_new_tokens


def test_contrastive_search_preserves_prompt(small_model):
    """Output must start with the original prompt."""
    prompt = torch.randint(0, 256, (1, 6))
    cfg = ContrastiveConfig(k=2, alpha=0.5, max_new_tokens=4)
    output = contrastive_search(small_model, prompt, cfg)
    assert torch.equal(output[:, :prompt.shape[1]], prompt)


def test_contrastive_search_eos_stops(small_model):
    """Generation must stop when eos_token_id is produced."""
    prompt = torch.randint(0, 256, (1, 4))
    # max_seq_len=64 for the small model; keep prompt+generated well within that limit
    cfg = ContrastiveConfig(k=2, alpha=0.6, max_new_tokens=20, eos_token_id=7)
    output = contrastive_search(small_model, prompt, cfg)
    # Must not exceed max_new_tokens
    assert output.shape[1] <= prompt.shape[1] + cfg.max_new_tokens


def test_alpha_zero_matches_greedy(small_model):
    """With alpha=0, contrastive search is equivalent to greedy (top-1) decoding."""
    torch.manual_seed(42)
    prompt = torch.randint(0, 256, (1, 4))
    cfg_cs = ContrastiveConfig(k=1, alpha=0.0, max_new_tokens=5)

    # Contrastive with alpha=0, k=1
    out_cs = contrastive_search(small_model, prompt, cfg_cs)

    # Manual greedy
    generated = prompt.clone()
    for _ in range(5):
        _, logits, _ = small_model(generated)
        next_tok = logits[0, -1].argmax().view(1, 1)
        generated = torch.cat([generated, next_tok], dim=1)

    assert torch.equal(out_cs, generated)

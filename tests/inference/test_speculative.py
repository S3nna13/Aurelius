"""Tests for speculative decoding."""

import torch

from src.inference.speculative import SpeculativeConfig, SpeculativeDecoder, _sample_from_logits
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def _make_model(n_layers=2, d_model=64):
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_sample_from_logits_valid_token():
    """_sample_from_logits must return a token in valid range."""
    logits = torch.randn(256)
    token = _sample_from_logits(logits, temperature=1.0, top_p=0.9)
    assert 0 <= token.item() < 256


def test_sample_from_logits_temperature():
    """Temperature 0.1 (sharp) should repeatedly pick near-argmax token."""
    logits = torch.zeros(256)
    logits[42] = 100.0  # huge logit
    torch.manual_seed(0)
    tokens = {_sample_from_logits(logits, temperature=0.1, top_p=1.0).item() for _ in range(10)}
    assert 42 in tokens  # dominant token should appear


def test_speculative_generates_tokens():
    """SpeculativeDecoder.generate must produce more tokens than the prompt."""
    draft = _make_model(n_layers=1)
    target = _make_model(n_layers=2)

    cfg = SpeculativeConfig(K=2, max_new_tokens=8, temperature=1.0, top_p=0.9)
    decoder = SpeculativeDecoder(draft, target, cfg)

    prompt = torch.randint(0, 256, (1, 4))
    output = decoder.generate(prompt)

    assert output.shape[0] == 1
    assert output.shape[1] > prompt.shape[1]
    assert output.shape[1] <= prompt.shape[1] + cfg.max_new_tokens + 1  # +1 for bonus tokens


def test_speculative_eos_stops_generation():
    """Generation must stop at eos_token_id."""
    draft = _make_model()
    target = _make_model()

    cfg = SpeculativeConfig(K=2, max_new_tokens=64, eos_token_id=5)
    decoder = SpeculativeDecoder(draft, target, cfg)

    torch.manual_seed(42)
    prompt = torch.randint(0, 256, (1, 4))
    output = decoder.generate(prompt)

    # Output must be finite length (not fill entire max_new_tokens)
    assert output.shape[1] <= prompt.shape[1] + cfg.max_new_tokens + 1


def test_speculative_preserves_prompt():
    """Output must start with the original prompt."""
    draft = _make_model()
    target = _make_model()

    cfg = SpeculativeConfig(K=2, max_new_tokens=6)
    decoder = SpeculativeDecoder(draft, target, cfg)

    prompt = torch.randint(0, 256, (1, 8))
    output = decoder.generate(prompt)

    assert torch.equal(output[:, : prompt.shape[1]], prompt)


def test_acceptance_rate_in_range():
    """acceptance_rate must return a value in [0, 1]."""
    draft = _make_model()
    target = _make_model()

    cfg = SpeculativeConfig(K=3, max_new_tokens=32)
    decoder = SpeculativeDecoder(draft, target, cfg)

    prompt = torch.randint(0, 256, (1, 4))
    rate = decoder.acceptance_rate(prompt, n_rounds=3)
    assert 0.0 <= rate <= 1.0

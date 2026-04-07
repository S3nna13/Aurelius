"""Tests for native SFT implementation."""
import pytest
import torch
from unittest.mock import MagicMock
from src.alignment.sft import format_chatml, tokenize_for_sft, _resolve_target_modules, _cosine_warmup, NativeSFTRunner, SFTConfig
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


def _make_fake_tokenizer(vocab_size=256):
    """Minimal tokenizer mock that encodes char by char."""
    tok = MagicMock()
    tok.encode = lambda text, **kw: [ord(c) % vocab_size for c in text]
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    return tok


def test_format_chatml_structure():
    """format_chatml must include all four special tokens."""
    result = format_chatml("sys", "user", "assistant")
    assert "<|system|>" in result
    assert "<|user|>" in result
    assert "<|assistant|>" in result
    assert "<|end|>" in result


def test_tokenize_sft_prompt_masked():
    """Prompt tokens must have labels=-100, response tokens must have labels=input_ids."""
    tok = _make_fake_tokenizer()
    # Build a string that has <|assistant|> in it
    text = format_chatml("sys", "hello", "world")
    result = tokenize_for_sft(text, tok, max_len=512)

    assert "input_ids" in result
    assert "labels" in result
    # At least some labels must be -100 (prompt)
    assert (result["labels"] == -100).any()
    # At least some labels must NOT be -100 (response)
    assert (result["labels"] != -100).any()


def test_tokenize_sft_truncation():
    """tokenize_for_sft must truncate to max_len."""
    tok = _make_fake_tokenizer()
    text = "a" * 1000  # very long text
    result = tokenize_for_sft(text, tok, max_len=32)
    assert result["input_ids"].shape[0] == 32
    assert result["labels"].shape[0] == 32


def test_resolve_target_modules():
    """_resolve_target_modules must return full dotted paths for matching submodules."""
    cfg = AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2, head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64)
    model = AureliusTransformer(cfg)

    paths = _resolve_target_modules(model, ("q_proj", "v_proj"))
    assert len(paths) > 0
    # All paths must end with q_proj or v_proj
    for p in paths:
        assert p.endswith("q_proj") or p.endswith("v_proj")
    # Should find one per layer (2 layers × 2 targets = 4)
    assert len(paths) == 4


def test_cosine_warmup_values():
    """_cosine_warmup must return 0 at step 0 and base_lr at warmup end."""
    import torch.optim as optim
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = optim.AdamW([param], lr=1e-3)

    # At step 0: lr should be ~0 (warmup start)
    lr0 = _cosine_warmup(optimizer, step=0, total_steps=100, warmup_steps=10)
    assert lr0 < 1e-4  # nearly zero

    # At step 10 (end of warmup): lr should be close to base_lr
    lr10 = _cosine_warmup(optimizer, step=10, total_steps=100, warmup_steps=10)
    assert lr10 > 9e-4  # close to 1e-3

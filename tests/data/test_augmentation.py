import pytest
import torch
from src.data.augmentation import RandomTokenMask, TokenDropout, SpanCorruption

def test_random_mask_shape_preserved():
    mask = RandomTokenMask(p=0.15, mask_id=1, vocab_size=256)
    x = torch.randint(2, 256, (20,))
    out = mask(x)
    assert out.shape == x.shape

def test_random_mask_uses_mask_id():
    mask = RandomTokenMask(p=1.0, mask_id=99, vocab_size=256)
    x = torch.zeros(10, dtype=torch.long)
    out = mask(x)
    assert (out == 99).all()

def test_random_mask_does_not_modify_input():
    mask = RandomTokenMask(p=0.5, mask_id=0, vocab_size=256)
    x = torch.randint(1, 256, (20,))
    original = x.clone()
    mask(x)
    assert torch.equal(x, original)

def test_token_dropout_shortens():
    drop = TokenDropout(p=0.5)
    torch.manual_seed(42)
    x = torch.randint(0, 100, (50,))
    out = drop(x)
    assert len(out) < len(x)
    assert len(out) >= 1

def test_token_dropout_p0_unchanged():
    drop = TokenDropout(p=0.0)
    x = torch.randint(0, 100, (20,))
    out = drop(x)
    assert torch.equal(out, x)

def test_token_dropout_p1_keeps_one():
    drop = TokenDropout(p=1.0)
    x = torch.randint(0, 100, (10,))
    out = drop(x)
    assert len(out) == 1

def test_span_corruption_shortens():
    sc = SpanCorruption(p=0.15, mean_span_length=3, sentinel_start=32000)
    torch.manual_seed(0)
    x = torch.randint(0, 1000, (100,))
    out = sc(x)
    assert len(out) < len(x)

def test_span_corruption_contains_sentinels():
    sc = SpanCorruption(p=0.5, mean_span_length=3, sentinel_start=32000)
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (50,))
    out = sc(x)
    assert (out >= 32000).any()

def test_span_corruption_empty_safe():
    sc = SpanCorruption(p=0.15, sentinel_start=32000)
    x = torch.tensor([5], dtype=torch.long)
    out = sc(x)
    assert len(out) >= 1

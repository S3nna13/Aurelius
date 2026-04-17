"""Tests for src/model/spacebyte.py (SpaceByte -- arXiv:2404.14408).

Coverage targets (14 tests):
 1.  Logits shape (B=1, T=arbitrary) -> (B, T, 256)
 2.  Gradient flow: all parameters receive finite gradients
 3.  Determinism: identical output under same seed
 4.  T=1 (single byte, single patch)
 5.  No spaces in input -> single patch covering all bytes
 6.  All spaces -> each byte is its own patch
 7.  find_patch_boundaries: correct indices for "hello world foo"
 8.  No NaN/Inf on zeros input (all-null bytes, 0x00)
 9.  No NaN/Inf on alternating bytes (0x00/0xFF)
10.  Global model processes exactly n_patches vectors
11.  Variable patch lengths (patches of size 1, 3, 7)
12.  Loss scalar returned when targets provided
13.  Byte vocab size = 256 (embedding table row count)
14.  Works with text-like bytes (ASCII words "hello world")
"""

from __future__ import annotations

import torch
import pytest

from src.model.spacebyte import SpaceByteConfig, SpaceByteModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed: int = 0, **kwargs) -> SpaceByteModel:
    torch.manual_seed(seed)
    cfg = SpaceByteConfig(**kwargs)
    m = SpaceByteModel(cfg)
    m.train(False)  # put in inference mode
    return m


def _bytes_tensor(text: str, batch: int = 1) -> torch.LongTensor:
    """Encode ASCII text to a (batch, T) LongTensor of byte ids."""
    ids = [b for b in text.encode("ascii")]
    t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)
    if batch > 1:
        t = t.expand(batch, -1)
    return t


# ---------------------------------------------------------------------------
# Test 1: Logits shape (B=1)
# ---------------------------------------------------------------------------

def test_logits_shape_b1():
    model = _make_model()
    x = _bytes_tensor("hello world foo")  # T=15, spaces at 5 and 11
    with torch.no_grad():
        logits = model(x)
    B, T, V = logits.shape
    assert B == 1
    assert T == 15
    assert V == 256


# ---------------------------------------------------------------------------
# Test 2: Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    model = _make_model()
    model.train(True)
    x = _bytes_tensor("hello world")
    targets = x.clone()
    loss, _ = model(x, targets=targets)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Test 3: Determinism under same seed
# ---------------------------------------------------------------------------

def test_determinism():
    model_a = _make_model(seed=42)
    model_b = _make_model(seed=42)
    x = _bytes_tensor("determinism check")
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b), "Same seed should produce identical outputs"


# ---------------------------------------------------------------------------
# Test 4: T=1 (single byte, single patch)
# ---------------------------------------------------------------------------

def test_single_byte():
    model = _make_model()
    # A non-space byte -> single patch of length 1
    x = torch.tensor([[0x41]], dtype=torch.long)  # 'A'
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 1, 256)


# ---------------------------------------------------------------------------
# Test 5: No spaces -> single patch covering all bytes
# ---------------------------------------------------------------------------

def test_no_spaces_single_patch():
    model = _make_model()
    text = "nospaces"  # no 0x20 -> one patch of length 8
    x = _bytes_tensor(text)
    boundaries = model.find_patch_boundaries(x[0])
    assert boundaries == [0], f"Expected single boundary [0], got {boundaries}"
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, len(text), 256)


# ---------------------------------------------------------------------------
# Test 6: All spaces -> each byte is its own patch
# ---------------------------------------------------------------------------

def test_all_spaces_each_is_patch():
    model = _make_model()
    T = 5
    x = torch.full((1, T), 0x20, dtype=torch.long)  # all spaces
    boundaries = model.find_patch_boundaries(x[0])
    # Byte 0 is always a boundary; bytes 1..T-1 are all spaces -> also boundaries
    expected = list(range(T))
    assert boundaries == expected, f"Expected {expected}, got {boundaries}"
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, T, 256)


# ---------------------------------------------------------------------------
# Test 7: find_patch_boundaries correct indices for "hello world foo"
# ---------------------------------------------------------------------------

def test_find_patch_boundaries_hello_world_foo():
    model = _make_model()
    text = "hello world foo"
    x = torch.tensor([b for b in text.encode("ascii")], dtype=torch.long)
    boundaries = model.find_patch_boundaries(x)
    space_positions = [i for i, c in enumerate(text) if c == ' ']
    expected = [0] + space_positions
    assert boundaries == expected, f"Expected {expected}, got {boundaries}"


# ---------------------------------------------------------------------------
# Test 8: No NaN/Inf on zeros input
# ---------------------------------------------------------------------------

def test_no_nan_inf_zeros():
    model = _make_model()
    x = torch.zeros(1, 8, dtype=torch.long)  # all 0x00 bytes
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf for zeros input"


# ---------------------------------------------------------------------------
# Test 9: No NaN/Inf on alternating bytes
# ---------------------------------------------------------------------------

def test_no_nan_inf_alternating():
    model = _make_model()
    ids = [0x00 if i % 2 == 0 else 0xFF for i in range(10)]
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf for alternating bytes"


# ---------------------------------------------------------------------------
# Test 10: Global model processes exactly n_patches vectors
# ---------------------------------------------------------------------------

def test_global_processes_n_patches():
    """Verify the global transformer hidden state has exactly n_patches positions."""
    model = _make_model()
    text = "one two three"  # spaces at 3 and 7 -> n_patches = 3
    x = _bytes_tensor(text)
    boundaries = model.find_patch_boundaries(x[0])
    n_patches_expected = len(boundaries)

    captured = {}

    def hook(module, inp, out):
        captured["shape"] = inp[0].shape  # (B, n_patches, d_g)

    handle = model.global_transformer.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    assert "shape" in captured
    assert captured["shape"][1] == n_patches_expected, (
        f"Global transformer saw {captured['shape'][1]} tokens, "
        f"expected n_patches={n_patches_expected}"
    )


# ---------------------------------------------------------------------------
# Test 11: Variable patch lengths handled (patches of size 7, 4, 1)
# ---------------------------------------------------------------------------

def test_variable_patch_lengths():
    """Build a sequence whose patches have lengths 7, 4, 1.

    ids: [0x41]*7 + [0x20] + [0x42]*3 + [0x20] + [0x43]
    patch_byte=0x20: boundaries at 0, 7, 11.
    Patch 0: [0..7)  = 7 bytes
    Patch 1: [7..11) = 4 bytes (0x20 + 3x0x42)
    Patch 2: [11..12)= 1 byte  (0x20)
    """
    model = _make_model()
    ids = [0x41] * 7 + [0x20] + [0x42] * 3 + [0x20] + [0x43]
    x = torch.tensor([ids], dtype=torch.long)
    T = len(ids)
    boundaries = model.find_patch_boundaries(x[0])
    assert 0 in boundaries
    assert 7 in boundaries   # first space
    assert 11 in boundaries  # second space
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, T, 256)


# ---------------------------------------------------------------------------
# Test 12: Loss scalar returned when targets provided
# ---------------------------------------------------------------------------

def test_loss_scalar_with_targets():
    model = _make_model()
    model.train(True)
    x = _bytes_tensor("hello world")
    targets = x.clone()
    result = model(x, targets=targets)
    assert isinstance(result, tuple) and len(result) == 2
    loss, logits = result
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss is non-finite"
    B, T, V = logits.shape
    assert V == 256


# ---------------------------------------------------------------------------
# Test 13: Byte vocab size = 256
# ---------------------------------------------------------------------------

def test_vocab_size_256():
    model = _make_model()
    assert model.byte_embed.num_embeddings == 256, (
        f"Expected embedding table with 256 rows, "
        f"got {model.byte_embed.num_embeddings}"
    )
    assert model.output_proj.out_features == 256, (
        f"Expected output projection with 256 outputs, "
        f"got {model.output_proj.out_features}"
    )


# ---------------------------------------------------------------------------
# Test 14: Works with text-like bytes (ASCII words)
# ---------------------------------------------------------------------------

def test_text_like_bytes_ascii():
    model = _make_model()
    texts = [
        "the quick brown fox",
        "SpaceByte rocks",
        "a b c d e",
    ]
    for text in texts:
        x = _bytes_tensor(text)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape[0] == 1
        assert logits.shape[1] == len(text.encode("ascii"))
        assert logits.shape[2] == 256
        assert torch.isfinite(logits).all(), f"Non-finite logits for text: {repr(text)}"
